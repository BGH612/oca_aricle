from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
from catboost import CatBoostClassifier, Pool
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


DATASETS = {
    "baseline": Path("datasets/merged_base.csv"),
    "oca": Path("datasets/merged_oca.csv"),
    "ta": Path("datasets/merged_ta.csv"),
    "oca_ta": Path("datasets/merged_oca_ta.csv"),
}

FLOW_FEATURES = {
    "inflow",
    "outflow",
    "net_flow",
    "inflow_rel",
    "outflow_rel",
    "inflow_ratio",
    "outflow_ratio",
    "rolling_netflow_3h",
    "rolling_netflow_24h",
    "rolling_netflow_3h_normed",
    "rolling_netflow_24h_normed",
    "rolling_inflow_mean_spike",
    "rolling_outflow_mean_spike",
    "rolling_netflow_mean_spike",
    "spike_inflow",
    "spike_outflow",
    "spike_netflow",
}

WHALE_FEATURES = {
    "whale_tx_count",
    "whale_volume",
    "whale_volume_share",
    "avg_whale_amount",
    "rolling_whale_share_mean_spike",
    "spike_whales",
}

GENERAL_VOLUME_FEATURES = {
    "tx_count",
    "total_volume",
    "mean_amount",
    "std_amount",
    "median_amount",
    "max_amount",
    "min_amount",
    "avg_tx_amount",
    "rolling_volume_3h",
    "rolling_volume_24h",
    "rolling_vol_mean_spike",
    "spike_volume",
}

SPIKE_FEATURES = {
    "rolling_tx_mean_spike",
    "rolling_vol_mean_spike",
    "rolling_netflow_mean_spike",
    "rolling_inflow_mean_spike",
    "rolling_outflow_mean_spike",
    "rolling_addresses_mean_spike",
    "rolling_whale_share_mean_spike",
    "spike_tx",
    "spike_volume",
    "spike_netflow",
    "spike_inflow",
    "spike_outflow",
    "spike_unique_addresses",
    "spike_whales",
}


@dataclass
class Config:
    date_col: str = "date"
    close_col: str = "close"
    train_end_date: str = "2024-11-01"
    test_start_date: str = "2024-11-01"
    test_end_date: str = "2025-11-01"
    corr_threshold: float = 0.85
    score_threshold: float = 0.55
    min_features: int = 15
    cv_splits: int = 4
    cv_gap: int = 7
    lookback: int = 30
    epochs: int = 10
    batch_size: int = 64
    random_state: int = 42
    fee: float = 0.002


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_scores(s: pd.Series) -> pd.Series:
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if len(s) == 0 or s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def prepare_dataset(path: Path, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)
    numeric = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    keep = [cfg.date_col] + numeric
    df = df[keep].copy()
    df[numeric] = df[numeric].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    df["future_return"] = df[cfg.close_col].shift(-1) / df[cfg.close_col] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    df = df.iloc[:-1].copy().reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
    exclude = {cfg.date_col, cfg.close_col, "future_return", "target"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def masks(df: pd.DataFrame, cfg: Config) -> Tuple[pd.Series, pd.Series]:
    train = df[cfg.date_col] < pd.Timestamp(cfg.train_end_date)
    test = (df[cfg.date_col] >= pd.Timestamp(cfg.test_start_date)) & (
        df[cfg.date_col] < pd.Timestamp(cfg.test_end_date)
    )
    return train, test


def cat_model(seed: int, iterations: int = 180) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=iterations,
        depth=5,
        learning_rate=0.045,
        l2_leaf_reg=6,
        loss_function="Logloss",
        eval_metric="F1",
        verbose=False,
        random_seed=seed,
        allow_writing_files=False,
    )


def xgb_model(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=220,
        max_depth=3,
        learning_rate=0.035,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_alpha=1.0,
        reg_lambda=4.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )


def predictive_power_scores(X: pd.DataFrame, y: pd.Series, seed: int) -> pd.Series:
    from sklearn.feature_selection import mutual_info_classif

    X = safe_fill(X)
    vals = mutual_info_classif(X, y, discrete_features=False, random_state=seed)
    return pd.Series(vals, index=X.columns)


def cluster_representatives(X: pd.DataFrame, y: pd.Series, cfg: Config) -> Tuple[List[str], Dict[int, List[str]]]:
    X = safe_fill(X)
    keep = X.columns[X.nunique(dropna=False) > 1].tolist()
    X = X[keep]
    if len(keep) <= 1:
        return keep, {0: keep}
    corr = X.corr(method="spearman").abs().fillna(0.0)
    dist = 1 - corr
    dist_values = dist.values.copy()
    np.fill_diagonal(dist_values, 0.0)
    Z = linkage(squareform(dist_values, checks=False), method="average")
    labels = fcluster(Z, t=1 - cfg.corr_threshold, criterion="distance")
    pp = predictive_power_scores(X, y, cfg.random_state)
    reps: List[str] = []
    clusters: Dict[int, List[str]] = {}
    for cl in sorted(np.unique(labels)):
        cols = X.columns[labels == cl].tolist()
        clusters[int(cl)] = cols
        reps.append(pp[cols].sort_values(ascending=False).index[0])
    return reps, clusters


def stability_scores(X: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.Series:
    counts = pd.Series(0.0, index=X.columns)
    tscv = TimeSeriesSplit(n_splits=cfg.cv_splits, gap=cfg.cv_gap)
    for tr, val in tscv.split(X):
        Xtr, Xval = X.iloc[tr], X.iloc[val]
        ytr, yval = y.iloc[tr], y.iloc[val]
        model = cat_model(cfg.random_state, iterations=120)
        model.fit(Xtr, ytr)
        try:
            perm = permutation_importance(
                model,
                Xval,
                yval,
                n_repeats=4,
                random_state=cfg.random_state,
                scoring="f1",
                n_jobs=1,
            )
            imp = normalize_scores(pd.Series(perm.importances_mean, index=X.columns))
        except Exception:
            imp = normalize_scores(pd.Series(model.get_feature_importance(), index=X.columns))
        counts += (imp >= 0.5).astype(float)
    return counts / cfg.cv_splits


def shap_scores(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.Series:
    sample = X.iloc[: min(len(X), 350)].copy()
    try:
        pool = Pool(sample, y.iloc[: len(sample)])
        vals = model.get_feature_importance(pool, type="ShapValues")
        imp = np.abs(vals[:, :-1]).mean(axis=0)
        return pd.Series(imp, index=X.columns)
    except Exception:
        return pd.Series(model.get_feature_importance(), index=X.columns)


def cv_score_for_prefix(X: pd.DataFrame, y: pd.Series, features: List[str], cfg: Config) -> float:
    scores = []
    tscv = TimeSeriesSplit(n_splits=cfg.cv_splits, gap=cfg.cv_gap)
    for tr, val in tscv.split(X):
        model = cat_model(cfg.random_state, iterations=120)
        model.fit(X.iloc[tr][features], y.iloc[tr])
        pred = model.predict(X.iloc[val][features]).astype(int)
        scores.append(f1_score(y.iloc[val], pred, zero_division=0))
    return float(np.mean(scores)) if scores else 0.0


def select_features(X: pd.DataFrame, y: pd.Series, cfg: Config) -> Tuple[List[str], pd.DataFrame, Dict[int, List[str]]]:
    reps, clusters = cluster_representatives(X, y, cfg)
    Xr = safe_fill(X[reps].copy())
    base = cat_model(cfg.random_state, iterations=180)
    base.fit(Xr, y)

    shap_s = shap_scores(base, Xr, y, cfg)
    split_idx = max(1, int(len(Xr) * 0.8))
    Xtr, Xval = Xr.iloc[:split_idx], Xr.iloc[split_idx:]
    ytr, yval = y.iloc[:split_idx], y.iloc[split_idx:]
    perm_model = cat_model(cfg.random_state, iterations=140)
    perm_model.fit(Xtr, ytr)
    try:
        perm = permutation_importance(
            perm_model, Xval, yval, n_repeats=6, random_state=cfg.random_state, scoring="f1", n_jobs=1
        )
        perm_s = pd.Series(perm.importances_mean, index=Xr.columns)
    except Exception:
        perm_s = pd.Series(0.0, index=Xr.columns)

    stab_s = stability_scores(Xr, y, cfg)
    uni_s = predictive_power_scores(Xr, y, cfg.random_state)

    scores = pd.DataFrame(
        {
            "shap": normalize_scores(shap_s),
            "permutation": normalize_scores(perm_s),
            "stability": normalize_scores(stab_s),
            "univariate": normalize_scores(uni_s),
        }
    )
    scores["aggregate_score"] = scores[["shap", "permutation", "stability", "univariate"]].mean(axis=1)
    for col in ["shap", "permutation", "stability", "univariate", "aggregate_score"]:
        scores[f"rank_{col}"] = scores[col].rank(ascending=False, method="average")
    scores["aggregate_rank"] = scores[
        ["rank_shap", "rank_permutation", "rank_stability", "rank_univariate", "rank_aggregate_score"]
    ].mean(axis=1)
    scores = scores.sort_values(["aggregate_score", "aggregate_rank"], ascending=[False, True])

    threshold_selected = scores.index[scores["aggregate_score"] >= cfg.score_threshold].tolist()
    if len(threshold_selected) < cfg.min_features:
        threshold_selected = scores.head(min(cfg.min_features, len(scores))).index.tolist()

    ordered = scores.index.tolist()
    candidate_sizes = sorted(
        set(
            [
                len(threshold_selected),
                cfg.min_features,
                min(20, len(ordered)),
                min(30, len(ordered)),
                min(45, len(ordered)),
                len(threshold_selected),
            ]
        )
    )
    candidate_sizes = [s for s in candidate_sizes if 1 <= s <= len(ordered)]
    best_features = threshold_selected
    best_cv = -1.0
    cv_by_size = {}
    for size in candidate_sizes:
        feats = ordered[:size]
        sc = cv_score_for_prefix(Xr, y, feats, cfg)
        cv_by_size[size] = sc
        if sc > best_cv:
            best_cv = sc
            best_features = feats
    scores["selected"] = scores.index.isin(best_features)
    scores["cv_best_f1"] = best_cv
    scores["cv_by_size"] = json.dumps(cv_by_size)
    return best_features, scores.reset_index(names="feature"), clusters


def binary_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    return out


class SequenceNet(nn.Module):
    def __init__(self, input_size: int, hidden: int = 48, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def build_sequences(X: pd.DataFrame, y: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    xv = X.values.astype(np.float32)
    yv = y.values.astype(np.float32)
    Xs, ys, idx = [], [], []
    for i in range(lookback - 1, len(X)):
        Xs.append(xv[i - lookback + 1 : i + 1])
        ys.append(yv[i])
        idx.append(X.index[i])
    return np.array(Xs), np.array(ys), idx


def fit_sequence_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, features: List[str], cfg: Config, bidirectional: bool
) -> Tuple[np.ndarray, np.ndarray, List[int], SequenceNet, StandardScaler]:
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(X_train[features]), columns=features, index=X_train.index)
    Xte = pd.DataFrame(scaler.transform(X_test[features]), columns=features, index=X_test.index)

    split = max(cfg.lookback + 1, int(len(Xtr) * 0.85))
    X_sub, y_sub, _ = build_sequences(Xtr.iloc[:split], y_train.iloc[:split], cfg.lookback)
    X_val_ctx = pd.concat([Xtr.iloc[:split].tail(cfg.lookback - 1), Xtr.iloc[split:]], axis=0)
    y_val_ctx = pd.concat([y_train.iloc[:split].tail(cfg.lookback - 1), y_train.iloc[split:]], axis=0)
    X_val, y_val, _ = build_sequences(X_val_ctx, y_val_ctx, cfg.lookback)

    model = SequenceNet(len(features), bidirectional=bidirectional)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_sub), torch.tensor(y_sub)), batch_size=cfg.batch_size, shuffle=False
    )
    best_state = None
    best_val = np.inf
    patience = 3
    stale = 0
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(torch.tensor(X_val)), torch.tensor(y_val)).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state:
        model.load_state_dict(best_state)

    X_context = pd.concat([Xtr.tail(cfg.lookback - 1), Xte], axis=0)
    y_context = pd.concat([y_train.tail(cfg.lookback - 1), y_test], axis=0)
    X_seq, y_seq, idx = build_sequences(X_context, y_context, cfg.lookback)
    test_idx = set(X_test.index)
    keep = [i for i, real_idx in enumerate(idx) if real_idx in test_idx]
    X_seq = X_seq[keep]
    y_seq = y_seq[keep]
    idx = [idx[i] for i in keep]
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(X_seq))).numpy()
    pred = (prob >= 0.5).astype(int)
    return prob, pred, idx, model, scaler


def fit_tabular_model(model_name: str, X_train, y_train, X_test, cfg: Config) -> Tuple[np.ndarray, np.ndarray, object]:
    if model_name == "catboost":
        model = cat_model(cfg.random_state, iterations=260)
    elif model_name == "xgboost":
        model = xgb_model(cfg.random_state)
    else:
        raise ValueError(model_name)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return prob, pred, model


def benchmark_predictions(df: pd.DataFrame, cfg: Config) -> Dict[str, pd.Series]:
    prev_ret = df[cfg.close_col].pct_change()
    out = {
        "momentum": (prev_ret > 0).astype(int),
        "always_up": pd.Series(1, index=df.index),
    }
    rng = np.random.default_rng(cfg.random_state)
    out["random"] = pd.Series(rng.integers(0, 2, len(df)), index=df.index)
    return out


def equity_curve(eval_df: pd.DataFrame, signal_col: str, cfg: Config) -> pd.DataFrame:
    out = eval_df[["date", "future_return", signal_col]].copy()
    sig = out[signal_col].fillna(0).astype(int)
    position = np.where(sig > 0, 1, -1)
    trades = pd.Series(position, index=out.index).diff().abs().fillna(0)
    out["strategy_return"] = position * out["future_return"].fillna(0) - trades * cfg.fee
    out["equity"] = 1 + out["strategy_return"].cumsum()
    out["position"] = position
    return out


def equity_stats(eq: pd.DataFrame) -> Dict[str, float]:
    r = eq["strategy_return"].fillna(0)
    total = eq["equity"].iloc[-1] - 1 if len(eq) else np.nan
    sharpe = (r.mean() / r.std()) * np.sqrt(365) if r.std() and not np.isnan(r.std()) else np.nan
    peak = eq["equity"].cummax()
    dd = (eq["equity"] / peak - 1).min()
    return {"total_return": total, "sharpe": sharpe, "max_drawdown": dd}


def label_regimes(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df[[cfg.date_col, cfg.close_col, "future_return", "target"]].copy()
    ret30 = out[cfg.close_col].pct_change(30)
    vol20 = out[cfg.close_col].pct_change().rolling(20).std()
    trend_thr = ret30.rolling(365, min_periods=60).std().fillna(ret30.std()) * 0.35
    out["trend_regime"] = np.select(
        [ret30 > trend_thr, ret30 < -trend_thr], ["bull", "bear"], default="sideways"
    )
    vol_med = vol20.rolling(365, min_periods=60).median().fillna(vol20.median())
    out["vol_regime"] = np.where(vol20 > vol_med, "high_vol", "low_vol")
    out["market_regime"] = out["trend_regime"] + "_" + out["vol_regime"]
    out["regime_transition"] = (out["market_regime"] != out["market_regime"].shift(1)).astype(int)
    return out


def model_importance(model_name: str, model, X: pd.DataFrame, y: pd.Series, features: List[str], cfg: Config) -> Tuple[pd.Series, pd.Series]:
    if len(X) < 20:
        empty = pd.Series(np.nan, index=features)
        return empty, empty
    if model_name == "catboost":
        try:
            pool = Pool(X[features], y)
            vals = model.get_feature_importance(pool, type="ShapValues")
            shap_imp = pd.Series(np.abs(vals[:, :-1]).mean(axis=0), index=features)
        except Exception:
            shap_imp = pd.Series(model.get_feature_importance(), index=features)
        pred_model = model
    elif model_name == "xgboost":
        try:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X[features])
            shap_imp = pd.Series(np.abs(vals).mean(axis=0), index=features)
        except Exception:
            shap_imp = pd.Series(model.feature_importances_, index=features)
        pred_model = model
    else:
        empty = pd.Series(np.nan, index=features)
        return empty, empty

    try:
        perm = permutation_importance(
            pred_model, X[features], y, n_repeats=5, random_state=cfg.random_state, scoring="f1", n_jobs=1
        )
        perm_imp = pd.Series(perm.importances_mean, index=features)
    except Exception:
        perm_imp = pd.Series(np.nan, index=features)
    return shap_imp, perm_imp


def run_dataset(name: str, path: Path, cfg: Config, out_dir: Path) -> Tuple[List[dict], List[dict], List[dict], List[pd.DataFrame]]:
    df = prepare_dataset(path, cfg)
    train_mask, test_mask = masks(df, cfg)
    feats = feature_columns(df, cfg)
    X = safe_fill(df[feats])
    y = df["target"].copy()
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    selected, scores, clusters = select_features(X_train, y_train, cfg)
    scores.insert(0, "dataset", name)
    scores.to_csv(out_dir / f"{name}_feature_selection_scores.csv", index=False)
    pd.DataFrame(
        [{"cluster": k, "features": ",".join(v), "representative": next((s for s in selected if s in v), "")} for k, v in clusters.items()]
    ).to_csv(out_dir / f"{name}_feature_clusters.csv", index=False)

    metrics_rows: List[dict] = []
    equity_rows: List[dict] = []
    regime_rows: List[dict] = []
    importance_rows: List[dict] = []
    equity_frames: List[pd.DataFrame] = []
    pred_store: Dict[str, pd.DataFrame] = {}
    fitted_models: Dict[str, object] = {}

    for model_name in ["catboost", "xgboost"]:
        prob, pred, model = fit_tabular_model(model_name, X_train[selected], y_train, X_test[selected], cfg)
        fitted_models[model_name] = model
        eval_df = df.loc[test_mask, ["date", "future_return", "target"]].copy()
        eval_df[f"{model_name}_prob"] = prob
        eval_df[f"{model_name}_pred"] = pred
        pred_store[model_name] = eval_df
        m = binary_metrics(eval_df["target"], pred, prob)
        metrics_rows.append({"dataset": name, "model": model_name, "n_features": len(selected), **m})
        eq = equity_curve(eval_df.rename(columns={f"{model_name}_pred": "signal"}), "signal", cfg)
        st = equity_stats(eq)
        equity_rows.append({"dataset": name, "model": model_name, **st})
        eq = eq.rename(columns={"equity": model_name})
        equity_frames.append(eq[["date", model_name]])

    for model_name, bidir in [("lstm", False), ("bilstm", True)]:
        prob, pred, idx, model, scaler = fit_sequence_model(X_train, y_train, X_test, y_test, selected, cfg, bidir)
        eval_df = df.loc[idx, ["date", "future_return", "target"]].copy()
        eval_df[f"{model_name}_prob"] = prob
        eval_df[f"{model_name}_pred"] = pred
        pred_store[model_name] = eval_df
        m = binary_metrics(eval_df["target"], pred, prob)
        metrics_rows.append({"dataset": name, "model": model_name, "n_features": len(selected), **m})
        eq = equity_curve(eval_df.rename(columns={f"{model_name}_pred": "signal"}), "signal", cfg)
        st = equity_stats(eq)
        equity_rows.append({"dataset": name, "model": model_name, **st})
        eq = eq.rename(columns={"equity": model_name})
        equity_frames.append(eq[["date", model_name]])

    bench = benchmark_predictions(df, cfg)
    for bname, sig in bench.items():
        eval_df = df.loc[test_mask, ["date", "future_return", "target"]].copy()
        eval_df["pred"] = sig.loc[test_mask].fillna(0).astype(int).values
        eval_df["prob"] = np.where(eval_df["pred"] == 1, 0.55, 0.45)
        m = binary_metrics(eval_df["target"], eval_df["pred"], eval_df["prob"])
        metrics_rows.append({"dataset": name, "model": bname, "n_features": 0, **m})
        eq = equity_curve(eval_df.rename(columns={"pred": "signal"}), "signal", cfg)
        st = equity_stats(eq)
        equity_rows.append({"dataset": name, "model": bname, **st})
        eq = eq.rename(columns={"equity": bname})
        equity_frames.append(eq[["date", bname]])

    regimes = label_regimes(df, cfg)
    for model_name, pred_df in pred_store.items():
        pred_col = f"{model_name}_pred"
        prob_col = f"{model_name}_prob"
        joined = pred_df.merge(regimes[["date", "trend_regime", "vol_regime", "market_regime", "regime_transition"]], on="date", how="left")
        for regime_col in ["trend_regime", "vol_regime", "market_regime"]:
            for regime, grp in joined.groupby(regime_col):
                if len(grp) < 15:
                    continue
                m = binary_metrics(grp["target"], grp[pred_col], grp[prob_col])
                regime_rows.append({"dataset": name, "model": model_name, "regime_type": regime_col, "regime": regime, "n": len(grp), **m})
        for trans_name, grp in [("transition", joined[joined["regime_transition"] == 1]), ("stable", joined[joined["regime_transition"] == 0])]:
            if len(grp) >= 15:
                m = binary_metrics(grp["target"], grp[pred_col], grp[prob_col])
                regime_rows.append({"dataset": name, "model": model_name, "regime_type": "transition_layer", "regime": trans_name, "n": len(grp), **m})

    for model_name in ["catboost", "xgboost"]:
        model = fitted_models[model_name]
        eval_base = X_test[selected].copy()
        y_base = y_test.copy()
        joined_idx = df.loc[test_mask, ["date"]].merge(regimes[["date", "market_regime"]], on="date", how="left")
        joined_idx.index = X_test.index
        for regime, idxs in joined_idx.groupby("market_regime").groups.items():
            if len(idxs) < 20:
                continue
            shap_imp, perm_imp = model_importance(model_name, model, eval_base.loc[idxs], y_base.loc[idxs], selected, cfg)
            tmp = pd.DataFrame({"feature": selected, "shap": shap_imp.reindex(selected).values, "permutation": perm_imp.reindex(selected).values})
            tmp = tmp.sort_values("shap", ascending=False).head(20)
            for _, row in tmp.iterrows():
                importance_rows.append({"dataset": name, "model": model_name, "regime": regime, **row.to_dict()})

    merged_eq = None
    for frame in equity_frames:
        merged_eq = frame if merged_eq is None else merged_eq.merge(frame, on="date", how="outer")
    if merged_eq is not None:
        merged_eq = merged_eq.sort_values("date")
        merged_eq.to_csv(out_dir / f"{name}_equity_curves.csv", index=False)
        plt.figure(figsize=(12, 6))
        for col in [c for c in merged_eq.columns if c != "date"]:
            plt.plot(pd.to_datetime(merged_eq["date"]), merged_eq[col], label=col, linewidth=1.4)
        plt.title(f"Equity curves: {name}")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_equity_curves.png", dpi=150)
        plt.close()

    return metrics_rows, equity_rows, regime_rows, importance_rows


def hypothesis_tables(out_dir: Path) -> None:
    fs = pd.concat([pd.read_csv(p) for p in out_dir.glob("*_feature_selection_scores.csv")], ignore_index=True)
    metrics = pd.read_csv(out_dir / "model_metrics.csv")
    equity = pd.read_csv(out_dir / "equity_stats.csv")
    regime_imp = pd.read_csv(out_dir / "regime_feature_importance.csv")

    selected = fs[fs["selected"] == True].copy()
    rows = []
    for ds, grp in selected.groupby("dataset"):
        feats = set(grp["feature"])
        rows.append(
            {
                "dataset": ds,
                "selected_n": len(feats),
                "flow_selected": len(feats & FLOW_FEATURES),
                "whale_selected": len(feats & WHALE_FEATURES),
                "spike_selected": len(feats & SPIKE_FEATURES),
                "general_volume_selected": len(feats & GENERAL_VOLUME_FEATURES),
                "selected_features": ", ".join(grp.sort_values("aggregate_score", ascending=False)["feature"].head(25)),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "hypothesis_feature_groups.csv", index=False)

    perf = metrics.pivot_table(index="model", columns="dataset", values="f1", aggfunc="mean")
    perf.to_csv(out_dir / "hypothesis_h2_model_dataset_f1.csv")

    if not regime_imp.empty:
        imp = regime_imp.copy()
        imp["is_onchain_key"] = imp["feature"].isin(FLOW_FEATURES | WHALE_FEATURES | SPIKE_FEATURES)
        reg = imp.groupby(["dataset", "model", "regime"])["is_onchain_key"].mean().reset_index()
        reg.to_csv(out_dir / "hypothesis_h3_onchain_importance_by_regime.csv", index=False)

    comp = selected.assign(
        group=np.select(
            [
                selected["feature"].isin(WHALE_FEATURES),
                selected["feature"].isin(GENERAL_VOLUME_FEATURES),
                selected["feature"].isin(SPIKE_FEATURES),
            ],
            ["whale", "general_volume", "spike"],
            default="other",
        )
    )
    comp.groupby(["dataset", "group"])["aggregate_score"].agg(["count", "mean", "max"]).reset_index().to_csv(
        out_dir / "hypothesis_h4_h5_group_scores.csv", index=False
    )
    equity.to_csv(out_dir / "hypothesis_equity_summary.csv", index=False)


def main() -> None:
    cfg = Config()
    set_seed(cfg.random_state)
    out_dir = Path("results/hypothesis_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[dict] = []
    all_equity: List[dict] = []
    all_regime: List[dict] = []
    all_importance: List[dict] = []

    for name, path in DATASETS.items():
        print(f"Running {name}...")
        metrics, equity, regime, importance = run_dataset(name, path, cfg, out_dir)
        all_metrics.extend(metrics)
        all_equity.extend(equity)
        all_regime.extend(regime)
        all_importance.extend(importance)

    pd.DataFrame(all_metrics).to_csv(out_dir / "model_metrics.csv", index=False)
    pd.DataFrame(all_equity).to_csv(out_dir / "equity_stats.csv", index=False)
    pd.DataFrame(all_regime).to_csv(out_dir / "regime_metrics.csv", index=False)
    pd.DataFrame(all_importance).to_csv(out_dir / "regime_feature_importance.csv", index=False)
    hypothesis_tables(out_dir)

    print("\nModel metrics:")
    print(pd.DataFrame(all_metrics).sort_values(["dataset", "f1"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
