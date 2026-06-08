from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    train_start_date: str | None = None
    first_forecast_date: str = "2024-11-01"
    last_forecast_date: str = "2025-11-01"
    outer_step_days: int = 30
    inner_cv_splits: int = 4
    inner_cv_gap: int = 7
    corr_threshold: float = 0.85
    score_threshold: float = 0.55
    min_features: int = 15
    max_features: int = 45
    shap_eval_size: int = 350
    permutation_repeats: int = 5
    threshold: float = 0.5
    lookback: int = 30
    sequence_epochs: int = 8
    sequence_batch_size: int = 64
    fee: float = 0.002
    annualization: int = 365
    random_state: int = 42
    run_sequence_models: bool = True
    feature_selection_mode: str = "fixed_train"  # "fixed_train" or "nested_walk_forward"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def normalize_scores(s: pd.Series) -> pd.Series:
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    if len(s) == 0 or s.max() == s.min():
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def prepare_dataframe(input_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = input_df.copy()
    df[cfg.date_col] = pd.to_datetime(df[cfg.date_col])
    df = df.sort_values(cfg.date_col).reset_index(drop=True)

    numeric_cols = df.select_dtypes(include=[np.number, bool]).columns.tolist()
    df = df[[cfg.date_col] + numeric_cols].copy()
    df[numeric_cols] = safe_fill(df[numeric_cols])

    df["future_return"] = df[cfg.close_col].shift(-1) / df[cfg.close_col] - 1
    df["target"] = (df["future_return"] > 0).astype(int)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def prepare_dataset(path: Path, cfg: Config) -> pd.DataFrame:
    return prepare_dataframe(pd.read_csv(path), cfg)


def feature_columns(df: pd.DataFrame, cfg: Config) -> List[str]:
    excluded = {
        cfg.date_col,
        cfg.close_col,
        "future_return",
        "target",
        "trend_regime",
        "vol_regime",
        "market_regime",
        "regime_transition",
    }
    return [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]


def outer_forecast_windows(df: pd.DataFrame, cfg: Config) -> Iterable[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(cfg.first_forecast_date)
    end = pd.Timestamp(cfg.last_forecast_date)
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=cfg.outer_step_days), end)
        if ((df[cfg.date_col] >= cur) & (df[cfg.date_col] < nxt)).any():
            yield cur, nxt
        cur = nxt


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


def rf_model(seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )


def tscv(cfg: Config, n_rows: int) -> TimeSeriesSplit:
    splits = min(cfg.inner_cv_splits, max(2, n_rows // 120))
    return TimeSeriesSplit(n_splits=splits, gap=min(cfg.inner_cv_gap, max(0, n_rows // 20)))


def predictive_power_scores(X: pd.DataFrame, y: pd.Series, seed: int) -> pd.Series:
    vals = mutual_info_classif(safe_fill(X), y, discrete_features=False, random_state=seed)
    return pd.Series(vals, index=X.columns)


def cluster_representatives(X: pd.DataFrame, y: pd.Series, cfg: Config) -> Tuple[List[str], pd.DataFrame]:
    X = safe_fill(X)
    keep = X.columns[X.nunique(dropna=False) > 1].tolist()
    X = X[keep]
    if len(keep) <= 1:
        return keep, pd.DataFrame({"cluster": [0] * len(keep), "feature": keep, "representative": keep})

    corr = X.corr(method="spearman").abs().fillna(0.0)
    dist = 1 - corr
    dist_values = dist.values.copy()
    np.fill_diagonal(dist_values, 0.0)
    labels = fcluster(linkage(squareform(dist_values, checks=False), method="average"), t=1 - cfg.corr_threshold, criterion="distance")

    pp = predictive_power_scores(X, y, cfg.random_state)
    reps: List[str] = []
    rows: List[dict] = []
    for cl in sorted(np.unique(labels)):
        cols = X.columns[labels == cl].tolist()
        rep = pp[cols].sort_values(ascending=False).index[0]
        reps.append(rep)
        for col in cols:
            rows.append({"cluster": int(cl), "feature": col, "representative": rep})
    return reps, pd.DataFrame(rows)


def validation_shap_scores(model, model_name: str, X_val: pd.DataFrame, y_val: pd.Series, features: List[str], cfg: Config) -> pd.Series:
    if len(X_val) == 0:
        return pd.Series(0.0, index=features)
    sample = X_val.tail(min(len(X_val), cfg.shap_eval_size)).copy()
    y_sample = y_val.loc[sample.index]
    try:
        if model_name == "catboost":
            pool = Pool(sample, y_sample)
            vals = model.get_feature_importance(pool, type="ShapValues")
            return pd.Series(np.abs(vals[:, :-1]).mean(axis=0), index=features)
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(sample)
        if isinstance(vals, list):
            vals = vals[1]
        return pd.Series(np.abs(vals).mean(axis=0), index=features)
    except Exception:
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=features)
        return pd.Series(0.0, index=features)


def inner_cv_feature_scores(X: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.DataFrame:
    X = safe_fill(X)
    shap_acc = pd.Series(0.0, index=X.columns)
    perm_acc = pd.Series(0.0, index=X.columns)
    stability = pd.Series(0.0, index=X.columns)
    n_folds = 0

    for tr_idx, val_idx in tscv(cfg, len(X)).split(X):
        Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[tr_idx], y.iloc[val_idx]
        model = cat_model(cfg.random_state, iterations=130)
        model.fit(Xtr, ytr)

        shap_s = validation_shap_scores(model, "catboost", Xval, yval, X.columns.tolist(), cfg)
        try:
            perm = permutation_importance(
                model,
                Xval,
                yval,
                n_repeats=cfg.permutation_repeats,
                random_state=cfg.random_state,
                scoring="f1",
                n_jobs=1,
            )
            perm_s = pd.Series(perm.importances_mean, index=X.columns)
        except Exception:
            perm_s = pd.Series(0.0, index=X.columns)

        shap_n = normalize_scores(shap_s)
        perm_n = normalize_scores(perm_s)
        shap_acc += shap_n
        perm_acc += perm_n
        stability += ((shap_n >= 0.5) | (perm_n >= 0.5)).astype(float)
        n_folds += 1

    uni_s = predictive_power_scores(X, y, cfg.random_state)
    scores = pd.DataFrame(
        {
            "shap": shap_acc / max(1, n_folds),
            "permutation": perm_acc / max(1, n_folds),
            "stability": stability / max(1, n_folds),
            "univariate": normalize_scores(uni_s),
        }
    )
    scores["aggregate_score"] = scores[["shap", "permutation", "stability", "univariate"]].mean(axis=1)
    for col in ["shap", "permutation", "stability", "univariate", "aggregate_score"]:
        scores[f"rank_{col}"] = scores[col].rank(ascending=False, method="average")
    scores["aggregate_rank"] = scores[
        ["rank_shap", "rank_permutation", "rank_stability", "rank_univariate", "rank_aggregate_score"]
    ].mean(axis=1)
    return scores.sort_values(["aggregate_score", "aggregate_rank"], ascending=[False, True])


def cv_score_prefix(X: pd.DataFrame, y: pd.Series, features: List[str], cfg: Config) -> float:
    vals = []
    for tr_idx, val_idx in tscv(cfg, len(X)).split(X):
        model = cat_model(cfg.random_state, iterations=120)
        model.fit(X.iloc[tr_idx][features], y.iloc[tr_idx])
        prob = model.predict_proba(X.iloc[val_idx][features])[:, 1]
        pred = (prob >= cfg.threshold).astype(int)
        vals.append(f1_score(y.iloc[val_idx], pred, zero_division=0))
    return float(np.mean(vals)) if vals else 0.0


def nested_select_features(X_train: pd.DataFrame, y_train: pd.Series, cfg: Config) -> Tuple[List[str], pd.DataFrame, pd.DataFrame]:
    reps, clusters = cluster_representatives(X_train, y_train, cfg)
    Xr = safe_fill(X_train[reps].copy())
    scores = inner_cv_feature_scores(Xr, y_train, cfg)

    threshold_selected = scores.index[scores["aggregate_score"] >= cfg.score_threshold].tolist()
    if len(threshold_selected) < cfg.min_features:
        threshold_selected = scores.head(min(cfg.min_features, len(scores))).index.tolist()

    ordered = scores.index.tolist()
    candidate_sizes = sorted(
        {
            len(threshold_selected),
            min(cfg.min_features, len(ordered)),
            min(20, len(ordered)),
            min(30, len(ordered)),
            min(cfg.max_features, len(ordered)),
        }
    )
    candidate_sizes = [s for s in candidate_sizes if 1 <= s <= len(ordered)]

    best_features = threshold_selected
    best_cv = -np.inf
    cv_by_size: Dict[int, float] = {}
    for size in candidate_sizes:
        cur = ordered[:size]
        score = cv_score_prefix(Xr, y_train, cur, cfg)
        cv_by_size[size] = score
        if score > best_cv:
            best_cv = score
            best_features = cur

    scores = scores.copy()
    scores["selected"] = scores.index.isin(best_features)
    scores["cv_best_f1"] = best_cv
    scores["cv_by_size"] = json.dumps(cv_by_size)
    return best_features, scores.reset_index(names="feature"), clusters


def binary_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "positive_rate": float(np.mean(y_pred)),
        "class_1_share": float(np.mean(y_true)),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    try:
        out["brier"] = brier_score_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))
    except Exception:
        out["brier"] = np.nan
    try:
        out["logloss"] = log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))
    except Exception:
        out["logloss"] = np.nan
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


def fit_sequence_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    cfg: Config,
    bidirectional: bool,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
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
    loader = DataLoader(TensorDataset(torch.tensor(X_sub), torch.tensor(y_sub)), batch_size=cfg.sequence_batch_size, shuffle=False)

    best_state = None
    best_val = np.inf
    stale = 0
    for _ in range(cfg.sequence_epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                cur_val = loss_fn(model(torch.tensor(X_val)), torch.tensor(y_val)).item()
            if cur_val < best_val:
                best_val = cur_val
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 3:
                    break
    if best_state:
        model.load_state_dict(best_state)

    X_context = pd.concat([Xtr.tail(cfg.lookback - 1), Xte], axis=0)
    y_context = pd.concat([y_train.tail(cfg.lookback - 1), y_test], axis=0)
    X_seq, _, idx = build_sequences(X_context, y_context, cfg.lookback)
    test_idx = set(X_test.index)
    keep = [i for i, real_idx in enumerate(idx) if real_idx in test_idx]
    X_seq = X_seq[keep]
    idx = [idx[i] for i in keep]

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(X_seq))).numpy()
    pred = (prob >= cfg.threshold).astype(int)
    return prob, pred, idx


def fit_tabular_predict(model_name: str, X_train, y_train, X_test, cfg: Config):
    if model_name == "catboost":
        model = cat_model(cfg.random_state, iterations=260)
    elif model_name == "xgboost":
        model = xgb_model(cfg.random_state)
    elif model_name == "random_forest":
        model = rf_model(cfg.random_state)
    else:
        raise ValueError(model_name)
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= cfg.threshold).astype(int)
    return prob, pred, model


def benchmark_predictions(df: pd.DataFrame, cfg: Config) -> Dict[str, pd.Series]:
    prev_ret = df[cfg.close_col].pct_change()
    rng = np.random.default_rng(cfg.random_state)
    return {
        "momentum": (prev_ret > 0).astype(int),
        "always_up": pd.Series(1, index=df.index),
        "random": pd.Series(rng.integers(0, 2, len(df)), index=df.index),
    }


def equity_curve(eval_df: pd.DataFrame, signal_col: str, cfg: Config, execution_lag: int) -> pd.DataFrame:
    out = eval_df[["date", "future_return", signal_col]].copy().sort_values("date")
    signal = out[signal_col].fillna(0).astype(int)

    # Trading convention used in the paper:
    # signal at date t is produced after observing information at t;
    # the position is opened at 00:00 on t+1 and held until 00:00 on t+2.
    # Since future_return at row t is the t -> t+1 return, the tradable
    # return for a signal at t is future_return at row t+1.
    tradable_return = out["future_return"].shift(-1)
    out["trade_open_date"] = out["date"].shift(-1)
    out["trade_close_date"] = out["date"].shift(-2)

    if execution_lag > 0:
        signal = signal.shift(execution_lag).fillna(0).astype(int)
    position = np.where(signal > 0, 1, -1)
    trades = pd.Series(position, index=out.index).diff().abs().fillna(0)
    out["position"] = position
    out["tradable_return"] = tradable_return
    out["strategy_return"] = position * out["tradable_return"].fillna(0) - trades * cfg.fee
    out = out.dropna(subset=["tradable_return"]).copy()
    out["equity"] = 1 + out["strategy_return"].cumsum()
    out["execution_lag"] = execution_lag
    out["execution_rule"] = "signal_t_open_t_plus_1_close_t_plus_2"
    return out


def equity_stats(eq: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    r = eq["strategy_return"].fillna(0)
    total = eq["equity"].iloc[-1] - 1 if len(eq) else np.nan
    std = r.std(ddof=0)
    sharpe = (r.mean() / std) * np.sqrt(cfg.annualization) if std > 0 else np.nan
    peak = eq["equity"].cummax()
    dd = (eq["equity"] / peak - 1).min() if len(eq) else np.nan
    return {"total_return": total, "sharpe": sharpe, "max_drawdown": dd, "win_rate": float((r > 0).mean())}


def recursive_regime_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df[[cfg.date_col, cfg.close_col, "future_return", "target"]].copy()
    ret30 = out[cfg.close_col].pct_change(30)
    vol20 = out[cfg.close_col].pct_change().rolling(20).std()
    trend_thr = ret30.expanding(min_periods=90).std().shift(1).fillna(ret30.iloc[:90].std()) * 0.35
    vol_thr = vol20.expanding(min_periods=90).median().shift(1).fillna(vol20.iloc[:90].median())
    out["trend_regime"] = np.select([ret30 > trend_thr, ret30 < -trend_thr], ["bull", "bear"], default="sideways")
    out["vol_regime"] = np.where(vol20 > vol_thr, "high_vol", "low_vol")
    out["market_regime"] = out["trend_regime"] + "_" + out["vol_regime"]
    out["regime_transition"] = (out["market_regime"] != out["market_regime"].shift(1)).astype(int)
    return out


def model_importance(model_name: str, model, X: pd.DataFrame, y: pd.Series, features: List[str], cfg: Config) -> Tuple[pd.Series, pd.Series]:
    if len(X) < 20:
        empty = pd.Series(np.nan, index=features)
        return empty, empty
    try:
        if model_name == "catboost":
            vals = model.get_feature_importance(Pool(X[features], y), type="ShapValues")
            shap_imp = pd.Series(np.abs(vals[:, :-1]).mean(axis=0), index=features)
        else:
            explainer = shap.TreeExplainer(model)
            vals = explainer.shap_values(X[features])
            if isinstance(vals, list):
                vals = vals[1]
            shap_imp = pd.Series(np.abs(vals).mean(axis=0), index=features)
    except Exception:
        shap_imp = pd.Series(getattr(model, "feature_importances_", np.zeros(len(features))), index=features)

    try:
        perm = permutation_importance(model, X[features], y, n_repeats=cfg.permutation_repeats, random_state=cfg.random_state, scoring="f1", n_jobs=1)
        perm_imp = pd.Series(perm.importances_mean, index=features)
    except Exception:
        perm_imp = pd.Series(np.nan, index=features)
    return shap_imp, perm_imp


def collect_prediction_metrics(predictions: Dict[str, pd.DataFrame]) -> List[dict]:
    rows = []
    for key, pred_df in predictions.items():
        ds, model = key.split("::", 1)
        rows.append({"dataset": ds, "model": model, "n": len(pred_df), **binary_metrics(pred_df["target"], pred_df["pred"], pred_df["prob"])})
    return rows


def run_dataset(name: str, path: Path, cfg: Config, out_dir: Path):
    df = prepare_dataset(path, cfg)
    return run_prepared_dataset(name, df, cfg, out_dir)


def run_prepared_dataset(name: str, df: pd.DataFrame, cfg: Config, out_dir: Path):
    features_all = feature_columns(df, cfg)
    X_all = safe_fill(df[features_all])
    y_all = df["target"].copy()
    regimes = recursive_regime_labels(df, cfg)

    model_names = ["catboost", "xgboost", "random_forest", "lstm", "bilstm", "momentum", "random", "always_up"]
    prediction_parts: Dict[str, List[pd.DataFrame]] = {f"{name}::{m}": [] for m in model_names}
    fs_rows: List[pd.DataFrame] = []
    cluster_rows: List[pd.DataFrame] = []
    importance_rows: List[dict] = []

    fixed_selected: List[str] | None = None
    fixed_fs: pd.DataFrame | None = None
    fixed_clusters: pd.DataFrame | None = None
    if cfg.feature_selection_mode == "fixed_train":
        fixed_train_mask = df[cfg.date_col] < pd.Timestamp(cfg.first_forecast_date)
        if cfg.train_start_date is not None:
            fixed_train_mask &= df[cfg.date_col] >= pd.Timestamp(cfg.train_start_date)
        fixed_selected, fixed_fs, fixed_clusters = nested_select_features(
            X_all.loc[fixed_train_mask],
            y_all.loc[fixed_train_mask],
            cfg,
        )
        fixed_fs = fixed_fs.copy()
        fixed_fs.insert(0, "selection_scope", "fixed_pre_holdout_train")
        fixed_fs.insert(0, "selection_end", pd.Timestamp(cfg.first_forecast_date))
        fixed_fs.insert(0, "outer_fold", 0)
        fixed_fs.insert(0, "dataset", name)
        fs_rows.append(fixed_fs)

        fixed_clusters = fixed_clusters.copy()
        fixed_clusters.insert(0, "selection_scope", "fixed_pre_holdout_train")
        fixed_clusters.insert(0, "selection_end", pd.Timestamp(cfg.first_forecast_date))
        fixed_clusters.insert(0, "outer_fold", 0)
        fixed_clusters.insert(0, "dataset", name)
        cluster_rows.append(fixed_clusters)
    elif cfg.feature_selection_mode != "nested_walk_forward":
        raise ValueError("feature_selection_mode must be 'fixed_train' or 'nested_walk_forward'")

    for fold_no, (start, end) in enumerate(outer_forecast_windows(df, cfg), start=1):
        train_mask = df[cfg.date_col] < start
        if cfg.train_start_date is not None:
            train_mask &= df[cfg.date_col] >= pd.Timestamp(cfg.train_start_date)
        test_mask = (df[cfg.date_col] >= start) & (df[cfg.date_col] < end)
        if train_mask.sum() < 250 or test_mask.sum() == 0:
            continue

        X_train, y_train = X_all.loc[train_mask], y_all.loc[train_mask]
        X_test, y_test = X_all.loc[test_mask], y_all.loc[test_mask]
        if cfg.feature_selection_mode == "fixed_train":
            selected = list(fixed_selected or [])
        else:
            selected, fs, clusters = nested_select_features(X_train, y_train, cfg)
            fs.insert(0, "selection_scope", "nested_walk_forward")
            fs.insert(0, "forecast_end", end)
            fs.insert(0, "forecast_start", start)
            fs.insert(0, "outer_fold", fold_no)
            fs.insert(0, "dataset", name)
            fs_rows.append(fs)
            clusters = clusters.copy()
            clusters.insert(0, "selection_scope", "nested_walk_forward")
            clusters.insert(0, "forecast_start", start)
            clusters.insert(0, "outer_fold", fold_no)
            clusters.insert(0, "dataset", name)
            cluster_rows.append(clusters)

        fitted_tabular = {}
        for model_name in ["catboost", "xgboost", "random_forest"]:
            prob, pred, model = fit_tabular_predict(model_name, X_train[selected], y_train, X_test[selected], cfg)
            fitted_tabular[model_name] = model
            eval_df = df.loc[test_mask, ["date", "future_return", "target"]].copy()
            eval_df["pred"] = pred
            eval_df["prob"] = prob
            eval_df["outer_fold"] = fold_no
            eval_df["n_features"] = len(selected)
            prediction_parts[f"{name}::{model_name}"].append(eval_df)

        if cfg.run_sequence_models:
            for model_name, bidir in [("lstm", False), ("bilstm", True)]:
                prob, pred, idx = fit_sequence_predict(X_train, y_train, X_test, y_test, selected, cfg, bidir)
                eval_df = df.loc[idx, ["date", "future_return", "target"]].copy()
                eval_df["pred"] = pred
                eval_df["prob"] = prob
                eval_df["outer_fold"] = fold_no
                eval_df["n_features"] = len(selected)
                prediction_parts[f"{name}::{model_name}"].append(eval_df)

        bench = benchmark_predictions(df, cfg)
        for bname in ["momentum", "random", "always_up"]:
            eval_df = df.loc[test_mask, ["date", "future_return", "target"]].copy()
            eval_df["pred"] = bench[bname].loc[test_mask].fillna(0).astype(int).values
            eval_df["prob"] = np.where(eval_df["pred"] == 1, 0.55, 0.45)
            eval_df["outer_fold"] = fold_no
            eval_df["n_features"] = 0
            prediction_parts[f"{name}::{bname}"].append(eval_df)

        test_regimes = regimes.loc[test_mask, ["date", "market_regime"]]
        X_test_sel = X_test[selected]
        for model_name, model in fitted_tabular.items():
            by_reg = test_regimes.copy()
            by_reg.index = X_test_sel.index
            for regime, idxs in by_reg.groupby("market_regime").groups.items():
                if len(idxs) < 20:
                    continue
                shap_imp, perm_imp = model_importance(model_name, model, X_test_sel.loc[idxs], y_test.loc[idxs], selected, cfg)
                top = pd.DataFrame({"feature": selected, "shap": shap_imp.reindex(selected).values, "permutation": perm_imp.reindex(selected).values})
                top = top.sort_values("shap", ascending=False).head(25)
                for _, row in top.iterrows():
                    importance_rows.append({"dataset": name, "outer_fold": fold_no, "model": model_name, "regime": regime, **row.to_dict()})

    predictions = {
        key: pd.concat(parts, ignore_index=True).sort_values("date")
        for key, parts in prediction_parts.items()
        if parts
    }
    if fs_rows:
        pd.concat(fs_rows, ignore_index=True).to_csv(out_dir / f"{name}_feature_selection_scores.csv", index=False)
    if cluster_rows:
        pd.concat(cluster_rows, ignore_index=True).to_csv(out_dir / f"{name}_feature_clusters.csv", index=False)

    return predictions, regimes, importance_rows


def summarize_equity(predictions: Dict[str, pd.DataFrame], cfg: Config, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    equity_rows, curve_frames = [], []
    for key, pred_df in predictions.items():
        ds, model = key.split("::", 1)
        pred_df = pred_df.sort_values("date")
        pred_df.to_csv(out_dir / f"{ds}_{model}_predictions.csv", index=False)
        eq = equity_curve(pred_df.rename(columns={"pred": "signal"}), "signal", cfg, execution_lag=0)
        st = equity_stats(eq, cfg)
        equity_rows.append(
            {
                "dataset": ds,
                "model": model,
                "execution_rule": "signal_t_open_t_plus_1_close_t_plus_2",
                **st,
            }
        )
        curve_frames.append(eq[["date", "trade_open_date", "trade_close_date", "equity"]].assign(dataset=ds, model=model))
    return pd.DataFrame(equity_rows), pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()


def summarize_regimes(predictions: Dict[str, pd.DataFrame], regimes_by_dataset: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, pred_df in predictions.items():
        ds, model = key.split("::", 1)
        joined = pred_df.merge(
            regimes_by_dataset[ds][["date", "trend_regime", "vol_regime", "market_regime", "regime_transition"]],
            on="date",
            how="left",
        )
        for regime_type in ["trend_regime", "vol_regime", "market_regime"]:
            for regime, grp in joined.groupby(regime_type):
                if len(grp) >= 15:
                    rows.append({"dataset": ds, "model": model, "regime_type": regime_type, "regime": regime, "n": len(grp), **binary_metrics(grp["target"], grp["pred"], grp["prob"])})
        for label, grp in [("transition", joined[joined["regime_transition"] == 1]), ("stable", joined[joined["regime_transition"] == 0])]:
            if len(grp) >= 15:
                rows.append({"dataset": ds, "model": model, "regime_type": "transition_layer", "regime": label, "n": len(grp), **binary_metrics(grp["target"], grp["pred"], grp["prob"])})
    return pd.DataFrame(rows)


def hypothesis_tables(
    out_dir: Path,
    predictions: Dict[str, pd.DataFrame],
    prepared_datasets: Dict[str, pd.DataFrame] | None = None,
) -> None:
    fs_files = list(out_dir.glob("*_feature_selection_scores.csv"))
    if not fs_files:
        return
    fs = pd.concat([pd.read_csv(p) for p in fs_files], ignore_index=True)
    fs = fs.drop_duplicates(subset=["dataset", "outer_fold", "feature"], keep="last")
    selected = fs[fs["selected"] == True].copy()

    all_candidates = fs.copy()
    all_candidates["group"] = np.select(
        [
            all_candidates["feature"].isin(WHALE_FEATURES),
            all_candidates["feature"].isin(GENERAL_VOLUME_FEATURES),
            all_candidates["feature"].isin(SPIKE_FEATURES),
            all_candidates["feature"].isin(FLOW_FEATURES),
        ],
        ["whale", "general_volume", "spike", "flow"],
        default="other",
    )
    all_candidates.groupby(["dataset", "group"])["aggregate_score"].agg(
        ["count", "mean", "median", "max"]
    ).reset_index().to_csv(out_dir / "hypothesis_group_scores_all_candidates.csv", index=False)

    all_candidates.to_csv(out_dir / "hypothesis_all_candidate_feature_scores.csv", index=False)

    selected_summary = (
        selected.assign(
            is_flow=selected["feature"].isin(FLOW_FEATURES),
            is_whale=selected["feature"].isin(WHALE_FEATURES),
            is_spike=selected["feature"].isin(SPIKE_FEATURES),
            is_general_volume=selected["feature"].isin(GENERAL_VOLUME_FEATURES),
        )
        .groupby("dataset", as_index=False)
        .agg(
            selected_n=("feature", "nunique"),
            flow_selected=("is_flow", "sum"),
            whale_selected=("is_whale", "sum"),
            spike_selected=("is_spike", "sum"),
            general_volume_selected=("is_general_volume", "sum"),
            mean_aggregate_score=("aggregate_score", "mean"),
        )
    )
    selected_summary.to_csv(out_dir / "hypothesis_feature_groups_selected.csv", index=False)

    metrics = pd.read_csv(out_dir / "model_metrics.csv")
    metrics.pivot_table(index="model", columns="dataset", values="f1", aggfunc="mean").to_csv(out_dir / "hypothesis_h2_model_dataset_f1.csv")

    group_scores = selected.assign(
        group=np.select(
            [
                selected["feature"].isin(WHALE_FEATURES),
                selected["feature"].isin(GENERAL_VOLUME_FEATURES),
                selected["feature"].isin(SPIKE_FEATURES),
                selected["feature"].isin(FLOW_FEATURES),
            ],
            ["whale", "general_volume", "spike", "flow"],
            default="other",
        )
    )
    group_scores.groupby(["dataset", "group"])["aggregate_score"].agg(["count", "mean", "median", "max"]).reset_index().to_csv(out_dir / "hypothesis_h4_h5_group_scores_selected.csv", index=False)

    flow_rows = []
    for key, pred_df in predictions.items():
        ds, model = key.split("::", 1)
        if ds not in ["oca", "oca_ta"] or model not in ["catboost", "xgboost", "random_forest"]:
            continue
        if prepared_datasets is not None and ds in prepared_datasets:
            df = prepared_datasets[ds]
        else:
            source_path = DATASETS[ds]
            df = prepare_dataset(source_path, Config())
        flow_cols = [c for c in FLOW_FEATURES if c in df.columns]
        for col in flow_cols:
            merged = pred_df[["date"]].merge(df[["date", "future_return", "target", col]], on="date", how="left").dropna()
            if len(merged) < 30:
                continue
            rho, p_val = spearmanr(merged[col], merged["future_return"])
            flow_rows.append({"dataset": ds, "model": model, "feature": col, "n": len(merged), "spearman_future_return": rho, "p_value": p_val})
    pd.DataFrame(flow_rows).to_csv(out_dir / "hypothesis_h1_flow_direction.csv", index=False)


def save_equity_plots(curves: pd.DataFrame, out_dir: Path) -> None:
    if curves.empty:
        return
    curves.to_csv(out_dir / "equity_curves_realistic_execution.csv", index=False)
    for ds, grp in curves.groupby("dataset"):
        plt.figure(figsize=(12, 6))
        for model, mg in grp.groupby("model"):
            plt.plot(pd.to_datetime(mg["date"]), mg["equity"], label=model, linewidth=1.3)
        plt.title(f"Equity curves, signal t -> trade t+1 to t+2: {ds}")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{ds}_equity_curves_realistic_execution.png", dpi=150)
        plt.close()


def run_full_pipeline(
    oca: pd.DataFrame,
    ta: pd.DataFrame,
    baseline_model: pd.DataFrame,
    oca_ta: pd.DataFrame,
    cfg: Config | None = None,
    out_dir: str | Path = "results/fixed_train_hypothesis_pipeline_notebook",
) -> Dict[str, object]:
    """Notebook-friendly entry point.

    Usage:
        artifacts = run_full_pipeline(
            df_merged_oca,
            df_merged_ta,
            df_merged_base,
            df_merged_oca_ta,
        )
    """
    if cfg is None:
        cfg = Config()
    set_seed(cfg.random_state)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    raw_datasets = {
        "oca": oca,
        "ta": ta,
        "baseline": baseline_model,
        "oca_ta": oca_ta,
    }
    prepared_datasets = {
        name: prepare_dataframe(df, cfg)
        for name, df in raw_datasets.items()
    }

    all_predictions: Dict[str, pd.DataFrame] = {}
    regimes_by_dataset: Dict[str, pd.DataFrame] = {}
    all_importance: List[dict] = []

    for name, prepared in prepared_datasets.items():
        print(f"Running {cfg.feature_selection_mode} pipeline for {name}...")
        preds, regimes, importance = run_prepared_dataset(name, prepared, cfg, out_path)
        all_predictions.update(preds)
        regimes_by_dataset[name] = regimes
        all_importance.extend(importance)

    model_metrics = pd.DataFrame(collect_prediction_metrics(all_predictions))
    model_metrics.to_csv(out_path / "model_metrics.csv", index=False)

    equity_stats_table, equity_curves = summarize_equity(all_predictions, cfg, out_path)
    equity_stats_table.to_csv(out_path / "equity_stats.csv", index=False)
    save_equity_plots(equity_curves, out_path)

    regime_metrics = summarize_regimes(all_predictions, regimes_by_dataset)
    regime_metrics.to_csv(out_path / "regime_metrics.csv", index=False)

    regime_feature_importance = pd.DataFrame(all_importance)
    regime_feature_importance.to_csv(out_path / "regime_feature_importance.csv", index=False)

    hypothesis_tables(out_path, all_predictions, prepared_datasets=prepared_datasets)

    feature_selection_files = list(out_path.glob("*_feature_selection_scores.csv"))
    feature_selection_scores = (
        pd.concat([pd.read_csv(p) for p in feature_selection_files], ignore_index=True)
        if feature_selection_files
        else pd.DataFrame()
    )

    cluster_files = list(out_path.glob("*_feature_clusters.csv"))
    feature_clusters = (
        pd.concat([pd.read_csv(p) for p in cluster_files], ignore_index=True)
        if cluster_files
        else pd.DataFrame()
    )

    hypothesis_artifacts = {}
    for file_name in [
        "hypothesis_all_candidate_feature_scores.csv",
        "hypothesis_group_scores_all_candidates.csv",
        "hypothesis_feature_groups_selected.csv",
        "hypothesis_h2_model_dataset_f1.csv",
        "hypothesis_h4_h5_group_scores_selected.csv",
        "hypothesis_h1_flow_direction.csv",
    ]:
        path = out_path / file_name
        if path.exists():
            hypothesis_artifacts[path.stem] = pd.read_csv(path)

    artifacts = {
        "config": cfg,
        "out_dir": out_path,
        "prepared_datasets": prepared_datasets,
        "predictions": all_predictions,
        "regimes_by_dataset": regimes_by_dataset,
        "model_metrics": model_metrics,
        "equity_stats": equity_stats_table,
        "equity_curves": equity_curves,
        "regime_metrics": regime_metrics,
        "regime_feature_importance": regime_feature_importance,
        "feature_selection_scores": feature_selection_scores,
        "feature_clusters": feature_clusters,
        "hypothesis_artifacts": hypothesis_artifacts,
    }

    print(f"\nModel metrics ({cfg.feature_selection_mode}):")
    print(model_metrics.sort_values(["dataset", "f1"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved notebook artifacts to {out_path}")
    return artifacts


def main() -> None:
    cfg = Config()
    set_seed(cfg.random_state)
    out_dir = Path("results/fixed_train_hypothesis_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_predictions: Dict[str, pd.DataFrame] = {}
    regimes_by_dataset: Dict[str, pd.DataFrame] = {}
    all_importance: List[dict] = []

    for name, path in DATASETS.items():
        print(f"Running {cfg.feature_selection_mode} pipeline for {name}...")
        preds, regimes, importance = run_dataset(name, path, cfg, out_dir)
        all_predictions.update(preds)
        regimes_by_dataset[name] = regimes
        all_importance.extend(importance)

    metrics = pd.DataFrame(collect_prediction_metrics(all_predictions))
    metrics.to_csv(out_dir / "model_metrics.csv", index=False)

    equity, curves = summarize_equity(all_predictions, cfg, out_dir)
    equity.to_csv(out_dir / "equity_stats.csv", index=False)
    save_equity_plots(curves, out_dir)

    regime_metrics = summarize_regimes(all_predictions, regimes_by_dataset)
    regime_metrics.to_csv(out_dir / "regime_metrics.csv", index=False)

    pd.DataFrame(all_importance).to_csv(out_dir / "regime_feature_importance.csv", index=False)
    hypothesis_tables(out_dir, all_predictions)

    print(f"\nModel metrics ({cfg.feature_selection_mode}):")
    print(metrics.sort_values(["dataset", "f1"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved article-ready results to {out_dir}")


if __name__ == "__main__":
    main()
