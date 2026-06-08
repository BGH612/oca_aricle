from __future__ import annotations

import json
import random
import warnings
from dataclasses import dataclass, replace
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
from scipy.stats import chi2, norm, spearmanr
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
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


DATASETS = {
    "baseline": Path("datasets/merged_base.csv"),
    "oca": Path("datasets/merged_oca.csv"),
    "ta": Path("datasets/merged_ta.csv"),
    "oca_ta": Path("datasets/oca_ta.csv"),
    "baseline_oca": Path("datasets/baseline_oca.csv"),
    "baseline_ta": Path("datasets/baseline_ta.csv"),
    "full": Path("datasets/full.csv"),
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
    sequence_epochs: int = 30
    sequence_batch_size: int = 64
    sequence_patience: int = 5
    sequence_hidden_grid: Tuple[int, ...] = (32, 64, 96)
    sequence_dropout_grid: Tuple[float, ...] = (0.0, 0.2)
    sequence_lr_grid: Tuple[float, ...] = (0.001, 0.0005)
    tune_sequence_models: bool = True
    sequence_tune_max_combinations: int = 12
    fee: float = 0.001
    annualization: int = 365
    random_state: int = 42
    run_sequence_models: bool = True
    feature_selection_mode: str = "fixed_train"  # "fixed_train" or "nested_walk_forward"
    bootstrap_iters: int = 1000
    block_bootstrap_iters: int = 1000
    block_size: int = 10
    fee_sensitivity_grid: Tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002)
    run_extra_diagnostics: bool = True


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


def read_csv_if_nonempty(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


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
    def __init__(self, input_size: int, hidden: int = 48, dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * (2 if bidirectional else 1), 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

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


def train_sequence_model(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_size: int,
    cfg: Config,
    bidirectional: bool,
    hidden: int,
    dropout: float,
    lr: float,
) -> Tuple[SequenceNet, float, int]:
    torch.manual_seed(cfg.random_state)
    model = SequenceNet(input_size, hidden=hidden, dropout=dropout, bidirectional=bidirectional)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_sub), torch.tensor(y_sub)),
        batch_size=cfg.sequence_batch_size,
        shuffle=False,
    )

    best_state = None
    best_val = np.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, cfg.sequence_epochs + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        if len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                cur_val = loss_fn(model(torch.tensor(X_val)), torch.tensor(y_val)).item()
            if cur_val < best_val:
                best_val = cur_val
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= cfg.sequence_patience:
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model, float(best_val), int(best_epoch)


def tune_sequence_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    cfg: Config,
    bidirectional: bool,
) -> Dict[str, float | int]:
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(X_train[features]), columns=features, index=X_train.index)
    split = max(cfg.lookback + 1, int(len(Xtr) * 0.85))
    if split >= len(Xtr):
        split = len(Xtr) - 1

    X_sub, y_sub, _ = build_sequences(Xtr.iloc[:split], y_train.iloc[:split], cfg.lookback)
    X_val_ctx = pd.concat([Xtr.iloc[:split].tail(cfg.lookback - 1), Xtr.iloc[split:]], axis=0)
    y_val_ctx = pd.concat([y_train.iloc[:split].tail(cfg.lookback - 1), y_train.iloc[split:]], axis=0)
    X_val, y_val, _ = build_sequences(X_val_ctx, y_val_ctx, cfg.lookback)

    combos = [
        (hidden, dropout, lr)
        for hidden in cfg.sequence_hidden_grid
        for dropout in cfg.sequence_dropout_grid
        for lr in cfg.sequence_lr_grid
    ][: cfg.sequence_tune_max_combinations]

    best = {"hidden": 48, "dropout": 0.0, "lr": 0.001, "val_loss": np.inf, "best_epoch": 0}
    for hidden, dropout, lr in combos:
        _, val_loss, best_epoch = train_sequence_model(
            X_sub,
            y_sub,
            X_val,
            y_val,
            input_size=len(features),
            cfg=cfg,
            bidirectional=bidirectional,
            hidden=int(hidden),
            dropout=float(dropout),
            lr=float(lr),
        )
        if val_loss < best["val_loss"]:
            best = {
                "hidden": int(hidden),
                "dropout": float(dropout),
                "lr": float(lr),
                "val_loss": float(val_loss),
                "best_epoch": int(best_epoch),
            }
    return best


def fit_sequence_predict(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    cfg: Config,
    bidirectional: bool,
    params: Dict[str, float | int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    scaler = StandardScaler()
    Xtr = pd.DataFrame(scaler.fit_transform(X_train[features]), columns=features, index=X_train.index)
    Xte = pd.DataFrame(scaler.transform(X_test[features]), columns=features, index=X_test.index)

    split = max(cfg.lookback + 1, int(len(Xtr) * 0.85))
    X_sub, y_sub, _ = build_sequences(Xtr.iloc[:split], y_train.iloc[:split], cfg.lookback)
    X_val_ctx = pd.concat([Xtr.iloc[:split].tail(cfg.lookback - 1), Xtr.iloc[split:]], axis=0)
    y_val_ctx = pd.concat([y_train.iloc[:split].tail(cfg.lookback - 1), y_train.iloc[split:]], axis=0)
    X_val, y_val, _ = build_sequences(X_val_ctx, y_val_ctx, cfg.lookback)

    params = params or {"hidden": 48, "dropout": 0.0, "lr": 0.001}
    model, _, _ = train_sequence_model(
        X_sub,
        y_sub,
        X_val,
        y_val,
        input_size=len(features),
        cfg=cfg,
        bidirectional=bidirectional,
        hidden=int(params.get("hidden", 48)),
        dropout=float(params.get("dropout", 0.0)),
        lr=float(params.get("lr", 0.001)),
    )

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


def predict_tabular_model(model, X, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= cfg.threshold).astype(int)
    return prob, pred


def benchmark_predictions(df: pd.DataFrame, cfg: Config) -> Dict[str, pd.Series]:
    prev_ret = df[cfg.close_col].pct_change()
    rng = np.random.default_rng(cfg.random_state)
    return {
        "momentum": (prev_ret > 0).astype(int),
        "always_up": pd.Series(1, index=df.index),
        "random": pd.Series(rng.integers(0, 2, len(df)), index=df.index),
    }


def equity_curve(eval_df: pd.DataFrame, signal_col: str, cfg: Config) -> pd.DataFrame:
    out = eval_df[["date", "future_return", signal_col]].copy().sort_values("date")
    signal = out[signal_col].fillna(0).astype(int)

    # Trading convention used in the paper:
    # signal at date t is traded over the same forecast interval t -> t+1.
    # This assumes that all predictors for row t are available before the
    # trading decision for that interval.
    tradable_return = out["future_return"]
    out["trade_open_date"] = out["date"]
    out["trade_close_date"] = out["date"].shift(-1)

    position = np.where(signal > 0, 1, -1)
    trades = pd.Series(position, index=out.index).diff().abs().fillna(0)
    out["position"] = position
    out["tradable_return"] = tradable_return
    out["strategy_return"] = position * out["tradable_return"].fillna(0) - trades * cfg.fee
    out = out.dropna(subset=["tradable_return"]).copy()
    out["equity"] = 1 + out["strategy_return"].cumsum()
    out["execution_rule"] = "signal_t_trade_t_to_t_plus_1"
    return out


def equity_stats(eq: pd.DataFrame, cfg: Config) -> Dict[str, float]:
    r = eq["strategy_return"].fillna(0)
    total = eq["equity"].iloc[-1] - 1 if len(eq) else np.nan
    std = r.std(ddof=0)
    sharpe = (r.mean() / std) * np.sqrt(cfg.annualization) if std > 0 else np.nan
    peak = eq["equity"].cummax()
    dd = (eq["equity"] / peak - 1).min() if len(eq) else np.nan
    if "position" in eq.columns and len(eq):
        position = eq["position"].fillna(0).astype(float)
        position_change = position.diff().abs().fillna(0)
        n_trades = int((position_change > 0).sum())
        turnover = float(position_change.mean())
    else:
        n_trades = 0
        turnover = np.nan
    return {
        "total_return": total,
        "sharpe": sharpe,
        "max_drawdown": dd,
        "win_rate": float((r > 0).mean()),
        "n_trades": n_trades,
        "turnover": turnover,
    }


def bootstrap_ci(values: List[float], seed: int = 42) -> Dict[str, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    return {
        "mean": float(np.mean(vals)),
        "ci_low": float(np.quantile(vals, 0.025)),
        "ci_high": float(np.quantile(vals, 0.975)),
    }


def bootstrap_metric_ci(y_true, y_pred, y_prob, metric_name: str, cfg: Config) -> Dict[str, float]:
    rng = np.random.default_rng(cfg.random_state)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    n = len(y_true)
    vals: List[float] = []
    if n == 0:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    for _ in range(cfg.bootstrap_iters):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]
        try:
            if metric_name == "f1":
                vals.append(f1_score(yt, yp, zero_division=0))
            elif metric_name == "balanced_accuracy":
                vals.append(balanced_accuracy_score(yt, yp))
            elif metric_name == "roc_auc":
                if len(np.unique(yt)) > 1:
                    vals.append(roc_auc_score(yt, ypr))
            elif metric_name == "mcc":
                vals.append(matthews_corrcoef(yt, yp) if len(np.unique(yt)) > 1 else np.nan)
        except Exception:
            continue
    return bootstrap_ci(vals, cfg.random_state)


def moving_block_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if block_size <= 1 or n <= block_size:
        return rng.integers(0, n, n)
    starts = np.arange(0, n - block_size + 1)
    idx: List[int] = []
    while len(idx) < n:
        s = int(rng.choice(starts))
        idx.extend(range(s, min(s + block_size, n)))
    return np.asarray(idx[:n], dtype=int)


def bootstrap_equity_ci(strategy_returns: pd.Series, cfg: Config) -> Dict[str, Dict[str, float]]:
    r = pd.Series(strategy_returns).fillna(0).astype(float).reset_index(drop=True)
    rng = np.random.default_rng(cfg.random_state)
    total_return_vals, sharpe_vals = [], []
    n = len(r)
    if n == 0:
        return {
            "total_return": {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan},
            "sharpe": {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan},
        }
    for _ in range(cfg.block_bootstrap_iters):
        idx = moving_block_indices(n, cfg.block_size, rng)
        rb = r.iloc[idx].reset_index(drop=True)
        total_return_vals.append(float(rb.sum()))
        std = rb.std(ddof=0)
        sharpe_vals.append(float((rb.mean() / std) * np.sqrt(cfg.annualization)) if std > 0 else np.nan)
    return {
        "total_return": bootstrap_ci(total_return_vals, cfg.random_state),
        "sharpe": bootstrap_ci(sharpe_vals, cfg.random_state),
    }


def mcnemar_test(y_true, pred_a, pred_b) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    denom = n01 + n10
    if denom == 0:
        return {"n01": n01, "n10": n10, "stat": 0.0, "p_value": 1.0}
    stat = (abs(n01 - n10) - 1) ** 2 / denom
    return {"n01": n01, "n10": n10, "stat": float(stat), "p_value": float(1 - chi2.cdf(stat, df=1))}


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


def collect_metric_ci_table(predictions: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    rows = []
    for key, pred_df in predictions.items():
        ds, model = key.split("::", 1)
        for metric_name in ["f1", "balanced_accuracy", "roc_auc", "mcc"]:
            ci = bootstrap_metric_ci(pred_df["target"], pred_df["pred"], pred_df["prob"], metric_name, cfg)
            rows.append({"dataset": ds, "model": model, "metric": metric_name, **ci})
    return pd.DataFrame(rows)


def _classification_metric_value(y_true, y_pred, y_prob, metric_name: str) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    try:
        if metric_name == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))
        if metric_name == "balanced_accuracy":
            return float(balanced_accuracy_score(y_true, y_pred))
        if metric_name == "mcc":
            return float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else np.nan
        if metric_name == "roc_auc":
            return float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    except Exception:
        return np.nan
    return np.nan


def infer_dataset_comparisons(predictions: Dict[str, pd.DataFrame]) -> List[Tuple[str, str, str]]:
    available = {key.split("::", 1)[0] for key in predictions}
    candidates: List[Tuple[str, str, str]] = []
    for name in ["BLOCK_ABLATION_COMPARISONS", "OCA_INTERNAL_COMPARISONS"]:
        candidates.extend(globals().get(name, []))
    seen = set()
    rows = []
    for comparison, with_block, reference in candidates:
        key = (comparison, with_block, reference)
        if with_block in available and reference in available and key not in seen:
            rows.append(key)
            seen.add(key)
    return rows


def paired_classification_block_bootstrap(
    joined: pd.DataFrame,
    metric_name: str,
    cfg: Config,
) -> Dict[str, float]:
    if joined.empty:
        return {
            "metric_with": np.nan,
            "metric_reference": np.nan,
            "delta": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "n": 0,
        }

    joined = joined.sort_values("date").reset_index(drop=True)
    y = joined["target"].to_numpy()
    pred_with = joined["pred_with"].to_numpy()
    pred_ref = joined["pred_reference"].to_numpy()
    prob_with = joined["prob_with"].to_numpy()
    prob_ref = joined["prob_reference"].to_numpy()
    n = len(joined)

    metric_with = _classification_metric_value(y, pred_with, prob_with, metric_name)
    metric_ref = _classification_metric_value(y, pred_ref, prob_ref, metric_name)
    observed = metric_with - metric_ref

    rng = np.random.default_rng(cfg.random_state)
    boot = []
    for _ in range(cfg.block_bootstrap_iters):
        idx = moving_block_indices(n, cfg.block_size, rng)
        val_with = _classification_metric_value(y[idx], pred_with[idx], prob_with[idx], metric_name)
        val_ref = _classification_metric_value(y[idx], pred_ref[idx], prob_ref[idx], metric_name)
        diff = val_with - val_ref
        if np.isfinite(diff):
            boot.append(diff)

    boot_arr = np.asarray(boot, dtype=float)
    if len(boot_arr) == 0 or not np.isfinite(observed):
        ci_low = ci_high = p_value = np.nan
    else:
        ci_low = float(np.quantile(boot_arr, 0.025))
        ci_high = float(np.quantile(boot_arr, 0.975))
        p_value = float(2 * min(np.mean(boot_arr <= 0), np.mean(boot_arr >= 0)))
        p_value = min(1.0, p_value)

    return {
        "metric_with": float(metric_with) if np.isfinite(metric_with) else np.nan,
        "metric_reference": float(metric_ref) if np.isfinite(metric_ref) else np.nan,
        "delta": float(observed) if np.isfinite(observed) else np.nan,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n": int(n),
    }


def collect_paired_metric_block_bootstrap_tests(
    predictions: Dict[str, pd.DataFrame],
    cfg: Config,
    comparisons: List[Tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    rows = []
    if comparisons is None:
        comparisons = infer_dataset_comparisons(predictions)

    pred_map: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, pred_df in predictions.items():
        ds, model = key.split("::", 1)
        pred_map[(ds, model)] = pred_df

    for comparison, with_block, reference in comparisons:
        models = sorted({m for ds, m in pred_map if ds == with_block} & {m for ds, m in pred_map if ds == reference})
        for model in models:
            left = pred_map[(with_block, model)][["date", "target", "pred", "prob"]].rename(
                columns={"pred": "pred_with", "prob": "prob_with"}
            )
            right = pred_map[(reference, model)][["date", "target", "pred", "prob"]].rename(
                columns={"pred": "pred_reference", "prob": "prob_reference"}
            )
            joined = left.merge(right, on=["date", "target"], how="inner")
            for metric_name in ["balanced_accuracy", "mcc", "roc_auc", "f1"]:
                test = paired_classification_block_bootstrap(joined, metric_name, cfg)
                rows.append(
                    {
                        "comparison": comparison,
                        "with_block": with_block,
                        "reference": reference,
                        "model": model,
                        "metric": metric_name,
                        **test,
                    }
                )
    return pd.DataFrame(rows)


def collect_regime_paired_metric_block_bootstrap_tests(
    predictions: Dict[str, pd.DataFrame],
    regimes_by_dataset: Dict[str, pd.DataFrame],
    cfg: Config,
    comparisons: List[Tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    rows = []
    if comparisons is None:
        comparisons = infer_dataset_comparisons(predictions)

    pred_map: Dict[Tuple[str, str], pd.DataFrame] = {}
    for key, pred_df in predictions.items():
        ds, model = key.split("::", 1)
        pred_map[(ds, model)] = pred_df

    regime_cols = ["trend_regime", "vol_regime", "market_regime", "regime_transition"]
    for comparison, with_block, reference in comparisons:
        if with_block not in regimes_by_dataset:
            continue
        regime_df = regimes_by_dataset[with_block][["date"] + regime_cols].copy()
        models = sorted({m for ds, m in pred_map if ds == with_block} & {m for ds, m in pred_map if ds == reference})
        for model in models:
            left = pred_map[(with_block, model)][["date", "target", "pred", "prob"]].rename(
                columns={"pred": "pred_with", "prob": "prob_with"}
            )
            right = pred_map[(reference, model)][["date", "target", "pred", "prob"]].rename(
                columns={"pred": "pred_reference", "prob": "prob_reference"}
            )
            joined = left.merge(right, on=["date", "target"], how="inner").merge(regime_df, on="date", how="left")
            for regime_type in ["trend_regime", "vol_regime", "market_regime"]:
                for regime, grp in joined.groupby(regime_type):
                    if len(grp) < 15:
                        continue
                    for metric_name in ["balanced_accuracy", "mcc", "roc_auc", "f1"]:
                        test = paired_classification_block_bootstrap(grp, metric_name, cfg)
                        rows.append(
                            {
                                "comparison": comparison,
                                "with_block": with_block,
                                "reference": reference,
                                "model": model,
                                "regime_type": regime_type,
                                "regime": regime,
                                "metric": metric_name,
                                **test,
                            }
                        )
            for label, grp in [("transition", joined[joined["regime_transition"] == 1]), ("stable", joined[joined["regime_transition"] == 0])]:
                if len(grp) < 15:
                    continue
                for metric_name in ["balanced_accuracy", "mcc", "roc_auc", "f1"]:
                    test = paired_classification_block_bootstrap(grp, metric_name, cfg)
                    rows.append(
                        {
                            "comparison": comparison,
                            "with_block": with_block,
                            "reference": reference,
                            "model": model,
                            "regime_type": "transition_layer",
                            "regime": label,
                            "metric": metric_name,
                            **test,
                        }
                    )
    return pd.DataFrame(rows)


def collect_mcnemar_tests(predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ds in sorted({key.split("::", 1)[0] for key in predictions}):
        ds_preds = {key.split("::", 1)[1]: pred for key, pred in predictions.items() if key.startswith(f"{ds}::")}
        real_models = [m for m in ds_preds if m not in {"always_up", "random", "momentum"}]
        if not real_models:
            continue
        best_model = max(real_models, key=lambda m: f1_score(ds_preds[m]["target"], ds_preds[m]["pred"], zero_division=0))
        references = ["momentum", "always_up", best_model]
        for ref in references:
            if ref not in ds_preds:
                continue
            ref_df = ds_preds[ref][["date", "target", "pred"]].rename(columns={"pred": "ref_pred"})
            for model, cur_df in ds_preds.items():
                if model == ref:
                    continue
                joined = ref_df.merge(cur_df[["date", "pred"]].rename(columns={"pred": "model_pred"}), on="date", how="inner")
                if joined.empty:
                    continue
                test = mcnemar_test(joined["target"], joined["model_pred"], joined["ref_pred"])
                rows.append({"dataset": ds, "model": model, "reference_model": ref, **test})
    return pd.DataFrame(rows)


def collect_equity_ci_table(equity_curves: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    if equity_curves.empty:
        return pd.DataFrame()
    for (ds, model), grp in equity_curves.groupby(["dataset", "model"]):
        ci = bootstrap_equity_ci(grp.sort_values("date")["strategy_return"], cfg)
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "total_return_mean": ci["total_return"]["mean"],
                "total_return_ci_low": ci["total_return"]["ci_low"],
                "total_return_ci_high": ci["total_return"]["ci_high"],
                "sharpe_mean": ci["sharpe"]["mean"],
                "sharpe_ci_low": ci["sharpe"]["ci_low"],
                "sharpe_ci_high": ci["sharpe"]["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def collect_equity_pairwise_tests(equity_curves: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    if equity_curves.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(cfg.random_state)
    for ds, ds_eq in equity_curves.groupby("dataset"):
        models = sorted(ds_eq["model"].unique())
        references = [m for m in ["always_up", "momentum"] if m in models]
        real_models = [m for m in models if m not in {"always_up", "random", "momentum"}]
        if real_models:
            totals = ds_eq.groupby("model")["strategy_return"].sum()
            references.append(str(totals.loc[real_models].idxmax()))
        for ref in dict.fromkeys(references):
            ref_df = ds_eq[ds_eq["model"] == ref][["date", "strategy_return"]].rename(columns={"strategy_return": "ref_return"})
            for model in models:
                if model == ref:
                    continue
                cur_df = ds_eq[ds_eq["model"] == model][["date", "strategy_return"]].rename(columns={"strategy_return": "model_return"})
                joined = cur_df.merge(ref_df, on="date", how="inner").sort_values("date")
                if joined.empty:
                    continue
                diff = (joined["model_return"] - joined["ref_return"]).reset_index(drop=True)
                observed = float(diff.sum())
                boot = []
                for _ in range(cfg.block_bootstrap_iters):
                    idx = moving_block_indices(len(diff), cfg.block_size, rng)
                    boot.append(float(diff.iloc[idx].sum()))
                p_value = float(2 * min(np.mean(np.asarray(boot) <= 0), np.mean(np.asarray(boot) >= 0)))
                rows.append(
                    {
                        "dataset": ds,
                        "model": model,
                        "reference_model": ref,
                        "return_diff": observed,
                        "ci_low": float(np.quantile(boot, 0.025)),
                        "ci_high": float(np.quantile(boot, 0.975)),
                        "p_value": min(1.0, p_value),
                    }
                )
    return pd.DataFrame(rows)


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray) -> Dict[str, float]:
    """Diebold-Mariano style test for equal predictive accuracy.

    Positive mean_loss_diff means model A has larger loss than model B.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 5:
        return {"mean_loss_diff": np.nan, "dm_stat": np.nan, "p_value": np.nan, "n": n}
    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    if var_d <= 0:
        return {"mean_loss_diff": mean_d, "dm_stat": 0.0, "p_value": 1.0, "n": n}
    dm_stat = mean_d / np.sqrt(var_d / n)
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return {"mean_loss_diff": mean_d, "dm_stat": float(dm_stat), "p_value": float(p_value), "n": n}


def collect_dm_tests(predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for ds in sorted({key.split("::", 1)[0] for key in predictions}):
        ds_preds = {key.split("::", 1)[1]: pred for key, pred in predictions.items() if key.startswith(f"{ds}::")}
        real_models = [m for m in ds_preds if m not in {"always_up", "random", "momentum"}]
        if not real_models:
            continue
        best_model = max(real_models, key=lambda m: f1_score(ds_preds[m]["target"], ds_preds[m]["pred"], zero_division=0))
        references = list(dict.fromkeys(["momentum", "always_up", best_model]))
        for ref in references:
            if ref not in ds_preds:
                continue
            ref_df = ds_preds[ref][["date", "target", "prob"]].rename(columns={"prob": "ref_prob"})
            for model, cur_df in ds_preds.items():
                if model == ref:
                    continue
                joined = cur_df[["date", "target", "prob"]].rename(columns={"prob": "model_prob"}).merge(
                    ref_df, on=["date", "target"], how="inner"
                )
                if joined.empty:
                    continue
                y = joined["target"].values.astype(float)
                model_prob = np.clip(joined["model_prob"].values.astype(float), 1e-6, 1 - 1e-6)
                ref_prob = np.clip(joined["ref_prob"].values.astype(float), 1e-6, 1 - 1e-6)
                losses = {
                    "brier": ((y - model_prob) ** 2, (y - ref_prob) ** 2),
                    "logloss": (
                        -(y * np.log(model_prob) + (1 - y) * np.log(1 - model_prob)),
                        -(y * np.log(ref_prob) + (1 - y) * np.log(1 - ref_prob)),
                    ),
                    "zero_one": ((model_prob >= 0.5).astype(int) != y, (ref_prob >= 0.5).astype(int) != y),
                }
                for loss_name, (loss_model, loss_ref) in losses.items():
                    test = dm_test(loss_model, loss_ref)
                    rows.append(
                        {
                            "dataset": ds,
                            "model": model,
                            "reference_model": ref,
                            "loss": loss_name,
                            **test,
                        }
                    )
    return pd.DataFrame(rows)


def collect_fee_sensitivity(predictions: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    rows = []
    for fee in cfg.fee_sensitivity_grid:
        fee_cfg = replace(cfg, fee=float(fee))
        for key, pred_df in predictions.items():
            ds, model = key.split("::", 1)
            eq = equity_curve(pred_df.rename(columns={"pred": "signal"}), "signal", fee_cfg)
            rows.append({"dataset": ds, "model": model, "fee": float(fee), **equity_stats(eq, fee_cfg)})
    return pd.DataFrame(rows)


def collect_stationarity_diagnostics(prepared_datasets: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except Exception as exc:
        return pd.DataFrame([{"error": f"statsmodels unavailable: {exc}"}])

    rows = []
    key_candidates = [
        "future_return",
        cfg.close_col,
        "btc_close",
        "dxy",
        "sp500_close",
        "net_flow",
        "inflow",
        "outflow",
        "rolling_netflow_mean_spike",
        "avg_whale_amount",
        "MFI",
        "macd_z",
    ]
    for ds, df in prepared_datasets.items():
        cols = [c for c in key_candidates if c in df.columns]
        for col in cols:
            s = pd.Series(df[col]).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
            if col == cfg.close_col:
                s = np.log(s.replace(0, np.nan)).diff().dropna()
                series_name = f"logdiff_{col}"
            else:
                series_name = col
            if len(s) < 30 or s.nunique() <= 2:
                continue
            try:
                adf_stat, adf_p, *_ = adfuller(s.values, autolag="AIC")
            except Exception:
                adf_stat, adf_p = np.nan, np.nan
            try:
                kpss_stat, kpss_p, *_ = kpss(s.values, regression="c", nlags="auto")
            except Exception:
                kpss_stat, kpss_p = np.nan, np.nan
            rows.append(
                {
                    "dataset": ds,
                    "series": series_name,
                    "n": len(s),
                    "adf_stat": adf_stat,
                    "adf_p_value": adf_p,
                    "kpss_stat": kpss_stat,
                    "kpss_p_value": kpss_p,
                    "adf_stationary_at_5pct": bool(adf_p < 0.05) if np.isfinite(adf_p) else np.nan,
                    "kpss_stationary_at_5pct": bool(kpss_p > 0.05) if np.isfinite(kpss_p) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def elastic_net_feature_robustness(
    prepared_datasets: Dict[str, pd.DataFrame],
    feature_selection_scores: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    rows = []
    selected_lookup = {}
    if not feature_selection_scores.empty:
        fs = feature_selection_scores.copy()
        fs = fs[fs.get("selected", False) == True]
        selected_lookup = {
            ds: set(grp["feature"].astype(str))
            for ds, grp in fs.groupby("dataset")
        }

    for ds, df in prepared_datasets.items():
        train_mask = df[cfg.date_col] < pd.Timestamp(cfg.first_forecast_date)
        if cfg.train_start_date is not None:
            train_mask &= df[cfg.date_col] >= pd.Timestamp(cfg.train_start_date)
        feats = feature_columns(df, cfg)
        X = safe_fill(df.loc[train_mask, feats])
        y = df.loc[train_mask, "target"].copy()
        if len(X) < 100 or y.nunique() < 2:
            continue
        keep = X.columns[X.nunique(dropna=False) > 1].tolist()
        X = X[keep]
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.2,
            max_iter=5000,
            class_weight="balanced",
            random_state=cfg.random_state,
            n_jobs=1,
        )
        try:
            model.fit(Xs, y)
            coefs = np.abs(model.coef_[0])
        except Exception:
            coefs = np.zeros(len(keep))
        coef_s = pd.Series(coefs, index=keep)
        nonzero = set(coef_s[coef_s > 1e-8].index.astype(str))
        selected = selected_lookup.get(ds, set())
        denom = len(selected | nonzero) if len(selected | nonzero) else np.nan
        overlap = len(selected & nonzero) / denom if denom else np.nan
        for feat, val in coef_s.sort_values(ascending=False).items():
            rows.append(
                {
                    "dataset": ds,
                    "feature": feat,
                    "abs_coef": float(val),
                    "elastic_net_selected": bool(val > 1e-8),
                    "main_selector_selected": bool(feat in selected),
                    "selected_overlap_jaccard": overlap,
                }
            )
    return pd.DataFrame(rows)


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
    overfit_rows: List[dict] = []
    sequence_params: Dict[str, Dict[str, float | int]] = {}

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

        if cfg.run_sequence_models and cfg.tune_sequence_models:
            fixed_X_train = X_all.loc[fixed_train_mask]
            fixed_y_train = y_all.loc[fixed_train_mask]
            sequence_params["lstm"] = tune_sequence_params(
                fixed_X_train, fixed_y_train, fixed_selected, cfg, bidirectional=False
            )
            sequence_params["bilstm"] = tune_sequence_params(
                fixed_X_train, fixed_y_train, fixed_selected, cfg, bidirectional=True
            )
            pd.DataFrame(
                [
                    {"dataset": name, "model": model_name, **params}
                    for model_name, params in sequence_params.items()
                ]
            ).to_csv(out_dir / f"{name}_sequence_tuning.csv", index=False)
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
            train_prob, train_pred = predict_tabular_model(model, X_train[selected], cfg)
            train_m = binary_metrics(y_train, train_pred, train_prob)
            test_m = binary_metrics(y_test, pred, prob)
            overfit_rows.append(
                {
                    "dataset": name,
                    "model": model_name,
                    "outer_fold": fold_no,
                    "forecast_start": start,
                    "forecast_end": end,
                    "n_features": len(selected),
                    "train_f1": train_m["f1"],
                    "test_f1": test_m["f1"],
                    "f1_generalization_gap": train_m["f1"] - test_m["f1"],
                    "train_balanced_accuracy": train_m["balanced_accuracy"],
                    "test_balanced_accuracy": test_m["balanced_accuracy"],
                    "balanced_accuracy_gap": train_m["balanced_accuracy"] - test_m["balanced_accuracy"],
                    "train_roc_auc": train_m["roc_auc"],
                    "test_roc_auc": test_m["roc_auc"],
                    "roc_auc_gap": train_m["roc_auc"] - test_m["roc_auc"],
                }
            )
            eval_df = df.loc[test_mask, ["date", "future_return", "target"]].copy()
            eval_df["pred"] = pred
            eval_df["prob"] = prob
            eval_df["outer_fold"] = fold_no
            eval_df["n_features"] = len(selected)
            prediction_parts[f"{name}::{model_name}"].append(eval_df)

        if cfg.run_sequence_models:
            for model_name, bidir in [("lstm", False), ("bilstm", True)]:
                prob, pred, idx = fit_sequence_predict(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    selected,
                    cfg,
                    bidir,
                    params=sequence_params.get(model_name),
                )
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
    if overfit_rows:
        pd.DataFrame(overfit_rows).to_csv(out_dir / f"{name}_overfit_diagnostics.csv", index=False)

    return predictions, regimes, importance_rows


def summarize_equity(predictions: Dict[str, pd.DataFrame], cfg: Config, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    equity_rows, curve_frames = [], []
    for key, pred_df in predictions.items():
        ds, model = key.split("::", 1)
        pred_df = pred_df.sort_values("date")
        pred_df.to_csv(out_dir / f"{ds}_{model}_predictions.csv", index=False)
        eq = equity_curve(pred_df.rename(columns={"pred": "signal"}), "signal", cfg)
        st = equity_stats(eq, cfg)
        equity_rows.append(
            {
                "dataset": ds,
                "model": model,
                "execution_rule": "signal_t_trade_t_to_t_plus_1",
                **st,
            }
        )
        curve_frames.append(
            eq[["date", "trade_open_date", "trade_close_date", "strategy_return", "equity"]].assign(
                dataset=ds, model=model
            )
        )
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
    curves.to_csv(out_dir / "equity_curves_same_day_execution.csv", index=False)
    for ds, grp in curves.groupby("dataset"):
        plt.figure(figsize=(12, 6))
        for model, mg in grp.groupby("model"):
            plt.plot(pd.to_datetime(mg["date"]), mg["equity"], label=model, linewidth=1.3)
        plt.title(f"Equity curves, signal t -> trade t to t+1: {ds}")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{ds}_equity_curves_same_day_execution.png", dpi=150)
        plt.close()


def run_pipeline_from_datasets(
    datasets: Dict[str, pd.DataFrame],
    cfg: Config | None = None,
    out_dir: str | Path = "results/ablation_pipeline",
) -> Dict[str, object]:
    """Run the article pipeline on an arbitrary named dataset dictionary.

    This is the preferred entry point for block-level ablation, where the
    dataset names are experimental specifications such as baseline, oca,
    baseline_oca, baseline_ta, oca_ta, and full.
    """
    if cfg is None:
        cfg = Config()
    set_seed(cfg.random_state)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    prepared_datasets = {
        name: prepare_dataframe(df, cfg)
        for name, df in datasets.items()
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

    metric_ci = collect_metric_ci_table(all_predictions, cfg)
    metric_ci.to_csv(out_path / "metric_confidence_intervals.csv", index=False)
    paired_metric_tests = collect_paired_metric_block_bootstrap_tests(all_predictions, cfg)
    paired_metric_tests.to_csv(out_path / "paired_metric_block_bootstrap_tests.csv", index=False)
    equity_ci = collect_equity_ci_table(equity_curves, cfg)
    equity_ci.to_csv(out_path / "equity_confidence_intervals.csv", index=False)
    mcnemar_tests = collect_mcnemar_tests(all_predictions)
    mcnemar_tests.to_csv(out_path / "mcnemar_tests.csv", index=False)
    equity_pairwise_tests = collect_equity_pairwise_tests(equity_curves, cfg)
    equity_pairwise_tests.to_csv(out_path / "equity_pairwise_tests.csv", index=False)
    dm_tests = collect_dm_tests(all_predictions)
    dm_tests.to_csv(out_path / "diebold_mariano_tests.csv", index=False)
    fee_sensitivity = collect_fee_sensitivity(all_predictions, cfg)
    fee_sensitivity.to_csv(out_path / "fee_sensitivity.csv", index=False)
    stationarity_diagnostics = collect_stationarity_diagnostics(prepared_datasets, cfg)
    stationarity_diagnostics.to_csv(out_path / "stationarity_diagnostics.csv", index=False)

    regime_metrics = summarize_regimes(all_predictions, regimes_by_dataset)
    regime_metrics.to_csv(out_path / "regime_metrics.csv", index=False)
    regime_paired_metric_tests = collect_regime_paired_metric_block_bootstrap_tests(all_predictions, regimes_by_dataset, cfg)
    regime_paired_metric_tests.to_csv(out_path / "regime_paired_metric_block_bootstrap_tests.csv", index=False)

    regime_feature_importance = pd.DataFrame(all_importance)
    regime_feature_importance.to_csv(out_path / "regime_feature_importance.csv", index=False)

    hypothesis_tables(out_path, all_predictions, prepared_datasets=prepared_datasets)

    feature_selection_files = list(out_path.glob("*_feature_selection_scores.csv"))
    feature_selection_scores = (
        pd.concat([pd.read_csv(p) for p in feature_selection_files], ignore_index=True)
        if feature_selection_files
        else pd.DataFrame()
    )
    elastic_net_robustness = elastic_net_feature_robustness(prepared_datasets, feature_selection_scores, cfg)
    elastic_net_robustness.to_csv(out_path / "elastic_net_feature_robustness.csv", index=False)

    cluster_files = list(out_path.glob("*_feature_clusters.csv"))
    feature_clusters = (
        pd.concat([pd.read_csv(p) for p in cluster_files], ignore_index=True)
        if cluster_files
        else pd.DataFrame()
    )
    sequence_tuning_files = list(out_path.glob("*_sequence_tuning.csv"))
    sequence_tuning = (
        pd.concat([pd.read_csv(p) for p in sequence_tuning_files], ignore_index=True)
        if sequence_tuning_files
        else pd.DataFrame()
    )
    overfit_files = list(out_path.glob("*_overfit_diagnostics.csv"))
    overfit_diagnostics = (
        pd.concat([pd.read_csv(p) for p in overfit_files], ignore_index=True)
        if overfit_files
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
            hypothesis_artifacts[path.stem] = read_csv_if_nonempty(path)

    artifacts = {
        "config": cfg,
        "out_dir": out_path,
        "prepared_datasets": prepared_datasets,
        "predictions": all_predictions,
        "regimes_by_dataset": regimes_by_dataset,
        "model_metrics": model_metrics,
        "equity_stats": equity_stats_table,
        "equity_curves": equity_curves,
        "metric_confidence_intervals": metric_ci,
        "paired_metric_block_bootstrap_tests": paired_metric_tests,
        "equity_confidence_intervals": equity_ci,
        "mcnemar_tests": mcnemar_tests,
        "equity_pairwise_tests": equity_pairwise_tests,
        "diebold_mariano_tests": dm_tests,
        "fee_sensitivity": fee_sensitivity,
        "stationarity_diagnostics": stationarity_diagnostics,
        "regime_metrics": regime_metrics,
        "regime_paired_metric_block_bootstrap_tests": regime_paired_metric_tests,
        "regime_feature_importance": regime_feature_importance,
        "feature_selection_scores": feature_selection_scores,
        "feature_clusters": feature_clusters,
        "elastic_net_feature_robustness": elastic_net_robustness,
        "overfit_diagnostics": overfit_diagnostics,
        "sequence_tuning": sequence_tuning,
        "hypothesis_artifacts": hypothesis_artifacts,
    }

    print(f"\nModel metrics ({cfg.feature_selection_mode}):")
    print(model_metrics.sort_values(["dataset", "f1"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved artifacts to {out_path}")
    return artifacts


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

    metric_ci = collect_metric_ci_table(all_predictions, cfg)
    metric_ci.to_csv(out_path / "metric_confidence_intervals.csv", index=False)
    paired_metric_tests = collect_paired_metric_block_bootstrap_tests(all_predictions, cfg)
    paired_metric_tests.to_csv(out_path / "paired_metric_block_bootstrap_tests.csv", index=False)
    equity_ci = collect_equity_ci_table(equity_curves, cfg)
    equity_ci.to_csv(out_path / "equity_confidence_intervals.csv", index=False)
    mcnemar_tests = collect_mcnemar_tests(all_predictions)
    mcnemar_tests.to_csv(out_path / "mcnemar_tests.csv", index=False)
    equity_pairwise_tests = collect_equity_pairwise_tests(equity_curves, cfg)
    equity_pairwise_tests.to_csv(out_path / "equity_pairwise_tests.csv", index=False)
    dm_tests = collect_dm_tests(all_predictions)
    dm_tests.to_csv(out_path / "diebold_mariano_tests.csv", index=False)
    fee_sensitivity = collect_fee_sensitivity(all_predictions, cfg)
    fee_sensitivity.to_csv(out_path / "fee_sensitivity.csv", index=False)
    stationarity_diagnostics = collect_stationarity_diagnostics(prepared_datasets, cfg)
    stationarity_diagnostics.to_csv(out_path / "stationarity_diagnostics.csv", index=False)

    regime_metrics = summarize_regimes(all_predictions, regimes_by_dataset)
    regime_metrics.to_csv(out_path / "regime_metrics.csv", index=False)
    regime_paired_metric_tests = collect_regime_paired_metric_block_bootstrap_tests(all_predictions, regimes_by_dataset, cfg)
    regime_paired_metric_tests.to_csv(out_path / "regime_paired_metric_block_bootstrap_tests.csv", index=False)

    regime_feature_importance = pd.DataFrame(all_importance)
    regime_feature_importance.to_csv(out_path / "regime_feature_importance.csv", index=False)

    hypothesis_tables(out_path, all_predictions, prepared_datasets=prepared_datasets)

    feature_selection_files = list(out_path.glob("*_feature_selection_scores.csv"))
    feature_selection_scores = (
        pd.concat([pd.read_csv(p) for p in feature_selection_files], ignore_index=True)
        if feature_selection_files
        else pd.DataFrame()
    )
    elastic_net_robustness = elastic_net_feature_robustness(prepared_datasets, feature_selection_scores, cfg)
    elastic_net_robustness.to_csv(out_path / "elastic_net_feature_robustness.csv", index=False)

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
            hypothesis_artifacts[path.stem] = read_csv_if_nonempty(path)

    sequence_tuning_files = list(out_path.glob("*_sequence_tuning.csv"))
    sequence_tuning = (
        pd.concat([pd.read_csv(p) for p in sequence_tuning_files], ignore_index=True)
        if sequence_tuning_files
        else pd.DataFrame()
    )
    overfit_files = list(out_path.glob("*_overfit_diagnostics.csv"))
    overfit_diagnostics = (
        pd.concat([pd.read_csv(p) for p in overfit_files], ignore_index=True)
        if overfit_files
        else pd.DataFrame()
    )

    artifacts = {
        "config": cfg,
        "out_dir": out_path,
        "prepared_datasets": prepared_datasets,
        "predictions": all_predictions,
        "regimes_by_dataset": regimes_by_dataset,
        "model_metrics": model_metrics,
        "equity_stats": equity_stats_table,
        "equity_curves": equity_curves,
        "metric_confidence_intervals": metric_ci,
        "paired_metric_block_bootstrap_tests": paired_metric_tests,
        "equity_confidence_intervals": equity_ci,
        "mcnemar_tests": mcnemar_tests,
        "equity_pairwise_tests": equity_pairwise_tests,
        "diebold_mariano_tests": dm_tests,
        "fee_sensitivity": fee_sensitivity,
        "stationarity_diagnostics": stationarity_diagnostics,
        "regime_metrics": regime_metrics,
        "regime_paired_metric_block_bootstrap_tests": regime_paired_metric_tests,
        "regime_feature_importance": regime_feature_importance,
        "feature_selection_scores": feature_selection_scores,
        "feature_clusters": feature_clusters,
        "elastic_net_feature_robustness": elastic_net_robustness,
        "overfit_diagnostics": overfit_diagnostics,
        "sequence_tuning": sequence_tuning,
        "hypothesis_artifacts": hypothesis_artifacts,
    }

    print(f"\nModel metrics ({cfg.feature_selection_mode}):")
    print(model_metrics.sort_values(["dataset", "f1"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved notebook artifacts to {out_path}")
    return artifacts


def run_holdout_robustness(
    oca: pd.DataFrame,
    ta: pd.DataFrame,
    baseline_model: pd.DataFrame,
    oca_ta: pd.DataFrame,
    holdout_periods: List[Tuple[str, str]] | None = None,
    cfg: Config | None = None,
    out_dir: str | Path = "results/holdout_robustness_pipeline",
) -> Dict[str, object]:
    """Run the same fixed-train design over multiple holdout periods.

    Example:
        robustness = run_holdout_robustness(
            df_merged_oca,
            df_merged_ta,
            df_merged_base,
            df_merged_oca_ta,
            holdout_periods=[
                ("2023-11-01", "2024-11-01"),
                ("2024-11-01", "2025-11-01"),
            ],
        )
    """
    if cfg is None:
        cfg = Config()
    if holdout_periods is None:
        holdout_periods = [
            ("2023-11-01", "2024-05-01"),
            ("2024-05-01", "2024-11-01"),
            ("2024-11-01", "2025-05-01"),
            ("2025-05-01", "2025-11-01"),
        ]

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    period_artifacts: Dict[str, Dict[str, object]] = {}
    summary_frames: Dict[str, List[pd.DataFrame]] = {
        "model_metrics": [],
        "equity_stats": [],
        "metric_confidence_intervals": [],
        "paired_metric_block_bootstrap_tests": [],
        "equity_confidence_intervals": [],
        "mcnemar_tests": [],
        "equity_pairwise_tests": [],
        "diebold_mariano_tests": [],
        "fee_sensitivity": [],
        "stationarity_diagnostics": [],
        "regime_paired_metric_block_bootstrap_tests": [],
        "elastic_net_feature_robustness": [],
        "overfit_diagnostics": [],
    }

    for start, end in holdout_periods:
        label = f"{start}_to_{end}".replace("-", "")
        period_cfg = replace(cfg, first_forecast_date=start, last_forecast_date=end)
        print(f"\n=== Holdout robustness: {start} to {end} ===")
        artifacts = run_full_pipeline(
            oca,
            ta,
            baseline_model,
            oca_ta,
            cfg=period_cfg,
            out_dir=root / label,
        )
        period_artifacts[label] = artifacts
        for key in summary_frames:
            frame = artifacts.get(key)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                tmp = frame.copy()
                tmp.insert(0, "holdout_start", start)
                tmp.insert(1, "holdout_end", end)
                tmp.insert(2, "holdout_label", label)
                summary_frames[key].append(tmp)

    summaries: Dict[str, pd.DataFrame] = {}
    for key, frames in summary_frames.items():
        summaries[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        summaries[key].to_csv(root / f"{key}_all_holdouts.csv", index=False)

    return {
        "config": cfg,
        "out_dir": root,
        "period_artifacts": period_artifacts,
        "summaries": summaries,
    }


BLOCK_ABLATION_COMPARISONS = [
    ("oca_vs_baseline", "oca", "baseline"),
    ("oca_added_to_baseline", "baseline_oca", "baseline"),
    ("oca_added_to_ta", "oca_ta", "ta"),
    ("oca_added_to_baseline_ta", "full", "baseline_ta"),
    ("ta_added_to_baseline", "baseline_ta", "baseline"),
    ("ta_added_to_baseline_oca", "full", "baseline_oca"),
    ("baseline_added_to_oca_ta", "full", "oca_ta"),
    ("full_vs_baseline", "full", "baseline"),
    ("full_vs_oca", "full", "oca"),
    ("full_vs_ta", "full", "ta"),
]


def load_ablation_csv_datasets(base_dir: str | Path = "datasets") -> Dict[str, pd.DataFrame]:
    """Load the seven block-level ablation datasets from CSV files."""
    base = Path(base_dir)
    paths = {
        "baseline": base / "merged_base.csv",
        "oca": base / "merged_oca.csv",
        "ta": base / "merged_ta.csv",
        "baseline_oca": base / "baseline_oca.csv",
        "baseline_ta": base / "baseline_ta.csv",
        "oca_ta": base / "oca_ta.csv",
        "full": base / "full.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing ablation dataset CSV files: {missing}")
    return {name: pd.read_csv(path) for name, path in paths.items()}


def _pairwise_delta_table(
    frame: pd.DataFrame,
    comparisons: List[Tuple[str, str, str]],
    metric_cols: List[str],
    index_cols: List[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    if index_cols is None:
        index_cols = [c for c in ["holdout_start", "holdout_end", "holdout_label", "model"] if c in frame.columns]

    rows = []
    available = set(frame["dataset"].unique()) if "dataset" in frame.columns else set()
    for comparison, with_block, reference in comparisons:
        if with_block not in available or reference not in available:
            continue
        left = frame[frame["dataset"] == with_block].copy()
        right = frame[frame["dataset"] == reference].copy()
        keep_cols = index_cols + metric_cols
        left = left[[c for c in keep_cols if c in left.columns]]
        right = right[[c for c in keep_cols if c in right.columns]]
        merged = left.merge(right, on=index_cols, suffixes=("_with", "_reference"), how="inner")
        for _, r in merged.iterrows():
            row = {
                "comparison": comparison,
                "with_block": with_block,
                "reference": reference,
            }
            for c in index_cols:
                row[c] = r[c]
            for metric in metric_cols:
                lw = f"{metric}_with"
                rr = f"{metric}_reference"
                if lw not in merged.columns or rr not in merged.columns:
                    continue
                row[f"{metric}_with"] = r[lw]
                row[f"{metric}_reference"] = r[rr]
                if metric in {"brier", "logloss"}:
                    row[f"delta_{metric}_improvement"] = r[rr] - r[lw]
                else:
                    row[f"delta_{metric}"] = r[lw] - r[rr]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_ablation_deltas(
    model_metrics: pd.DataFrame,
    equity_stats_table: pd.DataFrame,
    comparisons: List[Tuple[str, str, str]] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Create paired delta tables for block-level ablation comparisons."""
    if comparisons is None:
        comparisons = BLOCK_ABLATION_COMPARISONS
    metric_cols = [
        c
        for c in ["f1", "balanced_accuracy", "mcc", "roc_auc", "brier", "logloss"]
        if c in model_metrics.columns
    ]
    equity_cols = [
        c
        for c in ["total_return", "sharpe", "max_drawdown", "win_rate", "n_trades", "turnover"]
        if c in equity_stats_table.columns
    ]
    classification_deltas = _pairwise_delta_table(model_metrics, comparisons, metric_cols)
    equity_deltas = _pairwise_delta_table(equity_stats_table, comparisons, equity_cols)

    def aggregate(deltas: pd.DataFrame) -> pd.DataFrame:
        if deltas.empty:
            return pd.DataFrame()
        delta_cols = [c for c in deltas.columns if c.startswith("delta_")]
        group_cols = ["comparison", "with_block", "reference", "model"]
        rows = []
        for keys, grp in deltas.groupby(group_cols):
            row = dict(zip(group_cols, keys))
            for col in delta_cols:
                vals = pd.to_numeric(grp[col], errors="coerce")
                row[f"mean_{col}"] = vals.mean()
                row[f"std_{col}"] = vals.std(ddof=0)
                row[f"positive_{col}_count"] = int((vals > 0).sum())
                row[f"n_{col}"] = int(vals.notna().sum())
            rows.append(row)
        return pd.DataFrame(rows)

    return {
        "classification_deltas": classification_deltas,
        "equity_deltas": equity_deltas,
        "classification_delta_summary": aggregate(classification_deltas),
        "equity_delta_summary": aggregate(equity_deltas),
    }


def run_holdout_robustness_from_datasets(
    datasets: Dict[str, pd.DataFrame],
    holdout_periods: List[Tuple[str, str]] | None = None,
    cfg: Config | None = None,
    out_dir: str | Path = "results/block_ablation_pipeline",
    comparisons: List[Tuple[str, str, str]] | None = None,
) -> Dict[str, object]:
    """Run holdout robustness for an arbitrary ablation dataset dictionary."""
    if cfg is None:
        cfg = Config()
    if holdout_periods is None:
        holdout_periods = [
            ("2023-11-01", "2024-05-01"),
            ("2024-05-01", "2024-11-01"),
            ("2024-11-01", "2025-05-01"),
            ("2025-05-01", "2025-11-01"),
        ]

    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    period_artifacts: Dict[str, Dict[str, object]] = {}
    summary_frames: Dict[str, List[pd.DataFrame]] = {
        "model_metrics": [],
        "equity_stats": [],
        "metric_confidence_intervals": [],
        "paired_metric_block_bootstrap_tests": [],
        "equity_confidence_intervals": [],
        "mcnemar_tests": [],
        "equity_pairwise_tests": [],
        "diebold_mariano_tests": [],
        "fee_sensitivity": [],
        "stationarity_diagnostics": [],
        "regime_paired_metric_block_bootstrap_tests": [],
        "elastic_net_feature_robustness": [],
        "overfit_diagnostics": [],
    }

    for start, end in holdout_periods:
        label = f"{start}_to_{end}".replace("-", "")
        period_cfg = replace(cfg, first_forecast_date=start, last_forecast_date=end)
        print(f"\n=== Ablation holdout: {start} to {end} ===")
        artifacts = run_pipeline_from_datasets(
            datasets,
            cfg=period_cfg,
            out_dir=root / label,
        )
        period_artifacts[label] = artifacts
        for key in summary_frames:
            frame = artifacts.get(key)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                tmp = frame.copy()
                tmp.insert(0, "holdout_start", start)
                tmp.insert(1, "holdout_end", end)
                tmp.insert(2, "holdout_label", label)
                summary_frames[key].append(tmp)

    summaries: Dict[str, pd.DataFrame] = {}
    for key, frames in summary_frames.items():
        summaries[key] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        summaries[key].to_csv(root / f"{key}_all_holdouts.csv", index=False)

    delta_tables = summarize_ablation_deltas(
        summaries.get("model_metrics", pd.DataFrame()),
        summaries.get("equity_stats", pd.DataFrame()),
        comparisons=comparisons,
    )
    for key, frame in delta_tables.items():
        frame.to_csv(root / f"ablation_{key}.csv", index=False)

    return {
        "config": cfg,
        "out_dir": root,
        "period_artifacts": period_artifacts,
        "summaries": summaries,
        "ablation": delta_tables,
    }


def run_block_ablation_from_csvs(
    cfg: Config | None = None,
    holdout_periods: List[Tuple[str, str]] | None = None,
    out_dir: str | Path = "results/block_ablation_pipeline",
) -> Dict[str, object]:
    """Convenience wrapper: load CSV ablation datasets and run the pipeline."""
    datasets = load_ablation_csv_datasets()
    return run_holdout_robustness_from_datasets(
        datasets,
        holdout_periods=holdout_periods,
        cfg=cfg,
        out_dir=out_dir,
    )


def oca_feature_group(feature: str) -> str:
    f = str(feature).lower()
    if any(x in f for x in ["whale", "large"]):
        return "whale"
    if "spike" in f or "burst" in f:
        return "spike"
    if any(x in f for x in ["inflow", "outflow", "netflow", "net_flow", "flow"]):
        return "flow"
    if any(x in f for x in ["volume", "amount", "tx_count", "transaction", "transfer"]):
        return "volume"
    return "other"


def build_oca_internal_datasets(oca: pd.DataFrame, cfg: Config | None = None) -> Dict[str, pd.DataFrame]:
    """Build OCA-only internal ablation specifications."""
    if cfg is None:
        cfg = Config()
    protected = [cfg.date_col, cfg.close_col]
    feature_cols = [c for c in oca.columns if c not in protected]
    groups = {c: oca_feature_group(c) for c in feature_cols}

    def subset(name: str, cols: List[str]) -> Tuple[str, pd.DataFrame]:
        keep = protected + [c for c in cols if c in oca.columns]
        return name, oca[keep].copy()

    flow = [c for c, g in groups.items() if g == "flow"]
    whale = [c for c, g in groups.items() if g == "whale"]
    spike = [c for c, g in groups.items() if g == "spike"]
    volume = [c for c, g in groups.items() if g == "volume"]
    non_flow = [c for c in feature_cols if groups[c] != "flow"]
    non_whale = [c for c in feature_cols if groups[c] != "whale"]
    non_spike = [c for c in feature_cols if groups[c] != "spike"]

    specs = [
        ("oca_full", oca.copy()),
        subset("flow_only", flow),
        subset("whale_only", whale),
        subset("spike_only", spike),
        subset("volume_only", volume),
        subset("oca_without_flow", non_flow),
        subset("oca_without_whale", non_whale),
        subset("oca_without_spike", non_spike),
    ]
    return {name: df for name, df in specs if len(feature_columns(prepare_dataframe(df, cfg), cfg)) > 0}


OCA_INTERNAL_COMPARISONS = [
    ("flow_only_vs_oca_full", "flow_only", "oca_full"),
    ("whale_only_vs_oca_full", "whale_only", "oca_full"),
    ("spike_only_vs_oca_full", "spike_only", "oca_full"),
    ("volume_only_vs_oca_full", "volume_only", "oca_full"),
    ("remove_flow", "oca_full", "oca_without_flow"),
    ("remove_whale", "oca_full", "oca_without_whale"),
    ("remove_spike", "oca_full", "oca_without_spike"),
]


def run_oca_internal_ablation(
    oca: pd.DataFrame | None = None,
    cfg: Config | None = None,
    holdout_periods: List[Tuple[str, str]] | None = None,
    out_dir: str | Path = "results/oca_internal_ablation_pipeline",
) -> Dict[str, object]:
    """Run internal OCA ablation: flow/whale/spike/volume and leave-one-group-out."""
    if oca is None:
        oca = pd.read_csv("datasets/merged_oca.csv")
    if cfg is None:
        cfg = Config()
    datasets = build_oca_internal_datasets(oca, cfg)
    return run_holdout_robustness_from_datasets(
        datasets,
        holdout_periods=holdout_periods,
        cfg=cfg,
        out_dir=out_dir,
        comparisons=OCA_INTERNAL_COMPARISONS,
    )


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
    collect_metric_ci_table(all_predictions, cfg).to_csv(out_dir / "metric_confidence_intervals.csv", index=False)
    collect_paired_metric_block_bootstrap_tests(all_predictions, cfg).to_csv(out_dir / "paired_metric_block_bootstrap_tests.csv", index=False)
    collect_equity_ci_table(curves, cfg).to_csv(out_dir / "equity_confidence_intervals.csv", index=False)
    collect_mcnemar_tests(all_predictions).to_csv(out_dir / "mcnemar_tests.csv", index=False)
    collect_equity_pairwise_tests(curves, cfg).to_csv(out_dir / "equity_pairwise_tests.csv", index=False)
    collect_dm_tests(all_predictions).to_csv(out_dir / "diebold_mariano_tests.csv", index=False)
    collect_fee_sensitivity(all_predictions, cfg).to_csv(out_dir / "fee_sensitivity.csv", index=False)

    regime_metrics = summarize_regimes(all_predictions, regimes_by_dataset)
    regime_metrics.to_csv(out_dir / "regime_metrics.csv", index=False)
    collect_regime_paired_metric_block_bootstrap_tests(all_predictions, regimes_by_dataset, cfg).to_csv(
        out_dir / "regime_paired_metric_block_bootstrap_tests.csv", index=False
    )

    pd.DataFrame(all_importance).to_csv(out_dir / "regime_feature_importance.csv", index=False)
    hypothesis_tables(out_dir, all_predictions)

    print(f"\nModel metrics ({cfg.feature_selection_mode}):")
    print(metrics.sort_values(["dataset", "f1"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved article-ready results to {out_dir}")


if __name__ == "__main__":
    main()
