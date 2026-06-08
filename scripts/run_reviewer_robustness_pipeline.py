from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


DATASETS = {
    "merged_base": Path("datasets/merged_base.csv"),
    "merged_oca": Path("datasets/merged_oca.csv"),
    "merged_oca_ta": Path("datasets/merged_oca_ta.csv"),
    "merged_ta": Path("datasets/merged_ta.csv"),
}

MARKET_SENTIMENT = {
    "search_trends",
    "btc_close",
    "halving_index",
    "dxy",
    "sp500_close",
    "fed_rate",
    "trading_volume_usd",
    *{f"vmd_{i}" for i in range(1, 11)},
}

TECHNICAL = {
    "CDL_ENGULFING",
    "CDL_HAMMER",
    "CDL_HANGINGMAN",
    "CDL_DOJI",
    "CDL_MORNINGSTAR",
    "CDL_EVENINGSTAR",
    "CDL_3WHITESOLDIERS",
    "CDL_3BLACKCROWS",
    "CDL_PIERCING",
    "CDL_DARKCLOUD",
    "EMA_5",
    "EMA_30",
    "EMA_100",
    "macd",
    "macd_signal",
    "macd_diff",
    "macd_z",
    "macd_price_slope_diff",
    "divergence_strength",
    "rsi",
    "bb_ma",
    "bb_std",
    "bb_upper",
    "bb_lower",
    "bb_dist_upper",
    "bb_dist_lower",
    "bb_pos",
    "bb_break_upper",
    "bb_break_lower",
    "bb_break_lower_cumulative",
    "bb_break_upper_cumulative",
    "OBV",
    "AD",
    "CMF",
    "MFI",
    "volume_avg",
    "ForceIndex",
    "hammer_rsi_low",
    "engulfing_near_bb_lower",
    "mfi_oversold_breakout",
    "macd_bull_cross",
    "bb_rsi_overbought",
    "force_volume_spike",
    "3crows_rsi_high",
    "bullish_signals",
    "bearish_signals",
    "doji_rsi_extreme",
    "hammer_bb_low",
    "hangingman_bearish",
    "engulfing_with_divergence",
    "morningstar_reversal",
    "eveningstar_reversal",
    "soldiers_after_acc",
    "bullish_patterns",
    "bearish_patterns",
}

ONCHAIN = {
    "tx_count",
    "address_count",
    "receiver_count",
    "unique_addresses",
    "inflow",
    "outflow",
    "total_volume",
    "net_flow",
    "mean_amount",
    "std_amount",
    "median_amount",
    "max_amount",
    "min_amount",
    "whale_tx_count",
    "whale_volume",
    "address_entropy",
    "burstiness",
    "inflow_rel",
    "outflow_rel",
    "total_volume_rel",
    "inflow_ratio",
    "outflow_ratio",
    "avg_tx_amount",
    "value_volatility",
    "whale_volume_share",
    "avg_whale_amount",
    "velocity_tpm",
    "rolling_netflow_3h",
    "rolling_netflow_24h",
    "rolling_volume_3h",
    "rolling_volume_24h",
    "rolling_netflow_3h_normed",
    "rolling_netflow_24h_normed",
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

MECHANICAL_VOLUME = {
    "tx_count",
    "address_count",
    "receiver_count",
    "unique_addresses",
    "inflow",
    "outflow",
    "total_volume",
    "net_flow",
    "mean_amount",
    "median_amount",
    "max_amount",
    "min_amount",
    "total_volume_rel",
    "avg_tx_amount",
    "rolling_volume_3h",
    "rolling_volume_24h",
}

PRICE_DERIVED = TECHNICAL | {f"vmd_{i}" for i in range(1, 11)}


@dataclass
class Config:
    date_col: str = "date"
    close_col: str = "close"
    window: int = 1000
    horizon: int = 30
    step: int = 30
    top_k: int = 3
    rho: float = 0.5
    random_state: int = 42


def target_specs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    specs = []
    if "close" in df.columns:
        specs.append(("next_close_return", "close"))
    if "total_volume" in df.columns:
        specs.append(("next_onchain_volume_return", "total_volume"))
    if "trading_volume_usd" in df.columns:
        specs.append(("next_trading_volume_return", "trading_volume_usd"))
    return specs


def transform_frame(df: pd.DataFrame, target_source: str, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    out[cfg.date_col] = pd.to_datetime(out[cfg.date_col])
    out = out.sort_values(cfg.date_col).reset_index(drop=True)
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)

    src = out[target_source].astype(float)
    if (src > 0).all():
        out["target"] = np.log(src.shift(-1) / src)
    else:
        out["target"] = src.shift(-1) - src

    feature_cols = [c for c in numeric_cols if c != cfg.close_col]
    if target_source != cfg.close_col and target_source in feature_cols:
        # Keep current target source only in full/spec tests where explicitly selected.
        pass

    transformed = {cfg.date_col: out[cfg.date_col], "target": out["target"]}
    for col in feature_cols:
        s = out[col].astype(float)
        if (s > 0).all():
            transformed[col] = np.log(s / s.shift(1))
        else:
            transformed[col] = s.diff()
    res = pd.DataFrame(transformed).replace([np.inf, -np.inf], np.nan)
    feature_cols = [c for c in res.columns if c not in [cfg.date_col, "target"]]
    res[feature_cols] = res[feature_cols].ffill().fillna(0.0)
    return res.dropna(subset=["target"]).reset_index(drop=True)


def available(cols: Iterable[str], df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c in cols]


def feature_specs(data: pd.DataFrame, target_name: str) -> Dict[str, List[str]]:
    market = available(MARKET_SENTIMENT, data)
    ta = available(TECHNICAL, data)
    oca = available(ONCHAIN, data)
    all_features = [c for c in data.columns if c not in ["date", "target"]]

    specs: Dict[str, List[str]] = {"all": all_features}
    if market:
        specs["market_sentiment"] = market
    if oca:
        specs["onchain_only"] = oca
    if ta:
        specs["technical_only"] = ta
    if market and oca:
        specs["market_plus_onchain"] = [c for c in all_features if c in set(market) | set(oca)]

    if "volume" in target_name:
        no_mech = [c for c in all_features if c not in MECHANICAL_VOLUME]
    else:
        no_mech = [c for c in all_features if c not in PRICE_DERIVED]
    if no_mech and len(no_mech) < len(all_features):
        specs["all_no_mechanical"] = no_mech

    return {k: v for k, v in specs.items() if len(v) >= 1}


def minmax_train(X: pd.DataFrame) -> pd.DataFrame:
    mn = X.min()
    mx = X.max()
    denom = (mx - mn).replace(0, np.nan)
    return ((X - mn) / denom).fillna(0.0).clip(0, 1)


def grey_relational_grades(X: pd.DataFrame, y: pd.Series, rho: float) -> pd.Series:
    Xn = minmax_train(X)
    yn = (y - y.min()) / (y.max() - y.min()) if y.max() != y.min() else y * 0.0
    diffs = Xn.sub(yn, axis=0).abs()
    dmin = float(diffs.min().min())
    dmax = float(diffs.max().max())
    if dmax == 0:
        return pd.Series(1.0, index=X.columns)
    return ((dmin + rho * dmax) / (diffs + rho * dmax)).mean(axis=0).sort_values(ascending=False)


def top_elasticnet(X: pd.DataFrame, y: pd.Series, k: int, seed: int) -> List[str]:
    if X.shape[1] <= k:
        return X.columns.tolist()
    model = make_pipeline(
        StandardScaler(),
        ElasticNet(alpha=0.001, l1_ratio=0.8, random_state=seed, max_iter=3000),
    )
    model.fit(X, y)
    coef = np.abs(model.named_steps["elasticnet"].coef_)
    order = np.argsort(coef)[::-1]
    selected = [X.columns[i] for i in order[:k] if coef[i] > 0]
    if len(selected) < k:
        mi = mutual_info_regression(X, y, random_state=seed)
        for i in np.argsort(mi)[::-1]:
            if X.columns[i] not in selected:
                selected.append(X.columns[i])
            if len(selected) == k:
                break
    return selected


def top_rf(X: pd.DataFrame, y: pd.Series, k: int, seed: int) -> List[str]:
    if X.shape[1] <= k:
        return X.columns.tolist()
    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=5,
        min_samples_leaf=5,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(X, y)
    order = np.argsort(model.feature_importances_)[::-1]
    return [X.columns[i] for i in order[:k]]


def tri_membership(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if math.isinf(a) and a < 0:
        return np.where(x <= b, 1.0, np.where(x >= c, 0.0, (c - x) / (c - b)))
    if math.isinf(c) and c > 0:
        return np.where(x >= b, 1.0, np.where(x <= a, 0.0, (x - a) / (b - a)))
    return np.maximum(np.minimum((x - a) / (b - a), (c - x) / (c - b)), 0.0)


def make_mfs(values: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
    q20, q40, q50, q60, q80 = np.nanpercentile(values, [20, 40, 50, 60, 80])
    if len({q20, q40, q50, q60, q80}) < 5:
        std = np.nanstd(values) or 1.0
        med = np.nanmedian(values)
        q20, q40, q50, q60, q80 = med - std, med - 0.25 * std, med, med + 0.25 * std, med + std
    return {
        "low": (-math.inf, q20, q40),
        "medium": (q20, q50, q80),
        "high": (q60, q80, math.inf),
    }


class FuzzyGra:
    def __init__(self, features: List[str]):
        self.features = features
        self.input_mfs: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        self.output_mfs: Dict[str, Tuple[float, float, float]] = {}
        self.universe: np.ndarray | None = None
        self.fallback = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FuzzyGra":
        self.input_mfs = {c: make_mfs(X[c].values) for c in self.features}
        self.output_mfs = make_mfs(y.values)
        lo, hi = np.nanpercentile(y.values, [0.5, 99.5])
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        self.universe = np.linspace(lo, hi, 401)
        self.fallback = float(np.nanmedian(y.values))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        universe = self.universe
        assert universe is not None
        for _, row in X[self.features].iterrows():
            agg = np.zeros_like(universe)
            for state, weight in {"low": 0.88, "medium": 0.85, "high": 0.92}.items():
                fire = min(
                    float(tri_membership(np.array([row[c]]), *self.input_mfs[c][state])[0])
                    for c in self.features
                ) * weight
                agg = np.maximum(agg, np.minimum(fire, tri_membership(universe, *self.output_mfs[state])))
            preds.append(self.fallback if agg.sum() <= 1e-12 else float((universe * agg).sum() / agg.sum()))
        return np.array(preds)


def evaluate_predictions(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    err = pred - y
    rmse = math.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    denom = (np.abs(y) + np.abs(pred)) / 2
    smape = np.nanmean(np.where(denom == 0, 0.0, np.abs(err) / denom)) * 100
    direction = np.mean(np.sign(y) == np.sign(pred))

    cutoff = np.nanpercentile(y, 80)
    actual_high = y >= cutoff
    pred_high = pred >= np.nanpercentile(pred, 80)
    tp = np.sum(actual_high & pred_high)
    precision = tp / max(np.sum(pred_high), 1)
    recall = tp / max(np.sum(actual_high), 1)

    return {
        "rmse": rmse,
        "mae": mae,
        "smape_pct": float(smape),
        "direction_accuracy": float(direction),
        "top20_precision": float(precision),
        "top20_recall": float(recall),
    }


def dm_test(fuzzy_loss: np.ndarray, benchmark_loss: np.ndarray, lag: int = 5) -> Tuple[float, float]:
    d = np.asarray(fuzzy_loss - benchmark_loss, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    mean = d.mean()
    centered = d - mean
    var = np.dot(centered, centered) / n
    for l in range(1, min(lag, n - 1) + 1):
        gamma = np.dot(centered[l:], centered[:-l]) / n
        var += 2 * (1 - l / (lag + 1)) * gamma
    stat = mean / math.sqrt(var / n) if var > 0 else np.nan
    p = math.erfc(abs(stat) / math.sqrt(2)) if np.isfinite(stat) else np.nan
    return float(stat), float(p)


def run_one(dataset: str, raw: pd.DataFrame, target_name: str, target_source: str, cfg: Config, out_dir: Path, compact: bool = False) -> Tuple[List[dict], List[dict], pd.DataFrame]:
    data = transform_frame(raw, target_source, cfg)
    specs = feature_specs(data, target_name)
    if compact:
        specs = {k: v for k, v in specs.items() if k in {"all", "all_no_mechanical"}}
    metric_rows: List[dict] = []
    stability_rows: List[dict] = []
    prediction_frames = []

    for spec_name, features in specs.items():
        pred_path = out_dir / f"{dataset}_{target_name}_{spec_name}_predictions.csv"
        if pred_path.exists():
            pred_df = pd.read_csv(pred_path)
            prediction_frames.append(pred_df)
            stability_rows.append({
                "dataset": dataset,
                "target": target_name,
                "spec": spec_name,
                "window_start": pd.NaT,
                "window_end": pd.NaT,
                "gra_top": "loaded_from_existing_predictions",
                "elasticnet_top": "loaded_from_existing_predictions",
                "rf_top": "loaded_from_existing_predictions",
                "gra_elasticnet_overlap": np.nan,
                "gra_rf_overlap": np.nan,
            })
            # Metrics are computed below from pred_df exactly as for fresh runs.
            y = pred_df["actual"].values
            fuzzy_loss = (pred_df["fuzzy_gra"].values - y) ** 2
            for model_name in [
                "naive_zero",
                "hist_mean",
                "linear_all",
                "elasticnet_all",
                "random_forest",
                "gradient_boosting",
                "fuzzy_gra",
            ]:
                if model_name not in pred_df.columns:
                    continue
                m = evaluate_predictions(y, pred_df[model_name].values)
                dm_stat, dm_p = (np.nan, np.nan)
                if model_name != "fuzzy_gra":
                    dm_stat, dm_p = dm_test(fuzzy_loss, (pred_df[model_name].values - y) ** 2)
                metric_rows.append({
                    "dataset": dataset,
                    "target": target_name,
                    "spec": spec_name,
                    "model": model_name,
                    "n_predictions": len(pred_df),
                    "first_prediction_date": pred_df["date"].min(),
                    "last_prediction_date": pred_df["date"].max(),
                    **m,
                    "dm_stat_fuzzy_minus_model": dm_stat,
                    "dm_p_value": dm_p,
                })
            continue

        pred_rows = []
        for start in range(0, len(data) - cfg.window, cfg.step):
            train_end = start + cfg.window
            test_end = min(train_end + cfg.horizon, len(data))
            train = data.iloc[start:train_end]
            test = data.iloc[train_end:test_end]
            X_train = train[features].copy()
            y_train = train["target"].copy()
            X_test = test[features].copy()
            y_test = test["target"].copy()

            gra_rank = grey_relational_grades(X_train, y_train, cfg.rho)
            gra_top = gra_rank.head(min(cfg.top_k, len(features))).index.tolist()
            enet_top = top_elasticnet(X_train, y_train, min(cfg.top_k, len(features)), cfg.random_state)
            rf_top = top_rf(X_train, y_train, min(cfg.top_k, len(features)), cfg.random_state)
            stability_rows.append({
                "dataset": dataset,
                "target": target_name,
                "spec": spec_name,
                "window_start": train[cfg.date_col].iloc[0],
                "window_end": train[cfg.date_col].iloc[-1],
                "gra_top": ",".join(gra_top),
                "elasticnet_top": ",".join(enet_top),
                "rf_top": ",".join(rf_top),
                "gra_elasticnet_overlap": len(set(gra_top) & set(enet_top)) / len(gra_top),
                "gra_rf_overlap": len(set(gra_top) & set(rf_top)) / len(gra_top),
            })

            models = {
                "naive_zero": None,
                "hist_mean": None,
                "linear_all": LinearRegression(),
                "elasticnet_all": make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.8, random_state=cfg.random_state, max_iter=3000)),
                "random_forest": RandomForestRegressor(n_estimators=30, max_depth=5, min_samples_leaf=5, random_state=cfg.random_state, n_jobs=1),
                "gradient_boosting": GradientBoostingRegressor(n_estimators=40, max_depth=3, learning_rate=0.05, random_state=cfg.random_state),
            }

            preds: Dict[str, np.ndarray] = {
                "naive_zero": np.zeros(len(test)),
                "hist_mean": np.full(len(test), float(y_train.mean())),
            }

            for model_name in ["linear_all", "elasticnet_all", "random_forest", "gradient_boosting"]:
                model = models[model_name]
                assert model is not None
                model.fit(X_train, y_train)
                preds[model_name] = model.predict(X_test)

            fuzzy = FuzzyGra(gra_top).fit(X_train, y_train)
            preds["fuzzy_gra"] = fuzzy.predict(X_test)

            for i, idx in enumerate(test.index):
                row = {
                    "dataset": dataset,
                    "target": target_name,
                    "spec": spec_name,
                    "date": data.loc[idx, cfg.date_col],
                    "actual": float(y_test.iloc[i]),
                    "gra_top": ",".join(gra_top),
                    "elasticnet_top": ",".join(enet_top),
                    "rf_top": ",".join(rf_top),
                }
                for model_name, p in preds.items():
                    row[model_name] = float(p[i])
                pred_rows.append(row)

        pred_df = pd.DataFrame(pred_rows)
        if pred_df.empty:
            continue
        prediction_frames.append(pred_df)
        pred_df.to_csv(pred_path, index=False)

        y = pred_df["actual"].values
        fuzzy_loss = (pred_df["fuzzy_gra"].values - y) ** 2
        for model_name in [
            "naive_zero",
            "hist_mean",
            "linear_all",
            "elasticnet_all",
            "random_forest",
            "gradient_boosting",
            "fuzzy_gra",
        ]:
            m = evaluate_predictions(y, pred_df[model_name].values)
            dm_stat, dm_p = (np.nan, np.nan)
            if model_name != "fuzzy_gra":
                dm_stat, dm_p = dm_test(fuzzy_loss, (pred_df[model_name].values - y) ** 2)
            metric_rows.append({
                "dataset": dataset,
                "target": target_name,
                "spec": spec_name,
                "model": model_name,
                "n_predictions": len(pred_df),
                "first_prediction_date": pred_df["date"].min(),
                "last_prediction_date": pred_df["date"].max(),
                **m,
                "dm_stat_fuzzy_minus_model": dm_stat,
                "dm_p_value": dm_p,
            })

    all_preds = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    return metric_rows, stability_rows, all_preds


def main() -> None:
    cfg = Config()
    out_dir = Path("results/reviewer_robustness_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: List[dict] = []
    all_stability: List[dict] = []

    for dataset, path in DATASETS.items():
        raw = pd.read_csv(path)
        specs_to_run = target_specs(raw)
        compact = dataset != "merged_oca_ta"
        if compact:
            specs_to_run = [s for s in specs_to_run if s[0] == "next_close_return"]
        for target_name, target_source in specs_to_run:
            print(f"Running {dataset} / {target_name}...")
            metrics_rows, stability_rows, _ = run_one(dataset, raw, target_name, target_source, cfg, out_dir, compact=compact)
            all_metrics.extend(metrics_rows)
            all_stability.extend(stability_rows)

    metrics_df = pd.DataFrame(all_metrics)
    stability_df = pd.DataFrame(all_stability)
    metrics_df.to_csv(out_dir / "summary_metrics_long.csv", index=False)
    stability_df.to_csv(out_dir / "feature_selection_stability.csv", index=False)

    best = (
        metrics_df.sort_values(["dataset", "target", "spec", "rmse"])
        .groupby(["dataset", "target", "spec"], as_index=False)
        .first()
    )
    best.to_csv(out_dir / "best_model_by_spec.csv", index=False)

    ablation = metrics_df[
        (metrics_df["model"].isin(["linear_all", "elasticnet_all", "random_forest", "gradient_boosting", "fuzzy_gra"]))
    ].copy()
    ablation.to_csv(out_dir / "ablation_model_metrics.csv", index=False)

    print("\nBest model per dataset/target/spec:")
    cols = ["dataset", "target", "spec", "model", "rmse", "mae", "direction_accuracy", "top20_recall"]
    print(best[cols].to_string(index=False))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
