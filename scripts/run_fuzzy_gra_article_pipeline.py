from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


DATASETS = {
    "merged_base": Path("datasets/merged_base.csv"),
    "merged_oca": Path("datasets/merged_oca.csv"),
    "merged_oca_ta": Path("datasets/merged_oca_ta.csv"),
    "merged_ta": Path("datasets/merged_ta.csv"),
}


@dataclass
class Config:
    date_col: str = "date"
    close_col: str = "close"
    window: int = 1000
    horizon: int = 30
    step: int = 30
    rho: float = 0.5
    bootstrap_iters: int = 1000
    random_state: int = 42


def safe_numeric_transform(df: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col).reset_index(drop=True)

    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    keep = [date_col] + numeric_cols
    out = out[keep].copy()

    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    out[numeric_cols] = out[numeric_cols].ffill().fillna(0.0)

    out["target"] = np.log(out[close_col].shift(-1) / out[close_col])

    feature_cols = [c for c in numeric_cols if c != close_col]
    for col in feature_cols:
        s = out[col].astype(float)
        if (s > 0).all():
            out[col] = np.log(s / s.shift(1))
        else:
            out[col] = s.diff()

    out = out.replace([np.inf, -np.inf], np.nan)
    out[feature_cols] = out[feature_cols].ffill().fillna(0.0)
    out = out.dropna(subset=["target"]).reset_index(drop=True)
    return out[[date_col, "target"] + feature_cols]


def minmax_train_apply(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mn = train.min()
    mx = train.max()
    denom = (mx - mn).replace(0, np.nan)
    train_n = ((train - mn) / denom).fillna(0.0)
    test_n = ((test - mn) / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return train_n.clip(0, 1), test_n.clip(0, 1)


def grey_relational_grades(X: pd.DataFrame, y: pd.Series, rho: float = 0.5) -> pd.Series:
    Xn, _ = minmax_train_apply(X, X)
    yn = (y - y.min()) / (y.max() - y.min()) if y.max() != y.min() else y * 0.0
    diffs = Xn.sub(yn, axis=0).abs()
    dmin = float(diffs.min().min())
    dmax = float(diffs.max().max())
    if dmax == 0:
        return pd.Series(1.0, index=X.columns)
    grc = (dmin + rho * dmax) / (diffs + rho * dmax)
    return grc.mean(axis=0).sort_values(ascending=False)


def tri_membership(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if math.isinf(a) and a < 0:
        return np.where(x <= b, 1.0, np.where(x >= c, 0.0, (c - x) / (c - b)))
    if math.isinf(c) and c > 0:
        return np.where(x >= b, 1.0, np.where(x <= a, 0.0, (x - a) / (b - a)))
    left = np.where(b == a, 0.0, (x - a) / (b - a))
    right = np.where(c == b, 0.0, (c - x) / (c - b))
    return np.maximum(np.minimum(left, right), 0.0)


def make_mfs(values: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
    q20, q40, q50, q60, q80 = np.nanpercentile(values, [20, 40, 50, 60, 80])
    if len({q20, q40, q50, q60, q80}) < 5:
        spread = np.nanstd(values) or 1.0
        center = np.nanmedian(values)
        q20, q40, q50, q60, q80 = center - spread, center - 0.25 * spread, center, center + 0.25 * spread, center + spread
    return {
        "low": (-math.inf, q20, q40),
        "medium": (q20, q50, q80),
        "high": (q60, q80, math.inf),
    }


class FuzzyGraModel:
    def __init__(self, selected_features: List[str]):
        self.selected_features = selected_features
        self.input_mfs: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        self.output_mfs: Dict[str, Tuple[float, float, float]] = {}
        self.output_universe: np.ndarray | None = None
        self.fallback_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FuzzyGraModel":
        for col in self.selected_features:
            self.input_mfs[col] = make_mfs(X[col].values)
        self.output_mfs = make_mfs(y.values)
        lo, hi = np.nanpercentile(y.values, [0.5, 99.5])
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
        self.output_universe = np.linspace(lo, hi, 401)
        self.fallback_ = float(np.nanmedian(y.values))
        return self

    def predict_one(self, row: pd.Series) -> float:
        fires = {}
        for state, weight in {"low": 0.88, "medium": 0.85, "high": 0.92}.items():
            degrees = []
            for col in self.selected_features:
                a, b, c = self.input_mfs[col][state]
                degrees.append(float(tri_membership(np.array([row[col]]), a, b, c)[0]))
            fires[state] = min(degrees) * weight

        universe = self.output_universe
        assert universe is not None
        aggregated = np.zeros_like(universe)
        for state, fire in fires.items():
            a, b, c = self.output_mfs[state]
            aggregated = np.maximum(aggregated, np.minimum(fire, tri_membership(universe, a, b, c)))

        denom = aggregated.sum()
        if denom <= 1e-12:
            return self.fallback_
        return float((universe * aggregated).sum() / denom)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(row) for _, row in X[self.selected_features].iterrows()])


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.where(denom == 0, 0.0, np.abs(err) / denom)
    ape = np.where(np.abs(y_true) == 0, np.nan, np.abs(err / y_true))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "smape_pct": float(np.nanmean(smape) * 100),
        "mdape_pct": float(np.nanmedian(ape) * 100),
    }


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, cfg: Config) -> Dict[str, str]:
    rng = np.random.default_rng(cfg.random_state)
    n = len(y_true)
    vals = {"rmse": [], "mae": [], "smape_pct": [], "mdape_pct": []}
    for _ in range(cfg.bootstrap_iters):
        idx = rng.integers(0, n, n)
        m = metrics(y_true[idx], y_pred[idx])
        for k in vals:
            vals[k].append(m[k])
    return {k: f"[{np.percentile(v, 2.5):.6f}, {np.percentile(v, 97.5):.6f}]" for k, v in vals.items()}


def newey_west_dm(loss_diff: np.ndarray, lag: int = 5) -> Tuple[float, float]:
    d = np.asarray(loss_diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    mean = d.mean()
    centered = d - mean
    gamma0 = np.dot(centered, centered) / n
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        gamma = np.dot(centered[l:], centered[:-l]) / n
        var += 2 * (1 - l / (lag + 1)) * gamma
    stat = mean / math.sqrt(var / n) if var > 0 else np.nan
    p = math.erfc(abs(stat) / math.sqrt(2)) if np.isfinite(stat) else np.nan
    return float(stat), float(p)


def run_dataset(name: str, path: Path, cfg: Config, out_dir: Path) -> Dict[str, float]:
    raw = pd.read_csv(path)
    data = safe_numeric_transform(raw, cfg.date_col, cfg.close_col)
    feature_cols = [c for c in data.columns if c not in [cfg.date_col, "target"]]

    pred_rows = []
    gra_rows = []
    train_metric_rows = []

    for start in range(0, len(data) - cfg.window, cfg.step):
        train_end = start + cfg.window
        test_end = min(train_end + cfg.horizon, len(data))
        if test_end <= train_end:
            break

        train = data.iloc[start:train_end].copy()
        test = data.iloc[train_end:test_end].copy()
        X_train = train[feature_cols]
        y_train = train["target"]
        X_test = test[feature_cols]
        y_test = test["target"].values

        gra = grey_relational_grades(X_train, y_train, cfg.rho)
        selected = gra.head(3).index.tolist()
        for rank, (feat, score) in enumerate(gra.head(10).items(), start=1):
            gra_rows.append({
                "dataset": name,
                "window_start": train[cfg.date_col].iloc[0],
                "window_end": train[cfg.date_col].iloc[-1],
                "rank": rank,
                "feature": feat,
                "grg": score,
            })

        fuzzy = FuzzyGraModel(selected).fit(X_train, y_train)
        fuzzy_pred = fuzzy.predict(X_test)
        fuzzy_train_pred = fuzzy.predict(X_train)

        linear = LinearRegression()
        linear.fit(X_train[selected], y_train)
        linear_pred = linear.predict(X_test[selected])
        linear_train_pred = linear.predict(X_train[selected])

        train_metric_rows.append({
            "dataset": name,
            "window_start": train[cfg.date_col].iloc[0],
            "window_end": train[cfg.date_col].iloc[-1],
            "fuzzy_train_rmse": metrics(y_train.values, fuzzy_train_pred)["rmse"],
            "linear_train_rmse": metrics(y_train.values, linear_train_pred)["rmse"],
        })

        for i, idx in enumerate(test.index):
            pred_rows.append({
                "dataset": name,
                "date": data.loc[idx, cfg.date_col],
                "actual": y_test[i],
                "fuzzy_pred": fuzzy_pred[i],
                "linear_pred": linear_pred[i],
                "features": ",".join(selected),
                "window_start": train[cfg.date_col].iloc[0],
                "window_end": train[cfg.date_col].iloc[-1],
            })

    pred_df = pd.DataFrame(pred_rows)
    gra_df = pd.DataFrame(gra_rows)
    train_df = pd.DataFrame(train_metric_rows)

    pred_df.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    gra_df.to_csv(out_dir / f"{name}_gra_top10_by_window.csv", index=False)

    y = pred_df["actual"].values
    fuzzy_p = pred_df["fuzzy_pred"].values
    linear_p = pred_df["linear_pred"].values

    fuzzy_metrics = metrics(y, fuzzy_p)
    linear_metrics = metrics(y, linear_p)
    fuzzy_ci = bootstrap_ci(y, fuzzy_p, cfg)
    linear_ci = bootstrap_ci(y, linear_p, cfg)
    loss_diff = (fuzzy_p - y) ** 2 - (linear_p - y) ** 2
    dm_stat, dm_p = newey_west_dm(loss_diff, lag=5)

    top_features = (
        gra_df.groupby("feature")["grg"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_features.to_csv(out_dir / f"{name}_gra_mean_top10.csv", index=False)

    fuzzy_train_rmse = float(train_df["fuzzy_train_rmse"].mean())
    linear_train_rmse = float(train_df["linear_train_rmse"].mean())

    row = {
        "dataset": name,
        "rows_after_transform": len(data),
        "n_predictions": len(pred_df),
        "first_prediction_date": pred_df["date"].min(),
        "last_prediction_date": pred_df["date"].max(),
        "fuzzy_rmse": fuzzy_metrics["rmse"],
        "fuzzy_rmse_ci": fuzzy_ci["rmse"],
        "fuzzy_mae": fuzzy_metrics["mae"],
        "fuzzy_smape_pct": fuzzy_metrics["smape_pct"],
        "fuzzy_mdape_pct": fuzzy_metrics["mdape_pct"],
        "linear_rmse": linear_metrics["rmse"],
        "linear_rmse_ci": linear_ci["rmse"],
        "linear_mae": linear_metrics["mae"],
        "linear_smape_pct": linear_metrics["smape_pct"],
        "linear_mdape_pct": linear_metrics["mdape_pct"],
        "rmse_improvement_pct": (linear_metrics["rmse"] - fuzzy_metrics["rmse"]) / linear_metrics["rmse"] * 100,
        "mae_improvement_pct": (linear_metrics["mae"] - fuzzy_metrics["mae"]) / linear_metrics["mae"] * 100,
        "dm_stat_fuzzy_minus_linear": dm_stat,
        "dm_p_value": dm_p,
        "fuzzy_overfit_ratio": (fuzzy_metrics["rmse"] - fuzzy_train_rmse) / fuzzy_train_rmse,
        "linear_overfit_ratio": (linear_metrics["rmse"] - linear_train_rmse) / linear_train_rmse,
        "mean_top_features": "; ".join(top_features["feature"].head(5).tolist()),
    }
    return row


def main() -> None:
    cfg = Config()
    out_dir = Path("results/fuzzy_gra_article_pipeline")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, path in DATASETS.items():
        print(f"Running {name}...")
        rows.append(run_dataset(name, path, cfg, out_dir))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary_metrics.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved results to {out_dir}")


if __name__ == "__main__":
    main()
