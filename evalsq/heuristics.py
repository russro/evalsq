"""H1 / H2 / H3 benchmark evaluation heuristics.

Benchmark = directional accuracy. Business value = Sharpe of a long/short strategy
that trades the model's next-day call.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .data import FEATURES
from .models import make_models, predict, predict_proba

ANNUAL = np.sqrt(252)


def _score(preds: np.ndarray, w: pd.DataFrame) -> tuple[float, float]:
    """(accuracy, annualised Sharpe) for predictions over window w."""
    acc = float((preds == w["target"].values).mean())
    strat = np.where(preds == 1, w["next_ret"].values, -w["next_ret"].values)
    sharpe = float(strat.mean() / (strat.std() + 1e-8) * ANNUAL)
    return acc, sharpe


# ── H1: Benchmark Validity Modeling ─────────────────────────────────────────

# Candidate benchmarks. Each is "higher is better" so correlations with Sharpe share a sign.
METRICS = ["accuracy", "auc", "neg_logloss", "bull_acc", "bear_acc"]


def _metrics(preds: np.ndarray, proba: np.ndarray, w: pd.DataFrame) -> dict:
    """All candidate benchmark scores for one window. bull/bear = accuracy on days above/below the 50-day MA."""
    y = w["target"].values
    hit = preds == y
    up = w["ma50"].values < 0  # price above its 50-day mean
    two_class = len(np.unique(y)) == 2
    return {
        "accuracy": float(hit.mean()),
        "auc": float(roc_auc_score(y, proba)) if two_class else np.nan,
        "neg_logloss": float(-log_loss(y, proba, labels=[0, 1])),
        "bull_acc": float(hit[up].mean()) if up.sum() >= 3 else np.nan,
        "bear_acc": float(hit[~up].mean()) if (~up).sum() >= 3 else np.nan,
    }


def h1_validity(df: pd.DataFrame, scaler: StandardScaler, models: dict, freq: str = "M") -> pd.DataFrame:
    """Per-period benchmark scores (METRICS) vs. Sharpe (business value), one row per (period, model)."""
    rows = []
    for name, model in models.items():
        preds = pd.Series(predict(model, scaler, df), index=df.index)
        proba = pd.Series(predict_proba(model, scaler, df), index=df.index)
        for period, w in df.groupby(df.index.to_period(freq)):
            if len(w) < 10:
                continue
            _, sharpe = _score(preds.loc[w.index].values, w)
            m = _metrics(preds.loc[w.index].values, proba.loc[w.index].values, w)
            rows.append({"period": str(period), "model": name, **m, "sharpe": sharpe})
    return pd.DataFrame(rows)


def h1_correlation(h1: pd.DataFrame, metric: str = "accuracy") -> float:
    """Pooled corr(metric, Sharpe): how good a proxy is the benchmark overall?"""
    return float(h1[metric].corr(h1["sharpe"]))


def h1_rolling(h1: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Rolling corr(metric, Sharpe) per model and metric. Long format: period, model, metric, rolling_corr."""
    out = []
    for name, g in h1.groupby("model", sort=False):
        g = g.sort_values("period")
        for metric in METRICS:
            out.append(pd.DataFrame({
                "period": g["period"].values,
                "model": name,
                "metric": metric,
                "rolling_corr": g[metric].rolling(window, min_periods=window // 2).corr(g["sharpe"]).values,
            }))
    return pd.concat(out).dropna().reset_index(drop=True)


def h1_summary(h1: pd.DataFrame, roll: pd.DataFrame) -> pd.DataFrame:
    """One row per metric: pooled corr with Sharpe, and how much the rolling corr moves (min, std)."""
    r = roll.groupby("metric")["rolling_corr"]
    out = pd.DataFrame({
        "pooled_corr": {m: h1_correlation(h1, m) for m in METRICS},
        "rolling_min": r.min(),
        "rolling_mean": r.mean(),
        "rolling_std": r.std(),
    }).loc[METRICS]
    out.index.name = "metric"
    return out.reset_index()


# ── H2: Temporal Holdout ─────────────────────────────────────────────────────

def h2_temporal(df: pd.DataFrame, years: list[int] | None = None) -> pd.DataFrame:
    """Expanding-window train, single-year test. One row per test year; 'winner' = top model."""
    if years is None:
        years = list(range(2015, df.index.year.max() + 1))
    rows = []
    for yr in years:
        tr, te = df[df.index.year < yr], df[df.index.year == yr]
        if len(tr) < 200 or len(te) < 50:
            continue
        sc = StandardScaler().fit(tr[FEATURES])
        row = {"test_year": yr}
        for name, m in make_models().items():
            m.fit(sc.transform(tr[FEATURES]), tr["target"])
            row[name] = float((predict(m, sc, te) == te["target"]).mean())
        rows.append(row)
    h2 = pd.DataFrame(rows)
    names = list(make_models())
    h2["winner"] = h2[names].idxmax(axis=1)
    return h2


def h2_rank_flips(h2: pd.DataFrame) -> int:
    """Number of year-over-year changes in the top-ranked model."""
    w = h2["winner"]
    return int((w != w.shift()).iloc[1:].sum())


# ── H3: Predictability Test (meta-model) ─────────────────────────────────────

def h3_meta(df: pd.DataFrame, h2: pd.DataFrame, model: str = "RF") -> tuple[pd.DataFrame, float]:
    """
    Predict a model's yearly accuracy from market regime features (vol, trend).
    Returns (table, leave-one-out R²). LOO because n≈10: in-sample R² would flatter the fit.
    High R² → score tracks regime, not model quality. Low/negative → benchmark still discriminating.
    """
    rows = [{
        "year": yr,
        "vol": float(df.loc[df.index.year == yr, "vol20"].mean()),
        "trend": float(df.loc[df.index.year == yr, "ma50"].mean()),
        "acc": float(acc),
    } for yr, acc in zip(h2["test_year"], h2[model])]
    meta = pd.DataFrame(rows)
    if len(meta) < 4:
        return meta, float("nan")
    X, y = meta[["vol", "trend"]], meta["acc"]
    meta["pred_acc"] = cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut())
    return meta, float(r2_score(y, meta["pred_acc"]))
