"""Monthly deployment under a feedback lag: how much money does each selection metric make?

Each month we retrain the zoo on data we can already see, score every model on a recent
held-out window, deploy the argmax for the month, and book its long/short P&L. Labels
(next-day returns) are only visible L trading days late, so both training and selection
run on data that is L days stale.
"""

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from .heuristics import _metrics, combined

# Selection rules: pick the model with the highest value of this column on the selection window.
RULES = ["accuracy", "auc", "neg_logloss", "bear_acc", "lagged_pnl", "combined"]
# "combined" blends the other rules: each model's mean percentile rank across them that month.
COMBINED_OF = [r for r in RULES if r != "combined"]
# Reference lines, not rules a practitioner could run.
REFERENCES = ["oracle", "never_switch", "always_long"]
START = 100_000
# Default frictions for SPY: 1bp per unit of position traded (half spread + slippage, no commission),
# 0.5%/yr to borrow shares while short.
COST_BP = 1.0
BORROW = 0.005


def month_starts(df: pd.DataFrame, start_year: int) -> list[int]:
    """Row positions of the first trading day of each month from start_year on."""
    months = df.index.to_period("M")
    first = np.flatnonzero(months != np.roll(months, 1))
    return [int(p) for p in first if df.index[p].year >= start_year]


def visible_end(pos0: int, lag: int) -> int:
    """Rows [0, end) have labels we can see when deciding at row pos0.

    Row i's label is the return from close i to close i+1, known at pos i+1. With lag L we
    see it at pos i+1+L, so we need i+1+L <= pos0.
    """
    return pos0 - lag


def _score_month(df: pd.DataFrame, zoo: dict, pos0: int, pos1: int, lag: int,
                 sel_window: int, min_train: int) -> list[dict]:
    """Retrain every model on data visible at pos0, score on the selection window, book the live month."""
    end = visible_end(pos0, lag)
    train, sel, live = df.iloc[:end - sel_window], df.iloc[end - sel_window:end], df.iloc[pos0:pos1]
    if len(train) < min_train:
        return []
    month = str(df.index[pos0].to_period("M"))
    spy = float(np.prod(1 + live["next_ret"].values) - 1)
    rows = []
    for name, (feats, model) in zoo.items():
        m = clone(model).fit(train[feats], train["target"])
        s_pred, s_proba = m.predict(sel[feats]), m.predict_proba(sel[feats])[:, 1]
        l_pred = m.predict(live[feats])
        strat_sel = np.where(s_pred == 1, 1, -1) * sel["next_ret"].values
        pos = np.where(l_pred == 1, 1, -1)
        strat_live = pos * live["next_ret"].values
        rows.append({
            "month": month, "model": name,
            **_metrics(s_pred, s_proba, sel),
            "lagged_pnl": float(strat_sel.sum()),
            "month_acc": float((l_pred == live["target"].values).mean()),
            "month_ret": float(np.prod(1 + strat_live) - 1),
            "spy_ret": spy,
            # enough to charge costs later without keeping the daily series (see apply_costs)
            "flips": int((np.diff(pos) != 0).sum()),
            "short_days": int((pos == -1).sum()),
            "first_pos": int(pos[0]),
            "last_pos": int(pos[-1]),
        })
    return rows


def walk_forward(df: pd.DataFrame, zoo: dict, start_year: int, lag: int = 21,
                 sel_window: int = 63, min_train: int = 500, n_jobs: int = -1) -> pd.DataFrame:
    """One row per (month, model): selection-window scores and the realised result of deploying it that month.

    Months are independent, so they run in parallel (joblib). If the pool fails for any
    reason (odd hardware, sandboxed /dev/shm), fall back to a plain serial loop.
    """
    starts = month_starts(df, start_year)
    bounds = [(p, starts[k + 1] if k + 1 < len(starts) else len(df)) for k, p in enumerate(starts)]
    args = (lag, sel_window, min_train)
    try:
        per_month = Parallel(n_jobs=n_jobs)(delayed(_score_month)(df, zoo, a, b, *args) for a, b in bounds)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"parallel walk-forward failed ({e!r}); running serially")
        per_month = [_score_month(df, zoo, a, b, *args) for a, b in bounds]
    return pd.DataFrame([r for rows in per_month for r in rows])


def apply_rules(scores: pd.DataFrame) -> pd.DataFrame:
    """Month x rule table of the picked model and its month return. Ties go to the first model in zoo order."""
    order = list(dict.fromkeys(scores["model"]))
    scores = scores.assign(combined=combined(scores, COMBINED_OF, by="month"))
    out = []
    for rule in RULES + ["oracle"]:
        col = "month_ret" if rule == "oracle" else rule
        wide = scores.pivot(index="month", columns="model", values=col)[order]
        # a metric can be undefined for a month (e.g. no downtrend days); keep last month's pick
        ok = wide.notna().any(axis=1)
        pick = wide[ok].idxmax(axis=1).reindex(wide.index).ffill().fillna(order[0])
        out.append(pd.DataFrame({"month": wide.index, "rule": rule, "model": pick.values}))
    first = out[0]["model"].iloc[0]  # never switch: keep the model accuracy picked in month one
    months = out[0]["month"]
    out.append(pd.DataFrame({"month": months, "rule": "never_switch", "model": first}))
    picks = pd.concat(out, ignore_index=True)
    rets = scores.set_index(["month", "model"])["month_ret"]
    picks["month_ret"] = rets.loc[list(zip(picks["month"], picks["model"]))].values
    spy = scores.groupby("month")["spy_ret"].first()
    long = pd.DataFrame({"month": spy.index, "rule": "always_long", "model": "SPY", "month_ret": spy.values})
    return pd.concat([picks, long], ignore_index=True)


def apply_costs(picks: pd.DataFrame, scores: pd.DataFrame, cost_bp: float = COST_BP,
                borrow: float = BORROW) -> pd.DataFrame:
    """Same picks, with month_ret net of trading and borrow costs.

    Trades are paid at the close, before the next day's return: value *= (1 - c) per unit of
    position traded. A long/short flip is 2 units. Months are joined in order per rule, so a swap whose new
    model opens on the other side pays a flip too, and month one pays to enter from cash.
    While short, value *= 1 - borrow/252 per day. Both are multiplicative, so they factor
    out of the gross month return exactly.
    """
    c, b = cost_bp / 1e4, borrow / 252
    info = scores.set_index(["month", "model"])[["flips", "short_days", "first_pos", "last_pos"]]
    out = []
    for rule, g in picks.groupby("rule", sort=False):
        g = g.sort_values("month").copy()
        if rule == "always_long":  # buy once, hold
            f = pd.DataFrame({"flips": 0, "short_days": 0, "first_pos": 1, "last_pos": 1}, index=g.index)
        else:
            f = info.loc[list(zip(g["month"], g["model"]))].set_index(g.index)
        enter = (f["first_pos"] - f["last_pos"].shift(fill_value=0)).abs()
        units = 2 * f["flips"] + enter
        g["month_ret"] = (1 + g["month_ret"]) * (1 - c) ** units * (1 - b) ** f["short_days"] - 1
        out.append(g)
    return pd.concat(out).sort_index()


def equity(picks: pd.DataFrame, start: float = START) -> pd.DataFrame:
    """Month x rule table of portfolio value, starting from `start` dollars."""
    wide = picks.pivot(index="month", columns="rule", values="month_ret")
    return start * (1 + wide).cumprod()


def deploy_summary(picks: pd.DataFrame, start: float = START) -> pd.DataFrame:
    """Per rule: final dollars, max drawdown, number of model switches, distinct models used."""
    eq = equity(picks, start)
    rows = []
    for rule in RULES + REFERENCES:
        p = picks[picks["rule"] == rule].sort_values("month")["model"]
        v = eq[rule]
        rows.append({
            "rule": rule,
            "final_usd": float(v.iloc[-1]),
            "max_drawdown": float((v / v.cummax() - 1).min()),
            "switches": int((p != p.shift()).iloc[1:].sum()),
            "n_models": int(p.nunique()),
        })
    return pd.DataFrame(rows)


def winners_curse_monthly(scores: pd.DataFrame, ks: tuple = (1, 2, 5, 10, 25),
                          n_draws: int = 200, seed: int = 0) -> pd.DataFrame:
    """Month x k table: draw k candidates, pick the best selection-window accuracy, compare with its accuracy once deployed.

    The gap (bench minus deployed) is the winner's curse: with more near-tied candidates the
    winner is more often the luckiest one, so the bench promises more than it delivers.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for month, g in scores.groupby("month"):
        bench, real = g["accuracy"].values, g["month_acc"].values
        for k in [k for k in ks if k <= len(g)]:
            idx = np.array([rng.choice(len(g), k, replace=False) for _ in range(n_draws)])
            best = idx[np.arange(n_draws), bench[idx].argmax(axis=1)]
            rows.append({"month": month, "k": k, "bench_acc": bench[best].mean(), "deployed_acc": real[best].mean()})
    out = pd.DataFrame(rows)
    out["gap"] = out["bench_acc"] - out["deployed_acc"]
    return out


def winners_curse(scores: pd.DataFrame, ks: tuple = (1, 2, 5, 10, 25),
                  n_draws: int = 200, seed: int = 0) -> pd.DataFrame:
    """Winner's curse averaged over months, one row per k."""
    return curse_by_k(winners_curse_monthly(scores, ks, n_draws, seed))


def curse_by_k(monthly: pd.DataFrame) -> pd.DataFrame:
    """Average the month x k table over months."""
    out = monthly.drop(columns="month").groupby("k").mean().reset_index()
    out["gap"] = out["bench_acc"] - out["deployed_acc"]
    return out


def optimal_k(monthly: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Per month, the pool size k whose winner deployed best, plus the rolling std of k as its volatility."""
    best = monthly.loc[monthly.groupby("month")["deployed_acc"].idxmax(), ["month", "k", "deployed_acc"]]
    best = best.rename(columns={"k": "opt_k"}).sort_values("month").reset_index(drop=True)
    best["opt_k_vol"] = best["opt_k"].rolling(window, min_periods=window).std()
    return best
