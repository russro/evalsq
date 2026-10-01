"""Learned benchmark: skip the hand-picked metric and learn which bench signals predict next month's $.

Row = (month, model). Inputs = that model's selection-window metrics, demeaned across the zoo
that month. Target = its live month return minus the zoo average that month. Demeaning on both
sides strips the month effect, so the ridge learns which model to pick, not whether SPY went up.
No model identity goes in, so the weights say which signals matter, not which model is good.

A month's return is only fully visible about a month after it ends (lag L ~ one month), so the
pick for month i trains on months <= i - GAP.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

SIGNALS = ["accuracy", "auc", "neg_logloss", "bull_acc", "bear_acc", "lagged_pnl"]
REGIME = ["vol", "trend"]
MODES = ["expanding", "rolling", "static"]
GAP = 2


def regime(scores: pd.DataFrame, window: int = 6) -> pd.DataFrame:
    """Per month: trailing SPY vol and trend over the `window` months before it, z-scored over the sample."""
    spy = scores.groupby("month")["spy_ret"].first().shift(1)
    r = pd.DataFrame({"vol": spy.rolling(window, min_periods=2).std(), "trend": spy.rolling(window, min_periods=1).sum()})
    return ((r - r.mean()) / r.std()).fillna(0)


def meta_table(scores: pd.DataFrame, use_regime: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """(month, model, feature columns..., y). Signals are z-scored over the sample, then demeaned within month.

    An undefined metric (e.g. bear_acc with no downtrend days) becomes 0, the month average.
    With use_regime, each signal is also crossed with vol and trend, so its weight can depend on the regime.
    """
    s = scores[SIGNALS]
    z = (s - s.mean()) / s.std()
    z = (z - z.groupby(scores["month"]).transform("mean")).fillna(0)
    cols = list(SIGNALS)
    if use_regime:
        reg = regime(scores).loc[scores["month"]].set_index(scores.index)
        for r in REGIME:
            for c in SIGNALS:
                z[f"{c}*{r}"] = z[c] * reg[r]
                cols.append(f"{c}*{r}")
    y = scores["month_ret"] - scores.groupby("month")["month_ret"].transform("mean")
    return pd.concat([scores[["month", "model"]], z, y.rename("y")], axis=1), cols


def train_months(i: int, mode: str, n_train: int, gap: int = GAP) -> range:
    """Indices of months the meta-model may train on when picking month i. Empty if not enough yet."""
    last = i - gap
    if mode == "expanding":
        lo = 0
    elif mode == "rolling":
        lo = last - n_train + 1
    elif mode == "static":
        lo, last = 0, n_train - 1
    else:
        raise ValueError(f"mode must be one of {MODES}")
    if lo < 0 or last - lo + 1 < n_train or last > i - gap:
        return range(0)
    return range(lo, last + 1)


def learned_select(scores: pd.DataFrame, mode: str = "expanding", n_train: int = 36, alpha: float = 10.0,
                   use_regime: bool = False, gap: int = GAP, holdout: str | None = None
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk forward over months. Returns (preds, weights).

    preds: month, model, pred (NaN until the meta-model has n_train months to learn from).
    weights: month, feature, weight, one ridge fit per month (static mode repeats the frozen fit).
    holdout: leave this model out of training, to test whether the weights transfer to an unseen model.
    """
    tab, cols = meta_table(scores, use_regime)
    months = sorted(tab["month"].unique())
    pos = {m: k for k, m in enumerate(months)}
    tab["i"] = tab["month"].map(pos)
    fit_rows = tab
    if holdout is not None:  # its return must not leak in through the month mean of the target either
        fit_rows = tab[tab["model"] != holdout].copy()
        ret = scores.loc[fit_rows.index, "month_ret"]
        fit_rows["y"] = ret - ret.groupby(fit_rows["month"]).transform("mean")
    preds, weights, frozen = [], [], None
    for i, m in enumerate(months):
        idx = train_months(i, mode, n_train, gap)
        live = tab[tab["i"] == i]
        if not len(idx):
            preds.append(live[["month", "model"]].assign(pred=np.nan))
            continue
        if mode != "static" or frozen is None:
            tr = fit_rows[fit_rows["i"].isin(idx)]
            frozen = Ridge(alpha=alpha, fit_intercept=False).fit(tr[cols], tr["y"])
        preds.append(live[["month", "model"]].assign(pred=frozen.predict(live[cols])))
        weights.append(pd.DataFrame({"month": m, "feature": cols, "weight": frozen.coef_}))
    w = pd.concat(weights, ignore_index=True) if weights else pd.DataFrame(columns=["month", "feature", "weight"])
    return pd.concat(preds, ignore_index=True), w


def learned_picks(scores: pd.DataFrame, preds: pd.DataFrame, rule: str, fallback: str = "accuracy") -> pd.DataFrame:
    """Picks in apply_rules format: argmax pred, or the fallback metric's pick before the meta-model is ready."""
    order = list(dict.fromkeys(scores["model"]))
    p = preds.pivot(index="month", columns="model", values="pred")[order]
    fb = scores.pivot(index="month", columns="model", values=fallback)[order]
    pick = p.fillna(-np.inf).idxmax(axis=1).where(p.notna().any(axis=1), fb.idxmax(axis=1))
    rets = scores.set_index(["month", "model"])["month_ret"]
    out = pd.DataFrame({"month": pick.index, "rule": rule, "model": pick.values})
    out["month_ret"] = rets.loc[list(zip(out["month"], out["model"]))].values
    return out


def rank_ic(scores: pd.DataFrame, pred: pd.Series) -> pd.Series:
    """Per month: Spearman between a score and the realised month return across the zoo."""
    d = scores[["month", "month_ret"]].assign(pred=pred.values).dropna()
    # a month where every model returns the same (all long all month) has no ranking; skip it
    g = d.groupby("month")
    d = d[(g["month_ret"].transform("nunique") > 1) & (g["pred"].transform("nunique") > 1)]
    if d.empty:  # meta-model never got enough months to fit
        return pd.Series(dtype=float)
    return d.groupby("month").apply(lambda g: g["pred"].corr(g["month_ret"], method="spearman"), include_groups=False)


def lomo(scores: pd.DataFrame, **kw) -> pd.DataFrame:
    """Leave one model out: predict each model only from a meta-model that never saw it.

    Returns per month rank IC of the LOMO predictions, the in-sample (all models) predictions,
    and plain accuracy, for comparison.
    """
    parts = []
    for m in dict.fromkeys(scores["model"]):
        p, _ = learned_select(scores, holdout=m, **kw)
        parts.append(p[p["model"] == m])
    held = pd.concat(parts).set_index(["month", "model"])["pred"]
    full = learned_select(scores, **kw)[0].set_index(["month", "model"])["pred"]
    key = list(zip(scores["month"], scores["model"]))
    return pd.DataFrame({
        "lomo": rank_ic(scores, held.loc[key]),
        "learned": rank_ic(scores, full.loc[key]),
        "accuracy": rank_ic(scores, scores["accuracy"]),
        "bear_acc": rank_ic(scores, scores["bear_acc"]),
    })


def learned_summary(scores: pd.DataFrame, cutoff: str, n_train: int = 36, alpha: float = 10.0,
                    cost_bp: float = 1.0, borrow: float = 0.005) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Each mode deployed from `cutoff` on, net of costs. Returns (summary, weights, picks).

    summary: mode, final_usd, switches, rank IC (all models) and LOMO rank IC, mean and t-stat.
    """
    from .deploy import START, apply_costs

    rows, ws, pks = [], [], []
    for mode in MODES:
        preds, w = learned_select(scores, mode, n_train, alpha)
        pk = learned_picks(scores, preds, f"learned_{mode}")
        pk = apply_costs(pk[pk["month"] >= cutoff], scores, cost_bp, borrow)
        ic = lomo(scores, mode=mode, n_train=n_train, alpha=alpha)
        ic = ic[ic.index >= cutoff]
        t = lambda c: float(ic[c].mean() / ic[c].std() * np.sqrt(ic[c].count()))  # noqa: E731
        rows.append({"mode": mode, "final_usd": float(START * (1 + pk["month_ret"]).prod()),
                     "switches": int((pk["model"] != pk["model"].shift()).iloc[1:].sum()),
                     "ic": float(ic["learned"].mean()), "ic_t": t("learned"),
                     "lomo_ic": float(ic["lomo"].mean()), "lomo_t": t("lomo")})
        ws.append(w.assign(mode=mode))
        pks.append(pk)
    return pd.DataFrame(rows), pd.concat(ws, ignore_index=True), pd.concat(pks, ignore_index=True)


def oos_split(scores: pd.DataFrame, cutoff: str, alpha: float = 10.0, cost_bp: float = 1.0,
              borrow: float = 0.005) -> pd.DataFrame:
    """Split the deploy period in half. Each rule's net $ per half ($START fresh each half), plus a ridge
    frozen on everything before the second half (static mode), deployed on the second half only.

    Returns rule, half (1 or 2), final_usd. Picking the best rule on half 1 and reading half 2 is the OOS test.
    """
    from .deploy import RULES, START, apply_costs, apply_rules

    months = sorted(m for m in scores["month"].unique() if m >= cutoff)
    mid = months[len(months) // 2]
    all_months = sorted(scores["month"].unique())
    preds, _ = learned_select(scores, "static", all_months.index(mid) - GAP + 1, alpha)
    pk = pd.concat([apply_rules(scores), learned_picks(scores, preds, "learned")], ignore_index=True)
    pk = pk[pk["rule"].isin(RULES + ["always_long", "learned"])]
    rows = []
    for half, sel in ((1, (pk["month"] >= cutoff) & (pk["month"] < mid)), (2, pk["month"] >= mid)):
        p = pk[sel & ~((half == 1) & (pk["rule"] == "learned"))]
        p = apply_costs(p, scores, cost_bp, borrow)
        for rule, g in p.groupby("rule", sort=False):
            rows.append({"rule": rule, "half": half, "start": g["month"].min(), "end": g["month"].max(),
                         "final_usd": float(START * (1 + g["month_ret"]).prod())})
    return pd.DataFrame(rows)
