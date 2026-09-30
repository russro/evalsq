"""Figures for the workshop slides. Each function takes the result tables and writes one PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .heuristics import METRICS  # noqa: E402

MODEL_COLORS = {"LogReg": "#0072B2", "RF": "#E69F00"}
METRIC_COLORS = dict(zip(METRICS, ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]))
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "neg_logloss": "Log-loss (negated)",
    "bull_acc": "Accuracy, uptrend days",
    "bear_acc": "Accuracy, downtrend days",
}
SHADE = {"2020 crash": ("2020-02-19", "2020-04-30"), "2022 bear": ("2022-01-03", "2022-10-12")}

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _period_ts(periods: pd.Series) -> pd.DatetimeIndex:
    return pd.PeriodIndex(periods, freq="M").to_timestamp()


def plot_setup(df: pd.DataFrame, cutoff: int, path: Path) -> Path:
    """SPY close with the train/test split and the two stress periods shaded."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(df.index, df["close"], color="#333333", lw=1)
    split = pd.Timestamp(f"{cutoff}-01-01")
    ax.axvline(split, color="#555555", ls="--", lw=1)
    ymax = df["close"].max()
    ax.text(split, ymax, "  test", va="top", ha="left", color="#555555")
    ax.text(split, ymax, "train  ", va="top", ha="right", color="#555555")
    for (label, (a, b)), ha in zip(SHADE.items(), ["right", "left"]):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#D55E00", alpha=0.15, lw=0)
        x = pd.Timestamp(a) if ha == "right" else pd.Timestamp(b)
        ax.text(x, df["close"].min(), f" {label} ", fontsize=8, color="#D55E00", va="bottom", ha=ha)
    ax.set_ylabel("SPY close (USD)")
    return _save(fig, path)


def plot_h1(summary: pd.DataFrame, roll: pd.DataFrame, path: Path) -> Path:
    """(a) pooled corr of each metric with Sharpe. (b) rolling corr over time, averaged over models."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 2]})
    s = summary.set_index("metric").loc[METRICS]
    a.barh([METRIC_LABELS[m] for m in METRICS], s["pooled_corr"], color=[METRIC_COLORS[m] for m in METRICS])
    a.invert_yaxis()
    a.set_xlim(min(0, s["pooled_corr"].min()) - 0.05, 1)
    a.set_xlabel("corr(metric, Sharpe)")
    a.set_title("(a) Whole test period", loc="left")

    # average over models; reindex to every month so gaps (too few up/down days) break the line
    mean = roll.groupby(["period", "metric"])["rolling_corr"].mean().unstack()
    months = pd.period_range(mean.index.min(), mean.index.max(), freq="M").astype(str)
    mean = mean.reindex(months)
    for m in METRICS:
        b.plot(_period_ts(mean.index), mean[m], color=METRIC_COLORS[m], lw=1.5, label=METRIC_LABELS[m])
    for a_, b_ in SHADE.values():
        b.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color="#D55E00", alpha=0.1, lw=0)
    b.set_ylabel("rolling corr with Sharpe")
    b.set_title("(b) 24-month rolling window", loc="left")
    b.legend(fontsize=8, frameon=False, loc="lower left")
    return _save(fig, path)


def plot_h2(h2: pd.DataFrame, path: Path) -> Path:
    """Per-year accuracy of each model; years where the top model changes are marked."""
    models = [c for c in h2.columns if c in MODEL_COLORS]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    width = 0.8 / len(models)
    x = range(len(h2))
    for i, m in enumerate(models):
        ax.bar([j + (i - (len(models) - 1) / 2) * width for j in x], h2[m], width, color=MODEL_COLORS[m], label=m)
    flips = h2["winner"] != h2["winner"].shift()
    for j in [j for j in x if j > 0 and flips.iloc[j]]:
        ax.annotate("flip", (j, h2.loc[j, models].max() + 0.005), ha="center", fontsize=8, color="#D55E00")
    ax.axhline(0.5, color="#555555", lw=0.8, ls=":")
    ax.set_xticks(list(x), h2["test_year"].astype(str))
    ax.set_ylim(0.4, max(0.62, h2[models].max().max() + 0.03))
    ax.set_ylabel("accuracy")
    ax.legend(frameon=False, ncol=len(models), loc="upper left")
    return _save(fig, path)


def plot_h3(meta: pd.DataFrame, r2: float, path: Path) -> Path:
    """Meta-model prediction of yearly accuracy (leave-one-out) against the observed accuracy."""
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(meta["acc"], meta["pred_acc"], color="#0072B2")
    for _, r in meta.iterrows():
        ax.annotate(str(int(r["year"])), (r["acc"], r["pred_acc"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lo = min(meta["acc"].min(), meta["pred_acc"].min()) - 0.01
    hi = max(meta["acc"].max(), meta["pred_acc"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color="#555555", lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("observed accuracy")
    ax.set_ylabel("predicted accuracy")
    ax.set_title(f"leave-one-out R² = {r2:.2f}", loc="left")
    return _save(fig, path)


# ── Deployment (deploy.py) ───────────────────────────────────────────────────

RULE_COLORS = {**METRIC_COLORS, "lagged_pnl": "#D55E00"}
RULE_LABELS = {**METRIC_LABELS, "lagged_pnl": "Lagged P&L", "oracle": "Oracle (hindsight)",
               "never_switch": "Never switch", "always_long": "Always long SPY"}
REF_STYLE = {"oracle": ("#000000", ":"), "never_switch": ("#777777", "--"), "always_long": ("#333333", "-")}
ZOO_COLORS = ["#0072B2", "#56B4E9", "#E69F00", "#F0E442", "#009E73", "#8FD694", "#CC79A7", "#D55E00"]


def plot_deploy(eq: pd.DataFrame, rules: list[str], path: Path) -> Path:
    """Portfolio value under each selection rule, with reference lines. Log scale so the oracle fits."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = _period_ts(eq.index)
    for r in eq.columns:
        color, ls = REF_STYLE.get(r, (RULE_COLORS.get(r, "#999999"), "-"))
        lw = 1.8 if r in rules else 1.2
        ax.plot(x, eq[r], color=color, ls=ls, lw=lw, label=f"{RULE_LABELS.get(r, r)}  ${eq[r].iloc[-1] / 1e3:,.0f}k")
    for a_, b_ in SHADE.values():
        ax.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color="#D55E00", alpha=0.1, lw=0)
    ax.set_yscale("log")
    ax.set_ylabel("portfolio value (USD, log)")
    ax.legend(fontsize=8, frameon=False, loc="upper left", title="pick the model with the best...", title_fontsize=8)
    return _save(fig, path)


def plot_picks(picks: pd.DataFrame, rules: list[str], path: Path) -> Path:
    """Which model each rule deployed, month by month."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    models = [m for m in dict.fromkeys(picks["model"]) if m != "SPY"]
    code = {m: i for i, m in enumerate(models)}
    wide = picks[picks["rule"].isin(rules)].pivot(index="rule", columns="month", values="model").loc[rules]
    cmap = ListedColormap((ZOO_COLORS * 2)[:len(models)])
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(rules) + 1.6))
    ax.imshow(wide.map(code.get).values, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(models) - 0.5, interpolation="nearest")
    ax.set_yticks(range(len(rules)), [RULE_LABELS.get(r, r) for r in rules])
    years = [i for i, m in enumerate(wide.columns) if m.endswith("-01")]
    ax.set_xticks(years, [wide.columns[i][:4] for i in years])
    ax.grid(False)
    ax.legend(handles=[Patch(color=cmap(code[m]), label=m) for m in models], fontsize=8, frameon=False,
              ncol=min(len(models), 4), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, path)


def plot_winners_curse(wc: pd.DataFrame, path: Path) -> Path:
    """Accuracy the bench promised for the winner vs. what it delivered, as the candidate pool grows."""
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(wc["k"], wc["bench_acc"], "o-", color="#0072B2", label="bench (selection window)")
    ax.plot(wc["k"], wc["deployed_acc"], "o-", color="#D55E00", label="deployed (next month)")
    ax.fill_between(wc["k"], wc["deployed_acc"], wc["bench_acc"], color="#999999", alpha=0.2, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("candidates compared (k)")
    ax.set_ylabel("accuracy of the picked model")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def plot_gap_dist(monthly: pd.DataFrame, path: Path) -> Path:
    """Spread of the monthly winner's-curse gap at each pool size k, with the mean on top."""
    ks = sorted(monthly["k"].unique())
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.boxplot([monthly.loc[monthly["k"] == k, "gap"] * 100 for k in ks], positions=range(len(ks)),
               widths=0.5, showfliers=False, medianprops={"color": "#0072B2"})
    means = monthly.groupby("k")["gap"].mean() * 100
    ax.plot(range(len(ks)), means.loc[ks], "o-", color="#D55E00", label="mean over months")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("candidates compared (k)")
    ax.set_ylabel("gap, bench minus deployed (pt)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return _save(fig, path)


def plot_opt_k(opt: pd.DataFrame, path: Path) -> Path:
    """Which pool size k would have deployed best each month, and how much that answer moves."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True, height_ratios=[2, 1])
    x = _period_ts(opt["month"])
    a.step(x, opt["opt_k"], where="mid", color="#0072B2", lw=1.2)
    a.set_yscale("log", base=2)
    ks = sorted(opt["opt_k"].unique())
    a.set_yticks(ks, [str(k) for k in ks])
    a.minorticks_off()
    a.set_ylabel("best k that month")
    b.plot(x, opt["opt_k_vol"], color="#D55E00")
    b.set_ylabel("12-mo std of log2 k")
    for ax in (a, b):
        for a_, b_ in SHADE.values():
            ax.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color="#D55E00", alpha=0.1, lw=0)
    return _save(fig, path)
