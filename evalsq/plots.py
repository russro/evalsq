"""Figures for the workshop slides. Each function takes the result tables and writes one PNG."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .heuristics import METRICS  # noqa: E402

MODEL_COLORS = {"LogReg": "#00838A", "RF": "#FF6C2F"}
MODEL_LABELS = {"LogReg": "Logistic regression", "RF": "Random forest"}
METRIC_COLORS = dict(zip(METRICS, ["#00838A", "#FF6C2F", "#3D9A3F", "#765BA7", "#3255A4", "#1F1A17"]))
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "neg_logloss": "Log-loss (negated)",
    "bull_acc": "Accuracy, uptrend days",
    "bear_acc": "Accuracy, downtrend days",
    "combined": "Combined (mean rank)",
}
# Time gradient for scatter points: teal (early) → yellow → pink (late).
TIME_CMAP = LinearSegmentedColormap.from_list("riso_time", ["#00838A", "#FFB511", "#FF48B0"])
K_TICKS = [1, 5, 10, 15, 20, 25]
SHADE = {"2020 crash": ("2020-02-19", "2020-04-30"), "2022 bear": ("2022-01-03", "2022-10-12")}

# Risograph-style inks on cream paper + Space Grotesk (OFL, bundled), to stand apart from default-looking figs.
PAPER, INK = "#F6F1E7", "#1F1A17"
for _f in (Path(__file__).parent / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(_f))

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "Space Grotesk",
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.titleweight": "bold",
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "grid.color": "#CFC5B6",
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
    ax.plot(df.index, df["close"], color="#2B2522", lw=1)
    split = pd.Timestamp(f"{cutoff}-01-01")
    ax.axvline(split, color="#5E554F", ls="--", lw=1)
    ymax = df["close"].max()
    ax.text(split, ymax, "  test", va="top", ha="left", color="#5E554F")
    ax.text(split, ymax, "train  ", va="top", ha="right", color="#5E554F")
    for (label, (a, b)), ha in zip(SHADE.items(), ["right", "left"]):
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#FF48B0", alpha=0.15, lw=0)
        x = pd.Timestamp(a) if ha == "right" else pd.Timestamp(b)
        ax.text(x, df["close"].min(), f" {label} ", fontsize=8, color="#FF48B0", va="bottom", ha=ha)
    ax.set_ylabel("SPY close (USD)")
    return _save(fig, path)


def plot_h1(summary: pd.DataFrame, roll: pd.DataFrame, path: Path) -> Path:
    """(a) pooled corr of each metric with Sharpe. (b) rolling corr over time, averaged over models."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 2]})
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
        b.plot(_period_ts(mean.index), mean[m], color=METRIC_COLORS[m], lw=2.2 if m == "combined" else 1.5,
               label=METRIC_LABELS[m])
    for a_, b_ in SHADE.values():
        b.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color="#FF48B0", alpha=0.1, lw=0)
    b.set_ylabel("rolling corr with Sharpe")
    b.set_title("(b) 24-month rolling window", loc="left")
    b.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    return _save(fig, path)


def _meta_panel(ax, meta: pd.DataFrame, when: pd.Series, r2: float, title: str):
    """Observed vs. LOO-predicted accuracy, points colored by date."""
    sc = ax.scatter(meta["acc"], meta["pred_acc"], c=when, cmap=TIME_CMAP, s=28, edgecolor=INK, lw=0.3, zorder=3)
    lo = min(meta["acc"].min(), meta["pred_acc"].min()) - 0.01
    hi = max(meta["acc"].max(), meta["pred_acc"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color="#5E554F", lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("observed accuracy")
    ax.set_title(f"{title}, leave-one-out R² = {r2:.2f}", loc="left")
    return sc


def plot_h3(meta: pd.DataFrame, r2: float, monthly: pd.DataFrame, r2_m: float, path: Path) -> Path:
    """Meta-model prediction of accuracy (leave-one-out) against observed: (a) per year, (b) per month."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 4))
    t = monthly["month"].str[:4].astype(int) + (monthly["month"].str[5:7].astype(int) - 1) / 12
    lims = dict(vmin=min(meta["year"].min(), t.min()), vmax=max(meta["year"].max(), t.max()))
    _meta_panel(a, meta, meta["year"], r2, "(a) Per year").set_clim(**lims)
    sc = _meta_panel(b, monthly, t, r2_m, "(b) Per month")
    sc.set_clim(**lims)
    a.set_ylabel("predicted accuracy")
    cb = fig.colorbar(sc, ax=[a, b], pad=0.02, fraction=0.04)
    cb.set_ticks(range(int(lims["vmin"]), int(lims["vmax"]) + 1, 2))
    cb.outline.set_visible(False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Deployment (deploy.py) ───────────────────────────────────────────────────

RULE_COLORS = {**METRIC_COLORS, "lagged_pnl": "#FF48B0"}
RULE_LABELS = {**METRIC_LABELS, "lagged_pnl": "Lagged P&L", "oracle": "Oracle (hindsight)",
               "never_switch": "Never switch", "always_long": "Always long SPY"}
RULE_LABELS["bogle"] = "Bogleheads 3-fund (60/20/20)"
REF_STYLE = {"oracle": ("#1F1A17", ":"), "never_switch": ("#8A8079", "--"), "always_long": ("#2B2522", "-"),
             "bogle": ("#FFB511", "-")}
ZOO_COLORS = ["#00838A", "#3255A4", "#FF6C2F", "#FFB511", "#3D9A3F", "#9FD4C8", "#765BA7", "#FF48B0"]


def plot_deploy(eq: pd.DataFrame, rules: list[str], path: Path) -> Path:
    """Portfolio value under each selection rule, with reference lines. Log scale so the oracle fits."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = _period_ts(eq.index)
    for r in eq.columns:
        color, ls = REF_STYLE.get(r, (RULE_COLORS.get(r, "#A89E96"), "-"))
        lw = 2.4 if r in ("combined", "bogle") else 1.8 if r in rules else 1.2
        ax.plot(x, eq[r], color=color, ls=ls, lw=lw, label=f"{RULE_LABELS.get(r, r)}  ${eq[r].iloc[-1] / 1e3:,.0f}k")
    for a_, b_ in SHADE.values():
        ax.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color="#FF48B0", alpha=0.1, lw=0)
    ax.set_yscale("log")
    ax.set_ylabel("portfolio value (USD, log)")
    ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1),
              title="pick the model with the best...", title_fontsize=8)
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
    ax.plot(wc["k"], wc["bench_acc"], "o-", color="#00838A", label="bench (selection window)")
    ax.plot(wc["k"], wc["deployed_acc"], "o-", color="#FF48B0", label="deployed (next month)")
    ax.fill_between(wc["k"], wc["deployed_acc"], wc["bench_acc"], color="#A89E96", alpha=0.2, lw=0)
    ax.set_xticks(K_TICKS)
    ax.set_xlim(0, 26)
    ax.set_xlabel("candidates compared (k)")
    ax.set_ylabel("accuracy of the picked model")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, path)


def plot_gap_dist(monthly: pd.DataFrame, path: Path) -> Path:
    """Spread of the monthly winner's-curse gap at each pool size k, with the mean on top."""
    ks = sorted(monthly["k"].unique())
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.boxplot([monthly.loc[monthly["k"] == k, "gap"] * 100 for k in ks], positions=ks,
               widths=0.8, showfliers=False, medianprops={"color": "#00838A"}, manage_ticks=False)
    means = monthly.groupby("k")["gap"].mean() * 100
    ax.plot(ks, means.loc[ks], "o-", color="#FF48B0", label="mean over months")
    ax.axhline(0, color="#A89E96", lw=0.8)
    ax.set_xticks(K_TICKS)
    ax.set_xlim(-1, 27)
    ax.set_xlabel("candidates compared (k)")
    ax.set_ylabel("gap, bench minus deployed (pt)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return _save(fig, path)


def plot_opt_k(opt: pd.DataFrame, path: Path) -> Path:
    """Which pool size k would have deployed best each month, and how much that answer moves."""
    fig, (a, b) = plt.subplots(2, 1, figsize=(8, 4.4), sharex=True, height_ratios=[2, 1])
    x = _period_ts(opt["month"])
    a.plot(x, opt["opt_k"], "o-", color="#00838A", lw=0.8, ms=3, alpha=0.6, label="that month")
    a.plot(x, opt["opt_k"].rolling(12, min_periods=12).mean(), color="#1F1A17", lw=2, label="12-mo mean")
    a.set_yticks(K_TICKS)
    a.set_ylabel("best k")
    a.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    b.plot(x, opt["opt_k_vol"], color="#FF48B0")
    b.set_ylabel("12-mo std of k")
    for ax in (a, b):
        for a_, b_ in SHADE.values():
            ax.axvspan(pd.Timestamp(a_), pd.Timestamp(b_), color="#FF48B0", alpha=0.1, lw=0)
    return _save(fig, path)


# ── Learned benchmark (learned.py) ───────────────────────────────────────────

MODE_TITLES = {"expanding": "Refit monthly, all history", "rolling": "Refit monthly, last N months",
               "static": "Fit once on first N months, frozen"}


def plot_learned_weights(w: pd.DataFrame, summary: pd.DataFrame, cutoff: str, path: Path) -> Path:
    """Ridge weight on each bench signal over time, one panel per retraining mode. Shaded = deployed period."""
    modes = list(dict.fromkeys(w["mode"]))
    fig, axes = plt.subplots(len(modes), 1, figsize=(8, 2.3 * len(modes)), sharex=True, sharey=True)
    usd = summary.set_index("mode")["final_usd"]
    for ax, mode in zip(np.atleast_1d(axes), modes):
        wide = w[w["mode"] == mode].pivot(index="month", columns="feature", values="weight")
        x = _period_ts(wide.index)
        for c in wide.columns:
            ax.plot(x, wide[c], color=RULE_COLORS.get(c, "#A89E96"), lw=1.6, label=RULE_LABELS.get(c, c))
        ax.axhline(0, color=INK, lw=0.6)
        ax.axvspan(pd.Timestamp(f"{cutoff}-01"), x.max(), color="#FFB511", alpha=0.08, lw=0)
        ax.set_title(f"{MODE_TITLES.get(mode, mode)}  (deployed ${usd[mode] / 1e3:,.0f}k)", loc="left", fontsize=10)
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.supylabel("ridge weight")
    return _save(fig, path)


def plot_h2_folds(folds: dict, path: Path) -> Path:
    """Nested rule x fold heatmap: each rule gets one row per block size (shortest on top), cells span their fold's time.

    Color = annualized fold return (red < 0 < green), text = rank within the fold. Blocks must tile the same span.
    """
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    from .deploy import block_label

    blocks = sorted(folds)
    unit = blocks[0]  # x unit = shortest block
    ann = {b: (1 + f) ** (1 / b) - 1 for b, f in folds.items()}  # comparable color across block sizes
    rules = list(next(iter(folds.values())).columns)
    cmap = LinearSegmentedColormap.from_list("ret", ["#A50F15", "#FB6A4A", "#FFFFFF", "#74C476", "#00592C"])
    lim = max(np.abs(a.values).max() for a in ann.values())
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)
    top, big = blocks[-1], folds[blocks[-1]]
    span = len(big) * top / unit
    fig, ax = plt.subplots(figsize=(13, 1.5 * len(rules) + 0.6))
    sub, gap = 1.0, 0.6
    for i, rule in enumerate(rules):
        y0 = i * (len(blocks) * sub + gap)
        for k, b in enumerate(blocks):
            y, w = y0 + k * sub, b / unit
            rk = ann[b].rank(axis=1, ascending=False, method="min")
            for j, f in enumerate(ann[b].index):
                v, r = ann[b].loc[f, rule], int(rk.loc[f, rule])
                ax.add_patch(Rectangle((j * w, y), w, sub, facecolor=cmap(norm(v)), edgecolor=PAPER, lw=1.5))
                ax.text(j * w + w / 2, y + sub / 2, f"#{r}", ha="center", va="center", fontsize=10 + 2 * k,
                        color=PAPER if abs(v) > 0.55 * lim else INK, fontweight="bold" if r == 1 else None)
            ax.text(span + 0.1, y + sub / 2, block_label(b), va="center", fontsize=8, color="#888")
        ax.text(-0.15, y0 + len(blocks) * sub / 2, RULE_LABELS.get(rule, rule), ha="right", va="center", fontsize=11)
    w_top = top / unit
    for j in range(1, len(big)):
        ax.axvline(j * w_top, color=INK, lw=1.2)
    ax.set_xlim(0, span + 0.6)
    ax.set_ylim(len(rules) * (len(blocks) * sub + gap) - gap, -0.4)
    ax.set_xticks([j * w_top + w_top / 2 for j in range(len(big))], big.index)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, fraction=0.025, pad=0.02,
                      format=lambda x, _: f"{x:+.0%}")
    cb.set_label("annualized fold return (net)", fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title(f"Rank within each fold: {' / '.join(block_label(b) for b in blocks)} windows", loc="left", pad=28)
    return _save(fig, path)


def plot_h2_regret(summary: pd.DataFrame, rules: list[str], path: Path) -> Path:
    """Regret (return pts vs the best rule in hindsight) of trusting last fold's winner vs fixed rules and a random pick."""
    from .deploy import block_label

    strategies = ["follow_leader", "random"] + [f"fixed_{r}" for r in rules]
    labels = {"follow_leader": "Last fold's winner", "random": "Random rule",
              **{f"fixed_{r}": f"Always {RULE_LABELS.get(r, r)}" for r in rules}}
    colors = {"follow_leader": "#FFB511", "random": "#A89E96", **{f"fixed_{r}": RULE_COLORS.get(r, "#A89E96") for r in rules}}
    fig, axes = plt.subplots(1, len(summary), figsize=(5 * len(summary), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, summary.iterrows()):
        vals = row[strategies].astype(float) * 100
        ax.barh(range(len(strategies)), vals, color=[colors[s] for s in strategies])
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.1f}", va="center", fontsize=8)
        ax.set_yticks(range(len(strategies)), [labels[s] for s in strategies])
        ax.set_xlabel("regret (return pts per fold)")
        ax.set_title(f"{block_label(row['block_years'])} folds: winner repeats {row['hit_rate']:.0%}"
                     f" (chance {row['chance']:.0%})", loc="left", fontsize=10)
        ax.set_xlim(0, vals.max() * 1.2 + 1)
    axes[0].invert_yaxis()  # shared y: invert once
    fig.text(0.99, 0.01, "which fixed rule is best is only known in hindsight",
             ha="right", fontsize=8, color="#5E554F")
    return _save(fig, path)


def _heat(ax, m: pd.DataFrame, cmap, norm, fmt, labels: dict):
    """Annotated square heatmap; lower triangle + diagonal only when m is symmetric."""
    vals = m.values.astype(float)
    sym = np.allclose(np.nan_to_num(vals), np.nan_to_num(vals.T))
    shown = np.where(np.triu(np.ones_like(vals, bool), 1), np.nan, vals) if sym else vals
    ax.imshow(shown, cmap=cmap, norm=norm, interpolation="nearest")
    lim = max(abs(norm.vmin), abs(norm.vmax))
    for (i, j), v in np.ndenumerate(shown):
        if not np.isnan(v):
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9, color=PAPER if abs(v) > 0.6 * lim else INK)
    names = [labels.get(c, c) for c in m.columns]
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(m.index)), [labels.get(c, c) for c in m.index], fontsize=8)
    ax.grid(False)
    ax.tick_params(length=0)


def plot_h3_redundancy(corr: pd.DataFrame, agree: pd.DataFrame, path: Path) -> Path:
    """Left: within-month rank corr between evals (and the $ each model then earned). Right: how often two rules deploy the same model."""
    from matplotlib.colors import Normalize, TwoSlopeNorm

    labels = {**RULE_LABELS, "month_ret": "$ that month"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4))
    div = LinearSegmentedColormap.from_list("corr", ["#A50F15", "#FFFFFF", "#00838A"])
    _heat(a, corr, div, TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), lambda v: f"{v:.2f}", labels)
    a.set_title("Do two evals rank the 8 models alike?", loc="left", fontsize=10)
    seq = LinearSegmentedColormap.from_list("agree", ["#FFFFFF", "#765BA7"])
    _heat(b, agree, seq, Normalize(0, 1), lambda v: f"{v:.0%}", labels)
    b.set_title("How often do they deploy the same model?", loc="left", fontsize=10)
    fig.text(0.99, 0.01, "mean within-month Spearman over the zoo, 2018-2025", ha="right", fontsize=8, color="#5E554F")
    return _save(fig, path)


def plot_h3_complementarity(forward: pd.DataFrame, pair: pd.DataFrame, path: Path) -> Path:
    """Left: greedy forward selection, $ of the mean-rank combo vs # evals. Right: $ gained by adding B to A."""
    from matplotlib.colors import TwoSlopeNorm

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.1]})
    y = forward["final_usd"] / 1e3
    a.plot(forward["n"], y, color=INK, lw=1.4, zorder=1)
    a.scatter(forward["n"], y, s=60, c=[RULE_COLORS.get(r, "#A89E96") for r in forward["added"]], zorder=2)
    for n, v, r in zip(forward["n"], y, forward["added"]):
        a.annotate(("" if n == 1 else "+ ") + RULE_LABELS.get(r, r), (n, v), textcoords="offset points",
                   xytext=(6, 8 if n % 2 else -16), fontsize=8)
    a.set_xticks(forward["n"])
    a.set_xlabel("number of evals averaged (mean rank)")
    a.set_ylabel("deployed $ (k, net)")
    a.set_ylim(0, y.max() * 1.15)
    a.set_xlim(0.7, len(forward) + 0.9)
    a.set_title("Adding the next-best eval", loc="left", fontsize=10)
    lim = np.abs(pair.values).max() / 1e3
    cmap = LinearSegmentedColormap.from_list("gain", ["#A50F15", "#FB6A4A", "#FFFFFF", "#74C476", "#00592C"])
    _heat(b, pair / 1e3, cmap, TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), lambda v: f"{v:+.0f}" if v else "", RULE_LABELS)
    b.set_xlabel("add this eval (B)")
    b.set_ylabel("starting eval (A)")
    b.set_title("$k gained by adding B to A", loc="left", fontsize=10)
    return _save(fig, path)


def plot_learned_oos(oos: pd.DataFrame, path: Path) -> Path:
    """Net $ per rule in each half of the deploy period, rows ordered by half 1, so a reordering in half 2 is the OOS story."""
    from .deploy import START

    w = oos.pivot(index="rule", columns="half", values="final_usd") / 1e3
    w = w.sort_values(1, ascending=False, na_position="last")
    labels = {**RULE_LABELS, "always_long": "Always long SPY", "learned": "Learned (ridge, fit before half 2)"}
    colors = {**RULE_COLORS, "always_long": "#A89E96", "learned": "#FFB511"}
    span = oos.groupby("half").agg(start=("start", "min"), end=("end", "max"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, half in zip(axes, (1, 2)):
        v = w[half]
        ax.barh(range(len(v)), v.fillna(0), color=[colors.get(r, "#A89E96") for r in v.index])
        for i, x in enumerate(v):
            ax.text(x if not np.isnan(x) else 0, i, f" ${x:,.0f}k" if not np.isnan(x) else " not fit yet",
                    va="center", fontsize=8, color=INK if not np.isnan(x) else "#5E554F")
        rank = v.rank(ascending=False, method="min")
        for i, (r, x) in enumerate(v.items()):
            if not np.isnan(x):
                ax.text(4, i, f"#{int(rank[r])}", va="center", fontsize=8, color=PAPER, fontweight="bold")
        ax.axvline(START / 1e3, color=INK, lw=0.8, ls="--")
        ax.set_xlim(0, w.max().max() * 1.22)
        ax.set_xlabel(r"deployed \$k (net, \$100k start)")
        s, e = span.loc[half]
        ax.set_title(f"{s[:4]}-{e[:4]}: " + ("pick the winner here" if half == 1 else "then deploy it"), loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(w)), [labels.get(r, r) for r in w.index])
    axes[0].invert_yaxis()
    fig.text(0.99, 0.01, "ridge frozen on all months before the second half; single evals need no fitting",
             ha="right", fontsize=8, color="#5E554F")
    return _save(fig, path)
