"""CLI entry point — run the full eval2 demo."""

import argparse
from pathlib import Path

import pandas as pd

from .data import download_bogle, load
from . import plots
from .heuristics import h1_rolling, h1_summary, h1_validity, h2_rank_flips, h2_temporal, h3_meta, h3_meta_monthly
from .models import accuracy, fit_models, make_grid, make_zoo, train_test_split_by_year
from .learned import learned_summary
from .deploy import BORROW, COST_BP, REFERENCES, RULES, START, apply_costs, apply_rules, curse_by_k, deploy_summary, equity, optimal_k, walk_forward, winners_curse_monthly


def _section(title: str) -> None:
    print(f"\n── {title} ──")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="evalsq: benchmark evaluation heuristics demo")
    p.add_argument("--data", default="spy_daily.csv", help="cached SPY CSV (auto-downloaded if missing)")
    p.add_argument("--out", default="results", help="output dir for CSVs (default: results/)")
    p.add_argument("--cutoff", type=int, default=2018, help="train/test year cutoff (default: 2018)")
    p.add_argument("--figs", default="figures", help="output dir for PNGs (default: figures/)")
    p.add_argument("--lag", type=int, default=21, help="trading days before a label is visible (default: 21)")
    p.add_argument("--sel-window", type=int, default=63, help="selection window in trading days (default: 63)")
    p.add_argument("--grid", type=int, default=0, help="also run an N-config GBM grid for the winner's-curse backup (off by default; 25 is plenty, ~1 min)")
    p.add_argument("--jobs", type=int, default=-1, help="parallel workers for the walk-forward (default: all cores, 1 = serial)")
    p.add_argument("--cost-bp", type=float, default=COST_BP, help=f"trading cost in bp per unit of position traded (default: {COST_BP})")
    p.add_argument("--borrow", type=float, default=BORROW, help=f"annual borrow rate while short (default: {BORROW})")
    p.add_argument("--bogle", default="bogle_3fund.csv", help="cached VTI/VXUS/BND CSV for the Bogleheads line (auto-downloaded if missing)")
    p.add_argument("--no-bogle", action="store_true", help="skip the Bogleheads line (fig5b)")
    p.add_argument("--meta-start", type=int, default=2012, help="first year of walk-forward months; months before --cutoff only train the learned benchmark (default: 2012)")
    p.add_argument("--no-plots", action="store_true", help="skip figures")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.data)
    print(f"{len(df)} trading days, {df.index[0].date()} → {df.index[-1].date()}")

    train, test = train_test_split_by_year(df, args.cutoff)
    scaler, models = fit_models(train)
    print(f"\nBaseline accuracy, {args.cutoff}–present:")
    for name, m in models.items():
        print(f"  {name:7s} {accuracy(m, scaler, test):.3f}")

    _section("H1: Benchmark Validity (metric vs. Sharpe, monthly)")
    h1 = h1_validity(test, scaler, models)
    roll = h1_rolling(h1)
    summary = h1_summary(h1, roll)
    print(summary.round(3).to_string(index=False))
    h1.to_csv(out / "h1_validity.csv", index=False)
    roll.to_csv(out / "h1_rolling.csv", index=False)
    summary.to_csv(out / "h1_summary.csv", index=False)

    _section("H2: Temporal Holdout (rank stability)")
    h2 = h2_temporal(df)
    print(h2.round(4).to_string(index=False))
    print(f"\n  Rank flips: {h2_rank_flips(h2)} / {len(h2) - 1} transitions")
    h2.to_csv(out / "h2_temporal.csv", index=False)

    _section("H3: Predictability Test (meta-model)")
    meta, r2 = h3_meta(df, h2)
    print(meta.round(4).to_string(index=False))
    print(f"\n  Meta-model LOO R² = {r2:.3f}  (high → score tracks regime, not model)")
    meta.to_csv(out / "h3_meta.csv", index=False)
    meta_m, r2_m = h3_meta_monthly(df, list(h2["test_year"]))
    print(f"  Monthly version: LOO R² = {r2_m:.3f} over {len(meta_m)} months")
    meta_m.to_csv(out / "h3_meta_monthly.csv", index=False)

    _section(f"H1b: Monthly deployment, lag {args.lag} days (money per selection rule)")
    # months before the cutoff are identical to a cutoff-start run; they only feed the learned benchmark
    scores_all = walk_forward(df, make_zoo(), min(args.meta_start, args.cutoff), lag=args.lag,
                              sel_window=args.sel_window, n_jobs=args.jobs)
    cut = f"{args.cutoff}-01"
    scores = scores_all[scores_all["month"] >= cut].reset_index(drop=True)
    gross = apply_rules(scores)
    picks = apply_costs(gross, scores, args.cost_bp, args.borrow)
    dep = deploy_summary(picks)
    dep.insert(2, "gross_usd", deploy_summary(gross)["final_usd"])
    print(f"  net of {args.cost_bp}bp per unit traded + {args.borrow:.2%}/yr borrow")
    print(dep.round(3).to_string(index=False))
    sweep = pd.DataFrame({f"{bp}bp": deploy_summary(apply_costs(gross, scores, bp, args.borrow))["final_usd"]
                          for bp in (0, 1, 2, 5, 10)})
    sweep.insert(0, "rule", RULES + REFERENCES)
    print("\n  Final $ by trading cost:")
    print(sweep.round(0).to_string(index=False))
    scores.to_csv(out / "deploy_scores.csv", index=False)
    gross.to_csv(out / "deploy_picks_gross.csv", index=False)
    picks.to_csv(out / "deploy_picks.csv", index=False)
    dep.to_csv(out / "deploy_summary.csv", index=False)
    sweep.to_csv(out / "deploy_cost_sweep.csv", index=False)

    _section(f"Learned benchmark: ridge on bench signals, trained on {scores_all['month'].iloc[0]} on")
    lsum, lw, lpicks = learned_summary(scores_all, cut, cost_bp=args.cost_bp, borrow=args.borrow)
    print(lsum.round(3).to_string(index=False))
    scores_all.to_csv(out / "deploy_scores_all.csv", index=False)
    lsum.to_csv(out / "learned_summary.csv", index=False)
    lw.to_csv(out / "learned_weights.csv", index=False)
    lpicks.to_csv(out / "learned_picks.csv", index=False)

    wc = None
    if args.grid:
        _section(f"Backup: winner's curse over {args.grid} GBM configs")
        grid_scores = walk_forward(df, make_grid(args.grid), args.cutoff, lag=args.lag, sel_window=args.sel_window, n_jobs=args.jobs)
        wcm = winners_curse_monthly(grid_scores)
        wc = curse_by_k(wcm)
        opt = optimal_k(wcm)
        print(wc.round(4).to_string(index=False))
        grid_scores.to_csv(out / "grid_scores.csv", index=False)
        wc.to_csv(out / "winners_curse.csv", index=False)
        wcm.to_csv(out / "winners_curse_monthly.csv", index=False)
        opt.to_csv(out / "optimal_k.csv", index=False)
        print(f"\n  Best k per month: {opt['opt_k'].value_counts().sort_index().to_dict()}")

    print(f"\nCSVs → {out.resolve()}/")

    if not args.no_plots:
        figs = Path(args.figs)
        figs.mkdir(parents=True, exist_ok=True)
        plots.plot_setup(df, args.cutoff, figs / "fig1_setup.png")
        plots.plot_h1(summary, roll, figs / "fig2_h1_validity.png")
        plots.plot_h2(h2, figs / "fig3_h2_holdout.png")
        plots.plot_h3(meta, r2, meta_m, r2_m, figs / "fig4_h3_meta.png")
        eq = equity(picks)
        plots.plot_deploy(eq, RULES, figs / "fig5_deploy.png")
        if not args.no_bogle:
            bogle = download_bogle(args.bogle).reindex(eq.index).fillna(0)
            eq["bogle"] = START * (1 + bogle).cumprod()
            plots.plot_deploy(eq, RULES, figs / "fig5b_deploy_bogle.png")
        plots.plot_picks(picks, RULES + ["oracle"], figs / "fig6_picks.png")
        if not lw.empty:
            plots.plot_learned_weights(lw, lsum, cut, figs / "fig10_learned_weights.png")
        if wc is not None:
            plots.plot_winners_curse(wc, figs / "fig7_winners_curse.png")
            plots.plot_gap_dist(wcm, figs / "fig8_gap_dist.png")
            plots.plot_opt_k(opt, figs / "fig9_opt_k.png")
        print(f"Figures → {figs.resolve()}/")


if __name__ == "__main__":
    main()
