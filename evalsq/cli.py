"""CLI entry point — run the full eval2 demo."""

import argparse
from pathlib import Path

from .data import load
from . import plots
from .heuristics import h1_rolling, h1_summary, h1_validity, h2_rank_flips, h2_temporal, h3_meta
from .models import accuracy, fit_models, make_grid, make_zoo, train_test_split_by_year
from .deploy import RULES, apply_rules, deploy_summary, equity, walk_forward, winners_curse


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
    p.add_argument("--grid", type=int, default=0, help="also run an N-config GBM grid for the winner's-curse backup (slow)")
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

    _section(f"H1b: Monthly deployment, lag {args.lag} days (money per selection rule)")
    scores = walk_forward(df, make_zoo(), args.cutoff, lag=args.lag, sel_window=args.sel_window)
    picks = apply_rules(scores)
    dep = deploy_summary(picks)
    print(dep.round(3).to_string(index=False))
    scores.to_csv(out / "deploy_scores.csv", index=False)
    picks.to_csv(out / "deploy_picks.csv", index=False)
    dep.to_csv(out / "deploy_summary.csv", index=False)

    wc = None
    if args.grid:
        _section(f"Backup: winner's curse over {args.grid} GBM configs")
        grid_scores = walk_forward(df, make_grid(args.grid), args.cutoff, lag=args.lag, sel_window=args.sel_window)
        wc = winners_curse(grid_scores)
        print(wc.round(4).to_string(index=False))
        grid_scores.to_csv(out / "grid_scores.csv", index=False)
        wc.to_csv(out / "winners_curse.csv", index=False)

    print(f"\nCSVs → {out.resolve()}/")

    if not args.no_plots:
        figs = Path(args.figs)
        figs.mkdir(parents=True, exist_ok=True)
        plots.plot_setup(df, args.cutoff, figs / "fig1_setup.png")
        plots.plot_h1(summary, roll, figs / "fig2_h1_validity.png")
        plots.plot_h2(h2, figs / "fig3_h2_holdout.png")
        plots.plot_h3(meta, r2, figs / "fig4_h3_meta.png")
        plots.plot_deploy(equity(picks), RULES, figs / "fig5_deploy.png")
        plots.plot_picks(picks, RULES + ["oracle"], figs / "fig6_picks.png")
        if wc is not None:
            plots.plot_winners_curse(wc, figs / "fig7_winners_curse.png")
        print(f"Figures → {figs.resolve()}/")


if __name__ == "__main__":
    main()
