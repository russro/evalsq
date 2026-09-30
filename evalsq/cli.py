"""CLI entry point — run the full eval2 demo."""

import argparse
from pathlib import Path

from .data import load
from . import plots
from .heuristics import h1_rolling, h1_summary, h1_validity, h2_rank_flips, h2_temporal, h3_meta
from .models import accuracy, fit_models, train_test_split_by_year


def _section(title: str) -> None:
    print(f"\n── {title} ──")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="evalsq: benchmark evaluation heuristics demo")
    p.add_argument("--data", default="spy_daily.csv", help="cached SPY CSV (auto-downloaded if missing)")
    p.add_argument("--out", default="results", help="output dir for CSVs (default: results/)")
    p.add_argument("--cutoff", type=int, default=2018, help="train/test year cutoff (default: 2018)")
    p.add_argument("--figs", default="figures", help="output dir for PNGs (default: figures/)")
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

    print(f"\nCSVs → {out.resolve()}/")

    if not args.no_plots:
        figs = Path(args.figs)
        figs.mkdir(parents=True, exist_ok=True)
        plots.plot_setup(df, args.cutoff, figs / "fig1_setup.png")
        plots.plot_h1(summary, roll, figs / "fig2_h1_validity.png")
        plots.plot_h2(h2, figs / "fig3_h2_holdout.png")
        plots.plot_h3(meta, r2, figs / "fig4_h3_meta.png")
        print(f"Figures → {figs.resolve()}/")


if __name__ == "__main__":
    main()
