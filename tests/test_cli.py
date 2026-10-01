"""E2E: full CLI on synthetic cached data, no network."""

import numpy as np
import pandas as pd

from evalsq.data import BOGLE

from evalsq.cli import main


def test_cli_end_to_end(tmp_path, featured_df):
    data = tmp_path / "spy.csv"
    featured_df[["close"]].to_csv(data, index_label="date")
    out = tmp_path / "out"
    figs = tmp_path / "figs"
    bogle = tmp_path / "bogle.csv"  # synthetic cache so the test never downloads
    idx = featured_df.index
    pd.DataFrame({t: 100 + i + np.arange(len(idx)) * 0.01 for i, t in enumerate(BOGLE)}, index=idx).to_csv(bogle, index_label="date")
    main(["--data", str(data), "--out", str(out), "--figs", str(figs), "--cutoff", "2021", "--grid", "3",
          "--bogle", str(bogle)])
    for f in ["h1_validity", "h1_rolling", "h1_summary", "h3_yearly", "h3_meta", "h2_selectors", "h2_folds_1y", "h2_folds_2y",
              "h3_meta_monthly", "deploy_scores", "deploy_picks", "deploy_picks_gross", "deploy_summary", "deploy_cost_sweep", "grid_scores", "winners_curse", "winners_curse_monthly", "optimal_k",
              "deploy_scores_all", "learned_summary", "learned_weights", "learned_picks"]:
        assert not pd.read_csv(out / f"{f}.csv").empty
    pngs = sorted(p.name for p in figs.glob("*.png"))
    assert pngs == sorted(["fig1_setup.png", "fig2_h1_validity.png", "fig3_h2_folds.png", "fig3b_h2_regret.png", "fig4_h3_meta.png",
                    "fig5_deploy.png", "fig5b_deploy_bogle.png", "fig6_picks.png", "fig7_winners_curse.png",
                    "fig8_gap_dist.png", "fig9_opt_k.png", "fig10_learned_weights.png",
                    "fig11_h3_redundancy.png", "fig11b_h3_complement.png"])
    assert all((figs / p).stat().st_size > 5000 for p in pngs)


def test_cli_no_plots(tmp_path, featured_df):
    data = tmp_path / "spy.csv"
    featured_df[["close"]].to_csv(data, index_label="date")
    figs = tmp_path / "figs"
    main(["--data", str(data), "--out", str(tmp_path / "out"), "--figs", str(figs), "--cutoff", "2022", "--no-plots", "--no-bogle"])
    assert not figs.exists()


def test_cli_meta_start_does_not_change_deploy(tmp_path, featured_df):
    """Extra early months only train the learned benchmark; the per-rule results are untouched."""
    data = tmp_path / "spy.csv"
    featured_df[["close"]].to_csv(data, index_label="date")
    runs = {}
    for start in (2012, 2021):
        out = tmp_path / str(start)
        main(["--data", str(data), "--out", str(out), "--cutoff", "2021", "--meta-start", str(start), "--no-plots", "--no-bogle"])
        runs[start] = pd.read_csv(out / "deploy_summary.csv")
    pd.testing.assert_frame_equal(runs[2012], runs[2021])
