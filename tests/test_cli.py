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
    for f in ["h1_validity", "h1_rolling", "h1_summary", "h2_temporal", "h3_meta",
              "h3_meta_monthly", "deploy_scores", "deploy_picks", "deploy_summary", "grid_scores", "winners_curse", "winners_curse_monthly", "optimal_k"]:
        assert not pd.read_csv(out / f"{f}.csv").empty
    pngs = sorted(p.name for p in figs.glob("*.png"))
    assert pngs == ["fig1_setup.png", "fig2_h1_validity.png", "fig3_h2_holdout.png", "fig4_h3_meta.png",
                    "fig5_deploy.png", "fig5b_deploy_bogle.png", "fig6_picks.png", "fig7_winners_curse.png",
                    "fig8_gap_dist.png", "fig9_opt_k.png"]
    assert all((figs / p).stat().st_size > 5000 for p in pngs)


def test_cli_no_plots(tmp_path, featured_df):
    data = tmp_path / "spy.csv"
    featured_df[["close"]].to_csv(data, index_label="date")
    figs = tmp_path / "figs"
    main(["--data", str(data), "--out", str(tmp_path / "out"), "--figs", str(figs), "--cutoff", "2022", "--no-plots", "--no-bogle"])
    assert not figs.exists()
