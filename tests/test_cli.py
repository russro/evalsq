"""E2E: full CLI on synthetic cached data, no network."""

import pandas as pd

from evalsq.cli import main


def test_cli_end_to_end(tmp_path, featured_df):
    data = tmp_path / "spy.csv"
    featured_df[["close"]].to_csv(data, index_label="date")
    out = tmp_path / "out"
    main(["--data", str(data), "--out", str(out), "--cutoff", "2016"])
    for f in ["h1_validity", "h1_rolling", "h2_temporal", "h3_meta"]:
        assert not pd.read_csv(out / f"{f}.csv").empty
