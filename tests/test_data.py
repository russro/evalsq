import pandas as pd
import pytest

from evalsq.data import FEATURES, build_features, download_spy


def test_build_features_columns(featured_df):
    for col in FEATURES + ["target", "close", "next_ret"]:
        assert col in featured_df.columns


def test_build_features_no_nan(featured_df):
    assert not featured_df.isnull().any().any()


def test_target_is_next_day_direction(featured_df):
    assert ((featured_df["next_ret"] > 0).astype(int) == featured_df["target"]).all()
    # next_ret(t) == ret1(t+1): no lookahead in features, target is strictly future
    assert featured_df["next_ret"].iloc[:-1].values == pytest.approx(featured_df["ret1"].iloc[1:].values)


def test_minimum_rows(featured_df):
    assert len(featured_df) >= 500


def test_cache_legacy_multiheader(tmp_path):
    """yfinance's 3-row header CSV must load cleanly."""
    p = tmp_path / "spy.csv"
    p.write_text("Price,Close,Open\nTicker,SPY,SPY\nDate,,\n2020-01-02,100.0,99\n2020-01-03,101.0,100\n")
    raw = download_spy(p)
    assert list(raw.columns) == ["close"]
    assert len(raw) == 2 and isinstance(raw.index, pd.DatetimeIndex)


def test_cache_flat_roundtrip(tmp_path):
    p = tmp_path / "spy.csv"
    idx = pd.date_range("2020-01-01", periods=3, name="date")
    pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx).to_csv(p)
    assert download_spy(p)["close"].tolist() == [1.0, 2.0, 3.0]
