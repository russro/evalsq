"""Shared fixtures: a small synthetic DataFrame that mirrors the real SPY schema."""

import pytest
import pandas as pd
import numpy as np


def _make_df(n=3000, seed=42):
    """Synthetic daily data spanning 2012-01-01 forward (~12 years)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-01", periods=n, freq="B")  # 2012 → ~2023
    close = 200 + np.cumsum(rng.normal(0, 1, n))
    raw = pd.DataFrame({"Close": close, "Ticker": "SPY"}, index=dates)
    raw.columns = pd.MultiIndex.from_tuples([("Close", "SPY"), ("Ticker", "SPY")])
    return raw


@pytest.fixture
def raw_df():
    return _make_df()


@pytest.fixture
def featured_df(raw_df):
    from evalsq.data import build_features
    return build_features(raw_df)
