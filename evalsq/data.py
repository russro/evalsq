"""Data loading and feature engineering for the SPY demo dataset."""

from pathlib import Path

import pandas as pd

FEATURES = ["ret1", "ret5", "ret20", "ma10", "ma50", "vol20"]


def download_spy(cache_path: Path | str = "spy_daily.csv") -> pd.DataFrame:
    """Return SPY daily closes (single 'close' column); download + cache on first call."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        raw = pd.read_csv(cache_path, index_col=0)
        # legacy yfinance multi-header cache: drop 'Ticker'/'Date' rows
        raw = raw[pd.to_datetime(raw.index, errors="coerce", format="%Y-%m-%d").notna()]
        raw.index = pd.to_datetime(raw.index)
        return _close_only(raw.astype(float))
    import yfinance as yf
    print("Downloading SPY data (2010-2025)...")
    raw = _close_only(yf.download("SPY", start="2010-01-01", end="2026-01-01", auto_adjust=True, progress=False))
    raw.to_csv(cache_path, index_label="date")
    return raw


def _close_only(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise yfinance output (flat or MultiIndex columns) to a single 'close' column."""
    cols = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
    close = raw.iloc[:, [str(c).lower() == "close" for c in cols].index(True)]
    return pd.DataFrame({"close": close.astype(float)}, index=raw.index)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and next-day direction target. All features use data up to t only."""
    df = _close_only(raw)
    c = df["close"]
    df["ret1"] = c.pct_change(1)
    df["ret5"] = c.pct_change(5)
    df["ret20"] = c.pct_change(20)
    df["ma10"] = c.rolling(10).mean() / c - 1
    df["ma50"] = c.rolling(50).mean() / c - 1
    df["vol20"] = df["ret1"].rolling(20).std()
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["next_ret"] = df["ret1"].shift(-1)
    df["target"] = (df["next_ret"] > 0).astype(int)
    return df.dropna()


# Bogleheads three-fund portfolio: US total market, international, bonds. Rebalanced monthly.
BOGLE = {"VTI": 0.6, "VXUS": 0.2, "BND": 0.2}


def download_bogle(cache_path: Path | str = "bogle_3fund.csv") -> pd.Series:
    """Monthly returns of the three-fund portfolio, indexed by 'YYYY-MM'; download + cache on first call."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        import yfinance as yf
        print("Downloading VTI/VXUS/BND (Bogleheads three-fund)...")
        raw = yf.download(list(BOGLE), start="2010-01-01", end="2026-01-01", auto_adjust=True, progress=False)["Close"]
        raw.to_csv(cache_path, index_label="date")
    closes = pd.read_csv(cache_path, index_col=0, parse_dates=True)[list(BOGLE)]
    monthly = closes.resample("ME").last().pct_change().dropna()
    ret = (monthly * pd.Series(BOGLE)).sum(axis=1)
    ret.index = ret.index.to_period("M").astype(str)
    return ret.rename("bogle")


def load(cache_path: Path | str = "spy_daily.csv") -> pd.DataFrame:
    """Download (or load cache) and return feature-engineered DataFrame."""
    return build_features(download_spy(cache_path))
