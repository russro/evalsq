import math

import numpy as np
import pandas as pd
import pytest

from evalsq.heuristics import (
    BASE_METRICS, METRICS, _metrics, combined, _score, h1_correlation, h1_rolling, h1_summary, h1_validity,
    h3_meta, h3_meta_monthly, yearly_holdout,
)
from evalsq.models import fit_models, train_test_split_by_year

CUTOFF = 2016
TEST_YEARS = list(range(2020, 2024))


@pytest.fixture
def fitted(featured_df):
    train, test = train_test_split_by_year(featured_df, cutoff=CUTOFF)
    scaler, models = fit_models(train)
    return test, scaler, models


@pytest.fixture
def yearly(featured_df):
    return yearly_holdout(featured_df, years=TEST_YEARS)


def test_score_perfect_predictions_positive_sharpe():
    w = pd.DataFrame({"next_ret": [0.01, -0.02, 0.03, -0.01]})
    w["target"] = (w["next_ret"] > 0).astype(int)
    acc, sharpe = _score(w["target"].values, w)
    assert acc == 1.0 and sharpe > 0
    acc, sharpe = _score(1 - w["target"].values, w)
    assert acc == 0.0 and sharpe < 0


def test_h1_shape_and_ranges(fitted):
    test, scaler, models = fitted
    h1 = h1_validity(test, scaler, models)
    assert set(h1["model"]) == set(models)
    assert h1["accuracy"].between(0, 1).all()
    assert -1 <= h1_correlation(h1) <= 1


def test_metrics_perfect_and_inverted():
    w = pd.DataFrame({"next_ret": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03],
                      "ma50": [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1]})
    w["target"] = (w["next_ret"] > 0).astype(int)
    y = w["target"].values
    good = _metrics(y, np.where(y == 1, 0.9, 0.1), w)
    bad = _metrics(1 - y, np.where(y == 1, 0.1, 0.9), w)
    assert set(good) == set(BASE_METRICS)
    assert good["accuracy"] == good["auc"] == good["bull_acc"] == good["bear_acc"] == 1.0
    assert bad["accuracy"] == bad["auc"] == 0.0
    assert good["neg_logloss"] > bad["neg_logloss"]


def test_h1_all_metrics_and_summary(fitted):
    test, scaler, models = fitted
    h1 = h1_validity(test, scaler, models)
    roll = h1_rolling(h1, window=12)
    assert set(roll["metric"]) <= set(METRICS)
    summary = h1_summary(h1, roll)
    assert list(summary["metric"]) == METRICS
    assert summary["pooled_corr"].between(-1, 1).all()


def test_h1_rolling(fitted):
    test, scaler, models = fitted
    roll = h1_rolling(h1_validity(test, scaler, models), window=12)
    assert not roll.empty
    assert roll["rolling_corr"].between(-1, 1).all()


def test_yearly_holdout_one_row_per_year(yearly):
    assert list(yearly["test_year"]) == TEST_YEARS
    assert yearly[["LogReg", "RF"]].stack().between(0, 1).all()


def test_h3_loo_r2(featured_df, yearly):
    meta, r2 = h3_meta(featured_df, yearly)
    assert len(meta) == len(TEST_YEARS)
    assert "pred_acc" in meta.columns
    assert not math.isnan(r2)


def test_combined_is_mean_rank_and_skips_nan():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"], "x": [1, 2, 3, 4], "y": [2, 1, np.nan, 5]})
    c = combined(df, ["x", "y"], by="g")
    assert c.tolist() == [0.75, 0.75, 0.5, 1.0]
    assert combined(df, ["x"]).tolist() == [0.25, 0.5, 0.75, 1.0]


def test_h1_has_combined(fitted):
    test, scaler, models = fitted
    h1 = h1_validity(test, scaler, models)
    assert h1["combined"].between(0, 1).all()


def test_h3_monthly(featured_df):
    meta, r2 = h3_meta_monthly(featured_df, TEST_YEARS)
    assert len(meta) > 12 * (len(TEST_YEARS) - 1) and meta["month"].is_unique
    assert meta["acc"].between(0, 1).all() and not math.isnan(r2)
