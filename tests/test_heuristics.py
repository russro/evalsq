import math

import numpy as np
import pandas as pd
import pytest

from evalsq.heuristics import (
    _score, h1_correlation, h1_rolling, h1_validity,
    h2_rank_flips, h2_temporal, h3_meta,
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
def h2(featured_df):
    return h2_temporal(featured_df, years=TEST_YEARS)


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


def test_h1_rolling(fitted):
    test, scaler, models = fitted
    roll = h1_rolling(h1_validity(test, scaler, models), window=12)
    assert not roll.empty
    assert roll["rolling_corr"].between(-1, 1).all()


def test_h2_winner_is_argmax(h2):
    assert len(h2) == len(TEST_YEARS)
    for _, r in h2.iterrows():
        assert r[r["winner"]] == max(r["LogReg"], r["RF"])


def test_h2_rank_flips_counts_transitions_only():
    h2 = pd.DataFrame({"winner": ["RF", "RF", "LogReg", "LogReg", "RF"]})
    assert h2_rank_flips(h2) == 2
    assert h2_rank_flips(pd.DataFrame({"winner": ["RF"] * 4})) == 0


def test_h3_loo_r2(featured_df, h2):
    meta, r2 = h3_meta(featured_df, h2)
    assert len(meta) == len(TEST_YEARS)
    assert "pred_acc" in meta.columns
    assert not math.isnan(r2)
