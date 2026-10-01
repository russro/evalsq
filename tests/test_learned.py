import numpy as np
import pandas as pd
import pytest

from evalsq.learned import (MODES, SIGNALS, learned_picks, learned_select, learned_summary, lomo, meta_table,
                            rank_ic, train_months)

MODELS = list("ABCDEFGH")


def _scores(n_months=60, signal="bear_acc", strength=0.05, seed=0):
    """Random metrics; month_ret = month effect + strength * `signal` (if given) + noise."""
    rng = np.random.default_rng(seed)
    months = [str(p) for p in pd.period_range("2012-01", periods=n_months, freq="M")]
    rows = []
    for m in months:
        shock = rng.normal(0, 0.05)
        for k in MODELS:
            r = {"month": m, "model": k, **{c: rng.random() for c in SIGNALS}, "spy_ret": shock,
                 "flips": 1, "short_days": 5, "first_pos": 1, "last_pos": 1}
            r["month_ret"] = shock + (strength * r[signal] if signal else 0) + rng.normal(0, 0.01)
            rows.append(r)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("mode,i,expect", [
    ("expanding", 10, range(0, 9)), ("expanding", 5, range(0)),
    ("rolling", 10, range(3, 9)), ("rolling", 6, range(0)),
    ("static", 10, range(0, 6)), ("static", 20, range(0, 6)), ("static", 6, range(0)),
])
def test_train_months(mode, i, expect):
    assert train_months(i, mode, n_train=6) == expect


def test_train_months_never_sees_live_month():
    for mode in MODES:
        for i in range(30):
            assert all(j <= i - 2 for j in train_months(i, mode, 6))


def test_train_months_bad_mode():
    with pytest.raises(ValueError):
        train_months(10, "nope", 6)


def test_meta_table_demeaned_within_month():
    s = _scores(12)
    s.loc[0, "bear_acc"] = np.nan
    tab, cols = meta_table(s)
    assert cols == SIGNALS and not tab[cols].isna().any().any()
    sums = tab.groupby("month")[cols + ["y"]].sum()
    assert np.allclose(sums.values, 0)


def test_meta_table_regime_interactions():
    tab, cols = meta_table(_scores(12), use_regime=True)
    assert len(cols) == 3 * len(SIGNALS) and "bear_acc*vol" in cols


@pytest.mark.parametrize("mode", MODES)
def test_learned_select_recovers_planted_signal(mode):
    s = _scores(60)
    preds, w = learned_select(s, mode, n_train=24, alpha=1.0)
    last = w[w["month"] == w["month"].max()].set_index("feature")["weight"]
    assert last.idxmax() == "bear_acc"
    assert preds["pred"].isna().sum() == 8 * (24 + 1)  # first n_train + gap - 1 months have no meta-model


def test_static_weights_frozen():
    _, w = learned_select(_scores(40), "static", n_train=12)
    assert (w.groupby("feature")["weight"].nunique() == 1).all()


def test_holdout_model_not_trained_on():
    s = _scores(40)
    _, clean = learned_select(s, n_train=12, holdout="A")
    s.loc[s["model"] == "A", "month_ret"] = 99.0  # only changes the fit if A leaks into training
    _, poisoned = learned_select(s, n_train=12, holdout="A")
    assert np.allclose(clean["weight"], poisoned["weight"])
    _, leaky = learned_select(s, n_train=12)
    assert not np.allclose(clean["weight"], leaky["weight"])


def test_learned_picks_fallback_and_argmax():
    s = _scores(30)
    preds, _ = learned_select(s, n_train=12)
    pk = learned_picks(s, preds, "learned")
    assert len(pk) == 30 and set(pk["rule"]) == {"learned"}
    early = pk[pk["month"] < preds.dropna()["month"].min()]
    acc_pick = s.loc[s.groupby("month")["accuracy"].idxmax()].set_index("month")["model"]
    assert (early.set_index("month")["model"] == acc_pick.loc[early["month"]]).all()
    late = pk[pk["month"] >= preds.dropna()["month"].min()].set_index("month")["model"]
    pred_pick = preds.dropna().loc[preds.dropna().groupby("month")["pred"].idxmax()].set_index("month")["model"]
    assert (late == pred_pick).all()


def test_rank_ic_sign():
    s = _scores(20, strength=0.5)
    assert rank_ic(s, s["bear_acc"]).mean() > 0.8
    assert rank_ic(s, -s["bear_acc"]).mean() < -0.8


def test_rank_ic_skips_constant_months():
    s = _scores(4)
    s.loc[s["month"] == s["month"].iloc[0], "month_ret"] = 0.01
    assert len(rank_ic(s, s["accuracy"])) == 3


def test_lomo_transfers_planted_signal():
    ic = lomo(_scores(60, strength=0.1), n_train=24, alpha=1.0).dropna()
    assert ic["lomo"].mean() > 0.5 and ic["bear_acc"].mean() > 0.5


def test_lomo_no_signal_near_zero():
    ic = lomo(_scores(60, signal=None, seed=1), n_train=24).dropna()
    assert abs(ic["lomo"].mean()) < 0.15


def test_learned_summary_beats_random_with_signal():
    s = _scores(72, strength=0.05)
    summ, w, pk = learned_summary(s, "2015-01", n_train=24)
    assert list(summ["mode"]) == MODES
    assert (summ["lomo_ic"] > 0.3).all()
    assert set(w["mode"]) == set(MODES) and pk["month"].min() == "2015-01"
