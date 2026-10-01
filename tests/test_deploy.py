import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from evalsq.data import FEATURES
from evalsq.heuristics import combined
from evalsq.deploy import (COMBINED_OF, RULES, START, apply_rules, deploy_summary, equity, month_starts,
                           optimal_k, visible_end, walk_forward, winners_curse,
                           winners_curse_monthly)
from evalsq.models import make_grid, make_zoo

SMALL_ZOO = {
    "A": (FEATURES, LogisticRegression(max_iter=200)),
    "B": (["ret1", "ret5"], LogisticRegression(max_iter=200, C=0.01)),
}
SEL_COLS = ["accuracy", "auc", "neg_logloss", "lagged_pnl"]


def _scores(months, models, rng):
    """Handmade scores table with random metrics."""
    rows = [{"month": m, "model": k, **{c: rng.random() for c in RULES},
             "month_acc": rng.random(), "month_ret": rng.normal(0, 0.05), "spy_ret": 0.01}
            for m in months for k in models]
    return pd.DataFrame(rows)


def test_zoo_features_exist(featured_df):
    for feats, _ in list(make_zoo().values()) + list(make_grid(5).values()):
        assert set(feats) <= set(featured_df.columns)
    assert len(make_zoo()) == 8 and len(make_grid(5)) == 5


def test_month_starts(featured_df):
    starts = month_starts(featured_df, 2020)
    idx = featured_df.index[starts]
    assert idx.min().year == 2020
    assert (idx.to_period("M") != featured_df.index[[s - 1 for s in starts]].to_period("M")).all()


@pytest.mark.parametrize("lag", [0, 5, 21])
def test_no_lookahead(featured_df, lag):
    """Scrambling every label we can't see yet must not change what the selection window reports."""
    pos0 = month_starts(featured_df, 2020)[0]
    first = str(featured_df.index[pos0].to_period("M"))
    base = walk_forward(featured_df, SMALL_ZOO, 2020, lag=lag)
    future = featured_df.copy()
    end = visible_end(pos0, lag)
    rng = np.random.default_rng(1)
    future.iloc[end:, future.columns.get_loc("next_ret")] = rng.normal(0, 0.05, len(future) - end)
    future["target"] = (future["next_ret"] > 0).astype(int)
    scrambled = walk_forward(future, SMALL_ZOO, 2020, lag=lag)
    a = base[base["month"] == first].set_index("model")[SEL_COLS]
    b = scrambled[scrambled["month"] == first].set_index("model")[SEL_COLS]
    pd.testing.assert_frame_equal(a, b)


def test_walk_forward_shape(featured_df):
    s = walk_forward(featured_df, SMALL_ZOO, 2021)
    assert len(s) == 2 * s["month"].nunique()
    assert s["month"].min().startswith("2021")


def test_rules_pick_argmax():
    rng = np.random.default_rng(0)
    s = _scores([f"2020-{m:02d}" for m in range(1, 13)], ["A", "B", "C"], rng)
    picks = apply_rules(s)
    s["combined"] = combined(s, COMBINED_OF, by="month")  # apply_rules derives it from the other rules
    for rule in RULES + ["oracle"]:
        col = "month_ret" if rule == "oracle" else rule
        want = s.loc[s.groupby("month")[col].idxmax(), "model"].values
        got = picks[picks["rule"] == rule].sort_values("month")["model"].values
        assert (want == got).all()
    ns = picks[picks["rule"] == "never_switch"]["model"]
    assert ns.nunique() == 1 and ns.iloc[0] == picks[picks["rule"] == "accuracy"]["model"].iloc[0]


def test_nan_metric_keeps_last_pick():
    s = _scores(["2020-01", "2020-02"], ["A", "B"], np.random.default_rng(0))
    s.loc[s["month"] == "2020-02", "bear_acc"] = np.nan
    picks = apply_rules(s)
    bear = picks[picks["rule"] == "bear_acc"].sort_values("month")["model"]
    assert bear.iloc[0] == bear.iloc[1]


def test_oracle_beats_every_rule():
    s = _scores([f"2020-{m:02d}" for m in range(1, 13)], ["A", "B", "C"], np.random.default_rng(2))
    picks = apply_rules(s)
    wide = picks.pivot(index="month", columns="rule", values="month_ret")
    for r in RULES + ["never_switch"]:
        assert (wide["oracle"] >= wide[r]).all()
    summ = deploy_summary(picks).set_index("rule")["final_usd"]
    assert (summ["oracle"] >= summ[RULES + ["never_switch"]]).all()


def test_equity_compounds():
    picks = pd.DataFrame({"month": ["2020-01", "2020-02"], "rule": "x", "model": "A", "month_ret": [0.1, -0.1]})
    assert equity(picks)["x"].tolist() == pytest.approx([START * 1.1, START * 1.1 * 0.9])


def test_winners_curse_grows_with_k():
    """Bench and deployed scores are independent noise, so the best bench score is pure luck."""
    s = _scores([f"2020-{m:02d}" for m in range(1, 13)], [f"m{i}" for i in range(50)], np.random.default_rng(3))
    wc = winners_curse(s, ks=(1, 5, 50), n_draws=50).set_index("k")
    assert abs(wc.loc[1, "gap"]) < 0.1
    assert wc.loc[50, "gap"] > wc.loc[5, "gap"] > wc.loc[1, "gap"]


def test_walk_forward_parallel_matches_serial(featured_df):
    par = walk_forward(featured_df, SMALL_ZOO, 2021, n_jobs=2)
    ser = walk_forward(featured_df, SMALL_ZOO, 2021, n_jobs=1)
    pd.testing.assert_frame_equal(par, ser)


def test_walk_forward_falls_back_to_serial(featured_df, monkeypatch):
    import evalsq.deploy as dep

    def broken(*a, **k):
        raise OSError("no shared memory")

    monkeypatch.setattr(dep, "Parallel", broken)
    with pytest.warns(UserWarning, match="running serially"):
        s = dep.walk_forward(featured_df, SMALL_ZOO, 2021)
    pd.testing.assert_frame_equal(s, walk_forward(featured_df, SMALL_ZOO, 2021, n_jobs=1))


def test_winners_curse_monthly_matches_mean():
    s = _scores([f"2020-{m:02d}" for m in range(1, 13)], [f"m{i}" for i in range(20)], np.random.default_rng(4))
    m = winners_curse_monthly(s, ks=(1, 5, 20), n_draws=20)
    assert len(m) == 12 * 3 and set(m.columns) >= {"month", "k", "gap"}
    wc = winners_curse(s, ks=(1, 5, 20), n_draws=20).set_index("k")
    assert np.allclose(m.groupby("k")["gap"].mean(), wc["gap"])


def test_optimal_k_picks_best_deployed():
    m = pd.DataFrame({"month": ["a", "a", "b", "b"], "k": [1, 5, 1, 5], "deployed_acc": [0.5, 0.6, 0.7, 0.4]})
    opt = optimal_k(m, window=2)
    assert opt["opt_k"].tolist() == [5, 1]
    assert np.isnan(opt["opt_k_vol"].iloc[0]) and opt["opt_k_vol"].iloc[1] > 0
