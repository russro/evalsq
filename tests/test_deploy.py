import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from evalsq.data import FEATURES
from evalsq.heuristics import combined
from evalsq.deploy import (COMBINED_OF, RULES, START, apply_costs, apply_rules, block_label, deploy_summary, equity, fold_returns, h2_selectors, h3_complementarity, h3_redundancy, combo_usd, month_starts,
                           selector_stability,
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
    s["combined"] = combined(s, COMBINED_OF, by="month").round(9)  # apply_rules derives it from the other rules
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


def _cost_scores():
    """Two months, two models, with hand-set positions."""
    return pd.DataFrame({
        "month": ["2020-01", "2020-01", "2020-02", "2020-02"], "model": ["A", "B", "A", "B"],
        "month_ret": [0.1, 0.0, 0.0, 0.2], "spy_ret": [0.01, 0.01, 0.02, 0.02],
        "flips": [1, 0, 0, 2], "short_days": [3, 0, 0, 5],
        "first_pos": [1, 1, 1, -1], "last_pos": [-1, 1, 1, -1],
    })


def test_apply_costs_zero_is_gross():
    s = _cost_scores()
    picks = pd.DataFrame({"month": ["2020-01", "2020-02"], "rule": "x", "model": ["A", "B"], "month_ret": [0.1, 0.2]})
    pd.testing.assert_frame_equal(apply_costs(picks, s, 0, 0), picks)


def test_apply_costs_charges_entry_flips_swaps_and_borrow():
    s = _cost_scores()
    picks = pd.DataFrame({"month": ["2020-01", "2020-02"], "rule": "x", "model": ["A", "B"], "month_ret": [0.1, 0.2]})
    net = apply_costs(picks, s, cost_bp=10, borrow=0.252)["month_ret"].tolist()
    c, b = 1e-3, 1e-3
    # month 1: enter from cash (1 unit) + 1 flip (2 units), 3 short days
    # month 2: A ends -1, B opens -1 so no swap trade; 2 flips (4 units), 5 short days
    assert net == pytest.approx([1.1 * (1 - c) ** 3 * (1 - b) ** 3 - 1, 1.2 * (1 - c) ** 4 * (1 - b) ** 5 - 1])


def test_apply_costs_swap_to_other_side_pays_flip():
    s = _cost_scores()
    picks = pd.DataFrame({"month": ["2020-01", "2020-02"], "rule": "x", "model": ["B", "B"], "month_ret": [0.0, 0.2]})
    net = apply_costs(picks, s, cost_bp=10, borrow=0)["month_ret"].tolist()
    # B ends month 1 long, opens month 2 short: 2 units at the boundary + 4 inside
    assert net[1] == pytest.approx(1.2 * (1 - 1e-3) ** 6 - 1)


def test_apply_costs_always_long_pays_once():
    picks = pd.DataFrame({"month": ["2020-01", "2020-02"], "rule": "always_long", "model": "SPY", "month_ret": [0.01, 0.02]})
    net = apply_costs(picks, _cost_scores(), cost_bp=10, borrow=1.0)["month_ret"].tolist()
    assert net == pytest.approx([1.01 * (1 - 1e-3) - 1, 0.02])


def test_costs_never_help(featured_df):
    s = walk_forward(featured_df, SMALL_ZOO, 2021)
    gross = apply_rules(s)
    net = apply_costs(gross, s)
    assert (net["month_ret"] <= gross["month_ret"] + 1e-12).all()
    assert (s["flips"] >= 0).all() and set(s["first_pos"]) <= {-1, 1}


def test_walk_forward_skips_months_without_enough_history(featured_df):
    s = walk_forward(featured_df, SMALL_ZOO, featured_df.index[0].year, n_jobs=1)
    first = pd.Period(s["month"].min(), "M").to_timestamp()
    assert (featured_df.index < first).sum() >= 500 + 63 + 21


def _picks(rets: dict) -> pd.DataFrame:
    """Picks table from {rule: [monthly returns]}, 12 months per year from 2018."""
    n = len(next(iter(rets.values())))
    months = [str(p) for p in pd.period_range("2018-01", periods=n, freq="M")]
    return pd.DataFrame([{"month": m, "rule": r, "model": "A", "month_ret": v}
                         for r, vs in rets.items() for m, v in zip(months, vs)])


def test_fold_returns_compound_per_block():
    rets = {r: [0.0] * 36 for r in RULES}
    rets["bear_acc"] = [0.01] * 36
    picks = pd.concat([_picks(rets), pd.DataFrame([{"month": "2018-01", "rule": "oracle", "model": "A", "month_ret": 9.0}])])
    f1, f2 = fold_returns(picks, 1), fold_returns(picks, 2)
    assert list(f1.index) == ["2018", "2019", "2020"] and list(f1.columns) == RULES  # references dropped
    assert list(f2.index) == ["2018-19", "2020"]  # trailing partial block kept
    assert f1.loc["2018", "bear_acc"] == pytest.approx(1.01 ** 12 - 1)
    assert f2.loc["2018-19", "bear_acc"] == pytest.approx(1.01 ** 24 - 1)
    assert (f1["accuracy"] == 0).all()


def test_selector_stability_hit_rate_and_regret():
    folds = pd.DataFrame({"a": [0.3, 0.1, 0.0], "b": [0.0, 0.2, 0.4]}, index=["f1", "f2", "f3"])
    s = selector_stability(folds)
    # winners a, b, b -> one repeat out of two transitions
    assert s["hit_rate"] == 0.5 and s["chance"] == 0.5 and s["n_folds"] == 3
    # follow leader: f2 uses a (0.1 vs best 0.2), f3 uses b (0.4) -> regret 0.05
    assert s["follow_leader"] == pytest.approx(0.05)
    assert s["random"] == pytest.approx(((0.2 - 0.15) + (0.4 - 0.2)) / 2)
    assert s["fixed_a"] == pytest.approx((0.1 + 0.4) / 2) and s["fixed_b"] == 0


def test_h2_selectors_one_row_per_block():
    rng = np.random.default_rng(0)
    picks = _picks({r: rng.normal(0, 0.03, 48) for r in RULES})
    folds, summary = h2_selectors(picks)
    assert set(folds) == {0.5, 1, 2} and len(folds[0.5]) == 8 and len(folds[1]) == 4 and len(folds[2]) == 2
    assert list(folds[0.5].index[:2]) == ["2018H1", "2018H2"]
    assert list(summary["block_years"]) == [0.5, 1, 2]
    assert summary["hit_rate"].between(0, 1).all()
    assert (summary[[c for c in summary if c.startswith("fixed_") or c in ("follow_leader", "random")]] >= 0).all().all()


def test_half_year_folds_compound_to_year():
    rng = np.random.default_rng(1)
    picks = _picks({r: rng.normal(0, 0.03, 24) for r in RULES})
    h, y = fold_returns(picks, 0.5), fold_returns(picks, 1)
    assert np.allclose((1 + h.iloc[0]) * (1 + h.iloc[1]) - 1, y.iloc[0])
    assert [block_label(b) for b in (0.5, 1, 2)] == ["6mo", "1y", "2y"]


def test_h3_redundancy_identical_rules_agree():
    rng = np.random.default_rng(0)
    s = _scores([f"2018-{m:02d}" for m in range(1, 13)], list("ABCD"), rng)
    s["auc"] = s["accuracy"] * 2  # same ranking, different scale
    s.loc[s["month"] == "2018-01", "bear_acc"] = np.nan  # undefined month is skipped, not zeroed
    corr, agree = h3_redundancy(s, apply_rules(s))
    assert list(corr.columns) == COMBINED_OF + ["month_ret"]
    assert corr.loc["accuracy", "auc"] == pytest.approx(1) and agree.loc["accuracy", "auc"] == 1
    assert corr.notna().all().all() and np.allclose(corr, corr.T)


def test_combo_usd_single_rule_matches_rule():
    rng = np.random.default_rng(1)
    s = _scores([f"2018-{m:02d}" for m in range(1, 13)], list("ABC"), rng)
    s[["flips", "short_days", "first_pos", "last_pos"]] = [0, 0, 1, 1]
    picks = apply_costs(apply_rules(s), s)
    assert combo_usd(s, ["accuracy"]) == pytest.approx(equity(picks)["accuracy"].iloc[-1])
    assert combo_usd(s, COMBINED_OF) == pytest.approx(equity(picks)["combined"].iloc[-1])


def test_h3_complementarity_forward_and_pairs():
    rng = np.random.default_rng(2)
    s = _scores([f"2018-{m:02d}" for m in range(1, 13)], list("ABC"), rng)
    s[["flips", "short_days", "first_pos", "last_pos"]] = [0, 0, 1, 1]
    fwd, pair = h3_complementarity(s)
    assert list(fwd["n"]) == [1, 2, 3, 4, 5] and sorted(fwd["added"]) == sorted(COMBINED_OF)
    assert fwd["final_usd"].iloc[0] == max(combo_usd(s, [r]) for r in COMBINED_OF)
    assert fwd["final_usd"].iloc[-1] == pytest.approx(combo_usd(s, COMBINED_OF))
    assert (np.diag(pair) == 0).all()
    assert pair.loc["accuracy", "auc"] == pytest.approx(combo_usd(s, ["accuracy", "auc"]) - combo_usd(s, ["accuracy"]))
