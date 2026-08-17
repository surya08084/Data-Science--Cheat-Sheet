import numpy as np

from vapor_policy_impact.features.engineering import (
    aggregate_log_volume,
    build_aggregated_feature_series,
    build_feature_panel,
)


def test_build_feature_panel_adds_expected_columns(small_sim):
    fp = build_feature_panel(small_sim.panel, small_sim.policy_calendar)
    for col in ["log_volume", "event_week", "treated_flag", "not_yet_treated_flag", "volume_lag_1", "momentum_4v4"]:
        assert col in fp.columns
    assert len(fp) == len(small_sim.panel)


def test_treated_flag_matches_policy_calendar(small_sim):
    fp = build_feature_panel(small_sim.panel, small_sim.policy_calendar)
    for state, eff_week in small_sim.policy_calendar.treated_states.items():
        rows = fp[fp["state"] == state]
        assert (rows.loc[rows["week"] >= eff_week, "treated_flag"]).all()
        assert not (rows.loc[rows["week"] < eff_week, "treated_flag"]).any()


def test_aggregate_log_volume_sums_correctly(small_sim):
    total = aggregate_log_volume(small_sim.panel, category="Vapor")
    manual = (
        small_sim.panel[small_sim.panel["category"] == "Vapor"]
        .groupby(["state", "week"])["volume"]
        .sum()
        .reset_index()
    )
    merged = total.merge(manual, on=["state", "week"], suffixes=("_agg", "_manual"))
    assert np.allclose(merged["volume_agg"], merged["volume_manual"])


def test_aggregated_feature_series_has_no_duplicate_state_week(small_sim):
    """Regression test: this exact bug (duplicate (state, week) rows silently
    misaligning .groupby('state').shift(k)) produced a garbage alternating
    forecast in the quantile-GBM baseline forecaster before being fixed."""
    agg = build_aggregated_feature_series(small_sim.panel, small_sim.policy_calendar, category="Vapor")
    assert not agg.duplicated(subset=["state", "week"]).any()
    for col in ["log_volume", "volume_lag_1", "momentum_4v4", "fourier_sin_1"]:
        assert col in agg.columns
