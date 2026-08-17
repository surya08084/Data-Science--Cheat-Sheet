import numpy as np
import pandas as pd
import pytest

from vapor_policy_impact.causal.synthetic_control import fit_synthetic_control
from vapor_policy_impact.features.engineering import aggregate_log_volume, build_aggregated_feature_series
from vapor_policy_impact.forecasting.baseline import (
    BSTSBaselineForecaster,
    QuantileGBMForecaster,
    ensemble_baseline_forecast,
)


def _synthetic_exog(small_sim):
    eff_week = small_sim.target_state_effective_week
    sc = fit_synthetic_control(
        small_sim.panel,
        target_state=small_sim.target_state,
        donor_states=small_sim.donor_pool_states,
        pre_period_weeks=(max(0, eff_week - 60), eff_week - 1),
        category="Vapor",
    )
    return sc.series.set_index("week")["synthetic_log_volume"]


def test_bsts_forecast_shape_and_weeks(small_sim):
    synth = _synthetic_exog(small_sim)
    target_series = aggregate_log_volume(small_sim.panel, category="Vapor")
    target_series = target_series[target_series["state"] == small_sim.target_state].set_index("week")["log_volume"]

    res = BSTSBaselineForecaster(n_draws=60).fit_and_forecast(target_series, synth, horizon=13)
    assert res.draws.shape == (60, 13)
    assert list(res.weeks) == list(range(small_sim.target_state_effective_week, small_sim.target_state_effective_week + 13))
    assert np.all(res.q10 <= res.mean) and np.all(res.mean <= res.q90)


def test_gbm_forecast_is_smooth_not_alternating(small_sim):
    """Regression test for the duplicate-row bug: a badly-aligned lag feature set
    produced wildly alternating week-to-week forecasts."""
    agg = build_aggregated_feature_series(small_sim.panel, small_sim.policy_calendar, category="Vapor")
    res = QuantileGBMForecaster().fit_and_forecast(
        agg, donor_states=small_sim.donor_pool_states, target_state=small_sim.target_state, horizon=13
    )
    diffs = np.abs(np.diff(res.mean))
    assert diffs.max() < 1.0  # week-over-week log-volume jumps should be modest, not >100% swings


def test_gbm_rejects_duplicate_state_week_rows(small_sim):
    agg = build_aggregated_feature_series(small_sim.panel, small_sim.policy_calendar, category="Vapor")
    dup = pd.concat([agg, agg.iloc[:5]], ignore_index=True)
    with pytest.raises(ValueError, match="one row per"):
        QuantileGBMForecaster().fit_and_forecast(
            dup, donor_states=small_sim.donor_pool_states, target_state=small_sim.target_state, horizon=3
        )


def test_ensemble_pools_draws(small_sim):
    synth = _synthetic_exog(small_sim)
    target_series = aggregate_log_volume(small_sim.panel, category="Vapor")
    target_series = target_series[target_series["state"] == small_sim.target_state].set_index("week")["log_volume"]
    bsts = BSTSBaselineForecaster(n_draws=60).fit_and_forecast(target_series, synth, horizon=13)

    agg = build_aggregated_feature_series(small_sim.panel, small_sim.policy_calendar, category="Vapor")
    gbm = QuantileGBMForecaster().fit_and_forecast(
        agg, donor_states=small_sim.donor_pool_states, target_state=small_sim.target_state, horizon=13
    )
    ens = ensemble_baseline_forecast([bsts, gbm], weights=[0.5, 0.5])
    assert ens.draws.shape[1] == 13
    assert ens.draws.shape[0] > 0
