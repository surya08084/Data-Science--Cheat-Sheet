import numpy as np

from vapor_policy_impact.causal.synthetic_control import fit_synthetic_control


def test_weights_are_nonnegative_and_sum_to_one(small_sim):
    target = small_sim.target_state
    eff_week = small_sim.target_state_effective_week
    res = fit_synthetic_control(
        small_sim.panel,
        target_state=target,
        donor_states=small_sim.donor_pool_states,
        pre_period_weeks=(max(0, eff_week - 60), eff_week - 1),
        category="Vapor",
    )
    weights = np.array(list(res.weights.values()))
    assert (weights >= -1e-8).all()
    assert abs(weights.sum() - 1.0) < 1e-6


def test_pre_period_fit_is_reasonably_close(small_sim):
    target = small_sim.target_state
    eff_week = small_sim.target_state_effective_week
    res = fit_synthetic_control(
        small_sim.panel,
        target_state=target,
        donor_states=small_sim.donor_pool_states,
        pre_period_weeks=(max(0, eff_week - 60), eff_week - 1),
        category="Vapor",
    )
    # the small test fixture has far fewer donor states/weeks than the full simulation
    # (see test_pipeline_and_validation.py / examples/run_demo.py for a tight-fit check
    # on the full-size panel, where RMSE is typically ~0.07-0.09), so this is a loose bound
    assert res.pre_period_rmse < 0.6


def test_synthetic_series_projects_into_future_weeks(small_sim):
    """The synthetic series must extend past the target's last observed week --
    that's what lets it be used as an exogenous regressor for forecasting."""
    target = small_sim.target_state
    eff_week = small_sim.target_state_effective_week
    res = fit_synthetic_control(
        small_sim.panel,
        target_state=target,
        donor_states=small_sim.donor_pool_states,
        pre_period_weeks=(max(0, eff_week - 60), eff_week - 1),
        category="Vapor",
    )
    assert res.series["week"].max() >= eff_week + 5
    future_rows = res.series[res.series["week"] >= eff_week]
    assert future_rows["synthetic_log_volume"].notna().all()
