import numpy as np
import pytest

from vapor_policy_impact.pipeline import run_category_pipeline
from vapor_policy_impact.validation.loso import (
    loso_aggregate_metrics,
    run_loso,
    run_placebo_in_time,
    summarize_loso,
)

pytestmark = pytest.mark.slow  # these exercise the full stack end-to-end; still kept fast via small_sim + tiny bootstrap


def test_run_category_pipeline_end_to_end(small_sim, small_covariates):
    result = run_category_pipeline(
        small_sim.panel,
        small_sim.policy_calendar,
        small_covariates,
        target_state=small_sim.target_state,
        target_effective_week=small_sim.target_state_effective_week,
        donor_states=small_sim.donor_pool_states,
        category="Vapor",
        n_bootstrap=20,
        n_mc=500,
    )
    assert result.scenario.cumulative["scenario_a"]["mean"] > 0
    assert result.scenario.cumulative["scenario_b"]["mean"] > 0
    # the injected effect is a vapor decline -> policy scenario should be lower on average
    assert result.scenario.cumulative["impact"]["mean"] < result.scenario.cumulative["scenario_b"]["mean"]


def test_run_loso_produces_one_row_per_treated_state(small_sim, small_covariates):
    results = run_loso(
        small_sim.panel,
        small_sim.policy_calendar,
        small_covariates,
        small_sim.donor_pool_states,
        category="Vapor",
        n_bootstrap=15,
        n_mc=300,
        true_no_policy=small_sim.full_ground_truth_no_policy,
    )
    assert len(results) == len(small_sim.policy_calendar.treated_states)
    summary = summarize_loso(results)
    assert (summary["wape"] >= 0).all()
    assert (summary["coverage_80"] >= 0).all() and (summary["coverage_80"] <= 1).all()

    metrics = loso_aggregate_metrics(summary)
    assert "mean_wape" in metrics
    assert "sign_accuracy" in metrics  # ground truth was supplied


def test_placebo_in_time_is_fast_and_returns_expected_columns(small_sim):
    placebo_states = small_sim.donor_pool_states[:3]
    df = run_placebo_in_time(
        small_sim.panel,
        small_sim.donor_pool_states,
        placebo_states,
        category="Vapor",
        n_bootstrap=50,
    )
    assert len(df) == len(placebo_states)
    for col in ["mean_post_period_att", "ci_low_90pct", "ci_high_90pct", "false_positive_at_90pct"]:
        assert col in df.columns


def test_placebo_rejects_states_not_in_donor_pool(small_sim):
    with pytest.raises(ValueError):
        run_placebo_in_time(
            small_sim.panel,
            small_sim.donor_pool_states,
            [small_sim.target_state],  # not a donor-pool state
            category="Vapor",
            n_bootstrap=10,
        )
