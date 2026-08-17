"""End-to-end orchestration (methodology section 8): wires the three causal
layers, the baseline forecaster, and the scenario combiner into a single call
per (category, manufacturer_group) cut for a target state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vapor_policy_impact.causal.event_study import EventStudyResult, StaggeredEventStudy
from vapor_policy_impact.causal.heterogeneity import EffectTransportModel
from vapor_policy_impact.causal.synthetic_control import fit_synthetic_control
from vapor_policy_impact.config import PolicyCalendar
from vapor_policy_impact.features.engineering import aggregate_log_volume, build_aggregated_feature_series
from vapor_policy_impact.forecasting.baseline import (
    BaselineForecastResult,
    BSTSBaselineForecaster,
    QuantileGBMForecaster,
    ensemble_baseline_forecast,
)
from vapor_policy_impact.scenario.combiner import ScenarioResult, combine_scenarios

DEFAULT_COVARIATE_COLS = [
    "population_m",
    "urbanization_pct",
    "median_income_k",
    "border_state",
    "retail_density",
    "baseline_vapor_share",
]


def prepare_covariates(state_covariates: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Numeric-encode the state covariate table for Layer 3 (state-indexed)."""
    columns = columns or DEFAULT_COVARIATE_COLS
    cov = state_covariates.set_index("state").copy()
    if "border_state" in cov.columns:
        cov["border_state"] = cov["border_state"].astype(float)
    return cov[columns]


@dataclass
class CategoryPipelineResult:
    category: str
    manufacturer_group: str | None
    event_study: EventStudyResult
    transport_scale: float
    baseline: BaselineForecastResult
    scenario: ScenarioResult


def run_category_pipeline(
    panel: pd.DataFrame,
    policy_calendar: PolicyCalendar,
    covariates: pd.DataFrame,
    target_state: str,
    target_effective_week: int,
    donor_states: list[str],
    category: str | None,
    manufacturer_group: str | None = None,
    exclude_states: list[str] | None = None,
    pre_period_lookback: int = 120,
    horizon: int = 13,
    n_bootstrap: int = 200,
    n_mc: int = 4000,
    baseline_weights: tuple[float, float] = (0.5, 0.5),
    random_state: int = 0,
) -> CategoryPipelineResult:
    label = f"{category or 'TotalMarket'}/{manufacturer_group or 'All'}"

    # ---- Layer 1: pooled effect curve from historical treated states -------------
    event_study = StaggeredEventStudy(n_bootstrap=n_bootstrap, random_state=random_state).fit(
        panel,
        policy_calendar,
        category=category,
        manufacturer_group=manufacturer_group,
        exclude_states=exclude_states,
        label=label,
    )

    # ---- Layer 3: transport the effect to the target state's covariates ----------
    gt = event_study.att_by_group_time
    cohort_effects = gt[(gt["event_week"] >= 0) & (gt["event_week"] <= 12)].groupby("cohort_state")["att"].mean()
    transport = EffectTransportModel().fit(cohort_effects, covariates)
    pooled_curve = event_study.att_by_event_time.set_index("event_week")["att"]
    scale = transport.transport_scale(pooled_curve, covariates.loc[target_state])

    horizon_cols = [e for e in range(horizon) if e in event_study.bootstrap_curves.columns]
    effect_draws = event_study.bootstrap_curves[horizon_cols].to_numpy() * scale
    if effect_draws.shape[1] < horizon:
        pad = np.repeat(effect_draws[:, -1:], horizon - effect_draws.shape[1], axis=1)
        effect_draws = np.concatenate([effect_draws, pad], axis=1)

    # ---- Baseline forecaster (Scenario B): synthetic-control-corrected BSTS + GBM
    pre_start = max(0, target_effective_week - pre_period_lookback)
    sc = fit_synthetic_control(
        panel,
        target_state=target_state,
        donor_states=donor_states,
        pre_period_weeks=(pre_start, target_effective_week - 1),
        category=category,
        manufacturer_group=manufacturer_group,
    )
    target_series = aggregate_log_volume(panel, category=category, manufacturer_group=manufacturer_group)
    target_series = target_series[target_series["state"] == target_state].set_index("week")["log_volume"]
    synth_series = sc.series.set_index("week")["synthetic_log_volume"]
    bsts = BSTSBaselineForecaster(n_draws=max(300, n_mc // 4), random_state=random_state).fit_and_forecast(
        target_series, synth_series, horizon=horizon
    )

    agg_features = build_aggregated_feature_series(
        panel, policy_calendar, category=category, manufacturer_group=manufacturer_group
    )
    gbm = QuantileGBMForecaster(random_state=random_state).fit_and_forecast(
        agg_features, donor_states=donor_states, target_state=target_state, horizon=horizon
    )
    baseline = ensemble_baseline_forecast([bsts, gbm], weights=list(baseline_weights))

    # ---- Combine into Scenario A / B + impact ------------------------------------
    scenario = combine_scenarios(
        baseline.draws, effect_draws, weeks=baseline.weeks, n_mc=n_mc, random_state=random_state
    )

    return CategoryPipelineResult(
        category=category,
        manufacturer_group=manufacturer_group,
        event_study=event_study,
        transport_scale=scale,
        baseline=baseline,
        scenario=scenario,
    )
