"""Scenario combiner + Monte Carlo uncertainty propagation (methodology section 5, 12).

Combines the no-policy baseline forecast (Scenario B draws, log scale) with a
set of transported effect-curve draws (Scenario A = B x exp(effect), section
5 Step B/C) to produce Policy vs. No-Policy scenarios, weekly and cumulative
impact, credible intervals, and P(material impact) -- all as empirical Monte
Carlo distributions rather than a single point estimate (section 12).

Baseline-forecast draws and effect-curve draws are resampled independently
and paired at random. This treats the two uncertainty sources (the baseline
forecaster's process/parameter uncertainty, and Layer 1's state-cluster
bootstrap uncertainty) as independent, which is a reasonable simplification
since they come from unrelated estimation procedures over different data
subsets (target state's own history + donor pool, vs. the pooled historical
treated-state panel) -- but it is a simplification, and is documented as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ScenarioResult:
    weeks: np.ndarray
    weekly: pd.DataFrame  # week, scenario_a_*, scenario_b_*, impact_*, pct_impact_*
    cumulative: dict  # scenario_a / scenario_b / impact / pct_impact -> {mean, median, q10, q90}
    p_material_negative: float
    material_threshold_pct: float
    draws: dict = field(default_factory=dict)  # 'scenario_a', 'scenario_b' -> n_mc x horizon volume arrays


def summarize_draws(x: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
    }


def combine_scenarios(
    baseline_log_draws: np.ndarray,
    effect_log_draws: np.ndarray,
    weeks: np.ndarray,
    n_mc: int = 4000,
    material_threshold_pct: float = -0.05,
    random_state: int = 0,
) -> ScenarioResult:
    """
    Parameters
    ----------
    baseline_log_draws: (n_baseline x horizon) Scenario-B draws in log-volume space,
        e.g. ``BaselineForecastResult.draws`` from the ensemble forecaster.
    effect_log_draws: (n_effect x horizon) transported log-multiplicative effect draws
        for event weeks 0..horizon-1, e.g. Layer-1 bootstrap curves scaled by
        ``EffectTransportModel.transport_scale`` for the target state.
    """
    rng = np.random.default_rng(random_state)
    horizon = baseline_log_draws.shape[1]
    assert effect_log_draws.shape[1] == horizon, "baseline and effect draws must cover the same horizon"

    b_idx = rng.integers(0, baseline_log_draws.shape[0], size=n_mc)
    e_idx = rng.integers(0, effect_log_draws.shape[0], size=n_mc)
    baseline_sample = baseline_log_draws[b_idx]
    effect_sample = effect_log_draws[e_idx]

    scenario_b_log = baseline_sample
    scenario_a_log = baseline_sample + effect_sample
    scenario_b = np.exp(scenario_b_log)
    scenario_a = np.exp(scenario_a_log)

    impact = scenario_a - scenario_b
    pct_impact = scenario_a / scenario_b - 1.0

    weekly_rows = []
    for h in range(horizon):
        weekly_rows.append(
            {
                "week": weeks[h],
                "scenario_b_mean": scenario_b[:, h].mean(),
                "scenario_b_q10": np.quantile(scenario_b[:, h], 0.10),
                "scenario_b_q90": np.quantile(scenario_b[:, h], 0.90),
                "scenario_a_mean": scenario_a[:, h].mean(),
                "scenario_a_q10": np.quantile(scenario_a[:, h], 0.10),
                "scenario_a_q90": np.quantile(scenario_a[:, h], 0.90),
                "impact_mean": impact[:, h].mean(),
                "impact_q10": np.quantile(impact[:, h], 0.10),
                "impact_q90": np.quantile(impact[:, h], 0.90),
                "pct_impact_mean": pct_impact[:, h].mean(),
                "pct_impact_q10": np.quantile(pct_impact[:, h], 0.10),
                "pct_impact_q90": np.quantile(pct_impact[:, h], 0.90),
            }
        )
    weekly = pd.DataFrame(weekly_rows)

    cum_a = scenario_a.sum(axis=1)
    cum_b = scenario_b.sum(axis=1)
    cum_impact = cum_a - cum_b
    cum_pct_impact = cum_a / cum_b - 1.0

    cumulative = {
        "scenario_a": summarize_draws(cum_a),
        "scenario_b": summarize_draws(cum_b),
        "impact": summarize_draws(cum_impact),
        "pct_impact": summarize_draws(cum_pct_impact),
    }

    p_material_negative = float(np.mean(cum_pct_impact < material_threshold_pct))

    return ScenarioResult(
        weeks=weeks,
        weekly=weekly,
        cumulative=cumulative,
        p_material_negative=p_material_negative,
        material_threshold_pct=material_threshold_pct,
        draws={"scenario_a": scenario_a, "scenario_b": scenario_b},
    )
