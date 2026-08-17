"""Cross-category substitution (methodology section 6).

Two pieces:

* :func:`reconcile_categories` sums independently-estimated per-category
  Scenario A/B Monte Carlo draws (Vapor, Cigarettes, TDN, MST) into a
  total-market view, and reports each category's share of the gross
  cumulative movement -- the accounting-identity check the doc calls for
  (``Delta Total = Delta Vapor + Delta Cigarettes + Delta TDN + Delta MST``).
* :func:`category_residual_correlation` is a lightweight Seemingly-Unrelated-
  Regression-style diagnostic: each category's log-volume is first detrended
  against shared observable drivers (trend, seasonality, price, promo,
  distribution) via OLS, then the *residuals* are correlated across
  categories at the same state-week. Negative correlation between a
  category's residual and vapor's residual, concentrated in the post-policy
  window, is evidence of substitution beyond what the observable drivers
  explain.

Per-category impact estimates themselves still come from running Layer 1
independently per category (different categories have different baseline
dynamics/elasticities) -- this module only reconciles and cross-checks them,
per the doc's recommendation in section 6 to evaluate the four category
effects *together*, not just estimate them separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

from vapor_policy_impact.scenario.combiner import ScenarioResult, summarize_draws

_DETREND_COLS = ["fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2", "price", "promo_depth", "distribution_acv"]


@dataclass
class ReconciliationResult:
    total_market: dict  # scenario_a / scenario_b / impact / pct_impact -> {mean, median, q10, q90}
    category_share_of_gross_movement: pd.DataFrame  # category, mean_cumulative_impact, share_of_gross_movement


def reconcile_categories(category_results: dict[str, ScenarioResult]) -> ReconciliationResult:
    n_mc = min(r.draws["scenario_a"].shape[0] for r in category_results.values())

    total_a = np.zeros(n_mc)
    total_b = np.zeros(n_mc)
    cat_cum_impact_mean: dict[str, float] = {}
    for cat, res in category_results.items():
        a = res.draws["scenario_a"][:n_mc].sum(axis=1)
        b = res.draws["scenario_b"][:n_mc].sum(axis=1)
        total_a += a
        total_b += b
        cat_cum_impact_mean[cat] = float((a - b).mean())

    total_impact = total_a - total_b
    total_pct_impact = total_a / total_b - 1.0
    total_market = {
        "scenario_a": summarize_draws(total_a),
        "scenario_b": summarize_draws(total_b),
        "impact": summarize_draws(total_impact),
        "pct_impact": summarize_draws(total_pct_impact),
    }

    gross_movement = sum(abs(v) for v in cat_cum_impact_mean.values()) or 1.0
    share_rows = [
        {
            "category": cat,
            "mean_cumulative_impact": val,
            "share_of_gross_movement": abs(val) / gross_movement,
        }
        for cat, val in cat_cum_impact_mean.items()
    ]
    share_df = pd.DataFrame(share_rows).sort_values("mean_cumulative_impact").reset_index(drop=True)

    return ReconciliationResult(total_market=total_market, category_share_of_gross_movement=share_df)


def category_residual_correlation(
    feature_series_by_category: dict[str, pd.DataFrame],
    pre_period_only: bool = False,
) -> pd.DataFrame:
    """`feature_series_by_category`: category -> output of
    :func:`vapor_policy_impact.features.engineering.build_aggregated_feature_series`
    for that category (one row per state-week). Returns the cross-category residual
    correlation matrix.
    """
    resid_series = {}
    for cat, df in feature_series_by_category.items():
        d = df.dropna(subset=["log_volume"] + _DETREND_COLS).copy()
        if pre_period_only:
            d = d[~d["treated_flag"]]
        X = sm.add_constant(d[["week"] + _DETREND_COLS])
        y = d["log_volume"]
        model = sm.OLS(y, X).fit()
        resid = pd.Series(model.resid.to_numpy(), index=pd.MultiIndex.from_frame(d[["state", "week"]]))
        resid_series[cat] = resid

    resid_df = pd.DataFrame(resid_series)
    return resid_df.corr()
