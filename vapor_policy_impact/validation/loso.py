"""Leave-one-state-out pseudo-policy validation + placebo-in-time tests
(methodology section 10).

:func:`run_loso` implements exactly the framework the doc specifies: hold out
one historical treated state, fit everything on the rest, forecast the held-
out state's 13-week Scenario A as if it were a brand-new state, then compare
against that state's *actual* historical post-policy volume (which we do
have, since it already happened) -- repeated for every treated state.

Because the true no-policy counterfactual for a real historical state is
fundamentally unobservable (that's the entire premise of the problem), the
impact-bias / sign-accuracy / rank-correlation metrics can only be computed
when ground truth is available -- which is only true in the synthetic-data
setting (``true_no_policy`` from
:attr:`vapor_policy_impact.data.simulate.SimulatedData.full_ground_truth_no_policy`).
Against real data, only the forecast-accuracy metrics (WAPE, interval
coverage) are computable from LOSO; the impact metrics would instead need
:func:`run_placebo_in_time` (false-positive rate on genuinely untreated
states) as the practical substitute for "how good is our impact estimate."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vapor_policy_impact.causal.event_study import StaggeredEventStudy
from vapor_policy_impact.config import PolicyCalendar
from vapor_policy_impact.features.engineering import aggregate_log_volume
from vapor_policy_impact.pipeline import run_category_pipeline


@dataclass
class LOSOFoldResult:
    held_out_state: str
    category: str
    weeks: np.ndarray
    scenario_a_mean: np.ndarray
    scenario_a_q10: np.ndarray
    scenario_a_q90: np.ndarray
    scenario_b_mean: np.ndarray
    actual_volume: np.ndarray
    wape: float
    coverage_80: float
    predicted_cum_pct_impact: float
    true_cum_pct_impact: float | None = None
    impact_bias: float | None = None
    sign_correct: bool | None = None


def run_loso(
    panel: pd.DataFrame,
    policy_calendar: PolicyCalendar,
    covariates: pd.DataFrame,
    donor_pool_states: list[str],
    category: str | None,
    manufacturer_group: str | None = None,
    horizon: int = 13,
    n_bootstrap: int = 100,
    n_mc: int = 2000,
    true_no_policy: pd.DataFrame | None = None,
    random_state: int = 0,
) -> list[LOSOFoldResult]:
    treated_states = list(policy_calendar.treated_states.keys())
    # evaluation uses the REAL, untruncated actuals; the pipeline itself must only ever
    # see the held-out state's data as it would look with no post-policy history yet.
    actual_series = aggregate_log_volume(panel, category=category, manufacturer_group=manufacturer_group)

    results = []
    for held_out in treated_states:
        held_out_week = policy_calendar.treated_states[held_out]
        donors = [s for s in donor_pool_states if s != held_out]
        truncated_panel = _truncate_panel(panel, held_out, held_out_week)

        pipe = run_category_pipeline(
            truncated_panel,
            policy_calendar,
            covariates,
            target_state=held_out,
            target_effective_week=held_out_week,
            donor_states=donors,
            category=category,
            manufacturer_group=manufacturer_group,
            exclude_states=[held_out],  # the whole point: Layer 1/3 never see this state
            horizon=horizon,
            n_bootstrap=n_bootstrap,
            n_mc=n_mc,
            random_state=random_state,
        )

        weeks = pipe.scenario.weeks
        actual = actual_series[actual_series["state"] == held_out].set_index("week")["log_volume"].reindex(weeks)
        actual_vol = np.exp(actual.to_numpy())

        pred_a_mean = pipe.scenario.weekly["scenario_a_mean"].to_numpy()
        wape = float(np.nansum(np.abs(actual_vol - pred_a_mean)) / np.nansum(np.abs(actual_vol)))
        q10 = pipe.scenario.weekly["scenario_a_q10"].to_numpy()
        q90 = pipe.scenario.weekly["scenario_a_q90"].to_numpy()
        coverage = float(np.nanmean((actual_vol >= q10) & (actual_vol <= q90)))
        predicted_cum_pct_impact = pipe.scenario.cumulative["pct_impact"]["mean"]

        true_cum_pct_impact = impact_bias = sign_correct = None
        if true_no_policy is not None:
            tnp = true_no_policy[true_no_policy["state"] == held_out]
            if manufacturer_group is not None:
                tnp = tnp[tnp["manufacturer_group"] == manufacturer_group]
            if category is not None:
                tnp = tnp[tnp["category"] == category]
            true_b = tnp.groupby("week")["volume_no_policy"].sum().reindex(weeks).to_numpy()
            true_cum_b = float(np.nansum(true_b))
            true_cum_a = float(np.nansum(actual_vol))  # the historical actual IS the true with-policy outcome
            true_cum_pct_impact = true_cum_a / true_cum_b - 1.0
            impact_bias = predicted_cum_pct_impact - true_cum_pct_impact
            sign_correct = bool(np.sign(predicted_cum_pct_impact) == np.sign(true_cum_pct_impact))

        results.append(
            LOSOFoldResult(
                held_out_state=held_out,
                category=category or "TotalMarket",
                weeks=weeks,
                scenario_a_mean=pred_a_mean,
                scenario_a_q10=q10,
                scenario_a_q90=q90,
                scenario_b_mean=pipe.scenario.weekly["scenario_b_mean"].to_numpy(),
                actual_volume=actual_vol,
                wape=wape,
                coverage_80=coverage,
                predicted_cum_pct_impact=predicted_cum_pct_impact,
                true_cum_pct_impact=true_cum_pct_impact,
                impact_bias=impact_bias,
                sign_correct=sign_correct,
            )
        )
    return results


def summarize_loso(results: list[LOSOFoldResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state": r.held_out_state,
                "category": r.category,
                "wape": r.wape,
                "coverage_80": r.coverage_80,
                "predicted_cum_pct_impact": r.predicted_cum_pct_impact,
                "true_cum_pct_impact": r.true_cum_pct_impact,
                "impact_bias": r.impact_bias,
                "sign_correct": r.sign_correct,
            }
            for r in results
        ]
    )


def loso_aggregate_metrics(summary: pd.DataFrame) -> dict:
    out = {
        "mean_wape": float(summary["wape"].mean()),
        "mean_coverage_80": float(summary["coverage_80"].mean()),
        "n_folds": int(len(summary)),
    }
    if "impact_bias" in summary and summary["impact_bias"].notna().any():
        valid = summary.dropna(subset=["impact_bias", "true_cum_pct_impact"])
        out["mean_impact_bias"] = float(valid["impact_bias"].mean())
        out["sign_accuracy"] = float(valid["sign_correct"].mean())
        if len(valid) > 2:
            out["rank_correlation"] = float(
                valid["predicted_cum_pct_impact"].corr(valid["true_cum_pct_impact"], method="spearman")
            )
    return out


def _truncate_panel(panel: pd.DataFrame, state: str, cutoff_week: int) -> pd.DataFrame:
    return panel[~((panel["state"] == state) & (panel["week"] >= cutoff_week))].copy()


def run_placebo_in_time(
    panel: pd.DataFrame,
    donor_pool_states: list[str],
    placebo_states: list[str],
    category: str | None,
    manufacturer_group: str | None = None,
    fake_effective_week: int | None = None,
    min_event_week: int = -12,
    max_event_week: int = 12,
    n_bootstrap: int = 200,
    random_state: int = 0,
) -> pd.DataFrame:
    """Assign a fake policy date to a genuinely never-treated state and re-run *Layer
    1's own estimator* (not the full new-state forecasting pipeline) with that state
    as an additional treated cohort, using only the other never-treated donor-pool
    states as controls. The estimated post-period ATT(e) should be statistically
    indistinguishable from zero (section 10.2) -- this checks whether the DiD/parallel-
    trends identification itself is sound, independent of any real treated states.

    This is deliberately *not* built on the full Scenario A/B forecasting workflow
    (:func:`vapor_policy_impact.pipeline.run_category_pipeline`): that workflow's whole
    job is to assume a policy is (about to be) applied and project an effect forward,
    so running it on an untreated state would mechanically apply the real historical
    effect curve regardless of whether the target was actually treated, and would not
    test what a placebo check is supposed to test. Layer 1 in isolation is the right
    unit -- it's the piece that is supposed to output ~0 when nothing really happened.
    """
    fweek = fake_effective_week or int(panel["week"].max() - max_event_week - 10)
    rows = []
    for fake_state in placebo_states:
        if fake_state not in donor_pool_states:
            raise ValueError(f"{fake_state} must be one of the genuinely never-treated donor_pool_states.")
        control_states = [s for s in donor_pool_states if s != fake_state]
        placebo_panel = panel[panel["state"].isin(control_states + [fake_state])]
        fake_calendar = PolicyCalendar(treated_states={fake_state: fweek})

        est = StaggeredEventStudy(
            min_event_week=min_event_week, max_event_week=max_event_week, n_bootstrap=n_bootstrap, random_state=random_state
        )
        res = est.fit(
            placebo_panel, fake_calendar, category=category, manufacturer_group=manufacturer_group, label=f"placebo:{fake_state}"
        )
        post = res.att_by_event_time[res.att_by_event_time["event_week"] >= 0]
        mean_post_att = float(post["att"].mean())
        ci_low = float((post["att"] - 1.645 * post["se"]).mean())
        ci_high = float((post["att"] + 1.645 * post["se"]).mean())
        rows.append(
            {
                "state": fake_state,
                "fake_effective_week": fweek,
                "mean_post_period_att": mean_post_att,
                "ci_low_90pct": ci_low,
                "ci_high_90pct": ci_high,
                "false_positive_at_90pct": bool(ci_low > 0 or ci_high < 0),
            }
        )
    return pd.DataFrame(rows)
