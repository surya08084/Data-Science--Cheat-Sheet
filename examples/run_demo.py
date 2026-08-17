"""End-to-end demo of the vapor-policy-impact framework on simulated data.

Runs the full pipeline described in Vapor-Policy-Impact-Forecasting-Methodology.md
for a single "new" state (Z_NEW) that is about to implement a vapor policy,
using 9 historical treated states + 15 never-treated control states as the
training panel. Produces:

  * a 13-week Policy vs. No-Policy business table per category + total market
  * event-study and fan-chart visualizations
  * cross-category substitution reconciliation
  * Altria-vs-competitor decomposition for Vapor
  * a leave-one-state-out validation run (Vapor only, to keep runtime reasonable)
  * a placebo-in-time false-positive check

Runtime: roughly 10-15 minutes on a single core with the default settings
below (dominated by the quantile-GBM baseline forecaster and LOSO's 9 folds).
Lower `N_BOOTSTRAP_MAIN` / `N_BOOTSTRAP_VALIDATION` for a faster, noisier run.

Usage: `python examples/run_demo.py` from the repo root, with the project's
venv active (see requirements.txt).
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from vapor_policy_impact.config import CATEGORIES
from vapor_policy_impact.data.simulate import simulate_panel
from vapor_policy_impact.decomposition.manufacturer import decompose_manufacturer
from vapor_policy_impact.pipeline import prepare_covariates, run_category_pipeline
from vapor_policy_impact.reporting.business_output import build_executive_table
from vapor_policy_impact.reporting.visuals import (
    plot_event_study,
    plot_scenario_fan_chart,
    plot_share_shift,
    plot_substitution_waterfall,
)
from vapor_policy_impact.substitution.cross_category import category_residual_correlation, reconcile_categories
from vapor_policy_impact.features.engineering import build_aggregated_feature_series
from vapor_policy_impact.validation.loso import (
    loso_aggregate_metrics,
    run_loso,
    run_placebo_in_time,
    summarize_loso,
)

N_BOOTSTRAP_MAIN = 150
N_BOOTSTRAP_VALIDATION = 60
N_MC = 3000
OUTPUT_DIR = Path(__file__).parent / "output"


def _step(msg: str) -> float:
    print(f"\n=== {msg} ===")
    return time.time()


def _done(t0: float) -> None:
    print(f"    ({time.time() - t0:.1f}s)")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    t0 = _step("Simulating 9 treated + 15 control state panel (4 categories x 2 manufacturer groups)")
    sim = simulate_panel()
    covariates = prepare_covariates(sim.state_covariates)
    print(f"    panel rows: {len(sim.panel):,} | target state: {sim.target_state} "
          f"| target policy week: {sim.target_state_effective_week}")
    _done(t0)

    # ---- 1. Per-category pipeline (category totals, Altria+competitor combined) ----
    t0 = _step("Running the causal + forecasting pipeline for each category (total market)")
    category_results = {}
    for cat in CATEGORIES:
        r = run_category_pipeline(
            sim.panel, sim.policy_calendar, covariates,
            target_state=sim.target_state, target_effective_week=sim.target_state_effective_week,
            donor_states=sim.donor_pool_states, category=cat,
            n_bootstrap=N_BOOTSTRAP_MAIN, n_mc=N_MC,
        )
        category_results[cat] = r
        print(f"    {cat:12s} transport_scale={r.transport_scale:+.3f}  "
              f"cum %impact mean={r.scenario.cumulative['pct_impact']['mean']:+.1%}")
    _done(t0)

    exec_table = build_executive_table({c: r.scenario for c, r in category_results.items()})
    exec_table.to_csv(OUTPUT_DIR / "executive_table_total.csv", index=False)
    print("\nExecutive table (Total Market view):")
    print(exec_table.to_string(index=False))

    fig = plot_event_study(category_results["Vapor"].event_study, title="Vapor event study (pooled, 9 historical states)")
    fig.savefig(OUTPUT_DIR / "event_study_vapor.png", dpi=110)
    fig = plot_scenario_fan_chart(category_results["Vapor"].scenario, "Vapor")
    fig.savefig(OUTPUT_DIR / "fan_chart_vapor.png", dpi=110)

    # ---- 2. Cross-category substitution reconciliation ----
    t0 = _step("Cross-category substitution reconciliation")
    recon = reconcile_categories({c: r.scenario for c, r in category_results.items()})
    print(recon.category_share_of_gross_movement.to_string(index=False))
    print("Total market (reconciled from category sum):", recon.total_market["pct_impact"])
    fig = plot_substitution_waterfall(recon.category_share_of_gross_movement)
    fig.savefig(OUTPUT_DIR / "substitution_waterfall.png", dpi=110)

    feat_by_cat = {c: build_aggregated_feature_series(sim.panel, sim.policy_calendar, category=c) for c in CATEGORIES}
    corr = category_residual_correlation(feat_by_cat)
    corr.to_csv(OUTPUT_DIR / "category_residual_correlation.csv")
    _done(t0)

    # ---- 3. Altria vs. competitor decomposition (Vapor) ----
    t0 = _step("Altria vs. competitor decomposition (Vapor)")
    altria = run_category_pipeline(
        sim.panel, sim.policy_calendar, covariates,
        target_state=sim.target_state, target_effective_week=sim.target_state_effective_week,
        donor_states=sim.donor_pool_states, category="Vapor", manufacturer_group="Altria",
        n_bootstrap=N_BOOTSTRAP_MAIN, n_mc=N_MC,
    )
    competitor = run_category_pipeline(
        sim.panel, sim.policy_calendar, covariates,
        target_state=sim.target_state, target_effective_week=sim.target_state_effective_week,
        donor_states=sim.donor_pool_states, category="Vapor", manufacturer_group="Competitor",
        n_bootstrap=N_BOOTSTRAP_MAIN, n_mc=N_MC,
    )
    decomp = decompose_manufacturer("Vapor", altria.scenario, competitor.scenario)
    mfg_table = pd.concat(
        [
            build_executive_table({"Vapor": altria.scenario}).assign(View="Altria"),
            build_executive_table({"Vapor": competitor.scenario}).assign(View="Competitor"),
        ]
    )
    mfg_table = mfg_table[mfg_table["Metric"] == "Vapor"]
    mfg_table.to_csv(OUTPUT_DIR / "executive_table_by_manufacturer.csv", index=False)
    print(mfg_table.to_string(index=False))
    print(f"    reconciled total (Altria+Competitor) %impact mean: "
          f"{decomp.total_from_manufacturer_sum['pct_impact']['mean']:+.1%}"
          f"  (category-level estimate was {category_results['Vapor'].scenario.cumulative['pct_impact']['mean']:+.1%})")
    fig = plot_share_shift(decomp.share_shift, "Vapor")
    fig.savefig(OUTPUT_DIR / "share_shift_vapor.png", dpi=110)
    _done(t0)

    # ---- 4. Leave-one-state-out validation (Vapor only, for runtime) ----
    t0 = _step("Leave-one-state-out validation (Vapor) -- this is the slow part, ~9 folds")
    loso_results = run_loso(
        sim.panel, sim.policy_calendar, covariates, sim.donor_pool_states, category="Vapor",
        n_bootstrap=N_BOOTSTRAP_VALIDATION, n_mc=1500, true_no_policy=sim.full_ground_truth_no_policy,
    )
    loso_summary = summarize_loso(loso_results)
    loso_summary.to_csv(OUTPUT_DIR / "loso_summary.csv", index=False)
    print(loso_summary.to_string(index=False))
    print("Aggregate LOSO metrics:", loso_aggregate_metrics(loso_summary))
    _done(t0)

    # ---- 5. Placebo-in-time false-positive check ----
    t0 = _step("Placebo-in-time check (Layer 1 on genuinely untreated states)")
    placebo = run_placebo_in_time(
        sim.panel, sim.donor_pool_states, sim.donor_pool_states[:6], category="Vapor",
        n_bootstrap=200,
    )
    placebo.to_csv(OUTPUT_DIR / "placebo_in_time.csv", index=False)
    print(placebo.to_string(index=False))
    print(f"    false positive rate at 90% CI: {placebo['false_positive_at_90pct'].mean():.1%} "
          f"(expect roughly ~10% under a well-calibrated null)")
    _done(t0)

    print(f"\nAll tables/figures written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
