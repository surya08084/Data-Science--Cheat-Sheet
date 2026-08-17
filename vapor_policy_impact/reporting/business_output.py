"""Executive business output table (methodology section 13.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vapor_policy_impact.scenario.combiner import ScenarioResult


def build_executive_table(scenario_by_category: dict[str, ScenarioResult]) -> pd.DataFrame:
    """One row per category plus a reconciled "Total Market" row, matching the
    section 13.1 table: No Policy | Policy | Incremental Impact | % Impact | 80% CI.

    The Total Market row is computed by summing each category's Monte Carlo draws
    (not just averaging point estimates), so its interval correctly reflects the
    combined uncertainty rather than being a naive sum of per-category intervals.
    """
    rows = []
    n_mc = min(r.draws["scenario_a"].shape[0] for r in scenario_by_category.values())
    total_a_draws = np.zeros(n_mc)
    total_b_draws = np.zeros(n_mc)

    for cat, res in scenario_by_category.items():
        c = res.cumulative
        rows.append(
            {
                "Metric": cat,
                "No Policy": c["scenario_b"]["mean"],
                "Policy": c["scenario_a"]["mean"],
                "Incremental Impact": c["impact"]["mean"],
                "% Impact": c["pct_impact"]["mean"],
                "80% CI (Impact)": f"[{c['impact']['q10']:.2f}, {c['impact']['q90']:.2f}]",
            }
        )
        total_a_draws += res.draws["scenario_a"][:n_mc].sum(axis=1)
        total_b_draws += res.draws["scenario_b"][:n_mc].sum(axis=1)

    total_impact_draws = total_a_draws - total_b_draws
    total_pct_draws = total_a_draws / total_b_draws - 1.0
    rows.append(
        {
            "Metric": "Total Market",
            "No Policy": float(total_b_draws.mean()),
            "Policy": float(total_a_draws.mean()),
            "Incremental Impact": float(total_impact_draws.mean()),
            "% Impact": float(total_pct_draws.mean()),
            "80% CI (Impact)": f"[{np.quantile(total_impact_draws, 0.10):.2f}, {np.quantile(total_impact_draws, 0.90):.2f}]",
        }
    )
    return pd.DataFrame(rows)
