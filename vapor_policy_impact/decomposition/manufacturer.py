"""Altria vs. competitor decomposition (methodology section 7).

Runs on two already-fit :class:`~vapor_policy_impact.scenario.combiner.ScenarioResult`
objects (one per manufacturer_group, produced by
:func:`vapor_policy_impact.pipeline.run_category_pipeline` with
``manufacturer_group="Altria"`` / ``"Competitor"``) for the same category and
target state. Reconciles their sum against the category-level total (QA
check, mirrors section 6's accounting identity one level down), and computes
the market-share-shift metrics from section 7.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vapor_policy_impact.scenario.combiner import ScenarioResult, summarize_draws


@dataclass
class ManufacturerDecomposition:
    category: str
    altria: ScenarioResult
    competitor: ScenarioResult
    total_from_manufacturer_sum: dict  # scenario_a/b/impact/pct_impact, summed from Altria+Competitor draws
    share_shift: pd.DataFrame  # week, altria_share_{no_policy,policy}_mean, delta_share_{mean,q10,q90}


def decompose_manufacturer(
    category: str,
    altria: ScenarioResult,
    competitor: ScenarioResult,
) -> ManufacturerDecomposition:
    n_mc = min(altria.draws["scenario_a"].shape[0], competitor.draws["scenario_a"].shape[0])
    a_a, a_b = altria.draws["scenario_a"][:n_mc], altria.draws["scenario_b"][:n_mc]
    c_a, c_b = competitor.draws["scenario_a"][:n_mc], competitor.draws["scenario_b"][:n_mc]

    total_a = a_a + c_a
    total_b = a_b + c_b
    total_from_manufacturer_sum = {
        "scenario_a": summarize_draws(total_a.sum(axis=1)),
        "scenario_b": summarize_draws(total_b.sum(axis=1)),
        "impact": summarize_draws((total_a - total_b).sum(axis=1)),
        "pct_impact": summarize_draws(total_a.sum(axis=1) / total_b.sum(axis=1) - 1.0),
    }

    share_no_policy = a_b / total_b  # n_mc x horizon
    share_policy = a_a / total_a
    delta_share = share_policy - share_no_policy

    weeks = altria.weeks
    rows = []
    for h in range(len(weeks)):
        rows.append(
            {
                "week": weeks[h],
                "altria_share_no_policy_mean": float(share_no_policy[:, h].mean()),
                "altria_share_policy_mean": float(share_policy[:, h].mean()),
                "delta_share_mean": float(delta_share[:, h].mean()),
                "delta_share_q10": float(np.quantile(delta_share[:, h], 0.10)),
                "delta_share_q90": float(np.quantile(delta_share[:, h], 0.90)),
            }
        )
    share_shift = pd.DataFrame(rows)

    return ManufacturerDecomposition(
        category=category,
        altria=altria,
        competitor=competitor,
        total_from_manufacturer_sum=total_from_manufacturer_sum,
        share_shift=share_shift,
    )
