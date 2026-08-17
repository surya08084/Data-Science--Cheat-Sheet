"""Layer 2: Synthetic Control / donor-pool weighting (methodology section 4-5).

Fits non-negative, sum-to-one weights over a donor pool of control states so
the weighted combination best reproduces a target state's *pre-period* log-
volume trajectory, then treats the weighted combination as that state's
counterfactual for any week (pre or post). Used both retrospectively (as a
Layer-1 robustness check on historical treated states) and prospectively (as
the donor-weighting template for a brand-new state's no-policy baseline,
section 5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from vapor_policy_impact.features.engineering import aggregate_log_volume


@dataclass
class SyntheticControlResult:
    target_state: str
    weights: dict  # donor_state -> weight
    pre_period_rmse: float
    series: pd.DataFrame  # week, target_log_volume, synthetic_log_volume, gap


def _fit_weights(target_pre: np.ndarray, donors_pre: np.ndarray, regularization: float) -> np.ndarray:
    n_donors = donors_pre.shape[1]
    x0 = np.full(n_donors, 1.0 / n_donors)

    def objective(w: np.ndarray) -> float:
        resid = target_pre - donors_pre @ w
        return float(resid @ resid + regularization * (w @ w))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_donors
    result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500})
    w = np.clip(result.x, 0, None)
    total = w.sum()
    return w / total if total > 0 else np.full(n_donors, 1.0 / n_donors)


def fit_synthetic_control(
    panel: pd.DataFrame,
    target_state: str,
    donor_states: list[str],
    pre_period_weeks: tuple[int, int],
    category: str | None = None,
    manufacturer_group: str | None = None,
    regularization: float = 1e-3,
) -> SyntheticControlResult:
    """`pre_period_weeks` is an inclusive (start, end) week range used to fit weights."""
    series = aggregate_log_volume(panel, category=category, manufacturer_group=manufacturer_group)
    wide = series.pivot(index="week", columns="state", values="log_volume")

    donor_states = [s for s in donor_states if s in wide.columns]
    if target_state not in wide.columns or not donor_states:
        raise ValueError("target_state or donor_states not present in panel for this category/manufacturer cut.")

    pre_start, pre_end = pre_period_weeks
    pre_mask = (wide.index >= pre_start) & (wide.index <= pre_end)
    pre = wide.loc[pre_mask, [target_state] + donor_states].dropna()
    if pre.shape[0] < 4:
        raise ValueError("Not enough overlapping pre-period observations to fit synthetic control.")

    target_pre = pre[target_state].to_numpy()
    donors_pre = pre[donor_states].to_numpy()
    weights = _fit_weights(target_pre, donors_pre, regularization)

    # Use every week where the donor pool has data, not just weeks where the target
    # does -- this is what lets the synthetic series be *projected* into future weeks
    # for a target state whose post-policy data doesn't exist yet (section 5, baseline
    # forecaster exogenous regressor), while still reporting NaN gaps where the target
    # is genuinely unobserved.
    all_weeks = wide[donor_states].dropna(how="all").index
    donors_full = wide[donor_states].reindex(all_weeks)
    synthetic = (donors_full.to_numpy() * weights).sum(axis=1)
    target_full = wide[target_state].reindex(all_weeks).to_numpy()

    out = pd.DataFrame(
        {
            "week": all_weeks,
            "target_log_volume": target_full,
            "synthetic_log_volume": synthetic,
        }
    )
    out["gap"] = out["target_log_volume"] - out["synthetic_log_volume"]

    pre_fit = out[(out["week"] >= pre_start) & (out["week"] <= pre_end)]
    rmse = float(np.sqrt(np.nanmean(pre_fit["gap"] ** 2))) if len(pre_fit) else float("nan")

    return SyntheticControlResult(
        target_state=target_state,
        weights=dict(zip(donor_states, weights.tolist())),
        pre_period_rmse=rmse,
        series=out,
    )
