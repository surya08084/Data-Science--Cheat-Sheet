"""Causal inference + counterfactual forecasting framework for estimating the
impact of a state-level vapor policy before it is implemented in a new state.

Implements the methodology in ``Vapor-Policy-Impact-Forecasting-Methodology.md``:
a three-layer causal stack (staggered-adoption event study, synthetic control,
heterogeneity/transport model) combined with a no-policy baseline forecaster to
produce 13-week Policy vs. No-Policy scenarios, cross-category substitution
estimates, and Altria-vs-competitor decomposition.
"""

from vapor_policy_impact.config import CATEGORIES, MANUFACTURER_GROUPS, PanelSchema

__all__ = ["CATEGORIES", "MANUFACTURER_GROUPS", "PanelSchema"]
