"""Shared schema constants and small dataclasses used across the framework.

The analysis grain throughout is ``state x category x manufacturer_group x week``,
as specified in section 2.1 of the methodology doc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CATEGORIES: list[str] = ["Vapor", "Cigarettes", "TDN", "MST"]

MANUFACTURER_GROUPS: list[str] = ["Altria", "Competitor"]

#: Columns that uniquely identify a row in the analysis-ready panel.
PANEL_KEY_COLUMNS: list[str] = ["state", "category", "manufacturer_group", "week"]

#: Core outcome/feature columns produced by feature engineering (see features/engineering.py)
CORE_FEATURE_COLUMNS: list[str] = [
    "volume",
    "log_volume",
    "price",
    "promo_depth",
    "distribution_acv",
    "category_share",
]


@dataclass(frozen=True)
class PanelSchema:
    """Documents the expected columns of the raw and analysis-ready panels.

    ``raw_columns`` is what a real retail-scan extract is expected to provide
    (section "Available Data" of the methodology doc); ``derived_columns`` is
    what :mod:`vapor_policy_impact.features.engineering` adds on top.
    """

    raw_columns: tuple[str, ...] = (
        "state",
        "week",
        "category",
        "manufacturer",
        "manufacturer_group",
        "sku",
        "brand",
        "volume",
        "sales",
        "price",
        "promo_depth",
        "distribution_acv",
    )
    derived_columns: tuple[str, ...] = (
        "log_volume",
        "event_week",
        "treated_flag",
        "not_yet_treated_flag",
        "category_share",
        "volume_lag_1",
        "volume_lag_4",
        "volume_lag_13",
        "volume_lag_52",
        "rolling_mean_13",
        "rolling_std_13",
        "yoy_growth",
        "momentum_4v4",
        "week_of_year",
        "fourier_sin_1",
        "fourier_cos_1",
        "fourier_sin_2",
        "fourier_cos_2",
    )


@dataclass
class PolicyCalendar:
    """State -> policy effective week (and metadata) lookup.

    ``treated_states`` maps state -> effective week (as an integer week index,
    consistent with the ``week`` column of the panel). States not present here
    are treated as never-treated / potential donor-pool controls unless they
    appear in ``not_yet_treated_as_of``.
    """

    treated_states: dict[str, int] = field(default_factory=dict)
    policy_type: dict[str, str] = field(default_factory=dict)
    policy_scope: dict[str, str] = field(default_factory=dict)

    def is_treated(self, state: str) -> bool:
        return state in self.treated_states

    def effective_week(self, state: str) -> int | None:
        return self.treated_states.get(state)
