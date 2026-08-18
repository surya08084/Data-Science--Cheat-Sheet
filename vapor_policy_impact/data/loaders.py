"""Real-data loading: the counterpart to :mod:`vapor_policy_impact.data.simulate`
for pointing the framework at an actual retail-scan extract instead of the
synthetic simulator (methodology doc, "Available Data" section).

Everything downstream of this module -- :func:`vapor_policy_impact.features.engineering.build_feature_panel`,
:func:`vapor_policy_impact.pipeline.run_category_pipeline`, and everything built on
top of it -- only cares about ending up with a ``panel`` DataFrame matching
:class:`vapor_policy_impact.config.PanelSchema`'s ``raw_columns`` (grain
``state x category x manufacturer_group x week``, integer ``week``), a
state-indexed ``state_covariates`` DataFrame, and a
:class:`~vapor_policy_impact.config.PolicyCalendar`. This module's only job is
producing those three things from a real extract; nothing downstream changes.

Real scan data almost never matches our internal column names or uses an
integer week index (it has real dates, and whatever column names your
warehouse uses) -- that bridging is what the functions here do.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from vapor_policy_impact.config import PanelSchema, PolicyCalendar

_REQUIRED_RAW_COLUMNS = ["state", "week", "category", "manufacturer_group", "volume"]
_OPTIONAL_RAW_COLUMNS = [c for c in PanelSchema().raw_columns if c not in _REQUIRED_RAW_COLUMNS]

#: The demographic/state-level covariates Layer 3 (heterogeneity/effect-transport)
#: expects -- see `vapor_policy_impact.pipeline.DEFAULT_COVARIATE_COLS`. Only
#: `baseline_vapor_share` is derivable from a scan panel; the rest are external.
_COVARIATE_COLS = [
    "population_m",
    "urbanization_pct",
    "median_income_k",
    "border_state",
    "retail_density",
    "baseline_vapor_share",
]


def to_pandas(df) -> pd.DataFrame:
    """Accept either a Spark DataFrame or a pandas DataFrame, always return pandas.

    Duck-typed on `.toPandas()` rather than importing pyspark, so this module
    (and its tests) work fine outside Databricks/Spark entirely.
    """
    if isinstance(df, pd.DataFrame):
        return df
    if hasattr(df, "toPandas"):
        return df.toPandas()
    raise TypeError(f"Expected a pandas or Spark DataFrame, got {type(df)!r}")


def infer_week_zero_date(dates: pd.Series) -> pd.Timestamp:
    """The reference date that week index 0 corresponds to: the earliest date
    in the data, floored to the preceding Monday for clean week boundaries.
    """
    dates = pd.to_datetime(dates)
    earliest = dates.min()
    return earliest - pd.Timedelta(days=earliest.weekday())


def date_to_week_index(date, week_zero_date: pd.Timestamp) -> int:
    """Convert a single real calendar date to an integer week index relative
    to `week_zero_date` -- the same conversion applied to the panel's `week`
    column, so a policy effective date and the panel line up on one axis.
    """
    return int((pd.Timestamp(date) - week_zero_date).days // 7)


def load_real_panel(
    df,
    column_mapping: dict[str, str],
    week_is_date: bool = True,
    week_zero_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Build the raw analysis panel (`PanelSchema.raw_columns`) from a real extract.

    Parameters
    ----------
    df: a pandas or Spark DataFrame in your own schema.
    column_mapping: ``{our_name: your_column_name}`` for every column of yours
        that maps onto one of our raw columns, e.g.
        ``{"state": "STATE_CD", "week": "WEEK_END_DATE", "category": "CATEGORY_DESC",
        "manufacturer_group": "MFG_GROUP", "volume": "UNIT_VOLUME"}``. Only
        ``state, week, category, manufacturer_group, volume`` are required;
        ``manufacturer, sku, brand, sales, price, promo_depth, distribution_acv``
        are optional (features/pipeline code already tolerates their absence).
    week_is_date: if True (the default), the column mapped to ``week`` holds real
        calendar dates and is converted to an integer week index. If False, it's
        assumed to already be an integer week index (e.g. you've pre-aggregated).
    week_zero_date: the reference date for week 0. If None and `week_is_date` is
        True, it's inferred from the data itself (see `infer_week_zero_date`) --
        pass this explicitly so a panel and a separately-loaded policy calendar
        share the exact same week-zero reference.

    Returns
    -------
    (panel, week_zero_date): `week_zero_date` is `None` when `week_is_date=False`
    (there's no calendar reference to share), otherwise the one actually used --
    pass it to `load_policy_calendar` so dates convert consistently.
    """
    pdf = to_pandas(df)

    missing_source_cols = [our for our in _REQUIRED_RAW_COLUMNS if our not in column_mapping]
    if missing_source_cols:
        raise ValueError(
            f"column_mapping is missing required columns: {missing_source_cols}. "
            f"Required: {_REQUIRED_RAW_COLUMNS}"
        )
    unresolved = [src for src in column_mapping.values() if src not in pdf.columns]
    if unresolved:
        raise ValueError(f"column_mapping points at columns not present in the source data: {unresolved}")

    rename = {src: our for our, src in column_mapping.items()}
    panel = pdf.rename(columns=rename)[list(column_mapping.keys())].copy()

    if week_is_date:
        panel["week"] = pd.to_datetime(panel["week"])
        if week_zero_date is None:
            week_zero_date = infer_week_zero_date(panel["week"])
        panel["week"] = ((panel["week"] - week_zero_date).dt.days // 7).astype(int)
    else:
        panel["week"] = panel["week"].astype(int)
        week_zero_date = None

    for col in _OPTIONAL_RAW_COLUMNS:
        if col not in panel.columns:
            if col in ("manufacturer", "sku", "brand"):
                panel[col] = panel["manufacturer_group"]
            else:
                panel[col] = np.nan

    panel = panel.groupby(
        ["state", "week", "category", "manufacturer", "manufacturer_group", "sku", "brand"], as_index=False
    ).agg(
        volume=("volume", "sum"),
        sales=("sales", "sum"),
        price=("price", "mean"),
        promo_depth=("promo_depth", "mean"),
        distribution_acv=("distribution_acv", "mean"),
    )

    return panel[list(PanelSchema().raw_columns)], week_zero_date


def load_policy_calendar(
    mapping: dict[str, str] | pd.DataFrame,
    week_zero_date: pd.Timestamp,
    state_column: str = "state",
    date_column: str = "effective_date",
) -> PolicyCalendar:
    """Build a `PolicyCalendar` from a ``{state: date_string}`` dict or a small
    DataFrame with a state column and a date column. Dates are converted to
    integer week indices using the *same* `week_zero_date` as the main panel
    (from `load_real_panel`), so event-time alignment lines up correctly.
    """
    if isinstance(mapping, dict):
        items = mapping.items()
    else:
        items = zip(mapping[state_column], mapping[date_column])

    treated_states = {state: date_to_week_index(date, week_zero_date) for state, date in items}
    return PolicyCalendar(treated_states=treated_states)


def build_fallback_state_covariates(panel: pd.DataFrame) -> pd.DataFrame:
    """When no external state-demographics table is available, build the
    minimal covariates table Layer 3 needs: `baseline_vapor_share` computed
    directly from the panel's pre-policy-window volume (the one covariate
    genuinely derivable from scan data alone), with the remaining demographic
    columns defaulted to a neutral 0.0 (post-standardization "average state").

    This lets the pipeline run end-to-end without external data, at the cost
    of Layer 3 losing most of its ability to personalize the transported
    effect to a specific new state's characteristics -- it will fall back
    close to the pooled historical-cohort average. A warning is emitted.
    """
    warnings.warn(
        "No state covariates supplied -- falling back to baseline_vapor_share only "
        "(computed from the panel) with population/urbanization/income/border/retail-density "
        "all defaulted to 0.0. Layer 3's ability to personalize the transported effect to a "
        "new state's specific characteristics will be substantially degraded; supply a real "
        "state-covariates table via load_state_covariates(source=...) when you have one.",
        stacklevel=2,
    )
    vapor = panel[panel["category"] == "Vapor"].groupby(["state", "week"], as_index=False)["volume"].sum()
    totals = panel.groupby(["state", "week"], as_index=False)["volume"].sum().rename(columns={"volume": "total"})
    merged = vapor.merge(totals, on=["state", "week"])
    merged["share"] = merged["volume"] / merged["total"].replace(0, np.nan)
    baseline_share = merged.groupby("state")["share"].mean().rename("baseline_vapor_share")

    states = panel["state"].unique()
    out = pd.DataFrame({"state": states}).set_index("state")
    out = out.join(baseline_share).reset_index()
    out["baseline_vapor_share"] = out["baseline_vapor_share"].fillna(out["baseline_vapor_share"].mean())
    for col in _COVARIATE_COLS:
        if col not in out.columns:
            out[col] = 0.0
    return out[["state"] + _COVARIATE_COLS]


def load_state_covariates(
    source,
    states,
    column_mapping: dict[str, str] | None = None,
    panel_for_fallback: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load state-level covariates from `source` (a pandas/Spark DataFrame),
    mapping columns per `column_mapping` (same `{our_name: your_name}` shape
    as `load_real_panel`, must include "state" plus whichever of
    `population_m, urbanization_pct, median_income_k, border_state,
    retail_density, baseline_vapor_share` you have). If `source` is None,
    falls back to `build_fallback_state_covariates(panel_for_fallback)`.
    """
    if source is None:
        if panel_for_fallback is None:
            raise ValueError("Need either `source` or `panel_for_fallback` to build state covariates.")
        return build_fallback_state_covariates(panel_for_fallback)

    pdf = to_pandas(source)
    column_mapping = column_mapping or {c: c for c in ["state"] + _COVARIATE_COLS if c in pdf.columns}
    if "state" not in column_mapping:
        raise ValueError("column_mapping must include 'state'.")

    rename = {src: our for our, src in column_mapping.items()}
    out = pdf.rename(columns=rename)[list(column_mapping.keys())].copy()
    out = out[out["state"].isin(states)].drop_duplicates(subset="state")

    for col in _COVARIATE_COLS:
        if col not in out.columns:
            out[col] = 0.0
    if "border_state" in out.columns:
        out["border_state"] = out["border_state"].astype(float)
    return out[["state"] + _COVARIATE_COLS]
