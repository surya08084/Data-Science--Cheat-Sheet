"""Feature engineering: event-time alignment, transforms, and the core
confounder/momentum features from section 2 and section 9 of the methodology.

Input is the raw state x category x manufacturer x week panel (see
:mod:`vapor_policy_impact.data.simulate` for the schema, or substitute real
scan data with the same columns). Output is the analysis-ready panel that
every causal/forecasting component downstream reads from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vapor_policy_impact.config import PolicyCalendar

_LAGS = (1, 4, 13, 52)
_GROUP_COLS = ["state", "category", "manufacturer_group"]


def _add_event_time(df: pd.DataFrame, policy_calendar: PolicyCalendar) -> pd.DataFrame:
    df = df.copy()
    eff_week = df["state"].map(policy_calendar.treated_states)
    df["treated_flag"] = eff_week.notna() & (df["week"] >= eff_week)
    df["event_week"] = np.where(eff_week.notna(), df["week"] - eff_week, np.nan)
    # a state counts as a valid "not yet treated" control at calendar week t if it is
    # either never treated, or treated but its effective week is still in the future.
    df["not_yet_treated_flag"] = eff_week.isna() | (df["week"] < eff_week)
    return df


def _add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_volume"] = np.log(df["volume"].clip(lower=1e-9))
    df["week_of_year"] = df["week"] % 52
    for k in (1, 2):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * df["week"] / 52.0)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * df["week"] / 52.0)
    return df


def _add_category_share(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_totals = df.groupby(["state", "category", "week"])["volume"].transform("sum")
    df["category_share"] = df["volume"] / cat_totals.replace(0, np.nan)
    return df


def _add_lags_and_momentum(df: pd.DataFrame, group_cols: list[str] = _GROUP_COLS) -> pd.DataFrame:
    df = df.sort_values(group_cols + ["week"]).copy()
    grouped = df.groupby(group_cols, group_keys=False)["log_volume"]
    for lag in _LAGS:
        df[f"volume_lag_{lag}"] = grouped.shift(lag)
    df["rolling_mean_13"] = grouped.transform(lambda s: s.shift(1).rolling(13, min_periods=4).mean())
    df["rolling_std_13"] = grouped.transform(lambda s: s.shift(1).rolling(13, min_periods=4).std())
    df["yoy_growth"] = df["log_volume"] - df["volume_lag_52"]

    last4 = grouped.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    prior4 = grouped.transform(lambda s: s.shift(5).rolling(4, min_periods=2).mean())
    df["momentum_4v4"] = last4 - prior4
    return df


def build_feature_panel(panel: pd.DataFrame, policy_calendar: PolicyCalendar) -> pd.DataFrame:
    """Build the analysis-ready panel: transforms + event-time + confounder/momentum features.

    Idempotent and safe to call on any subset of states (e.g. a LOSO training fold),
    since all engineered features are computed within-group.
    """
    df = _add_transforms(panel)
    df = _add_event_time(df, policy_calendar)
    df = _add_category_share(df)
    df = _add_lags_and_momentum(df)
    return df.reset_index(drop=True)


def build_aggregated_feature_series(
    panel: pd.DataFrame,
    policy_calendar: PolicyCalendar,
    category: str | None = None,
    manufacturer_group: str | None = None,
) -> pd.DataFrame:
    """Like :func:`build_feature_panel`, but collapsed to exactly one row per
    (state, week) -- summing volume over whichever of category/manufacturer_group
    is left unspecified -- before computing lags/momentum/seasonality.

    This is the required input shape for anything that does time-indexed
    ``.shift()``/rolling-window feature engineering keyed on ``state`` alone
    (the forecasters in :mod:`vapor_policy_impact.forecasting.baseline`):
    feeding those a frame with multiple rows per (state, week) -- e.g. one row
    per manufacturer_group -- silently misaligns every lag/shift, since
    ``.groupby("state").shift(k)`` shifts by *row position*, not by calendar
    week.
    """
    df = panel
    if category is not None:
        df = df[df["category"] == category]
    if manufacturer_group is not None:
        df = df[df["manufacturer_group"] == manufacturer_group]

    agg_kwargs = {"volume": ("volume", "sum")}
    for optional_col in ("price", "promo_depth", "distribution_acv"):
        if optional_col in df.columns:
            agg_kwargs[optional_col] = (optional_col, "mean")
    grouped = df.groupby(["state", "week"], as_index=False).agg(**agg_kwargs)

    grouped = _add_transforms(grouped)
    grouped = _add_event_time(grouped, policy_calendar)
    grouped = _add_lags_and_momentum(grouped, group_cols=["state"])
    return grouped.sort_values(["state", "week"]).reset_index(drop=True)


def aggregate_log_volume(
    panel: pd.DataFrame,
    category: str | None = None,
    manufacturer_group: str | None = None,
) -> pd.DataFrame:
    """Collapse the panel to one `log_volume` series per (state, week), summing
    over whichever of category/manufacturer_group is left unspecified.

    ``category=None`` gives the total-market series; ``manufacturer_group=None``
    gives the Altria+competitor combined series -- used by Layer 1 to run the
    same event-study machinery at category, manufacturer, or total-market grain.
    """
    df = panel
    if category is not None:
        df = df[df["category"] == category]
    if manufacturer_group is not None:
        df = df[df["manufacturer_group"] == manufacturer_group]
    grouped = df.groupby(["state", "week"], as_index=False)["volume"].sum()
    grouped["log_volume"] = np.log(grouped["volume"].clip(lower=1e-9))
    return grouped


def pre_period_momentum(
    feature_panel: pd.DataFrame,
    state: str,
    as_of_week: int,
    window: int = 13,
) -> pd.DataFrame:
    """Section 9 "pre-policy momentum" covariate: growth rate & volatility in the
    `window` weeks immediately before `as_of_week`, by category x manufacturer_group.
    """
    mask = (
        (feature_panel["state"] == state)
        & (feature_panel["week"] < as_of_week)
        & (feature_panel["week"] >= as_of_week - window)
    )
    sub = feature_panel.loc[mask]
    out = (
        sub.groupby(["category", "manufacturer_group"])["log_volume"]
        .agg(pre_policy_growth=lambda s: (s.iloc[-1] - s.iloc[0]) / max(len(s) - 1, 1) if len(s) > 1 else np.nan,
             pre_policy_volatility="std")
        .reset_index()
    )
    return out
