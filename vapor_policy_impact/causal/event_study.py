"""Layer 1: staggered-adoption-robust event study / DiD (methodology section 4-5).

A simplified, from-scratch implementation of the Callaway & Sant'Anna (2021)
group-time ATT idea: each treated state is its own "cohort" `g` (its policy
effective week), compared against the pool of *not-yet-treated* states at
every calendar week `t`, using `g-1` as the fixed pre-treatment base period.
Group-time effects ATT(g,t) are then re-indexed to event time `e = t - g` and
averaged across cohorts to give the pooled effect curve ATT(e) the rest of
the pipeline consumes.

This intentionally does *not* implement the full doubly-robust / covariate-
adjusted CS estimator or its closed-form asymptotic variance -- both would
require the `differences-in-differences`/`csdid`-style machinery that isn't
available in this environment. Uncertainty is instead obtained by a
state-level cluster bootstrap (resampling states with replacement and
recomputing the whole aggregate), which is the right unit of resampling
given that the entire small-N-treated-states problem (section 14) lives at
the state level, not the row level.

Implementation note: the core group-time ATT computation is fully vectorized
with numpy (state x week matrices + an eligibility mask) rather than looping
with pandas ``.loc`` scalar access, since the bootstrap needs to repeat it
hundreds of times.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vapor_policy_impact.config import PolicyCalendar
from vapor_policy_impact.features.engineering import aggregate_log_volume


@dataclass
class EventStudyResult:
    att_by_group_time: pd.DataFrame  # cohort_state, g, t, event_week, att
    att_by_event_time: pd.DataFrame  # event_week, att, se, ci_low, ci_high, n_cohorts
    label: str  # e.g. "Vapor / Altria"
    bootstrap_curves: pd.DataFrame | None = None  # replicate x event_week pooled ATT draws,
    # used by the scenario combiner for a joint (correlated-across-weeks) Monte Carlo
    # draw of the effect curve, rather than sampling each event-week independently.


class _PanelMatrix:
    """A state x week log-volume matrix plus fast index lookups, built once and
    reused across the real fit and every bootstrap replicate.
    """

    def __init__(self, series: pd.DataFrame):
        wide = series.pivot(index="state", columns="week", values="log_volume")
        self.states = list(wide.index)
        self.weeks = np.asarray(wide.columns, dtype=float)
        self.state_idx = {s: i for i, s in enumerate(self.states)}
        self.arr = wide.to_numpy(dtype=float)  # n_states x n_weeks, NaN where missing

    def row(self, state: str) -> np.ndarray:
        return self.arr[self.state_idx[state]]


def _group_time_atts_vectorized(
    mat: _PanelMatrix,
    treated_states: list[str],
    effective_week: dict[str, int],
    min_event_week: int,
    max_event_week: int,
) -> pd.DataFrame:
    n_states, n_weeks = mat.arr.shape
    eff_arr = np.array([effective_week.get(s, np.inf) for s in mat.states], dtype=float)
    # eligible[i, j] = state i is a valid "not-yet-treated" control at week j
    eligible = mat.weeks[None, :] < eff_arr[:, None]
    valid = ~np.isnan(mat.arr)

    records = []
    for g_state in treated_states:
        g_row = mat.state_idx[g_state]
        g_week = effective_week[g_state]
        base_week = g_week - 1
        base_pos = np.searchsorted(mat.weeks, base_week)
        if base_pos >= n_weeks or mat.weeks[base_pos] != base_week:
            continue
        g_base_val = mat.arr[g_row, base_pos]
        if np.isnan(g_base_val):
            continue

        elig_mask = eligible.copy()
        elig_mask[g_row, :] = False
        elig_and_valid = elig_mask & valid  # n_states x n_weeks

        ctrl_counts = elig_and_valid.sum(axis=0)  # per week
        safe_arr = np.where(elig_and_valid, mat.arr, 0.0)
        ctrl_sums = safe_arr.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ctrl_means = np.where(ctrl_counts > 0, ctrl_sums / np.maximum(ctrl_counts, 1), np.nan)

        ctrl_base = ctrl_means[base_pos]
        if np.isnan(ctrl_base):
            continue

        g_t_vals = mat.arr[g_row, :]
        event_weeks = mat.weeks - g_week
        keep = (
            (~np.isnan(g_t_vals))
            & (~np.isnan(ctrl_means))
            & (event_weeks >= min_event_week)
            & (event_weeks <= max_event_week)
        )
        atts = (g_t_vals[keep] - g_base_val) - (ctrl_means[keep] - ctrl_base)
        for e, att, n_c in zip(event_weeks[keep], atts, ctrl_counts[keep]):
            records.append({"cohort_state": g_state, "g": g_week, "event_week": int(e), "att": att, "n_controls": int(n_c)})

    return pd.DataFrame.from_records(records)


class StaggeredEventStudy:
    """Fits Layer 1 for one outcome series (a given category / manufacturer_group
    cut, or the total market) and produces the pooled ATT(e) curve.
    """

    def __init__(
        self,
        min_event_week: int = -12,
        max_event_week: int = 16,
        n_bootstrap: int = 300,
        random_state: int = 0,
    ):
        self.min_event_week = min_event_week
        self.max_event_week = max_event_week
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state

    def fit(
        self,
        panel: pd.DataFrame,
        policy_calendar: PolicyCalendar,
        category: str | None = None,
        manufacturer_group: str | None = None,
        exclude_states: list[str] | None = None,
        label: str | None = None,
    ) -> EventStudyResult:
        exclude_states = set(exclude_states or [])
        series = aggregate_log_volume(panel, category=category, manufacturer_group=manufacturer_group)
        series = series[~series["state"].isin(exclude_states)]

        effective_week = {s: w for s, w in policy_calendar.treated_states.items() if s not in exclude_states}
        mat = _PanelMatrix(series)
        treated_states = [s for s in effective_week if s in mat.state_idx]
        if not treated_states:
            raise ValueError("No treated states with policy dates overlap this panel/category cut.")

        gt = _group_time_atts_vectorized(mat, treated_states, effective_week, self.min_event_week, self.max_event_week)
        if gt.empty:
            raise ValueError("No group-time ATT estimates could be computed -- check policy calendar/panel overlap.")

        point = gt.groupby("event_week").agg(att=("att", "mean"), n_cohorts=("cohort_state", "nunique")).reset_index()
        point = point.sort_values("event_week")

        curves = self._bootstrap(mat, treated_states, effective_week)
        se = curves.std(axis=0, ddof=1).rename("se").reset_index().rename(columns={"index": "event_week"})
        merged = point.merge(se, on="event_week", how="left")
        merged["ci_low"] = merged["att"] - 1.645 * merged["se"]
        merged["ci_high"] = merged["att"] + 1.645 * merged["se"]

        return EventStudyResult(
            att_by_group_time=gt,
            att_by_event_time=merged,
            label=label or f"{category or 'TotalMarket'} / {manufacturer_group or 'All'}",
            bootstrap_curves=curves,
        )

    def _bootstrap(
        self,
        mat: _PanelMatrix,
        treated_states: list[str],
        effective_week: dict[str, int],
    ) -> pd.DataFrame:
        """Returns a (replicate x event_week) DataFrame of pooled ATT(e) draws, so
        downstream Monte Carlo (the scenario combiner) can sample whole correlated
        curves rather than treating each event-week's uncertainty as independent.
        """
        rng = np.random.default_rng(self.random_state)
        rows: list[dict] = []

        for rep in range(self.n_bootstrap):
            sampled = rng.choice(mat.states, size=len(mat.states), replace=True)
            boot_states = [f"b{i}" for i in range(len(sampled))]
            boot_arr = np.stack([mat.arr[mat.state_idx[s]] for s in sampled])
            boot_mat = _PanelMatrix.__new__(_PanelMatrix)
            boot_mat.states = boot_states
            boot_mat.weeks = mat.weeks
            boot_mat.state_idx = {s: i for i, s in enumerate(boot_states)}
            boot_mat.arr = boot_arr

            boot_treated = [f"b{i}" for i, s in enumerate(sampled) if s in treated_states]
            boot_eff = {f"b{i}": effective_week[s] for i, s in enumerate(sampled) if s in treated_states}
            if not boot_treated:
                continue

            gt_boot = _group_time_atts_vectorized(
                boot_mat, boot_treated, boot_eff, self.min_event_week, self.max_event_week
            )
            if gt_boot.empty:
                continue
            agg = gt_boot.groupby("event_week")["att"].mean()
            rows.append({"replicate": rep, **agg.to_dict()})

        curves = pd.DataFrame(rows).set_index("replicate")
        curves.columns.name = "event_week"
        return curves
