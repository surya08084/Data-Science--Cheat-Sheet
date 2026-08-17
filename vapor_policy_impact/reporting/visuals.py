"""Recommended visualizations (methodology section 13.2)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe; callers can savefig() the returned Figure

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vapor_policy_impact.causal.event_study import EventStudyResult
from vapor_policy_impact.scenario.combiner import ScenarioResult


def plot_event_study(result: EventStudyResult, title: str | None = None) -> plt.Figure:
    df = result.att_by_event_time.sort_values("event_week")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvline(0, color="0.6", linestyle="--", linewidth=1)
    ax.axhline(0, color="0.6", linestyle="-", linewidth=1)
    ax.fill_between(df["event_week"], df["ci_low"], df["ci_high"], alpha=0.25, color="C0", label="90% CI")
    ax.plot(df["event_week"], df["att"], color="C0", marker="o", markersize=3, label="Pooled ATT(e)")
    ax.set_xlabel("Weeks since policy (event time)")
    ax.set_ylabel("Log-multiplicative ATT")
    ax.set_title(title or f"Event study: {result.label}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_scenario_fan_chart(result: ScenarioResult, category: str) -> plt.Figure:
    w = result.weekly
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(w["week"], w["scenario_b_q10"], w["scenario_b_q90"], alpha=0.2, color="C0")
    ax.plot(w["week"], w["scenario_b_mean"], color="C0", label="No Policy (Scenario B)")
    ax.fill_between(w["week"], w["scenario_a_q10"], w["scenario_a_q90"], alpha=0.2, color="C3")
    ax.plot(w["week"], w["scenario_a_mean"], color="C3", label="Policy (Scenario A)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Volume")
    ax.set_title(f"{category}: 13-week Policy vs. No-Policy scenarios")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_substitution_waterfall(category_share: pd.DataFrame) -> plt.Figure:
    """`category_share`: output of
    :func:`vapor_policy_impact.substitution.cross_category.reconcile_categories`'s
    ``category_share_of_gross_movement`` field.
    """
    df = category_share.sort_values("mean_cumulative_impact")
    colors = ["C3" if v < 0 else "C2" for v in df["mean_cumulative_impact"]]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["category"], df["mean_cumulative_impact"], color=colors)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_ylabel("Mean cumulative volume impact")
    ax.set_title("Cross-category substitution: 13-week impact by category")
    fig.tight_layout()
    return fig


def plot_share_shift(share_shift: pd.DataFrame, category: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(share_shift["week"], share_shift["delta_share_q10"], share_shift["delta_share_q90"], alpha=0.2, color="C4")
    ax.plot(share_shift["week"], share_shift["delta_share_mean"], color="C4", marker="o", markersize=3)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xlabel("Week")
    ax.set_ylabel("Altria share (Policy - No Policy)")
    ax.set_title(f"{category}: Altria market-share shift under policy")
    fig.tight_layout()
    return fig
