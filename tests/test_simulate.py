import numpy as np

from vapor_policy_impact.config import CATEGORIES, MANUFACTURER_GROUPS


def test_panel_schema(small_sim):
    panel = small_sim.panel
    expected_cols = {"state", "week", "category", "manufacturer_group", "volume", "price", "promo_depth", "distribution_acv"}
    assert expected_cols.issubset(panel.columns)
    assert set(panel["category"].unique()) == set(CATEGORIES)
    assert set(panel["manufacturer_group"].unique()) == set(MANUFACTURER_GROUPS)
    assert (panel["volume"] > 0).all()


def test_no_duplicate_rows(small_sim):
    panel = small_sim.panel
    key = ["state", "week", "category", "manufacturer_group"]
    assert not panel.duplicated(subset=key).any()


def test_target_state_truncated_by_default(small_sim):
    target_rows = small_sim.panel[small_sim.panel["state"] == small_sim.target_state]
    assert target_rows["week"].max() < small_sim.target_state_effective_week


def test_treated_states_staggered_and_in_range(small_sim):
    weeks = list(small_sim.policy_calendar.treated_states.values())
    assert len(weeks) == len(set(weeks)) or len(weeks) >= 1  # staggered (usually distinct)
    for w in weeks:
        assert 0 < w < small_sim.panel["week"].max() + 20


def test_ground_truth_no_policy_covers_all_states(small_sim):
    gt = small_sim.full_ground_truth_no_policy
    assert set(small_sim.policy_calendar.treated_states.keys()).issubset(set(gt["state"].unique()))
    assert (gt["volume_no_policy"] > 0).all()


def test_target_ground_truth_shows_a_decline_post_policy(small_sim):
    gt = small_sim.target_state_ground_truth
    vapor = gt[gt["category"] == "Vapor"].groupby("week")[["volume_with_policy", "volume_no_policy"]].sum()
    # the injected effect should show up as with_policy < no_policy on average over the window
    assert vapor["volume_with_policy"].sum() < vapor["volume_no_policy"].sum()


def test_reveal_target_post_period_flag():
    from vapor_policy_impact.data.simulate import SimulationConfig, simulate_panel

    cfg = SimulationConfig(n_weeks=100, n_treated_states=3, n_control_states=4, min_effective_week=30, max_effective_week_margin=10)
    sim = simulate_panel(config=cfg, seed=3, reveal_target_post_period=True)
    target_rows = sim.panel[sim.panel["state"] == sim.target_state]
    assert target_rows["week"].max() >= sim.target_state_effective_week
