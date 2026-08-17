from vapor_policy_impact.causal.event_study import StaggeredEventStudy


def test_event_study_pre_trends_near_zero(small_sim):
    est = StaggeredEventStudy(min_event_week=-10, max_event_week=10, n_bootstrap=40, random_state=1)
    res = est.fit(small_sim.panel, small_sim.policy_calendar, category="Vapor")
    pre = res.att_by_event_time[res.att_by_event_time["event_week"] < 0]
    # noisy small-sample estimate, but should not be systematically large
    assert abs(pre["att"].mean()) < 0.25


def test_event_study_post_period_shows_negative_vapor_effect(small_sim):
    est = StaggeredEventStudy(min_event_week=-10, max_event_week=10, n_bootstrap=40, random_state=1)
    res = est.fit(small_sim.panel, small_sim.policy_calendar, category="Vapor")
    post = res.att_by_event_time[res.att_by_event_time["event_week"] >= 4]
    assert post["att"].mean() < 0


def test_bootstrap_curves_shape_matches_event_time_range(small_sim):
    est = StaggeredEventStudy(min_event_week=-6, max_event_week=6, n_bootstrap=25, random_state=2)
    res = est.fit(small_sim.panel, small_sim.policy_calendar, category="Vapor")
    assert res.bootstrap_curves.shape[0] <= 25
    assert set(res.bootstrap_curves.columns).issubset(set(range(-6, 7)))


def test_exclude_states_removes_cohort(small_sim):
    treated = list(small_sim.policy_calendar.treated_states.keys())
    held_out = treated[0]
    est = StaggeredEventStudy(n_bootstrap=10, random_state=0)
    res = est.fit(small_sim.panel, small_sim.policy_calendar, category="Vapor", exclude_states=[held_out])
    assert held_out not in set(res.att_by_group_time["cohort_state"])
