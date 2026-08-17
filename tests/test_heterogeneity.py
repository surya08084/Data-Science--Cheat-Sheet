import pandas as pd

from vapor_policy_impact.causal.event_study import StaggeredEventStudy
from vapor_policy_impact.causal.heterogeneity import EffectTransportModel


def test_transport_model_fit_predict(small_sim, small_covariates):
    est = StaggeredEventStudy(n_bootstrap=20, random_state=0)
    res = est.fit(small_sim.panel, small_sim.policy_calendar, category="Vapor")
    gt = res.att_by_group_time
    cohort_effects = gt[(gt["event_week"] >= 0) & (gt["event_week"] <= 8)].groupby("cohort_state")["att"].mean()

    model = EffectTransportModel().fit(cohort_effects, small_covariates)
    pred = model.predict(small_covariates.loc[small_sim.target_state])
    assert isinstance(pred, float)

    pooled_curve = res.att_by_event_time.set_index("event_week")["att"]
    curve = model.transport_curve(pooled_curve, small_covariates.loc[small_sim.target_state])
    assert isinstance(curve, pd.Series)
    assert len(curve) == len(pooled_curve)


def test_transport_scale_is_finite_and_reasonable(small_sim, small_covariates):
    est = StaggeredEventStudy(n_bootstrap=20, random_state=0)
    res = est.fit(small_sim.panel, small_sim.policy_calendar, category="Vapor")
    gt = res.att_by_group_time
    cohort_effects = gt[(gt["event_week"] >= 0) & (gt["event_week"] <= 8)].groupby("cohort_state")["att"].mean()
    model = EffectTransportModel().fit(cohort_effects, small_covariates)
    pooled_curve = res.att_by_event_time.set_index("event_week")["att"]
    scale = model.transport_scale(pooled_curve, small_covariates.loc[small_sim.target_state])
    assert scale == scale  # not NaN
    assert -5 < scale < 5  # sane magnitude, not a degenerate blow-up
