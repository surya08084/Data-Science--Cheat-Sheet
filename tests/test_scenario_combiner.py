import numpy as np

from vapor_policy_impact.scenario.combiner import combine_scenarios


def test_zero_effect_gives_scenario_a_close_to_b():
    rng = np.random.default_rng(0)
    horizon = 5
    baseline = rng.normal(1.0, 0.05, size=(500, horizon))
    zero_effect = np.zeros((300, horizon))
    weeks = np.arange(100, 100 + horizon)

    result = combine_scenarios(baseline, zero_effect, weeks=weeks, n_mc=2000, random_state=0)
    for h in range(horizon):
        row = result.weekly.iloc[h]
        assert abs(row["scenario_a_mean"] - row["scenario_b_mean"]) < 0.01
        assert abs(row["pct_impact_mean"]) < 0.01


def test_negative_effect_produces_negative_impact_and_high_p_material():
    rng = np.random.default_rng(1)
    horizon = 4
    baseline = rng.normal(2.0, 0.02, size=(500, horizon))
    negative_effect = np.full((300, horizon), -0.3) + rng.normal(0, 0.01, size=(300, horizon))
    weeks = np.arange(50, 50 + horizon)

    result = combine_scenarios(baseline, negative_effect, weeks=weeks, n_mc=2000, material_threshold_pct=-0.05, random_state=0)
    assert result.cumulative["impact"]["mean"] < 0
    assert result.cumulative["pct_impact"]["mean"] < -0.1
    assert result.p_material_negative > 0.95


def test_weekly_frame_has_expected_columns_and_length():
    rng = np.random.default_rng(2)
    horizon = 6
    baseline = rng.normal(0.0, 0.1, size=(200, horizon))
    effect = rng.normal(-0.1, 0.02, size=(200, horizon))
    weeks = np.arange(horizon)
    result = combine_scenarios(baseline, effect, weeks=weeks, n_mc=1000, random_state=0)
    assert len(result.weekly) == horizon
    for col in ["scenario_a_mean", "scenario_b_mean", "impact_mean", "pct_impact_mean"]:
        assert col in result.weekly.columns
