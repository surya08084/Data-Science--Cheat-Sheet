import numpy as np
import pandas as pd

from vapor_policy_impact.decomposition.manufacturer import decompose_manufacturer
from vapor_policy_impact.scenario.combiner import combine_scenarios
from vapor_policy_impact.substitution.cross_category import category_residual_correlation, reconcile_categories


def _fake_scenario(mean_a: float, mean_b: float, horizon: int = 4, n: int = 500, seed: int = 0):
    rng = np.random.default_rng(seed)
    baseline = rng.normal(np.log(mean_b), 0.02, size=(n, horizon))
    effect = np.full((n, horizon), np.log(mean_a / mean_b))
    weeks = np.arange(horizon)
    return combine_scenarios(baseline, effect, weeks=weeks, n_mc=1000, random_state=seed)


def test_reconcile_categories_total_matches_sum_of_categories():
    vapor = _fake_scenario(mean_a=8.0, mean_b=10.0, seed=1)
    cigs = _fake_scenario(mean_a=10.3, mean_b=10.0, seed=2)
    recon = reconcile_categories({"Vapor": vapor, "Cigarettes": cigs})

    manual_total_b = vapor.cumulative["scenario_b"]["mean"] + cigs.cumulative["scenario_b"]["mean"]
    assert abs(recon.total_market["scenario_b"]["mean"] - manual_total_b) < manual_total_b * 0.05
    assert set(recon.category_share_of_gross_movement["category"]) == {"Vapor", "Cigarettes"}
    # vapor should dominate gross movement since its cumulative impact is larger in magnitude
    shares = recon.category_share_of_gross_movement.set_index("category")["share_of_gross_movement"]
    assert shares["Vapor"] > shares["Cigarettes"]


def test_decompose_manufacturer_reconciles_and_share_shift_direction():
    altria = _fake_scenario(mean_a=3.0, mean_b=3.0, seed=3)  # unaffected -> share should not shift down
    competitor = _fake_scenario(mean_a=4.0, mean_b=7.0, seed=4)  # competitor declines a lot

    decomp = decompose_manufacturer("Vapor", altria, competitor)
    total_b = decomp.total_from_manufacturer_sum["scenario_b"]["mean"]
    assert total_b > 9.5  # roughly 3 + 7

    # Altria holds steady while competitor shrinks -> Altria's share should rise under policy
    assert decomp.share_shift["delta_share_mean"].mean() > 0


def test_category_residual_correlation_runs_and_is_symmetric():
    rng = np.random.default_rng(5)
    n = 200
    states = np.repeat([f"S{i}" for i in range(5)], n // 5)
    weeks = np.tile(np.arange(n // 5), 5)
    base = pd.DataFrame(
        {
            "state": states,
            "week": weeks,
            "log_volume": rng.normal(1.0, 0.1, n),
            "price": rng.normal(5, 0.5, n),
            "promo_depth": rng.uniform(0, 0.3, n),
            "distribution_acv": rng.normal(80, 5, n),
            "fourier_sin_1": np.sin(weeks),
            "fourier_cos_1": np.cos(weeks),
            "fourier_sin_2": np.sin(2 * weeks),
            "fourier_cos_2": np.cos(2 * weeks),
            "treated_flag": False,
        }
    )
    feature_series_by_category = {"Vapor": base, "Cigarettes": base.assign(log_volume=rng.normal(2.0, 0.1, n))}
    corr = category_residual_correlation(feature_series_by_category)
    assert corr.shape == (2, 2)
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T, atol=1e-8)
    assert np.allclose(np.diag(corr.to_numpy()), 1.0, atol=1e-6)
