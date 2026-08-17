import pytest

from vapor_policy_impact.data.simulate import SimulationConfig, simulate_panel
from vapor_policy_impact.pipeline import prepare_covariates


@pytest.fixture(scope="session")
def small_sim():
    """A much smaller simulated panel than the full demo default, so the test suite
    runs in seconds instead of minutes. Shares the same schema/behavior."""
    cfg = SimulationConfig(
        n_weeks=120,
        n_treated_states=4,
        n_control_states=6,
        min_effective_week=40,
        max_effective_week_margin=15,
        target_state_effective_week_margin=13,
    )
    return simulate_panel(config=cfg, seed=11)


@pytest.fixture(scope="session")
def small_covariates(small_sim):
    return prepare_covariates(small_sim.state_covariates)
