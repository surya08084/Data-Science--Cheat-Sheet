"""Synthetic state x category x manufacturer x week retail-scan panel.

Generates data that mimics the structure described in the methodology doc's
"Available Data" section, with a *known, injected* vapor-policy causal effect
(including cross-category substitution and Altria-vs-competitor heterogeneity)
so every downstream component (event study, synthetic control, baseline
forecaster, scenario combiner, validation) can be built and checked against
ground truth before ever touching real scan data.

This is a stand-in for real data only -- the schema matches
:class:`vapor_policy_impact.config.PanelSchema` so real scan extracts can be
substituted without changing any downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vapor_policy_impact.config import CATEGORIES, MANUFACTURER_GROUPS, PolicyCalendar

RNG_DEFAULT_SEED = 7


@dataclass
class SimulationConfig:
    n_weeks: int = 208  # 4 years of weekly data
    n_treated_states: int = 9
    n_control_states: int = 15
    #: earliest/latest week (exclusive of the final 13) a treated state's
    #: policy may take effect, so every treated state has pre-trend history
    #: and at least a full 13-week post window observed.
    min_effective_week: int = 60
    max_effective_week_margin: int = 20  # margin before n_weeks - 13
    #: the state used as the "brand new, about to implement policy" case for
    #: demos/validation. Its post-policy data is withheld from the returned
    #: panel (see `simulate_panel(..., reveal_target_post_period=False)`).
    target_state_effective_week_margin: int = 13


@dataclass
class SimulatedData:
    panel: pd.DataFrame
    state_covariates: pd.DataFrame
    policy_calendar: PolicyCalendar
    ground_truth_log_effect: dict  # state -> category -> mfg_group -> np.ndarray (indexed by event_week 0..)
    target_state: str
    target_state_effective_week: int
    target_state_ground_truth: pd.DataFrame  # post-period true volumes, policy & no-policy, by category/mfg
    donor_pool_states: list = field(default_factory=list)
    #: state x week x category x manufacturer_group true no-policy volume for *every*
    #: state (including the 9 historical treated states) -- test/validation-only, since
    #: this is exactly the thing a real deployment can never observe.
    full_ground_truth_no_policy: pd.DataFrame = field(default_factory=pd.DataFrame)


def _fourier_seasonal(week: np.ndarray, period: float = 52.0, n_harmonics: int = 2) -> np.ndarray:
    out = np.zeros_like(week, dtype=float)
    for k in range(1, n_harmonics + 1):
        out += 0.05 * np.sin(2 * np.pi * k * week / period) / k
        out += 0.03 * np.cos(2 * np.pi * k * week / period) / k
    return out


def _make_state_covariates(states: list[str], rng: np.random.Generator) -> pd.DataFrame:
    n = len(states)
    return pd.DataFrame(
        {
            "state": states,
            "population_m": rng.lognormal(mean=1.3, sigma=0.6, size=n).round(2),
            "urbanization_pct": np.clip(rng.normal(65, 15, n), 20, 98).round(1),
            "median_income_k": np.clip(rng.normal(62, 12, n), 35, 110).round(1),
            "border_state": rng.random(n) < 0.35,
            "retail_density": np.clip(rng.normal(1.0, 0.25, n), 0.4, 1.8).round(3),
            "baseline_vapor_share": np.clip(rng.normal(0.18, 0.04, n), 0.06, 0.32).round(3),
        }
    )


def _policy_stringency(rng: np.random.Generator, n: int) -> np.ndarray:
    return np.clip(rng.normal(0.6, 0.2, n), 0.15, 1.0).round(3)


def _true_effect_curve(asymptote: float, decay: float, n_event_weeks: int) -> np.ndarray:
    """Log-multiplicative ATT(e) for e = 0..n_event_weeks-1: ramps in then plateaus."""
    e = np.arange(n_event_weeks)
    return asymptote * (1 - np.exp(-e / decay))


def simulate_panel(
    config: SimulationConfig | None = None,
    seed: int = RNG_DEFAULT_SEED,
    reveal_target_post_period: bool = False,
) -> SimulatedData:
    """Simulate the full panel plus ground truth.

    Parameters
    ----------
    reveal_target_post_period:
        If False (the default, and the realistic production setting), the
        returned ``panel`` contains no rows for the target ("new") state past
        its hypothetical policy week -- exactly what you'd have for a state
        that hasn't implemented the policy yet. The true post-period paths
        (both with- and without-policy) are still returned separately via
        ``target_state_ground_truth`` for evaluation purposes only.
    """
    cfg = config or SimulationConfig()
    rng = np.random.default_rng(seed)

    treated_states = [f"T{i+1}" for i in range(cfg.n_treated_states)]
    control_states = [f"C{i+1}" for i in range(cfg.n_control_states)]
    target_state = "Z_NEW"
    all_states = treated_states + control_states + [target_state]

    state_cov = _make_state_covariates(all_states, rng).set_index("state")

    # Staggered effective weeks for the 9 historical treated states.
    max_eff = cfg.n_weeks - 13 - cfg.max_effective_week_margin
    effective_weeks = {
        s: int(w)
        for s, w in zip(
            treated_states,
            np.sort(rng.integers(cfg.min_effective_week, max_eff, size=cfg.n_treated_states)),
        )
    }
    # Target state's policy is right at the edge: only 13 post weeks exist in history,
    # mimicking "we are 0 weeks away from a new state's policy taking effect."
    target_effective_week = cfg.n_weeks - cfg.target_state_effective_week_margin

    policy_calendar = PolicyCalendar(
        treated_states=dict(effective_weeks),
        policy_type={s: rng.choice(["flavor_ban", "sales_restriction", "licensing"]) for s in treated_states},
        policy_scope={s: rng.choice(["statewide", "statewide_with_local_exceptions"]) for s in treated_states},
    )
    stringency = dict(zip(treated_states, _policy_stringency(rng, len(treated_states))))

    # ---- category/manufacturer base structure -----------------------------------
    category_base_share = {"Vapor": 0.22, "Cigarettes": 0.55, "TDN": 0.06, "MST": 0.17}
    category_trend = {"Vapor": 0.0022, "Cigarettes": -0.0016, "TDN": 0.0040, "MST": -0.0004}
    altria_share_by_category = {"Vapor": 0.32, "Cigarettes": 0.48, "TDN": 0.30, "MST": 0.55}

    weeks = np.arange(cfg.n_weeks)
    rows = []

    ground_truth_log_effect: dict = {s: {} for s in treated_states + [target_state]}

    for state in all_states:
        cov = state_cov.loc[state]
        state_scale = np.log(cov["population_m"]) + 0.15 * (cov["retail_density"] - 1.0)
        is_treated_hist = state in treated_states
        eff_week = effective_weeks.get(state) if is_treated_hist else (
            target_effective_week if state == target_state else None
        )
        # heterogeneity multiplier on effect magnitude driven by covariates (used later
        # to check whether Layer 3 can recover it from state characteristics alone).
        het_multiplier = (
            1.0
            + 0.35 * (cov["urbanization_pct"] - 65) / 40
            + 0.25 * (cov["baseline_vapor_share"] - 0.18) / 0.08
            + (0.30 * stringency.get(state, 0.6) if state in stringency else 0.30 * 0.6)
        )

        for category in CATEGORIES:
            cat_base = np.log(category_base_share[category]) + state_scale
            trend = category_trend[category]
            seasonal = _fourier_seasonal(weeks, n_harmonics=2 if category != "TDN" else 1)

            price = np.clip(rng.normal(6.5 if category == "Vapor" else 7.5, 0.6, cfg.n_weeks), 3, 14)
            promo_depth = np.clip(rng.normal(0.12, 0.05, cfg.n_weeks), 0, 0.4)
            distribution_acv = np.clip(rng.normal(78, 8, cfg.n_weeks) + 3 * np.log1p(weeks / 52), 40, 99)

            price_effect = -0.35 * (np.log(price) - np.log(price.mean()))
            promo_effect = 0.9 * promo_depth
            distribution_effect = 0.006 * (distribution_acv - distribution_acv.mean())

            # --- true policy + substitution effect (log-multiplicative), event-time indexed
            vapor_log_effect = np.zeros(cfg.n_weeks)
            if eff_week is not None:
                event_week = weeks - eff_week
                post_mask = event_week >= 0
                n_post = int(post_mask.sum())
                if n_post > 0:
                    vapor_curve = _true_effect_curve(asymptote=-0.55, decay=5.0, n_event_weeks=n_post) * het_multiplier
                    if category == "Vapor":
                        vapor_log_effect[post_mask] = vapor_curve
                    elif category == "Cigarettes":
                        # ~35% of the (multiplicative) vapor loss reappears as a cigarette gain
                        vapor_log_effect[post_mask] = -0.35 * vapor_curve * 0.55
                    elif category == "MST":
                        vapor_log_effect[post_mask] = -0.25 * vapor_curve * 0.55
                    elif category == "TDN":
                        vapor_log_effect[post_mask] = -0.10 * vapor_curve * 0.55
                    if state in ground_truth_log_effect and category == "Vapor":
                        ground_truth_log_effect[state]["Vapor"] = vapor_curve

            noise = rng.normal(0, 0.045, cfg.n_weeks)
            log_vol_total = cat_base + trend * weeks + seasonal + price_effect + promo_effect \
                + distribution_effect + vapor_log_effect + noise

            for mfg in MANUFACTURER_GROUPS:
                altria_share = altria_share_by_category[category]
                mfg_share = altria_share if mfg == "Altria" else (1 - altria_share)
                # small extra manufacturer-level heterogeneity in exposure to the policy
                mfg_policy_tilt = (0.06 if mfg == "Altria" else -0.06) if category == "Vapor" else 0.0
                log_vol = log_vol_total + np.log(mfg_share) + mfg_policy_tilt * (vapor_log_effect < 0)
                volume = np.exp(log_vol)
                sales = volume * price * (1.0 + 0.02 * rng.standard_normal(cfg.n_weeks))

                df_part = pd.DataFrame(
                    {
                        "state": state,
                        "week": weeks,
                        "category": category,
                        "manufacturer_group": mfg,
                        "manufacturer": f"{mfg}_{category[:3]}",
                        "brand": f"{mfg}_{category[:3]}_brand",
                        "sku": f"{state}_{category[:3]}_{mfg[:3]}_sku",
                        "volume": volume,
                        "sales": sales,
                        "price": price,
                        "promo_depth": promo_depth,
                        "distribution_acv": distribution_acv,
                    }
                )
                rows.append(df_part)

    panel = pd.concat(rows, ignore_index=True)

    # ---- assemble target-state ground truth (true with/without-policy post-period) ----
    target_rows = panel[panel["state"] == target_state].copy()
    post_mask = target_rows["week"] >= target_effective_week
    target_post = target_rows[post_mask].copy()
    target_post = target_post.rename(columns={"volume": "volume_with_policy"})

    # Recompute the true no-policy counterfactual by regenerating the same state with
    # the effect forced to zero, reusing an identical RNG stream so everything else matches.
    rng_cf = np.random.default_rng(seed)
    cf_rows = []
    for state in all_states:
        cov = state_cov.loc[state]
        state_scale = np.log(cov["population_m"]) + 0.15 * (cov["retail_density"] - 1.0)
        for category in CATEGORIES:
            cat_base = np.log(category_base_share[category]) + state_scale
            trend = category_trend[category]
            seasonal = _fourier_seasonal(weeks, n_harmonics=2 if category != "TDN" else 1)
            price = np.clip(rng_cf.normal(6.5 if category == "Vapor" else 7.5, 0.6, cfg.n_weeks), 3, 14)
            promo_depth = np.clip(rng_cf.normal(0.12, 0.05, cfg.n_weeks), 0, 0.4)
            distribution_acv = np.clip(rng_cf.normal(78, 8, cfg.n_weeks) + 3 * np.log1p(weeks / 52), 40, 99)
            price_effect = -0.35 * (np.log(price) - np.log(price.mean()))
            promo_effect = 0.9 * promo_depth
            distribution_effect = 0.006 * (distribution_acv - distribution_acv.mean())
            noise = rng_cf.normal(0, 0.045, cfg.n_weeks)
            log_vol_total = cat_base + trend * weeks + seasonal + price_effect + promo_effect \
                + distribution_effect + noise  # NOTE: no policy effect
            for mfg in MANUFACTURER_GROUPS:
                altria_share = altria_share_by_category[category]
                mfg_share = altria_share if mfg == "Altria" else (1 - altria_share)
                log_vol = log_vol_total + np.log(mfg_share)
                cf_rows.append(
                    pd.DataFrame(
                        {
                            "state": state,
                            "week": weeks,
                            "category": category,
                            "manufacturer_group": mfg,
                            "volume_no_policy": np.exp(log_vol),
                        }
                    )
                )
    cf_df = pd.concat(cf_rows, ignore_index=True)

    # Full-panel true no-policy counterfactual for every state (including the 9 historical
    # treated states, not just the target) -- lets the LOSO validation framework be checked
    # against genuine ground truth, not just real-vs-forecast actuals. A real deployment
    # obviously has no equivalent of this and relies on real actuals only (see validation.loso).
    full_ground_truth_no_policy = cf_df.copy()

    cf_post = cf_df[(cf_df["state"] == target_state) & (cf_df["week"] >= target_effective_week)]
    target_ground_truth = target_post.merge(
        cf_post, on=["state", "week", "category", "manufacturer_group"], how="left"
    )[["state", "week", "category", "manufacturer_group", "volume_with_policy", "volume_no_policy"]]

    if not reveal_target_post_period:
        panel = panel[~((panel["state"] == target_state) & (panel["week"] >= target_effective_week))].copy()

    panel = panel.sort_values(["state", "category", "manufacturer_group", "week"]).reset_index(drop=True)

    return SimulatedData(
        panel=panel,
        state_covariates=state_cov.reset_index(),
        policy_calendar=policy_calendar,
        ground_truth_log_effect=ground_truth_log_effect,
        target_state=target_state,
        target_state_effective_week=target_effective_week,
        target_state_ground_truth=target_ground_truth,
        donor_pool_states=control_states,
        full_ground_truth_no_policy=full_ground_truth_no_policy,
    )
