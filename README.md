# Vapor Policy Impact — Causal Inference & Counterfactual Forecasting

A working Python implementation of the methodology in
[`Vapor-Policy-Impact-Forecasting-Methodology.md`](./Vapor-Policy-Impact-Forecasting-Methodology.md):
estimating the causal impact of a state-level vapor policy on U.S. tobacco
retail volume *before* it is implemented in a new state, using the states
that have already implemented similar policies as a natural experiment.

This branch is self-contained: the methodology write-up (`Vapor-Policy-Impact-Forecasting-Methodology.md`),
the implementation plan (`PLAN.md`), and this code are the only things here —
it shares no history with the rest of the repository.

## What this is (and isn't)

This is a **reference implementation you can actually run**, built against a
**synthetic, simulated retail-scan panel** with a known, injected policy
effect — real Altria/competitor scan data was never available in this
environment. The synthetic simulator (`vapor_policy_impact/data/simulate.py`)
generates data matching the schema real scan data would have, with a known
ground-truth effect baked in, specifically so every component can be built
and checked against a right answer before ever touching real data. Swapping
in real data means replacing the simulator's output with a real extract in
the same schema — nothing downstream needs to change.

Every simplification relative to the full methodology doc is called out
directly in the relevant module's docstring (e.g. the event-study estimator
is a from-scratch simplified Callaway–Sant'Anna, not the full doubly-robust
version; the Bayesian structural time series baseline is a frequentist
state-space fit via `statsmodels`, not a full MCMC treatment). Read the
module docstrings before treating any single number as gospel.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `vapor_policy_impact` importable

python examples/run_demo.py     # full end-to-end run, ~10-15 minutes
pytest tests/ -q                 # unit + integration tests, ~1-2 minutes
```

`examples/run_demo.py` runs the whole pipeline for a simulated "new" state
about to implement a vapor policy: per-category 13-week Policy vs. No-Policy
scenarios, the executive business table, cross-category substitution
reconciliation, Altria-vs-competitor decomposition, a leave-one-state-out
validation run, and a placebo-in-time false-positive check. Outputs (tables
as CSV, charts as PNG) are written to `examples/output/` (gitignored).

## Package structure -> methodology section

| Module | Methodology section | What it does |
|---|---|---|
| `vapor_policy_impact/data/simulate.py` | "Available Data" | Synthetic state x category x manufacturer x week panel with a known injected effect |
| `vapor_policy_impact/features/engineering.py` | 2. Data preparation | Event-time alignment, transforms, lags/momentum/seasonality |
| `vapor_policy_impact/causal/event_study.py` | 4-5. Causal methodology (Layer 1) | Staggered-adoption event study / DiD -> pooled ATT(e) |
| `vapor_policy_impact/causal/synthetic_control.py` | 4-5 (Layer 2) | Donor-pool synthetic control, used both retrospectively and as the new-state baseline template |
| `vapor_policy_impact/causal/heterogeneity.py` | 4-5, 15 (Layer 3) | Transports the pooled effect to a new state via its covariates |
| `vapor_policy_impact/forecasting/baseline.py` | 5, 11 | No-policy baseline forecaster: BSTS-style state-space model + quantile-GBM ensemble |
| `vapor_policy_impact/scenario/combiner.py` | 5, 12 | Combines baseline + effect draws into Scenario A/B, impact, and uncertainty |
| `vapor_policy_impact/substitution/cross_category.py` | 6 | Cross-category reconciliation + residual-correlation substitution diagnostic |
| `vapor_policy_impact/decomposition/manufacturer.py` | 7 | Altria vs. competitor decomposition + market-share shift |
| `vapor_policy_impact/validation/loso.py` | 10 | Leave-one-state-out pseudo-policy validation + placebo-in-time tests |
| `vapor_policy_impact/reporting/` | 13 | Executive table + recommended visualizations |
| `vapor_policy_impact/pipeline.py` | 8 | Wires everything into one call per (state, category, manufacturer_group) |

## Tests

`tests/` covers each module in isolation with a small, fast simulated fixture
(`tests/conftest.py`), plus slower end-to-end integration tests
(`test_pipeline_and_validation.py`) exercising the full stack including LOSO
and placebo validation. A few of these tests are regression tests for real
bugs caught during development — e.g. `test_gbm_forecast_is_smooth_not_alternating`
guards against a duplicate-row misalignment bug that previously made the
quantile-GBM baseline forecast oscillate wildly week to week.
