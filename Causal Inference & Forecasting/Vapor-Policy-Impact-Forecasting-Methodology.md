# Predicting the Impact of a State-Level Vapor Policy Before It Happens
### A Causal Inference + Counterfactual Forecasting Methodology for U.S. Tobacco Retail Scan Data

---

## Executive Summary

This is **not** a plain forecasting problem — a standard time-series model has no way to represent "what would have happened if the world had branched differently on the policy date." It is a **causal inference problem wrapped in a forecasting problem**:

1. First, **learn the causal effect of the policy** from the 8–9 states where it has already happened, using their pre/post behavior and untreated states as controls (panel causal inference: staggered-adoption event-study / Difference-in-Differences, Synthetic Control, Synthetic DiD, Bayesian Structural Time Series).
2. Then, **transport that learned effect** onto a new state that hasn't yet implemented the policy, by building a no-policy baseline forecast for that state and adding the estimated effect curve to it.

The two 13-week outputs the business wants — "Policy" and "No-Policy" — are literally `Baseline Forecast` and `Baseline Forecast + Estimated Treatment Effect`. Everything below is about doing both halves (the causal estimation and the baseline forecast) rigorously, at category/manufacturer granularity, with defensible uncertainty and a validation framework that works with only 8–9 treated units.

---

## 1. Problem Formulation

| Element | Definition |
|---|---|
| **Unit of analysis** | State × Category (Vapor, Cigarettes, TDN, MST) × Manufacturer/Brand (Altria vs. named competitors), observed weekly |
| **Treatment** | Binary indicator `Policy_t = 1` for a state from its policy effective date onward (optionally continuous/graded — see §14 for partial compliance & policy strength) |
| **Intervention date** | State-specific `T0_s` (staggered across the 8–9 treated states — this is a **staggered adoption** design, not a single-shot DiD) |
| **Outcome** | Weekly volume (primary), with sales/revenue and market share as secondary outcomes, at state-category-manufacturer grain |
| **Treatment group** | The 8–9 states with an implemented vapor policy, each analyzed relative to its own `T0_s` |
| **Control group** | (a) Not-yet-treated states in the same calendar window (used in staggered-adoption estimators), and (b) a synthetic donor pool of never-treated states weighted to match each treated state's pre-period trajectory (used in Synthetic Control) |
| **Counterfactual** | The volume path a state *would have followed from `T0_s` onward had the policy not been implemented* — estimated, never observed, and is the entire object this methodology exists to produce |

**Framing:** this is a **combination** of:
- **Panel causal inference** (to estimate the treatment-effect function over "event time" — weeks since policy — from the historical treated states), and
- **Counterfactual time-series forecasting** (to project a state-specific no-policy baseline for a *new* state that has no post-treatment data at all yet), with
- The causal effect curve **transported** onto that baseline to produce the "with-policy" scenario.

This is deliberately *not* framed as "train an ML model to forecast volume given a policy flag as a feature." A single regression/GBM with a policy dummy will pick up confounding (states that adopt policies may already be trending differently) and cannot separate "what the policy caused" from "what was already happening in that state." Causal identification has to happen first; forecasting is downstream of it.

---

## 2. Data Preparation

### 2.1 Grain
- **Base grain for causal modeling:** `state × week × category × manufacturer (Altria / named competitor / other)`, rolled up from SKU-level scan data (SKU detail is used for feature construction — e.g., new product launches — but the causal model should not run at SKU grain; too sparse and too much launch/discontinuation noise).
- **Base grain for the business output:** `state × category` (with an Altria-vs-competitor split as a secondary cut).

### 2.2 Core Transformations
- **Log or log1p volume** as the modeling target — policy effects are usually more stable as % changes than absolute-unit changes across states of different sizes.
- **Event time** `e = week − T0_s`, so all treated states can be pooled/aligned on a common axis (`e = -12 … -1` pre-period, `e = 0 … 12+` post-period for the 13-week window).
- **Indexing to a pre-period baseline** (e.g., volume relative to the mean of weeks `e = -13..-1`) for visualization and comparability across states of very different absolute size.
- **Deseasonalization / calendar adjustment**: remove ISO week-of-year seasonality and holiday effects (e.g., via STL decomposition or Fourier terms in the regression) *before* comparing pre/post — otherwise seasonal timing differences between states' policy dates will be misread as policy effect.
- **Detrending**: state-specific linear/local trend removed or explicitly modeled (see mixed-effects / synthetic control below), since pre-existing divergent trends are the main threat to identification.

### 2.3 Policy-Event Alignment
Build a **panel keyed on event time, not calendar time**, for the causal estimation step:

```
state | category | manufacturer | event_week (e) | calendar_week | volume | log_volume | treated_flag | ...features
```
Calendar time is retained as a covariate (for macro/seasonal controls and for the not-yet-treated comparison group) but the core regression axis for the effect curve is event time.

### 2.4 Handling Different Policy Dates Across States
This is exactly the staggered-adoption problem in the DiD literature. Two implications:
1. Do **not** use classic two-way-fixed-effects DiD naively — it is known to produce biased estimates under staggered adoption with heterogeneous effects (the "bad comparison" / negative-weighting problem). Use a staggered-adoption-robust estimator (Callaway–Sant'Anna, Sun–Abraham) — see §4.
2. Each treated state's post-period overlaps a *different* macro/FDA/calendar window. Calendar-time fixed effects (or a smooth macro control) in the model absorb national-level shocks (FDA guidance, national supply issues, excise tax changes) that are common across states in a given calendar week, so they aren't misattributed to any one state's policy.

### 2.5 Confounders to Explicitly Control For
- **Pricing** (own and competitor, list and promoted)
- **Promotion depth/frequency** (own and competitor)
- **Distribution / ACV (% stores carrying)**
- **Product launches and discontinuations** (a big vape launch/delisting in the same window as a policy can masquerade as a policy effect — flag and, if material, control for it or exclude affected SKUs from category aggregates with a footnote)
- **Concurrent regulation** (federal FDA actions, flavor bans, PMTA enforcement, local excise tax changes) — build a **regulatory calendar** covariate, separate from the state vapor policy flag, at national and state level
- **Store/channel mix shifts** (if scan panel composition changes over time)
- **Macro** (state unemployment, population, urbanization — mostly for donor-pool matching, see §3)

---

## 3. Exploratory Analysis of the 8–9 Historical Policy States

1. **Event-study plots per state**: index volume to `e=-1`, plot `e = -13..+13` for every treated state on one chart, category by category. This is the single most important diagnostic — it tells you visually whether effects are consistent, delayed, transient, or heterogeneous across states.
2. **Parallel pre-trends check**: for each treated state and its candidate controls/donor pool, regress pre-period volume on time and compare slopes; formally test (e.g., via a placebo/pre-period DiD on `e = -13..-1` only — the estimated "effect" should be statistically indistinguishable from zero). States that fail this are flagged as having non-parallel pre-trends and need a trend-adjusted estimator (synthetic control or DiD with state-specific linear trends) rather than plain DiD.
3. **Comparability / donor-pool suitability**: cluster all states (treated and not-yet-treated) on pre-period features — category mix, per-capita volume, urbanization, historical growth rate, retail channel mix, border-state status — using k-means or hierarchical clustering on standardized pre-period trajectories. This defines which never-treated states are legitimate synthetic-control donors for which treated states (and, later, for the new target state).
4. **Magnitude and shape clustering of post-period effects**: once effect curves are estimated per state (§4), cluster the *shapes* of the effect curves themselves (immediate sharp drop vs. gradual decline vs. delayed effect vs. no effect) — this determines whether one pooled effect curve is appropriate or whether state characteristics need to moderate the effect (motivates the causal-forest heterogeneity step, §4/§15).
5. **Cross-category co-movement**: at the same time as vapor's event-study plot, plot cigarettes/TDN/MST for the same states/weeks to visually pre-screen for substitution before formal modeling.

---

## 4. Causal Methodology — Options, Evaluation, Recommendation

| Method | What it does | Strength here | Weakness here |
|---|---|---|---|
| **Classic DiD (2×2)** | Compares treated vs. control, before vs. after | Simple, interpretable | Breaks down with staggered adoption + heterogeneous effects (biased/negatively-weighted estimates); assumes parallel trends |
| **Event Study (staggered-adoption robust: Callaway–Sant'Anna, Sun–Abraham)** | DiD generalized to many treatment dates, estimates effect **by event-time**, avoids "bad comparisons" | Directly produces the `e=0..13` effect curve we need; robust to staggered timing and heterogeneous effects; testable pre-trends | Still assumes no anticipation and needs a valid control pool per state |
| **Synthetic Control (SC)** | Builds a weighted combination of never-treated (or not-yet-treated) states that best reproduces each treated state's pre-period path, then treats the gap post-period as the effect | Excellent for a **small number of treated units** (exactly our 8–9 state case); highly visual/interpretable; doesn't need parallel-trends assumption, just good pre-fit | One state at a time; classic inference (permutation/placebo) is approximate; needs a good donor pool |
| **Synthetic DiD (SDiD)** | Combines DiD's use of multiple treated units + regularization with SC's unit-weighting and adds time-weighting | Best of both — pools information across all 8–9 states while keeping SC's robustness to level differences; more efficient standard errors than SC alone | More complex to implement; still developing tooling maturity vs. DiD/SC |
| **CausalImpact / Bayesian Structural Time Series (BSTS)** | Fits a Bayesian state-space model on a treated series using correlated control series as regressors, forecasts the counterfactual with full posterior | Gives a **full posterior over the counterfactual path** — exactly the uncertainty quantification the business wants (§13); good per-state diagnostic | One series (state) at a time; less natural for *pooling* the 8 states into one effect estimate |
| **Panel regression / mixed-effects (state & time random/fixed effects)** | Pools all states/categories in one regression with `Policy × event-time` interaction, state random intercepts/slopes | Efficient use of all data at once; natural place to add confounder controls (price, promo, distribution) and category/manufacturer interactions | Needs correct staggered-adoption specification (use with Callaway–Sant'Anna weighting, not naive TWFE) |
| **Causal forests / heterogeneous treatment effects (CATE)** | Machine-learning estimator of how the treatment effect *varies* by state/category/manufacturer characteristics | Directly answers "which state characteristics predict a bigger/smaller effect" → needed to **extrapolate to a new state that looks somewhat different from the 8–9 historical ones** | Needs reasonably large N; with only 8–9 treated states this is used at the *state-covariate* level (many state-category-manufacturer rows), not literally 8–9 observations |

### Recommendation

Use a **three-layer causal stack**, not a single method:

1. **Layer 1 — Effect-curve estimation (the workhorse):** Staggered-adoption event-study (Callaway–Sant'Anna) run on the pooled state-category-manufacturer panel, with not-yet-treated states as the control group and event-time fixed effects. This produces `ATT(e)` — the average effect at each week since policy, by category and manufacturer group, with confidence bands, pooled sensibly across all 8–9 states.
2. **Layer 2 — Per-state robustness / donor-pool counterfactual:** Synthetic Control (or Synthetic DiD) fit **per treated state**, to (a) sanity-check Layer 1 state-by-state, (b) supply a state-specific counterfactual baseline used later as the *template* for how to build a baseline for a brand-new state, and (c) support placebo-based inference given the small N.
3. **Layer 3 — Heterogeneity model:** A causal forest (or simpler: interact Layer-1 effect with state covariates in a meta-regression, given small N) trained on `state-category-manufacturer` rows to learn **how the estimated effect varies with observable state/policy characteristics** (urban share, pre-policy vapor category share, policy strength/scope, border-state exposure, retail density). This is what lets you move from "the average effect across 8 states" to "the *predicted* effect for a specific new state with its own characteristics" — required because the new state won't be identical to the historical 8.

BSTS/CausalImpact is used as a **secondary, per-state validation tool** (does its counterfactual agree with the SC/DiD counterfactual?) and as the **engine for the new-state no-policy baseline forecast** in §6, because it naturally produces the full posterior needed for uncertainty quantification.

---

## 5. 13-Week Counterfactual Forecasting for a New State

For a state `Z` about to implement the policy at date `T0_Z`, with historical pre-period data but (by definition) no post-period data yet:

**Step A — No-Policy Baseline (Scenario B).**
Forecast `Z`'s volume for `T0_Z .. T0_Z+13` *as if untreated*, using:
- `Z`'s own pre-period history (trend, seasonality, momentum),
- a synthetic-control-style donor pool of comparable never/not-yet-treated states (weighted to match `Z`'s pre-period trajectory — same technique as §4 Layer 2, just used prospectively instead of retrospectively) to correct for any idiosyncratic near-term deviation `Z`'s own history wouldn't have captured,
- known non-policy confounders projected forward (planned promotions, seasonal calendar, any known product launches).

This is a standard (but carefully donor-pool-corrected) probabilistic time-series forecast — implemented with BSTS (gives a full posterior directly) or a quantile-GBM/LightGBM ensemble (gives quantile forecasts) — see §11 for the choice.

**Step B — Transport the Effect Curve (Scenario A).**
Take the `ATT(e)` effect curve for weeks `e=0..12` from Layer 1 (or the heterogeneity-adjusted version from Layer 3, using `Z`'s own state characteristics, category mix, and policy strength/scope), and apply it multiplicatively (since effects were estimated in log space) to the Scenario-B baseline:

```
Scenario_A[e] = Scenario_B[e] × exp( ATT_hat(e | Z's covariates) )
```

Uncertainty from both the baseline forecast and the transported effect estimate are propagated together (see §13) — Scenario A's interval is *wider* than Scenario B's, since it carries both forecast uncertainty and causal-estimate uncertainty.

**Step C — Impact.**
```
Policy Impact(e) = Scenario_A[e] − Scenario_B[e]     (absolute)
% Impact(e)       = Scenario_A[e] / Scenario_B[e] − 1  (relative)
```
reported per week and cumulated over the 13-week window.

Why this two-stage design instead of one model that ingests a "policy" feature and forecasts directly: a single model conditioned on a policy flag would need the *new* state's post-treatment behavior to learn from, which doesn't exist — it can only ever have learned "association between policy flag and outcome" from the 8-9 historical states, which is exactly the confounding problem in §1. Separating "what would have happened anyway" (pure forecasting problem, uses `Z`'s own rich history) from "what the policy changes on top of that" (pure causal-transport problem, uses the *pooled* historical treated states) is what makes both halves tractable and auditable.

---

## 6. Cross-Category Substitution (Vapor / Cigarettes / TDN / MST)

**Recommendation: model jointly, not independently**, via a **system approach**:

- Run the *same* staggered-adoption event-study specification (§4, Layer 1) separately by category to get a clean, category-specific `ATT(e)` — independent estimation of the effect itself is fine and necessary (different categories have different baseline dynamics, price elasticities, etc.).
- But **evaluate and reconcile the four category effects together**, not in isolation, using a **market-level accounting identity**:

```
Δ Total Nicotine Volume (weighted for product strength/format if needed)
   = Δ Vapor + Δ Cigarettes + Δ TDN + Δ MST
```

  If `Δ Vapor` is strongly negative and `Δ Cigarettes/TDN/MST` are positive and roughly offsetting, that's evidence of **substitution**. If the sum is strongly negative with no offsetting categories, that's evidence of **net category attrition** (consumers leaving nicotine categories altogether, at least in the legal/scanned channel — see cross-border/illicit-channel caveat in §14).
- Formally test substitution with a **Seemingly Unrelated Regression (SUR)** or a multivariate panel model where the four category equations share the state/time structure and error correlations across categories are estimated — this both increases statistical efficiency and gives a direct cross-category correlation of residual shocks, which is informative about whether categories move together beyond what's explained by observables.
- Report substitution **flows**, not just category totals: cross-reference the *timing* (does cigarette/MST volume start rising within the same weeks vapor drops, in the same state?) and, where possible, panel/loyalty-linked data if available (outside pure scan data) to see actual switching rather than inferring it solely from aggregate co-movement.

---

## 7. Altria vs. Competitor Impact

- Extend the grain one level: `state × category × manufacturer (Altria brand family / named competitor / other)`.
- Run Layer 1 (event study) at this grain too — this gives `ATT(e)` **separately for Altria's brands and competitors within the same category and state set**, which is the direct answer to "does the policy hit Altria differently than competitors."
- **Total category impact** = sum of manufacturer-level impacts (should reconcile with the category-level model in §6 as a QA check).
- **Market-share shift**: track `share_t = volume_Altria,t / volume_category,t` under both scenarios; `Δ Share = Share_A(policy) − Share_B(no-policy)` isolates whether the policy is share-neutral (hits everyone proportionally) or share-shifting (e.g., if compliance/enforcement, product mix, or price positioning make one manufacturer's portfolio more/less exposed to the policy's mechanics — e.g., a flavor restriction hits disproportionately if Altria/competitor flavor mix differs).
- **Substitution effects across manufacturers within a category** (e.g., consumers switching from a restricted competitor SKU to an Altria SKU that remains compliant, or vice versa) are visible as *manufacturer-level effects with opposite signs within the same category and state* — flag any state where the category-level effect is small but manufacturer-level effects are large and offsetting, since that is a within-category share story, not a demand-destruction story.

---

## 8. Model Architecture (Production Pipeline)

```
┌────────────────────┐
│ Raw Weekly Scan Data│  (state, week, SKU, brand, manufacturer, category, volume,
│ + Policy Calendar    │   $, price, promo, distribution/ACV, share)
│ + Reg. Calendar      │  (state policy effective dates + type/scope, FDA/national actions)
└─────────┬───────────┘
          ▼
┌────────────────────┐
│ Feature Engineering  │  rollups to state-category-manufacturer-week,
│ & Panel Construction │  event-time alignment, deseasonalization, confounder features (§10),
│                      │  donor-pool / comparability features
└─────────┬───────────┘
          ▼
┌────────────────────┐
│ Policy / Event Panel │  state × category × manufacturer × event_week,
│ Dataset (analysis-   │  treated_flag, calendar controls, all features — this is the
│ ready)               │  single feature store both the causal layer and forecaster read from
└─────────┬───────────┘
          ▼
┌──────────────────────────────┐     ┌─────────────────────────────┐
│ Causal Effect Model            │     │ No-Policy Baseline Forecaster │
│ (Layer 1 event-study/SDiD,     │     │ (BSTS / quantile GBM using     │
│  Layer 2 synthetic control,    │     │  donor-pool + own history for  │
│  Layer 3 heterogeneity/CATE)   │     │  the *new* target state)       │
│ → ATT(e) curve + posterior/CI  │     │ → Scenario B + intervals        │
└───────────────┬────────────────┘     └───────────────┬─────────────┘
                │                                        │
                └───────────────┬────────────────────────┘
                                 ▼
                     ┌─────────────────────┐
                     │ Counterfactual        │
                     │ Generator / Scenario   │   Scenario_A = Scenario_B × exp(ATT_hat(e))
                     │ Combiner               │   propagate uncertainty (§13)
                     └──────────┬────────────┘
                                 ▼
                     ┌─────────────────────┐
                     │ 13-Week Forecast       │  weekly Scenario A & B, by category & manufacturer
                     └──────────┬────────────┘
                                 ▼
                     ┌─────────────────────┐
                     │ Policy vs. No-Policy   │  Δ, %Δ, credible intervals, P(material impact)
                     │ Comparison Layer       │
                     └──────────┬────────────┘
                                 ▼
                     ┌─────────────────────┐
                     │ Business Impact Output │  exec table, visuals, narrative (§13-14)
                     └─────────────────────┘
```

**Productionization notes:**
- Feature store and the analysis panel are refreshed on the normal scan-data cadence (weekly).
- The causal effect model (Layers 1–3) is **retrained periodically** (e.g., quarterly, or whenever a new state crosses its policy date and a 13-week+ post window becomes available) — not every week; it doesn't need weekly-fresh data since it's estimating a structural relationship, not chasing short-term noise.
- The baseline forecaster **does** refresh weekly/on each new scan-data drop for the target state, since it's a normal rolling forecast.
- Expose the pipeline as a **scenario API/service**: input = target state + optional hypothetical policy date; output = the 13-week Scenario A/B table with intervals, at category and manufacturer grain, versioned against the causal-model version that produced `ATT_hat`.
- Model registry: version the Layer 1/2/3 causal models and the baseline forecaster independently, log which combination produced any given business output for auditability (regulatory/commercial decisions will need to trace back to model version).

---

## 9. Feature Engineering

| Group | Features |
|---|---|
| **Historical volume/trend** | Lagged volume (1,4,13,52 wk), rolling mean/std, YoY growth, trailing trend slope, momentum (last-4-week vs. prior-4-week growth) |
| **Seasonality** | ISO week-of-year, month, holiday flags, Fourier seasonal terms, category-specific seasonal indices |
| **Pricing** | Own list/promo price, price gap vs. category, competitor price, price elasticity proxies |
| **Promotion** | Promo depth/frequency/duration, feature/display flags, own & competitor |
| **Distribution** | ACV %, store count, distribution change rate, new-store vs. existing-store growth |
| **Market share** | Category share, share trend, share volatility, Altria vs. competitor share gap |
| **Competitor activity** | Competitor volume/price/promo/distribution changes, competitor launch/exit flags |
| **Category-level trends** | Adjacent-category growth rates (for substitution features), total nicotine category trend |
| **State characteristics** | Population, urbanization %, median income, border-state flag, historical category category mix, retail channel mix (convenience vs. grocery vs. other) |
| **Policy characteristics** | Policy type (flavor ban, tax, licensing, sales restriction, age/ID enforcement change), scope (statewide vs. local), stringency/strength score, enforcement mechanism, anticipation lead time between announcement and effective date |
| **Pre-policy momentum** | Growth rate and volatility in the 4/13/26 weeks immediately pre-`T0` — a strong predictor of how sharply a state reacts |
| **Macro/environmental** | State excise tax level/changes, national FDA regulatory actions, unemployment, gas prices as a mobility/cross-border-shopping proxy, neighboring-state policy status (spillover feature — see §14) |

Feature engineering should produce **both**: (a) contemporaneous/lagged features for the baseline forecaster, and (b) static, pre-period-summarized covariates (state characteristics, policy characteristics, pre-policy momentum) for the Layer 3 heterogeneity model that predicts effect size for a *new* state.

---

## 10. Validation Strategy (Leave-One-State-Out / Pseudo-Policy Framework)

With only 8–9 treated states, k-fold CV on "rows" is meaningless (rows within a state are correlated, and the real unit of generalization is the *state*). Validate at the **state level**:

### 10.1 Leave-One-State-Out (LOSO) Pseudo-Policy Test
For each treated state `i` in `{A..I}`:
1. **Train** the full stack (Layer 1 event-study + Layer 3 heterogeneity model) on the remaining 7–8 treated states plus the control pool.
2. **Predict** state `i`'s effect curve and 13-week Scenario A/B **as if it were a new, unseen state**, using only its pre-period data and its state/policy covariates (exactly the real production scenario).
3. **Compare** the predicted Scenario A against `i`'s *actual observed* post-policy volume (which we have, since it's historical) — this is the only way to get a true out-of-sample read on both the forecast and the causal transport, given the small N.
4. Repeat for every state (`A..H → I`, `A..G+I → H`, etc.), producing 8–9 independent out-of-sample evaluations.

### 10.2 Placebo-in-Time Tests (on control states)
Assign a **fake** policy date to a never-treated (or not-yet-treated) state and run the exact same pipeline — the estimated "effect" should be statistically indistinguishable from zero. This directly tests for false positives / model bias independent of the treated-state sample size limitation.

### 10.3 Metrics

**For the 13-week level forecast (Scenario B baseline, and Scenario A against actuals):**
- WAPE / MAPE (weighted, since categories vary hugely in scale)
- MASE (scale-free, comparable across states/categories)
- Prediction interval coverage (does the 80%/95% interval actually contain the actual ~80%/95% of the time across the 8–9 LOSO runs?)

**For the estimated policy impact specifically (the number that matters most to the business):**
- Bias: mean(`predicted impact` − `actual realized impact`) across the LOSO runs, both in absolute and % terms
- Sign accuracy: did the model get the *direction* of impact right for each held-out state and category?
- Interval coverage for the impact estimate (does the credible interval for Δ contain the realized Δ ~as often as its stated confidence level?)
- Rank correlation between predicted and actual effect magnitude across the 8–9 states (does the model at least order states correctly by impact size, even if point estimates aren't perfect — useful for prioritization even under high uncertainty)

---

## 11. Model Comparison / Practical Stack

| Model class | Where used | Why |
|---|---|---|
| **Classical statistical (ETS, ARIMA/SARIMA)** | Baseline forecaster's simplest benchmark; per-state naive seasonal baseline | Fast, interpretable floor to beat; useful sanity check for the fancier baseline forecaster |
| **Econometric panel models (staggered-adoption DiD, mixed-effects)** | Layer 1 causal effect estimation | The right tool for identifying a causal effect from panel data with controls — this is the core of the whole methodology |
| **Synthetic Control / Synthetic DiD** | Layer 2 causal estimation + template for new-state no-policy baseline construction | Best-suited to small-N treated groups; highly auditable/visual for stakeholders |
| **Bayesian Structural Time Series (BSTS/CausalImpact)** | New-state Scenario B baseline forecaster (primary), and per-state causal robustness check | Natively gives a full posterior — exactly what's needed for §13's uncertainty requirements; handles multiple correlated donor-state regressors well |
| **Gradient boosting (LightGBM/XGBoost, quantile loss)** | Alternative/ensemble baseline forecaster when many engineered features (promo, price, distribution) are needed and nonlinearity/interactions matter | Handles rich feature sets and nonlinear confounder relationships better than BSTS; use quantile regression for interval estimates |
| **Causal forests / meta-learners (e.g., X-learner)** | Layer 3 heterogeneity — predicting effect size for a new state from its covariates | Purpose-built for CATE estimation; needed to extrapolate beyond the exact 8–9 historical states |
| **Ensemble** | Final Scenario B forecast = weighted blend of BSTS and quantile-GBM (weights chosen by LOSO backtest performance per category); final `ATT_hat` = Layer 1 pooled estimate adjusted by Layer 3 heterogeneity correction | Reduces single-model risk; the LOSO framework (§10) is exactly what should decide ensemble weights, not in-sample fit |

---

## 12. Uncertainty Quantification

The business needs a distribution, not a number, at every step:

1. **Point estimate**: median/mean of the posterior (BSTS) or median of quantile forecasts (GBM), for Scenario A and B separately.
2. **Prediction interval** (baseline forecast uncertainty): from BSTS posterior draws, or from GBM quantile heads (e.g., p10/p50/p90).
3. **Credible interval for the policy impact**: since `Impact = Scenario_A − Scenario_B`, and both scenarios share the same baseline draw (Scenario A is baseline × effect), propagate uncertainty **jointly**, not by naively subtracting two independent intervals (which would overstate uncertainty) — draw baseline-forecast samples and effect-curve posterior samples together (Monte Carlo), compute `Impact` per draw, and take the empirical interval of that distribution. This is straightforward if the baseline forecaster is BSTS (posterior draws are native) and the effect curve comes with its own posterior/bootstrap draws (e.g., via placebo permutation from Synthetic Control, or Bayesian panel model posterior).
4. **Probability of material impact**: define a business-relevant threshold (e.g., "more than 5% category volume decline"), then report `P(Impact% < -5%)` directly from the Monte Carlo draws in (3) — this converts a wide, hard-to-act-on interval into a decision-relevant statement ("82% probability the policy causes at least a 5% vapor volume decline in the first 13 weeks").

---

## 13. Business Output

### 13.1 Core Table (per target state, 13-week cumulative)

| Metric | No Policy | Policy | Incremental Impact | % Impact | 80% CI (Impact) |
|---|---:|---:|---:|---:|---:|
| Vapor | | | | | |
| Cigarettes | | | | | |
| TDN | | | | | |
| MST | | | | | |
| **Total Market** | | | | | |

Produce this table **twice**: once for Altria only, once for total category (Altria + competitors), plus a third "competitor-only" derived view (`Total − Altria`), so all three (Altria, competitor, total) are consistent and reconcilable.

### 13.2 Recommended Visualizations
- **Event-study plot** (per category): historical states' indexed volume around `e=0`, with the new state's projected Scenario A/B overlaid from `e=0`.
- **Synthetic control fit plot**: target state's actual pre-period vs. its synthetic donor-pool counterfactual, to build stakeholder trust in the counterfactual method before showing the future projection.
- **13-week fan chart**: Scenario A and B lines with shaded prediction intervals, side by side, for each category.
- **Substitution waterfall**: category-by-category Δ volume bars (vapor loss vs. cigarette/TDN/MST gains) summing to the total market Δ.
- **Altria vs. competitor share-shift chart**: share trajectory under Policy vs. No-Policy scenarios.
- **Probability-of-impact gauge/bar**: `P(material negative impact)` per category, from §12.4.

### 13.3 Example Executive Narrative Pattern
> "Based on the experience of the 8 states that have already implemented similar vapor policies, we project [State Z]'s vapor category volume would be **X% lower** over the first 13 weeks post-implementation than it would have been otherwise (80% CI: [a%, b%]), with an estimated **P% probability** of a decline exceeding 10%. Roughly **Y%** of that vapor volume loss appears to shift into cigarettes and MST rather than leaving the tracked nicotine category outright. Altria's exposure is [larger/smaller/in line with] the category average, driven by [specific covariate, e.g., higher flavor-SKU mix in this state]."

---

## 14. Advanced Considerations & Mitigations

| Challenge | Mitigation |
|---|---|
| Only 8–9 treated states | Pool via staggered-adoption DiD/SDiD (not single 2×2 DiD) for statistical efficiency; use LOSO (§10) instead of holdout splits; prefer methods designed for small-N (Synthetic Control, placebo-based inference) over asymptotic-N-assuming methods; be conservative/wide with intervals |
| Different policy definitions across states | Encode a **policy taxonomy** (flavor ban, flat sales ban, licensing/age enforcement, tax) as a categorical/feature set rather than one binary flag; either model each policy type's effect separately if enough states exist per type, or include policy-type as a Layer 3 heterogeneity covariate |
| Different implementation dates | Staggered-adoption event-study design (§4) is built exactly for this; calendar-time controls absorb concurrent national shocks |
| Anticipation effects | Check for pre-period volume/pricing/inventory anomalies in the weeks immediately before `T0` (e.g., stockpiling, early compliance); if present, either exclude the immediate pre-window from the "clean" pre-trend estimate or explicitly model an anticipation window as its own event-time bucket |
| Partial compliance | Add a compliance-intensity or enforcement-strength covariate where data allows (e.g., % of stores confirmed compliant); if unavailable, treat policy stringency as a Layer 3 covariate and flag effect estimates from low-compliance states as noisier |
| Concurrent regulations (FDA/other) | Maintain the separate national regulatory calendar (§2.5) as a control; calendar-time fixed effects in the panel absorb regulation that hits all states simultaneously; state-specific concurrent actions get their own dummy if material and dated |
| State-specific trends | State fixed effects (or random slopes in a mixed-effects specification) plus the synthetic-control donor weighting, which explicitly matches pre-trend, not just pre-level |
| Spillover / neighboring-state effects | Add a "neighboring-state policy exposure" feature (share of bordering states with an active policy); exclude or down-weight geographically adjacent states from a target state's control/donor pool if cross-border substitution is suspected, to avoid contaminated controls |
| Cross-border purchasing | Where possible, incorporate border-distance/border-store features; interpret category "loss" cautiously near borders — some volume may be relocating to neighboring-state scan data rather than truly disappearing; flag border states explicitly in the output |
| Product launches/discontinuations | Track as explicit event flags in the feature set; for major launches coincident with a policy date, consider a sensitivity check excluding that SKU/brand from the category rollup |
| Competitor response (repricing, reformulation, new compliant SKUs) | Captured naturally by the manufacturer-level model (§7) and competitor-activity features (§9); if a competitor response happens *within* the 13-week window, it's part of the true "policy" scenario (that's the real world), but should be called out narratively since it's a second-order effect layered on the direct policy effect |
| Changing consumer behavior over time | Periodic retraining of Layer 1/3 (§8) as more post-policy states accumulate data, so the effect curve isn't frozen to an early cohort's behavior |
| FDA vs. state-level regulation disentanglement | The dual regulatory calendar (state + national) plus calendar-time fixed effects is the core tool; where a national action coincides tightly with a specific state's `T0`, that state's estimate should be flagged low-confidence and down-weighted in pooling |
| Structural breaks | Monitor residuals of the baseline forecaster for the target state pre-launch; a break unrelated to policy (e.g., distribution disruption) should be modeled/controlled before attributing subsequent changes to policy |
| Seasonality | Deseasonalize before modeling (§2.2); ensure the 13-week comparison window is seasonally matched between Scenario A and B (trivial since both are the same calendar weeks) but also seasonally comparable across historical treated states' effect curves (event-time pooling can otherwise mix winter and summer post-periods — include calendar-month controls) |
| Data sparsity (small brands, thin categories like TDN in early years) | Use manufacturer/category rollups rather than SKU-level for the causal model; apply shrinkage/partial pooling (hierarchical/mixed-effects) so sparse categories borrow strength from the pooled category-level estimate rather than producing noisy standalone estimates |

---

## 15. Recommended Final Solution (Concrete)

**Models used:**
- **Layer 1 (causal effect):** Callaway–Sant'Anna staggered-adoption event-study / DiD, estimated on the pooled state-category-manufacturer panel, producing `ATT(e)` for `e = 0..12` with confidence bands, by category and by manufacturer group.
- **Layer 2 (robustness + template):** Synthetic Control / Synthetic DiD, fit per treated state per category, used to cross-validate Layer 1 and to define the donor-weighting methodology reused prospectively for new states.
- **Layer 3 (heterogeneity/transport):** A meta-regression / causal-forest-style model over state-category-manufacturer rows, regressing the Layer-1/Layer-2 estimated effect on state and policy covariates (§9), so it can output a **predicted** `ATT_hat(e | new state's covariates)` for a state with no post-treatment data.
- **Baseline forecaster (Scenario B):** BSTS as primary (posterior-native, handles donor-state regressors naturally), with a quantile-LightGBM model as a secondary/ensemble check where richer feature interactions (promo, price, distribution) matter.

**Why this combination:** no single method satisfies every requirement simultaneously — staggered dates (needs Callaway–Sant'Anna, not plain DiD), only 8–9 treated units (needs Synthetic Control-style small-N robustness and LOSO validation), need to generalize to a genuinely new, previously-unseen state (needs the heterogeneity/transport layer, not just an average historical effect), and need full uncertainty propagation for a business impact number (needs a posterior-native forecaster). Each layer is individually well-established causal/forecasting methodology; combining them is what fits this specific business problem's constraints.

**Training dataset:** the state-category-manufacturer-week panel (§2.3) spanning all history for the 8–9 treated states plus all never/not-yet-treated states used as controls/donors, with event-time alignment for the treated states and calendar-time alignment for control-state comparisons.

**Target variable:** `log(weekly volume)` at state-category-manufacturer grain (secondary targets: log sales/$ and category share, modeled the same way for the value and share views of impact).

**Features:** the full set in §9 — historical volume/trend/seasonality/momentum, pricing, promotion, distribution, share, competitor activity, cross-category trends, state characteristics, policy characteristics, and the national/state regulatory calendar.

**Treatment/control definition:** treatment = state-level vapor policy adoption from its effective date (`T0_s`), with policy type/scope as covariates; control = not-yet-treated states (staggered DiD) and a covariate-matched donor pool of never-treated states (synthetic control), explicitly excluding geographically adjacent states from a target's donor pool where cross-border contamination is a concern.

**Counterfactual generation:** for a new state, forecast its own no-policy baseline (Scenario B) from its pre-period history plus its synthetic donor pool; obtain `ATT_hat(e)` from Layer 3 using the new state's own covariates; combine multiplicatively to get Scenario A; Monte-Carlo-propagate both sources of uncertainty jointly for the impact interval.

**13-week forecast production:** run the baseline forecaster and the Layer 3 transport weekly as new data lands for the target state pre-launch, refreshed automatically; freeze/version the specific forecast used for any given business decision.

**Policy impact calculation:** `Scenario_A − Scenario_B` (and `%`), computed at category and manufacturer grain, aggregated to the executive table (§13.1), with the cross-category accounting identity (§6) used as an internal consistency check on every run.

**Validation:** LOSO pseudo-policy backtesting across all 8–9 treated states (§10.1) plus placebo-in-time tests on control states (§10.2), reported both as forecast-accuracy metrics and impact-estimate bias/coverage/sign-accuracy metrics — this is the validation story to put in front of stakeholders, since it's the only way to demonstrate out-of-sample credibility with this sample size.

**Productionization:** the pipeline in §8 — a periodically-retrained causal layer (Layers 1–3), a weekly-refreshed baseline forecaster, combined behind a versioned scenario API that outputs the business table/visuals for any target state on demand, with model-version lineage tracked for auditability.

---

## 16. Step-by-Step Implementation Roadmap

**Phase 1 — Data Foundation (weeks 1–3)**
- Build the state-category-manufacturer-week panel from raw scan data.
- Compile the policy calendar (8–9 treated states: effective dates, type, scope) and the national/state regulatory calendar.
- Implement event-time alignment, deseasonalization, and the core confounder features (price, promo, distribution, share).

**Phase 2 — Exploratory Analysis & Comparability (weeks 3–5)**
- Event-study plots per state/category; pre-trend parallelism checks; state clustering for donor-pool definition.
- Decide, per treated state, whether plain event-study/DiD is safe or a trend-adjusted/synthetic-control approach is needed.

**Phase 3 — Causal Effect Estimation on Historical States (weeks 5–9)**
- Implement Layer 1 (Callaway–Sant'Anna staggered event-study) at category and manufacturer grain.
- Implement Layer 2 (Synthetic Control / SDiD) per state as robustness check and donor-weighting template.
- Cross-check Layer 1 vs. Layer 2 estimates; reconcile discrepancies before proceeding.

**Phase 4 — Heterogeneity / Transport Layer (weeks 9–11)**
- Assemble state/policy covariate table; fit the Layer 3 meta-model regressing estimated effects on covariates.
- Sanity check: does it recover roughly the right effect for each of the 8–9 states when using leave-one-out covariate prediction (this is effectively a preview of §10.1)?

**Phase 5 — Validation Framework (weeks 11–13)**
- Implement the full LOSO pseudo-policy loop and placebo-in-time tests.
- Establish the metrics dashboard (WAPE/MASE/coverage for forecasts; bias/sign-accuracy/coverage for impact) and a go/no-go bar for production use.

**Phase 6 — Counterfactual Forecasting Engine for New States (weeks 13–16)**
- Build the BSTS (and quantile-GBM ensemble) baseline forecaster with donor-pool correction.
- Wire Scenario A/B combination logic and joint Monte Carlo uncertainty propagation.

**Phase 7 — Cross-Category & Manufacturer Decomposition (weeks 16–18)**
- Implement the SUR/multivariate cross-category model and the accounting-identity reconciliation check.
- Implement the Altria-vs-competitor manufacturer-level rollups and share-shift metrics.

**Phase 8 — Business Reporting Layer (weeks 18–20)**
- Build the executive table generator and the recommended visualization set.
- Draft narrative-generation templates (auto-populated from the numeric outputs) for stakeholder-ready summaries.

**Phase 9 — Production Deployment & Monitoring (weeks 20+)**
- Stand up the scenario API/service, model registry, and version lineage tracking.
- Set retraining cadence (causal layer: quarterly or on new-state-crossing-13-weeks-post; baseline forecaster: weekly).
- Monitor: track realized outcomes against Scenario A once a new state's actual post-policy data arrives, feeding it back into the treated-state pool for the next causal-layer retrain — this is how the methodology keeps improving as more states adopt the policy over time.
