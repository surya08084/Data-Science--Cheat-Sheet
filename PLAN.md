# Plan: Isolate Methodology into a Standalone Orphan Branch

## Context (this revision)

The user now wants the vapor-policy methodology deliverable isolated into its **own orphan branch** (`vapor-policy-methodology`) that shares no git history with the rest of the repo and contains **only two files**: this plan document and the methodology write-up — none of the repo's existing PDFs, folders, or README. This supersedes the earlier approach of adding the doc alongside existing repo content on `claude/vapor-policy-impact-forecast-jn9zgm` (that branch/PR is left as-is; this is a separate, additional branch).

## Steps

1. **Create a true orphan branch** off the current checkout: `git checkout --orphan vapor-policy-methodology`. This starts a new root commit with no parent, so it shares no commit history with `main` or `claude/vapor-policy-impact-forecast-jn9zgm`.
2. **Clear the working tree** of everything currently tracked/staged (`git rm -rf .` after the orphan checkout, since `--orphan` keeps existing working-tree files present but unstaged) so no PDFs, topic folders, or the original `README.md` remain.
3. **Add exactly two files** at the repo root:
   - `Vapor-Policy-Impact-Forecasting-Methodology.md` — the same comprehensive methodology content already authored on the other branch (problem formulation, data prep, causal methodology, 13-week counterfactual design, cross-category substitution, Altria vs. competitor decomposition, architecture, features, validation, uncertainty, business output, advanced considerations, recommended solution, roadmap).
   - `PLAN.md` — a copy of this plan file's final content (the plan describing this orphan-branch task), written at commit time from `/root/.claude/plans/you-are-a-seasoned-fluffy-widget.md`.
4. **Commit** both files as the single root commit on the new branch.
5. **Push** with `git push -u origin vapor-policy-methodology` (new branch, no force needed since it doesn't exist on the remote).
6. Leave the current branch/working tree situation clean afterward — return to the original branch (`claude/vapor-policy-impact-forecast-jn9zgm`) locally so the previously pushed work there is undisturbed.

## Verification

- `git log --oneline vapor-policy-methodology` should show exactly one commit.
- `git show --stat vapor-policy-methodology` should list only `Vapor-Policy-Impact-Forecasting-Methodology.md` and `PLAN.md`.
- `git merge-base main vapor-policy-methodology` should fail / return nothing (confirming no shared history — true orphan).
- Confirm the local working directory is restored to the prior branch (`claude/vapor-policy-impact-forecast-jn9zgm`) with a clean `git status` after pushing.

---

## Context (original methodology-authoring task, for reference)

The user needs a rigorous, end-to-end methodology for estimating the causal impact of a state-level vapor policy on retail volume *before* the policy is implemented in a new U.S. state, using the ~8–9 states that have already implemented similar policies as a natural experiment. This is a domain-expert consulting/design deliverable (causal inference + counterfactual forecasting), not a code-implementation task.

Repo context (`Data-Science--Cheat-Sheet`, branch `claude/vapor-policy-impact-forecast-jn9zgm`): the repository is currently a flat collection of curated third-party PDF cheat sheets, one Title-Case-named folder per topic (e.g. `Machine Learning/`, `Statistics/`, `Deep Learning/`), with no original authored markdown content and only a 1-line `README.md`. There is no existing causal-inference, forecasting, or case-study content. Since the task requires original authored analysis (not a PDF upload), the natural fit is a new, repo-authored Markdown document placed in a new topic folder, following the repo's existing "Title Case folder name" convention.

The deliverable will be a single comprehensive Markdown file containing the full methodology, then committed and pushed to the designated branch.

## Deliverable

**New folder:** `Causal Inference & Forecasting/`
**New file:** `Causal Inference & Forecasting/Vapor-Policy-Impact-Forecasting-Methodology.md`

This mirrors the repo's Title-Case topic-folder pattern while making clear this is an original methodology write-up (Markdown) rather than a PDF cheat sheet.

Also update the root `README.md` to add a short table-of-contents style link to the new file/folder, since it's the first authored content in the repo and currently nothing links anywhere.

## Document Structure

The markdown doc will directly answer all 15 requested sections, organized as:

1. **Executive Summary** — one paragraph framing this as a panel-data causal inference + counterfactual forecasting problem (not plain forecasting).
2. **Problem Formulation** — treatment (state-level vapor policy adoption), outcome (weekly volume by state/category/brand/manufacturer), intervention date, treatment vs. control group definitions, precise counterfactual definition, and framing as a combination of panel causal inference (DiD/SDiD/Synthetic Control for estimating the effect in *historical* states) + a counterfactual forecasting layer (for projecting that estimated effect onto a *new* state).
3. **Data Preparation** — grain (state × week × category × brand/manufacturer, rolled up from SKU), transformations (log volume, indexed/relative time to policy, deseasonalization), event-time alignment across staggered adoption dates, confounder handling (promo, price, distribution/ACV, launches/discontinuations, macro).
4. **Exploratory Analysis** — cross-state comparability checks, pre-trend parallelism tests, clustering states by pre-policy trajectory, visual event-study plots.
5. **Causal Methodology Comparison** — DiD, Event Study, Synthetic Control, Synthetic DiD, CausalImpact/BSTS, panel/mixed-effects regression, causal forests/CATE, staggered-adoption-robust DiD (Callaway–Sant'Anna, Sun–Abraham) — with an explicit recommendation and rationale (staggered-adoption DiD/event-study + Synthetic Control / SDiD as the core, BSTS as a per-state robustness check, causal forest for heterogeneity across state characteristics).
6. **13-Week Counterfactual Forecasting Design for a New (Untreated-So-Far) State** — two-stage approach: (a) estimate the *treatment effect function over event-time* from historical treated states (pooled event-study / SDiD effect curve, or donor-pool synthetic control), (b) build a no-policy baseline forecast for the new state using its own pre-period + synthetic donor pool of never/not-yet-treated states, (c) Scenario A = baseline forecast + estimated treatment-effect curve applied at the appropriate event-time weeks 1–13; Scenario B = baseline forecast alone.
7. **Cross-Category Substitution** — treat as a joint/system model (seemingly unrelated regression or multivariate panel with cross-category effects) rather than fully independent single-category models, so category impacts are estimated with a shared policy-effect design but category-specific coefficients and an implied "total nicotine market" reconciliation check (vapor loss ≈ sum of gains elsewhere + true attrition).
8. **Altria vs. Competitor Impact** — model manufacturer/brand as a nested level under category (state-category-manufacturer grain), decompose category-level impact into Altria vs. competitor shares, market-share-shift metrics, substitution flags.
9. **Model Architecture** — the pipeline diagram: Raw Scan Data → Feature Engineering → Policy/Event Panel Dataset → Causal Effect Estimation (historical states) → Counterfactual Baseline Generator (new state) → 13-Week Scenario Forecast (A & B) → Policy vs. No-Policy Comparison → Business Impact Layer, described as production components (batch feature store, model registry per category, scenario API).
10. **Feature Engineering** — full feature list requested (historical volume/trend/seasonality, pricing, promo, distribution, share, competitor activity, category trends, state characteristics, policy characteristics/strength, pre-policy momentum, macro/demographic).
11. **Validation Strategy** — leave-one-state-out (LOSO) pseudo-policy validation loop exactly as specified (train on 7–8, hold out 1, repeat), plus placebo-in-time tests on control states, metrics for both the 13-week level forecast (MAPE/WAPE/MASE, coverage of prediction intervals) and for the *estimated impact* (bias vs. the held-out state's actual realized effect, sign accuracy, interval coverage).
12. **Model Comparison / Stack** — when to use classical statistical (ETS/ARIMA baselines), econometric panel models, gradient boosting (LightGBM/XGBoost for baseline + feature-rich forecasting), Bayesian structural time series (per-state effect + uncertainty), causal forests (heterogeneity), and how they're ensembled.
13. **Uncertainty Quantification** — point estimate, prediction intervals (quantile GBM or BSTS posterior), credible interval for the impact (posterior of Scenario A − Scenario B), probability of material positive/negative impact (posterior probability mass beyond a business-defined threshold).
14. **Business Output** — the requested comparison table (Vapor/Cigarettes/TDN/MST/Total Market × No Policy/Policy/Incremental/% Impact) plus recommended visuals (event-study plot, synthetic control fit plot, fan chart for the 13-week scenarios, waterfall of category substitution, Altria-vs-competitor share-shift chart) and example executive narrative bullets.
15. **Advanced Considerations** — explicit treatment of each listed challenge (small N=8–9 treated units, heterogeneous policy definitions/dates, anticipation effects, partial compliance, concurrent FDA regulation, spillovers/cross-border purchasing, structural breaks, sparsity) and the specific mitigation used for each.
16. **Recommended Final Solution (concrete)** — a single opinionated recommendation: staggered-adoption-robust event-study/DiD (Callaway–Sant'Anna) combined with Synthetic Control / Synthetic-DiD per state to build the effect curve, applied on top of a BSTS or LightGBM-quantile no-policy baseline forecast for the new state; states as panel of donor pool; target variable (log weekly volume by state-category-manufacturer); full feature set; treatment/control definition; counterfactual generation mechanics; validation via LOSO; productionization plan.
17. **Step-by-Step Implementation Roadmap** — phased roadmap (data foundation → EDA/comparability → causal effect estimation on historical states → validation via LOSO → counterfactual forecasting engine for new states → cross-category/manufacturer decomposition → uncertainty & business reporting layer → production deployment & monitoring), with rough phase sequencing.

## Verification

- Since this is a documentation deliverable (no executable code), verification is: re-read the final markdown for consistency between sections (e.g., the "Recommended Final Solution" must match the methodology chosen in section 5/6), confirm all 15 requested numbered topics from the user's prompt are addressed, confirm tables/diagrams render as valid Markdown (check fenced code blocks and pipe-tables are well-formed), and confirm the file is placed and linked per repo convention.
- After writing, `git status`/`git diff` to confirm only the new file + README edit are staged, then commit with a descriptive message and push to `claude/vapor-policy-impact-forecast-jn9zgm` per the branch instructions.
