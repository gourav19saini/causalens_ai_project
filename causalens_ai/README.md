# 🧭 CausaLens AI — Automated Causal Inference & Policy Optimization Engine

**Hackathon Problem Statement PS-04:** Estimate the causal effect of interventions,
identify confounders, estimate treatment effects for individuals/subgroups, and
recommend policies based on estimated benefits and costs.

This is a **working Streamlit application**, not a slide deck. It runs fully
offline / locally — no OpenAI/Anthropic/paid API key required.

---

## 1. Project Structure

```
causalens_ai/
│
├── app.py                     # Streamlit UI (all pages), imports src/ for logic
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml     # theme
│
├── data/
│   └── sample_data.csv        # pre-generated SYNTHETIC demo dataset (2000 rows)
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # sample data generator, CSV loading, summaries
│   ├── validation.py          # plain-language config validation
│   ├── causal_analysis.py     # DAG, propensity scores, overlap & balance diagnostics
│   ├── estimators.py          # ATE/CATE estimation (EconML if available, else AIPW fallback)
│   ├── policy_optimization.py # cost-benefit policy rules & policy comparison
│   ├── evaluation.py          # sample counts, placebo refutation, uplift/Qini-style curve
│   └── report_generation.py   # CSV / HTML report builders
│
└── tests/
    └── test_core.py           # unit tests for everything in src/ (no Streamlit needed)
```

Causal-ML logic is fully separated from the UI: everything in `src/` can be
imported and unit-tested without ever starting Streamlit.

---

## 2. Setup & Installation

```bash
# 1. Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note on EconML / DoWhy:** these are listed as recommended, optional
> dependencies. If they fail to install on your machine (they have native
> build dependencies on some platforms), **comment out those two lines in
> `requirements.txt` and reinstall** — the app will automatically detect
> their absence and use its built-in AIPW/T-learner fallback estimator
> instead, with a clear on-screen note explaining that a fallback was used.

---

## 3. Running the App

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`).

---

## 4. Running Tests

```bash
pytest tests/test_core.py -v
```

**What was actually tested before delivery, and how:** `streamlit`, `plotly`,
`econml`, and `dowhy` are not installable in the sandbox this project was
built in (no network access), so the automated tests below were run for
real, but `streamlit run app.py` itself was verified by a separate
lightweight harness that stubs `streamlit`/`plotly` and executes every page's
code path directly (not shipped, since it's a dev aid, not a product file).
You should still run `streamlit run app.py` yourself as a final check —
the logic underneath has been fully exercised, but real Streamlit's own
widget rendering was not observed pixel-for-pixel in this environment.

- ✅ 28 unit tests covering: sample data generation, CSV loading, dataset
  validation, treatment/outcome/feature validation, DAG construction,
  propensity scoring, overlap/balance diagnostics, the AIPW/T-learner
  fallback estimator (including small-sample and no-confounder edge cases),
  policy recommendation logic (including budget constraints), policy
  comparison, placebo refutation, the uplift/gain curve, and report
  generation — **all passing**.
- ✅ Every one of the 8 Streamlit pages executed successfully end-to-end
  (including button-triggered code paths: run estimation, save config,
  run placebo test, generate report) via the stub harness described above.
- ✅ `python -m py_compile` passed on every source file.

---

## 5. Sample Dataset

`data/sample_data.csv` (also regenerable from the "Upload Dataset" page) is a
**SYNTHETIC** customer-discount dataset with a known, programmed
data-generating process — built specifically so the app has something
realistic to demonstrate confounding and heterogeneous effects on:

| Column | Meaning |
|---|---|
| `customer_id` | Unique customer identifier |
| `age` | Customer age (confounder) |
| `income` | Annual income (confounder) |
| `previous_purchases` | Count of past purchases (confounder) |
| `website_visits` | Recent site visits (confounder) |
| `previous_spending` | Historical spending (confounder) |
| `discount_received` | **Treatment** (0/1) — engaged customers are deliberately *more likely* to receive a discount, creating real confounding |
| `purchase_amount` | **Outcome** — depends on confounders *and* a heterogeneous treatment effect (varies with income & previous purchases) |

**This dataset does not represent real customers or a validated real-world
effect.** It exists purely to demonstrate the causal-inference mechanics
(confounding → biased naive comparison → adjusted causal estimate).

---

## 6. Causal ML Methodology

### Why not just predict the outcome?
A predictive model answers *"what purchase amount do treated customers tend
to have?"*. It cannot answer *"how much MORE would this customer have spent
BECAUSE of the discount?"* — that requires reasoning about the
**counterfactual** (what would have happened otherwise), which is what this
app estimates.

### Estimator selection logic (`src/estimators.py`)
1. If **EconML** is installed and there are ≥200 usable rows, the app tries
   `CausalForestDML` (heterogeneous, cross-fitted machine-learning based DML).
   If it's not installed, or fitting fails for any reason, the app **falls
   back automatically** — it never silently produces a worse number without
   saying so.
2. **Fallback estimator (what will actually run out-of-the-box unless you
   install EconML):**
   - **CATE:** a *T-learner* — two separate outcome models are fit (one on
     the treated group, one on the control group) using Gradient Boosting
     Regression (or Linear Regression with a treatment interaction term in
     small-sample mode), then `CATE(x) = μ₁(x) − μ₀(x)`.
   - **ATE:** *AIPW (Augmented Inverse Propensity Weighting)*, a standard
     **doubly-robust** estimator:
     ```
     ψᵢ = (μ₁(xᵢ) − μ₀(xᵢ)) + Tᵢ·(Yᵢ − μ₁(xᵢ))/e(xᵢ) − (1−Tᵢ)·(Yᵢ − μ₀(xᵢ))/(1−e(xᵢ))
     ATE = mean(ψ),  SE = std(ψ)/√n,  95% CI = ATE ± 1.96·SE
     ```
     where `e(x)` is the propensity score. AIPW is consistent if **either**
     the outcome models or the propensity model is correctly specified —
     this is why it's a defensible fallback rather than an ad hoc shortcut.
3. **Propensity scores** are estimated via `LogisticRegression` on the
   selected confounders, with cross-validated AUC reported so you can judge
   how predictable treatment is from features (very high AUC ⇒ possible
   overlap problems).

### Core concepts explained in-app
- **Treatment, Outcome, Confounder** — defined on the Home page.
- **ATE vs CATE** — defined on the Treatment Effects page.
- **DAG (causal graph)** — an **assumption you declare**, not something
  auto-discovered from correlations (Causal Analysis page).
- **Overlap / positivity** and **covariate balance (SMD)** — diagnostics
  that tell you whether the data can even support a credible estimate.
- **Placebo refutation** — treatment is randomly shuffled and the *same*
  pipeline is re-run; a trustworthy pipeline should show a near-zero effect
  on the shuffled data (Evaluation page). This mirrors DoWhy's
  `placebo_treatment_refuter` and is re-implemented directly so it works
  even without DoWhy installed.
- **Qini / uplift curve** — only labelled "Qini" for a binary outcome; for
  the continuous `purchase_amount` outcome, the app shows the equivalent
  cumulative-gain-by-predicted-uplift-rank curve instead of mislabeling a
  metric that wasn't designed for continuous outcomes.

### Policy optimization (`src/policy_optimization.py`)
```
estimated_benefit = CATE(x) × benefit_ratio     # benefit_ratio configurable by the user,
                                                  # e.g. 0.3 if only 30% of an outcome-unit
                                                  # increase is actually profit margin
net_benefit(x) = estimated_benefit − treatment_cost
recommend(x) = True  if net_benefit(x) > 0 AND CATE(x) ≥ min_effect_threshold
```
An optional **budget** ranks eligible customers by net benefit and treats as
many as the budget allows. **Treat Everyone / Treat No One / Model-Based
Targeting** are compared side-by-side on the Policy Optimization page.

---

## 7. Hackathon Demo Flow (~5 minutes)

1. **Home** — explain the problem statement and correlation-vs-causation.
2. **Upload Dataset** → click "Generate & Load Sample Dataset" (defaults are
   pre-filled: `discount_received` as treatment, `purchase_amount` as
   outcome, all four confounders selected) → click **Validate & Save
   Configuration**.
3. **Data Exploration** — show the outcome-by-treatment box plot and note
   that "treated" customers already look different on confounders (that's
   the confounding the DAG page will name explicitly).
4. **Causal Analysis** — show the DAG, propensity score overlap, and the SMD
   balance table (point out which confounders are meaningfully imbalanced).
5. **Treatment Effects** — click **Run Causal Effect Estimation** → show
   ATE, CI, and the CATE histogram/uplift ranking.
6. **Policy Optimization** — set a treatment cost (e.g. 10), show the
   recommended-customer count and net benefit, then show the
   Treat-All/Treat-None/Model-Based comparison chart.
7. **Evaluation** — run the placebo refutation test live, and show the
   uplift/gain curve.
8. **Download Report** — download the HTML report as the closing artifact.
9. Close by stating the assumptions & limitations (no unmeasured
   confounding, overlap, consistency, SUTVA) — this is the difference
   between this project and a plain regression demo.

---

## 8. Common Errors & Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | Dependencies not installed | `pip install -r requirements.txt` |
| EconML/DoWhy fail to build during `pip install` | Native build toolchain missing on your OS | Remove those two lines from `requirements.txt` and reinstall; the app's fallback estimator will be used automatically (this is expected and clearly labelled, not an error state) |
| "Treatment column must be binary" | Selected treatment has >2 values or is continuous | Pick a 0/1 (or two-value) column as treatment, or recode your data first |
| "Outcome column must be numeric" | Selected outcome is text/categorical | Pick a numeric outcome column |
| "No confounders selected" warnings | Feature list is empty | Select at least one confounder on the Upload Dataset page for an adjusted (not naive) estimate |
| Poor overlap warning | Treatment is highly predictable from features | Either accept the estimate with caution (and say so in your presentation), or restrict analysis to the region of common support |
| "Only N usable rows... using AIPW fallback" | Dataset too small for EconML's cross-fitting | Expected behavior — the fallback is real, documented, and doubly-robust, not an error |

---

## 9. Features Actually Implemented

- ✅ Home dashboard with concept explanations
- ✅ Sample synthetic dataset generator + CSV upload, with clear "synthetic" labeling
- ✅ Dataset summary: shape, missing values, numeric/categorical detection, descriptive stats
- ✅ Treatment/outcome/feature/ID/cost configuration with plain-language validation
- ✅ Exploratory data analysis (treatment & outcome distributions, group comparisons, confounder-by-treatment plots, correlation heatmap)
- ✅ Assumption-based DAG visualization (networkx + matplotlib)
- ✅ Propensity score estimation, overlap diagnostics, covariate balance (SMD) table
- ✅ ATE/CATE estimation: EconML CausalForestDML when installed & data-sufficient, otherwise a transparent AIPW (doubly-robust) + T-learner fallback — never fabricated
- ✅ CATE distribution, treatment-effect-by-feature chart, uplift ranking of top/bottom segments
- ✅ Policy engine: net benefit, configurable benefit ratio, minimum-effect threshold, optional budget constraint, individual-level recommendation table
- ✅ Policy comparison: Treat Everyone / Treat No One / Model-Based Targeting
- ✅ Evaluation: sample counts, propensity AUC, placebo/refutation test, uplift/cumulative-gain curve
- ✅ CSV and HTML report generation/download
- ✅ 28 passing unit tests covering the entire `src/` package

## 10. Known Limitations (Stated Explicitly, Not Hidden)

- The **fallback AIPW estimator** is only as good as the outcome/propensity
  models used inside it (Gradient Boosting / Logistic Regression); it is a
  transparent, defensible baseline, not a state-of-the-art causal ML method.
- **EconML's CausalForestDML confidence intervals** (when that path is used)
  assume the forest is well-specified and should be treated as approximate.
- The **placebo refutation test** is one sanity check, not a full causal
  validity guarantee; DoWhy's additional refuters (e.g. random common cause,
  data-subset refuter) are not implemented separately, since the placebo
  test already re-uses the exact same pipeline logic and covers the most
  common failure mode.
- **Qini score** is only computed/labelled for binary outcomes; for the
  continuous demo outcome, a cumulative-gain analogue is shown instead and
  explicitly not called "Qini" to avoid reporting an invalid metric.
- No causal discovery: the DAG is **assumption-based**, entered implicitly
  via your confounder selection — the app does not attempt to discover
  causal structure from data automatically (this is a stated design choice
  for transparency, not an oversight).
- The bundled sample dataset is **synthetic** and must never be presented as
  a real-world validated causal effect.
- This app was built and tested in a sandboxed environment without internet
  access, so `streamlit`, `plotly`, `econml`, and `dowhy` could not be
  `pip install`-ed there. All causal-ML logic (`src/`) was fully unit-tested
  with real execution (28/28 passing) using the scientific-Python stack that
  *was* available (pandas, NumPy, scikit-learn). Every Streamlit page's code
  path was additionally executed via a stub harness that fakes the
  `streamlit`/`plotly` APIs, to catch logic and reference errors ahead of
  time. **You should still do one `streamlit run app.py` smoke test on your
  own machine before the hackathon demo** — this covers real widget
  rendering, which could not be observed directly in the build environment.
