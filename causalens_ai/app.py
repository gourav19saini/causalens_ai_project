"""
CausaLens AI — Automated Causal Inference & Policy Optimization Engine
=======================================================================
Main Streamlit application entry point.

Run with:  streamlit run app.py

This file only handles UI/navigation/state. All causal-ML logic lives in
the `src/` package so it can be unit-tested independently of Streamlit.
"""

import io

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.causal_analysis import (
    assess_overlap,
    build_dag_edges,
    fit_propensity_model,
    standardized_mean_difference,
)
from src.data_processing import (
    basic_clean,
    generate_sample_dataset,
    load_csv,
    summarize_dataset,
)
from src.estimators import estimate_causal_effect
from src.evaluation import (
    diagnostics_summary,
    placebo_treatment_refutation,
    qini_curve,
    sample_counts,
)
from src.policy_optimization import PolicyConfig, compare_policies, individual_recommendations, policy_summary
from src.report_generation import build_csv_report, build_html_report
from src.validation import validate_configuration

# ---------------------------------------------------------------------------
# Page config & style
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CausaLens AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#6C63FF"
DARK = "#4B3F72"
ACCENT = "#00C2A8"

st.markdown(
    f"""
    <style>
    .metric-card {{
        background: linear-gradient(135deg, {PRIMARY}22, {ACCENT}11);
        border: 1px solid {PRIMARY}44;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }}
    .section-banner {{
        background: #FFF4E5;
        border-left: 5px solid #FFA726;
        padding: 10px 16px;
        border-radius: 4px;
        margin: 10px 0 18px 0;
    }}
    .info-banner {{
        background: #E8F0FE;
        border-left: 5px solid {PRIMARY};
        padding: 10px 16px;
        border-radius: 4px;
        margin: 10px 0 18px 0;
    }}
    h1, h2, h3 {{ color: {DARK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
DEFAULTS = {
    "df": None,
    "dataset_source": None,
    "treatment_col": None,
    "outcome_col": None,
    "feature_cols": [],
    "id_col": None,
    "cost_mode": "fixed",
    "fixed_cost": 10.0,
    "cost_col": None,
    "estimator_result": None,
    "propensity_result": None,
    "config_valid": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"<h2 style='color:{PRIMARY};'>🧭 CausaLens AI</h2>", unsafe_allow_html=True)
st.sidebar.caption("Automated Causal Inference & Policy Optimization Engine")

PAGES = [
    "1. Home",
    "2. Upload Dataset",
    "3. Data Exploration",
    "4. Causal Analysis",
    "5. Treatment Effects",
    "6. Policy Optimization",
    "7. Evaluation",
    "8. Download Report",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.divider()
if st.session_state.df is not None:
    st.sidebar.success(f"Dataset loaded: {st.session_state.df.shape[0]} rows × {st.session_state.df.shape[1]} cols")
    if st.session_state.dataset_source == "synthetic":
        st.sidebar.warning("⚠️ Using SYNTHETIC sample data")
else:
    st.sidebar.info("No dataset loaded yet.")

st.sidebar.divider()
st.sidebar.caption("PS-04 · Automated Causal Inference & Policy Optimization Engine")


# ===========================================================================
# PAGE 1: HOME
# ===========================================================================
if page == "1. Home":
    st.title("🧭 CausaLens AI")
    st.subheader("Automated Causal Inference & Policy Optimization Engine")

    st.markdown(
        """
        <div class="info-banner">
        <b>Problem Statement (PS-04):</b> Estimate the causal effect of interventions,
        identify confounding variables, estimate treatment effects for individuals or
        subgroups, and recommend policies based on estimated benefits and costs — going
        beyond simple outcome prediction to genuine causal reasoning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    for col, label, value in zip(
        [col1, col2, col3, col4],
        ["Core Task", "Method Family", "Interface", "Runs Offline"],
        ["Causal Inference", "DML / AIPW / T-learner", "Streamlit", "✅ Yes"],
    ):
        col.markdown(f"<div class='metric-card'><b>{label}</b><br><span style='font-size:1.3em'>{value}</span></div>", unsafe_allow_html=True)

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.markdown("### 🤔 What is Causal Inference?")
        st.write(
            "Causal inference is the process of estimating **what would happen to an "
            "outcome if we changed a treatment**, rather than just predicting the outcome "
            "as-is. It asks *'what if?'* using the mathematical framework of **potential "
            "outcomes**: for each unit, we imagine both the outcome *had they been treated* "
            "(Y¹) and the outcome *had they not been treated* (Y⁰). We only ever observe one "
            "of the two — the other is the **counterfactual**."
        )
        st.markdown("### ❗ Correlation vs. Causation")
        st.write(
            "A predictive model can tell you that customers who received a discount spent "
            "more — but that could simply mean the business **already targets its best "
            "customers** with discounts (confounding), not that the discount caused the "
            "extra spending. Causal inference tries to strip out that confounding to "
            "isolate the discount's true effect."
        )

    with right:
        st.markdown("### 🧩 Key Definitions")
        st.markdown(
            """
            - **Treatment (T):** The intervention being evaluated (e.g. receiving a discount).
            - **Outcome (Y):** What we care about (e.g. purchase amount).
            - **Confounders (X):** Variables that influence *both* who gets treated and the
              outcome (e.g. a highly engaged customer is both more likely to get a discount
              *and* more likely to spend more, regardless of the discount).
            - **ATE:** Average Treatment Effect across the whole population.
            - **CATE:** Conditional Average Treatment Effect — the effect for a specific
              individual or subgroup, given their features.
            - **Propensity score:** P(Treatment = 1 | X) — how likely a unit was to be
              treated given its characteristics.
            """
        )
        st.markdown("### ⚙️ How This System Works")
        st.markdown(
            """
            1. Load / upload data → 2. Configure treatment, outcome & confounders →
            3. Explore descriptive patterns → 4. Declare causal assumptions (DAG) →
            5. Estimate propensity scores & check overlap → 6. Estimate ATE / CATE →
            7. Turn effects into a treat/no-treat **policy** using cost-benefit logic →
            8. Compare policies & check robustness → 9. Export a report.
            """
        )

    st.markdown("---")
    st.markdown("### 📦 Sample Use Case: Customer Discount Optimization")
    st.write(
        "*Should we give a discount to a given customer?* We estimate, for each customer, "
        "how much **extra** they would spend because of the discount (not just how much "
        "customers who got discounts spent), and weigh that estimated benefit against the "
        "cost of the discount to decide who should receive one."
    )
    st.markdown(
        "<div class='section-banner'>➡️ Head to <b>2. Upload Dataset</b> in the sidebar to get started, "
        "or load the built-in synthetic sample dataset there for a quick demo.</div>",
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE 2: UPLOAD DATASET
# ===========================================================================
elif page == "2. Upload Dataset":
    st.title("📂 Upload Dataset")

    tab1, tab2 = st.tabs(["Use Sample Dataset", "Upload Your Own CSV"])

    with tab1:
        st.markdown(
            "<div class='section-banner'>⚠️ This is a <b>SYNTHETIC</b> dataset generated with a known "
            "data-generating process, for demonstration only. It does NOT represent a real, "
            "validated causal effect.</div>",
            unsafe_allow_html=True,
        )
        n_rows = st.slider("Number of synthetic customers to generate", 200, 5000, 2000, step=100)
        if st.button("Generate & Load Sample Dataset", type="primary"):
            st.session_state.df = generate_sample_dataset(n=n_rows)
            st.session_state.dataset_source = "synthetic"
            st.session_state.treatment_col = "discount_received"
            st.session_state.outcome_col = "purchase_amount"
            st.session_state.feature_cols = ["age", "income", "previous_purchases", "website_visits", "previous_spending"]
            st.session_state.id_col = "customer_id"
            st.success(f"Synthetic sample dataset generated: {n_rows} rows.")

    with tab2:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is not None:
            try:
                df = load_csv(uploaded)
                st.session_state.df = df
                st.session_state.dataset_source = "uploaded"
                st.session_state.treatment_col = None
                st.session_state.outcome_col = None
                st.session_state.feature_cols = []
                st.success(f"Loaded '{uploaded.name}': {df.shape[0]} rows × {df.shape[1]} columns.")
            except ValueError as e:
                st.error(str(e))

    st.markdown("---")

    if st.session_state.df is not None:
        df = st.session_state.df
        summary = summarize_dataset(df)

        st.markdown("### 👀 Dataset Preview")
        st.dataframe(df.head(20), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", summary.n_rows)
        c2.metric("Columns", summary.n_cols)
        c3.metric("Numeric Columns", len(summary.numeric_columns))
        c4.metric("Categorical Columns", len(summary.categorical_columns))

        st.markdown("### 🧮 Missing Value Summary")
        missing_df = pd.DataFrame(
            {"missing_count": summary.missing_counts, "missing_pct": summary.missing_pct}
        )
        missing_df = missing_df[missing_df["missing_count"] > 0]
        if missing_df.empty:
            st.success("No missing values detected.")
        else:
            st.dataframe(missing_df, use_container_width=True)
            st.info("Missing values in selected treatment/outcome/feature columns will be handled during configuration.")

        st.markdown("### 📊 Descriptive Statistics")
        st.dataframe(summary.describe, use_container_width=True)

        st.markdown("---")
        st.markdown("## ⚙️ Treatment / Outcome / Feature Configuration")

        cols = list(df.columns)
        binary_like_cols = [
            c for c in cols
            if df[c].dropna().nunique() == 2
        ]

        cfg1, cfg2 = st.columns(2)
        with cfg1:
            default_t = st.session_state.treatment_col if st.session_state.treatment_col in cols else (binary_like_cols[0] if binary_like_cols else cols[0])
            treatment_col = st.selectbox("Treatment column (binary)", options=cols, index=cols.index(default_t) if default_t in cols else 0)
            id_col = st.selectbox("Customer/ID column (optional)", options=["<None>"] + cols,
                                   index=(["<None>"] + cols).index(st.session_state.id_col) if st.session_state.id_col in cols else 0)
        with cfg2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            default_o = st.session_state.outcome_col if st.session_state.outcome_col in numeric_cols else (numeric_cols[0] if numeric_cols else cols[0])
            outcome_col = st.selectbox("Outcome column (numeric)", options=numeric_cols if numeric_cols else cols,
                                        index=(numeric_cols.index(default_o) if numeric_cols and default_o in numeric_cols else 0))

        candidate_features = [c for c in cols if c not in (treatment_col, outcome_col, id_col)]
        default_features = [f for f in st.session_state.feature_cols if f in candidate_features] or candidate_features
        feature_cols = st.multiselect("Confounder / feature columns", options=candidate_features, default=default_features)

        st.markdown("#### 💰 Treatment Cost")
        cost_mode = st.radio("How is treatment cost specified?", ["Fixed cost per unit", "From a column"], horizontal=True)
        if cost_mode == "Fixed cost per unit":
            fixed_cost = st.number_input("Fixed treatment cost (in outcome units, e.g. $)", min_value=0.0, value=float(st.session_state.fixed_cost), step=1.0)
            st.session_state.cost_mode = "fixed"
            st.session_state.fixed_cost = fixed_cost
            st.session_state.cost_col = None
        else:
            cost_candidates = [c for c in numeric_cols if c not in (outcome_col,)]
            cost_col = st.selectbox("Cost column", options=cost_candidates) if cost_candidates else None
            st.session_state.cost_mode = "column"
            st.session_state.cost_col = cost_col

        if st.button("Validate & Save Configuration", type="primary"):
            st.session_state.treatment_col = treatment_col
            st.session_state.outcome_col = outcome_col
            st.session_state.feature_cols = feature_cols
            st.session_state.id_col = None if id_col == "<None>" else id_col

            report = validate_configuration(df, treatment_col, outcome_col, feature_cols)
            for m in report.messages:
                if m.level == "error":
                    st.error(m.message)
                elif m.level == "warning":
                    st.warning(m.message)
                elif m.level == "success":
                    st.success(m.message)
                else:
                    st.info(m.message)
            st.session_state.config_valid = not report.has_errors
    else:
        st.info("Load the sample dataset or upload a CSV to continue.")


# ===========================================================================
# PAGE 3: DATA EXPLORATION
# ===========================================================================
elif page == "3. Data Exploration":
    st.title("🔍 Exploratory Data Analysis")
    st.caption("Descriptive only — none of the charts on this page establish causation by themselves.")

    if st.session_state.df is None or not st.session_state.config_valid:
        st.warning("Please load data and validate a configuration on the **Upload Dataset** page first.")
    else:
        df = st.session_state.df
        t_col, o_col, f_cols = st.session_state.treatment_col, st.session_state.outcome_col, st.session_state.feature_cols
        clean_df = basic_clean(df, [t_col, o_col] + f_cols)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Treatment Distribution")
            vc = clean_df[t_col].value_counts().rename_axis(t_col).reset_index(name="count")
            fig = px.bar(vc, x=t_col, y="count", color=t_col, title=f"Distribution of {t_col}")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("#### Outcome Distribution")
            fig = px.histogram(clean_df, x=o_col, nbins=40, title=f"Distribution of {o_col}")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Outcome by Treatment Group (Descriptive Only)")
        fig = px.box(clean_df, x=t_col, y=o_col, color=t_col, points="outliers",
                     title=f"{o_col} by {t_col} — descriptive comparison, NOT a causal effect")
        st.plotly_chart(fig, use_container_width=True)

        group_stats = clean_df.groupby(t_col)[o_col].agg(["count", "mean", "median", "std"]).reset_index()
        st.dataframe(group_stats, use_container_width=True)
        st.markdown(
            "<div class='section-banner'>The difference in group means above is a <b>naive/associational</b> "
            "comparison. It can be misleading if treated and control customers differ systematically "
            "(confounding) — see the Causal Analysis page.</div>",
            unsafe_allow_html=True,
        )

        if f_cols:
            st.markdown("#### Confounder Distributions by Treatment Group")
            feat_pick = st.selectbox("Choose a feature to inspect", f_cols)
            if pd.api.types.is_numeric_dtype(clean_df[feat_pick]):
                fig = px.histogram(clean_df, x=feat_pick, color=clean_df[t_col].astype(str),
                                    barmode="overlay", opacity=0.6, title=f"{feat_pick} distribution by treatment group")
            else:
                ct = pd.crosstab(clean_df[feat_pick], clean_df[t_col])
                fig = px.bar(ct, barmode="group", title=f"{feat_pick} counts by treatment group")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Feature Correlations (Descriptive)")
            numeric_f = [c for c in f_cols if pd.api.types.is_numeric_dtype(clean_df[c])]
            if numeric_f:
                corr = clean_df[numeric_f + [o_col]].corr()
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation matrix (association, not causation)")
                st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 4: CAUSAL ANALYSIS (DAG + Propensity)
# ===========================================================================
elif page == "4. Causal Analysis":
    st.title("🧠 Causal Analysis: Assumptions & Propensity Scores")

    if st.session_state.df is None or not st.session_state.config_valid:
        st.warning("Please load data and validate a configuration on the **Upload Dataset** page first.")
    else:
        df = st.session_state.df
        t_col, o_col, f_cols = st.session_state.treatment_col, st.session_state.outcome_col, st.session_state.feature_cols
        clean_df = basic_clean(df, [t_col, o_col] + f_cols)

        st.markdown("## 🗺️ Causal Graph (DAG)")
        st.markdown(
            "<div class='info-banner'>This graph encodes an <b>assumption</b>, not a discovery: we assume the "
            "selected features are confounders that affect both treatment and outcome. If this assumption "
            "is wrong (e.g. an important confounder is missing), the effect estimate below can be biased.</div>",
            unsafe_allow_html=True,
        )

        edges = build_dag_edges(f_cols, t_col, o_col)
        G = nx.DiGraph()
        G.add_edges_from(edges)

        pos = {}
        pos[t_col] = (0, 0)
        pos[o_col] = (2, 0)
        for i, c in enumerate(f_cols):
            pos[c] = (1, 1.2 - i * (2.4 / max(len(f_cols) - 1, 1)) if len(f_cols) > 1 else 0.6)

        fig, ax = plt.subplots(figsize=(8, max(4, 0.9 * len(f_cols))))
        node_colors = []
        for n in G.nodes():
            if n == t_col:
                node_colors.append("#6C63FF")
            elif n == o_col:
                node_colors.append("#00C2A8")
            else:
                node_colors.append("#FFA726")
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2200, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, font_color="white", ax=ax)
        nx.draw_networkx_edges(G, pos, arrowstyle="-|>", arrowsize=18, connectionstyle="arc3,rad=0.05", ax=ax)
        ax.axis("off")
        st.pyplot(fig)
        st.caption("🟣 Treatment · 🟢 Outcome · 🟠 Confounders")

        st.markdown("---")
        st.markdown("## 🎯 Propensity Score Analysis")
        st.write("Estimating P(Treatment = 1 | Confounders) using logistic regression.")

        if not f_cols:
            st.error("No confounders selected — propensity scores cannot be estimated. Go back and select at least one feature.")
        else:
            with st.spinner("Fitting propensity model..."):
                prop_result = fit_propensity_model(clean_df, t_col, f_cols)
            st.session_state.propensity_result = prop_result

            c1, c2 = st.columns(2)
            with c1:
                if prop_result.auc is not None:
                    st.metric("Propensity Model AUC (cross-validated)", f"{prop_result.auc:.3f}")
                    if prop_result.auc > 0.9:
                        st.warning("Very high AUC — treatment is highly predictable from features, which can signal poor overlap.")
            with c2:
                st.metric("Features used", len(prop_result.feature_names))

            plot_df = pd.DataFrame({
                "propensity_score": prop_result.scores,
                t_col: clean_df[t_col].astype(str).values,
            })
            fig = px.histogram(plot_df, x="propensity_score", color=t_col, barmode="overlay", opacity=0.6, nbins=40,
                                title="Propensity Score Distribution & Overlap")
            st.plotly_chart(fig, use_container_width=True)

            overlap = assess_overlap(prop_result.scores, clean_df[t_col].astype(int).values)
            c1, c2, c3 = st.columns(3)
            c1.metric("% Treated with score < 0.1", f"{overlap.pct_treated_below_0_1}%")
            c2.metric("% Control with score > 0.9", f"{overlap.pct_control_above_0_9}%")
            c3.metric("Min treated / Max control", f"{overlap.min_treated:.2f} / {overlap.max_control:.2f}")
            if overlap.overlap_warning:
                st.warning(overlap.overlap_warning)
            else:
                st.success("Overlap looks reasonable across treatment groups.")

            st.markdown("### ⚖️ Covariate Balance (Standardized Mean Difference)")
            smd_df = standardized_mean_difference(clean_df, t_col, f_cols)
            if not smd_df.empty:
                st.dataframe(smd_df, use_container_width=True)
                n_imbalanced = int(smd_df["imbalanced"].sum())
                if n_imbalanced > 0:
                    st.warning(f"{n_imbalanced} feature(s) show meaningful imbalance (|SMD| > 0.1) between treatment groups — real confounding is present, which is exactly what the causal estimator on the next page needs to adjust for.")
                else:
                    st.info("No numeric features show strong imbalance by this rule of thumb.")
            else:
                st.info("No numeric features available for SMD balance check.")


# ===========================================================================
# PAGE 5: TREATMENT EFFECTS (ATE / CATE / Uplift)
# ===========================================================================
elif page == "5. Treatment Effects":
    st.title("📈 Treatment Effect Estimation")

    if st.session_state.df is None or not st.session_state.config_valid:
        st.warning("Please load data and validate a configuration on the **Upload Dataset** page first.")
    elif not st.session_state.feature_cols:
        st.error("No confounders selected. Please select at least one feature column on the Upload page.")
    else:
        df = st.session_state.df
        t_col, o_col, f_cols = st.session_state.treatment_col, st.session_state.outcome_col, st.session_state.feature_cols
        clean_df = basic_clean(df, [t_col, o_col] + f_cols)

        prefer_econml = st.checkbox("Prefer EconML (CausalForestDML) when available and data is sufficient", value=True)

        if st.button("▶️ Run Causal Effect Estimation", type="primary"):
            with st.spinner("Estimating causal effect... this may take a moment."):
                try:
                    result = estimate_causal_effect(clean_df, t_col, o_col, f_cols, prefer_econml=prefer_econml)
                    st.session_state.estimator_result = result
                except Exception as e:  # noqa: BLE001
                    st.error(f"Estimation failed: {e}")
                    st.session_state.estimator_result = None

        result = st.session_state.estimator_result
        if result is None:
            st.info("Click 'Run Causal Effect Estimation' to compute ATE / CATE.")
        else:
            st.markdown(f"### Estimator used: `{result.estimator_name}`")

            c1, c2, c3 = st.columns(3)
            c1.markdown(f"<div class='metric-card'><b>ATE</b><br><span style='font-size:1.6em'>{result.ate:.3f}</span></div>", unsafe_allow_html=True)
            if result.ate_ci:
                c2.markdown(f"<div class='metric-card'><b>95% CI</b><br><span style='font-size:1.3em'>({result.ate_ci[0]:.3f}, {result.ate_ci[1]:.3f})</span></div>", unsafe_allow_html=True)
            else:
                c2.markdown("<div class='metric-card'><b>95% CI</b><br>Not available</div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><b>Units of effect</b><br>{o_col}</div>", unsafe_allow_html=True)

            st.info(
                f"**Interpretation:** On average, receiving the treatment ('{t_col}') is estimated to change "
                f"'{o_col}' by **{result.ate:.3f} units**, holding the selected confounders fixed, under the "
                f"assumptions listed below."
            )

            if result.warnings:
                for w in result.warnings:
                    st.warning(w)

            with st.expander("📋 Assumptions this estimate relies on"):
                for a in result.assumptions:
                    st.write(f"- {a}")
            with st.expander("⚠️ Known limitations of this estimate"):
                for l in result.limitations:  # noqa: E741
                    st.write(f"- {l}")

            st.markdown("---")

            if result.cate is not None:
                st.markdown("## 🎯 CATE / Individual Treatment Effect Analysis")
                st.caption("ATE = Average Treatment Effect (one number for everyone). CATE = Conditional Average "
                           "Treatment Effect (varies by individual/subgroup characteristics). These are ESTIMATED "
                           "quantities from a model, not guaranteed outcomes for any specific person.")

                cate_df = clean_df.loc[result.row_index].copy()
                cate_df["estimated_treatment_effect"] = result.cate

                c1, c2, c3 = st.columns(3)
                c1.metric("Mean CATE", f"{np.mean(result.cate):.3f}")
                c2.metric("Std Dev of CATE", f"{np.std(result.cate):.3f}")
                c3.metric("Range", f"{result.cate.min():.2f} to {result.cate.max():.2f}")

                fig = px.histogram(cate_df, x="estimated_treatment_effect", nbins=40,
                                    title="Distribution of Estimated Individual Treatment Effects")
                fig.add_vline(x=result.ate, line_dash="dash", line_color="red", annotation_text="ATE")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### Treatment Effect by Feature")
                feat_pick = st.selectbox("Choose a feature to segment by", f_cols, key="cate_feat")
                if pd.api.types.is_numeric_dtype(cate_df[feat_pick]):
                    fig = px.scatter(cate_df, x=feat_pick, y="estimated_treatment_effect", trendline="lowess",
                                      title=f"Estimated Treatment Effect vs. {feat_pick}", opacity=0.5)
                else:
                    grp = cate_df.groupby(feat_pick)["estimated_treatment_effect"].mean().reset_index()
                    fig = px.bar(grp, x=feat_pick, y="estimated_treatment_effect",
                                 title=f"Mean Estimated Treatment Effect by {feat_pick}")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("#### 🏆 Uplift Ranking: High vs. Low Benefit Segments")
                n_show = min(10, len(cate_df))
                display_cols = [st.session_state.id_col] if st.session_state.id_col else []
                display_cols += f_cols + ["estimated_treatment_effect"]
                display_cols = [c for c in dict.fromkeys(display_cols) if c in cate_df.columns]

                top_col, bottom_col = st.columns(2)
                with top_col:
                    st.markdown("**Top segments (highest estimated benefit)**")
                    st.dataframe(cate_df.sort_values("estimated_treatment_effect", ascending=False)[display_cols].head(n_show), use_container_width=True)
                with bottom_col:
                    st.markdown("**Bottom segments (lowest / negative estimated benefit)**")
                    st.dataframe(cate_df.sort_values("estimated_treatment_effect", ascending=True)[display_cols].head(n_show), use_container_width=True)
            else:
                st.info("The estimator used here (naive mean-difference) does not support individual-level CATE.")


# ===========================================================================
# PAGE 6: POLICY OPTIMIZATION
# ===========================================================================
elif page == "6. Policy Optimization":
    st.title("💡 Policy Optimization & Cost-Benefit Analysis")

    result = st.session_state.estimator_result
    if result is None or result.cate is None:
        st.warning("Run causal estimation with a CATE-producing estimator on the **Treatment Effects** page first.")
    else:
        df = st.session_state.df
        t_col, o_col, f_cols = st.session_state.treatment_col, st.session_state.outcome_col, st.session_state.feature_cols
        clean_df = basic_clean(df, [t_col, o_col] + f_cols).loc[result.row_index]

        st.markdown("### ⚙️ Policy Settings")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.session_state.cost_mode == "fixed":
                cost = st.number_input("Treatment cost per unit", min_value=0.0, value=float(st.session_state.fixed_cost), step=1.0)
            else:
                cost = float(clean_df[st.session_state.cost_col].mean()) if st.session_state.cost_col else 0.0
                st.metric("Avg. treatment cost (from column)", f"{cost:.2f}")
        with c2:
            benefit_ratio = st.slider(
                "Benefit ratio (fraction of the outcome-unit effect counted as real benefit)",
                min_value=0.0, max_value=1.0, value=1.0, step=0.05,
                help="E.g. if 'purchase_amount' has a 30% margin, set this to 0.3 so net benefit reflects profit, not revenue.",
            )
        with c3:
            min_effect = st.number_input("Minimum required estimated effect to treat", value=0.0, step=1.0)

        use_budget = st.checkbox("Apply a total budget constraint")
        budget = st.number_input("Total budget", min_value=0.0, value=1000.0, step=50.0) if use_budget else None

        config = PolicyConfig(treatment_cost=cost, benefit_ratio=benefit_ratio, min_effect_threshold=min_effect, budget=budget)
        table = individual_recommendations(result.cate, config)
        summary = policy_summary(table)

        st.markdown("### 📊 Policy Recommendation Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers Recommended for Treatment", summary["n_recommended"])
        c2.metric("% of Population", f"{summary['pct_recommended']}%")
        c3.metric("Total Estimated Benefit", f"{summary['total_estimated_benefit']:.2f}")
        c4.metric("Total Estimated Net Benefit", f"{summary['total_net_benefit']:.2f}")
        if summary["budget_binding"]:
            st.warning("Budget constraint is binding: not all eligible customers can be treated within budget. Customers are ranked by estimated net benefit.")

        st.markdown(
            f"<div class='info-banner'>Net benefit = (estimated effect × benefit ratio) − treatment cost. "
            f"With benefit ratio = {benefit_ratio}, a 1-unit increase in '{o_col}' is counted as "
            f"{benefit_ratio} units of benefit.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### 📋 Individual-Level Recommendations")
        display_df = clean_df.copy()
        display_df["estimated_effect"] = table["estimated_effect"].values
        display_df["net_benefit"] = table["net_benefit"].values
        display_df["recommended"] = table["recommended"].values
        cols_to_show = ([st.session_state.id_col] if st.session_state.id_col else []) + ["estimated_effect", "net_benefit", "recommended"]
        cols_to_show = [c for c in dict.fromkeys(cols_to_show) if c in display_df.columns]
        st.dataframe(display_df[cols_to_show].sort_values("net_benefit", ascending=False), use_container_width=True)

        st.markdown("---")
        st.markdown("## ⚔️ Policy Comparison")
        comparison_df = compare_policies(result.cate, config)
        st.dataframe(comparison_df, use_container_width=True)

        fig = px.bar(comparison_df, x="policy", y="estimated_net_benefit", color="policy",
                     title="Estimated Net Benefit by Policy")
        st.plotly_chart(fig, use_container_width=True)

        if comparison_df["budget_violation"].any():
            st.warning("'Treat Everyone' would violate the specified budget.")

        st.markdown(
            "<div class='section-banner'>These figures are <b>model-based estimates under stated assumptions</b>, "
            "not guaranteed business outcomes. Observational data limitations (e.g. unmeasured confounding) "
            "can bias these numbers.</div>",
            unsafe_allow_html=True,
        )

        st.session_state.policy_table = table
        st.session_state.policy_summary = summary
        st.session_state.policy_comparison = comparison_df
        st.session_state.policy_config = config


# ===========================================================================
# PAGE 7: EVALUATION
# ===========================================================================
elif page == "7. Evaluation":
    st.title("🧪 Evaluation & Diagnostics")

    result = st.session_state.estimator_result
    if result is None:
        st.warning("Run causal estimation on the **Treatment Effects** page first.")
    else:
        df = st.session_state.df
        t_col, o_col, f_cols = st.session_state.treatment_col, st.session_state.outcome_col, st.session_state.feature_cols
        clean_df = basic_clean(df, [t_col, o_col] + f_cols)

        st.markdown("### 👥 Sample Counts")
        counts = sample_counts(clean_df[t_col].values)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total", counts["n_total"])
        c2.metric("Treated", counts["n_treated"])
        c3.metric("Control", counts["n_control"])

        st.markdown("### 📝 Diagnostic Notes")
        notes = diagnostics_summary(clean_df, t_col, o_col, f_cols, result.diagnostics.get("propensity_auc"))
        for n in notes:
            st.write(f"- {n}")

        st.markdown("---")
        st.markdown("### 🧪 Placebo / Refutation Test")
        st.caption("Shuffles the treatment column at random and re-runs the SAME estimation pipeline. "
                   "A trustworthy pipeline should show a placebo effect near zero.")
        if st.button("Run Placebo Refutation Test"):
            with st.spinner("Running placebo test (re-estimating on shuffled treatment)..."):
                refutation = placebo_treatment_refutation(clean_df, t_col, o_col, f_cols)
            c1, c2 = st.columns(2)
            c1.metric("Original ATE", f"{refutation.original_ate:.3f}")
            c2.metric("Placebo (shuffled) ATE", f"{refutation.refuted_ate:.3f}")
            if refutation.passed:
                st.success(refutation.note)
            else:
                st.error(refutation.note)

        st.markdown("---")
        st.markdown("### 📈 Uplift / Gain Curve")
        if result.cate is not None:
            outcome_vals = clean_df.loc[result.row_index, o_col].values
            treatment_vals = clean_df.loc[result.row_index, t_col].astype(int).values
            curve_df, is_binary = qini_curve(outcome_vals, treatment_vals, result.cate)
            if is_binary:
                st.caption("Outcome is binary — this is a standard Qini-style cumulative gain curve.")
            else:
                st.caption("Outcome is continuous, so a binary-outcome 'Qini score' would not be a valid metric here. "
                           "Showing the analogous cumulative-gain-by-predicted-uplift-rank curve instead.")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve_df["fraction_targeted"], y=curve_df["cumulative_gain"], mode="lines", name="Model targeting"))
            fig.update_layout(title="Cumulative Gain by Fraction Targeted (ranked by estimated uplift)",
                               xaxis_title="Fraction of population targeted", yaxis_title="Cumulative incremental outcome")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("CATE is required for uplift evaluation; the current estimator does not provide it.")

        st.markdown("---")
        st.markdown(
            "<div class='section-banner'><b>Causal assumptions this whole analysis relies on:</b><br>"
            "1) No unmeasured confounding &nbsp;·&nbsp; 2) Positivity/overlap &nbsp;·&nbsp; "
            "3) Consistency &nbsp;·&nbsp; 4) SUTVA. None of these can be fully verified from data alone.</div>",
            unsafe_allow_html=True,
        )


# ===========================================================================
# PAGE 8: DOWNLOAD REPORT
# ===========================================================================
elif page == "8. Download Report":
    st.title("📥 Download Report")

    result = st.session_state.estimator_result
    if st.session_state.df is None:
        st.warning("Load a dataset first.")
    elif result is None:
        st.warning("Run causal estimation on the **Treatment Effects** page first.")
    else:
        df = st.session_state.df
        t_col, o_col, f_cols = st.session_state.treatment_col, st.session_state.outcome_col, st.session_state.feature_cols

        dataset_info = {
            "Source": st.session_state.dataset_source,
            "Rows": df.shape[0],
            "Columns": df.shape[1],
        }
        config_info = {
            "Treatment column": t_col,
            "Outcome column": o_col,
            "Feature/confounder columns": ", ".join(f_cols),
        }

        cate_summary = None
        if result.cate is not None:
            cate_summary = {
                "Mean CATE": round(float(np.mean(result.cate)), 4),
                "Std Dev CATE": round(float(np.std(result.cate)), 4),
                "Min CATE": round(float(np.min(result.cate)), 4),
                "Max CATE": round(float(np.max(result.cate)), 4),
            }

        policy_sum = st.session_state.get("policy_summary")
        policy_comp = st.session_state.get("policy_comparison")

        if policy_sum is None:
            st.info("Tip: visit the Policy Optimization page first to include policy results in the report.")
            policy_sum = {"note": "Policy optimization was not run in this session."}

        st.markdown("### Report Preview")
        st.json({"dataset_info": dataset_info, "config": config_info, "ate": result.ate, "policy_summary": policy_sum})

        csv_bytes = build_csv_report(
            dataset_info, config_info, result.estimator_name, result.ate, result.ate_ci,
            policy_sum, result.assumptions, result.limitations,
        )
        html_report = build_html_report(
            dataset_info, config_info, result.estimator_name, result.ate, result.ate_ci,
            cate_summary, policy_sum, policy_comp, result.assumptions, result.limitations, result.warnings,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download CSV Report", data=csv_bytes, file_name="causalens_report.csv", mime="text/csv")
        with c2:
            st.download_button("⬇️ Download HTML Report", data=html_report.encode("utf-8"), file_name="causalens_report.html", mime="text/html")

        st.markdown(
            "<div class='section-banner'>This report documents estimated effects under explicit statistical "
            "assumptions. It is not proof of real-world causality, particularly for the synthetic demo dataset.</div>",
            unsafe_allow_html=True,
        )
