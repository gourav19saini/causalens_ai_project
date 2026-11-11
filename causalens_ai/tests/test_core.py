"""
tests/test_core.py
-------------------
Unit tests for the causal-ML core (everything under src/), independent of
Streamlit. Run with:  pytest tests/test_core.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import pytest

from src.data_processing import generate_sample_dataset, load_csv, summarize_dataset, basic_clean
from src.validation import validate_configuration, validate_treatment, validate_outcome
from src.causal_analysis import build_dag_edges, fit_propensity_model, assess_overlap, standardized_mean_difference
from src.estimators import estimate_causal_effect
from src.policy_optimization import PolicyConfig, individual_recommendations, policy_summary, compare_policies
from src.evaluation import sample_counts, placebo_treatment_refutation, qini_curve
from src.report_generation import build_csv_report, build_html_report


TREATMENT = "discount_received"
OUTCOME = "purchase_amount"
FEATURES = ["age", "income", "previous_purchases", "website_visits", "previous_spending"]


@pytest.fixture(scope="module")
def sample_df():
    return generate_sample_dataset(n=600, seed=1)


# --------------------------------------------------------------------------
# Data processing
# --------------------------------------------------------------------------
def test_sample_dataset_shape(sample_df):
    assert sample_df.shape[0] == 600
    expected_cols = {"customer_id", "age", "income", "previous_purchases",
                      "website_visits", "previous_spending", "discount_received", "purchase_amount"}
    assert expected_cols.issubset(set(sample_df.columns))


def test_sample_dataset_treatment_binary(sample_df):
    assert set(sample_df["discount_received"].unique()).issubset({0, 1})
    assert sample_df["discount_received"].sum() > 0
    assert (1 - sample_df["discount_received"]).sum() > 0


def test_load_csv_roundtrip(tmp_path, sample_df):
    path = tmp_path / "test.csv"
    sample_df.to_csv(path, index=False)
    loaded = load_csv(str(path))
    assert loaded.shape == sample_df.shape


def test_load_csv_empty_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError):
        load_csv(str(path))


def test_summarize_dataset(sample_df):
    summary = summarize_dataset(sample_df)
    assert summary.n_rows == 600
    assert "age" in summary.numeric_columns


def test_basic_clean_imputes_missing(sample_df):
    df = sample_df.copy()
    df.loc[0:5, "age"] = np.nan
    cleaned = basic_clean(df, ["age"])
    assert cleaned["age"].isna().sum() == 0


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def test_validate_configuration_valid(sample_df):
    report = validate_configuration(sample_df, TREATMENT, OUTCOME, FEATURES)
    assert not report.has_errors


def test_validate_treatment_rejects_nonbinary(sample_df):
    df = sample_df.copy()
    df["bad_treatment"] = np.random.randint(0, 3, size=len(df))
    msgs = validate_treatment(df, "bad_treatment")
    assert any(m.level == "error" for m in msgs)


def test_validate_outcome_rejects_non_numeric(sample_df):
    df = sample_df.copy()
    df["bad_outcome"] = "text_value"
    msgs = validate_outcome(df, "bad_outcome")
    assert any(m.level == "error" for m in msgs)


def test_validate_configuration_missing_column(sample_df):
    report = validate_configuration(sample_df, "does_not_exist", OUTCOME, FEATURES)
    assert report.has_errors


# --------------------------------------------------------------------------
# Causal analysis: DAG, propensity, overlap, balance
# --------------------------------------------------------------------------
def test_build_dag_edges():
    edges = build_dag_edges(["age", "income"], "T", "Y")
    assert ("age", "T") in edges
    assert ("age", "Y") in edges
    assert ("T", "Y") in edges
    assert len(edges) == 2 * 2 + 1


def test_fit_propensity_model(sample_df):
    result = fit_propensity_model(sample_df, TREATMENT, FEATURES)
    assert len(result.scores) == len(sample_df)
    assert np.all((result.scores >= 0) & (result.scores <= 1))
    assert result.auc is None or 0 <= result.auc <= 1


def test_assess_overlap(sample_df):
    result = fit_propensity_model(sample_df, TREATMENT, FEATURES)
    overlap = assess_overlap(result.scores, sample_df[TREATMENT].values)
    assert 0 <= overlap.pct_treated_below_0_1 <= 100
    assert 0 <= overlap.pct_control_above_0_9 <= 100


def test_standardized_mean_difference(sample_df):
    smd_df = standardized_mean_difference(sample_df, TREATMENT, FEATURES)
    assert set(["feature", "mean_treated", "mean_control", "smd"]).issubset(smd_df.columns)
    assert len(smd_df) == len(FEATURES)


# --------------------------------------------------------------------------
# Estimators (fallback path -- econml/dowhy not required for these tests)
# --------------------------------------------------------------------------
def test_estimate_causal_effect_fallback(sample_df):
    result = estimate_causal_effect(sample_df, TREATMENT, OUTCOME, FEATURES, prefer_econml=False)
    assert isinstance(result.ate, float)
    assert result.cate is not None
    assert len(result.cate) == len(result.row_index)
    assert result.ate_ci is not None
    assert result.ate_ci[0] <= result.ate <= result.ate_ci[1]
    # Known DGP: true average effect is around 25 + heterogeneity mean, should be positive
    assert result.ate > 0


def test_estimate_causal_effect_no_features(sample_df):
    result = estimate_causal_effect(sample_df, TREATMENT, OUTCOME, [], prefer_econml=False)
    assert "Unadjusted" in result.estimator_name
    assert result.cate is None


def test_estimate_causal_effect_small_sample():
    small_df = generate_sample_dataset(n=80, seed=2)
    result = estimate_causal_effect(small_df, TREATMENT, OUTCOME, FEATURES, prefer_econml=False)
    assert isinstance(result.ate, float)
    assert "small-sample" in result.estimator_name or "Fallback" in result.estimator_name


# --------------------------------------------------------------------------
# Policy optimization
# --------------------------------------------------------------------------
def test_individual_recommendations_basic():
    cate = np.array([50, 5, -10, 100, 0])
    config = PolicyConfig(treatment_cost=20, benefit_ratio=1.0)
    table = individual_recommendations(cate, config)
    assert list(table["recommended"]) == [True, False, False, True, False]


def test_individual_recommendations_budget():
    cate = np.array([50, 40, 30, 20, 10])
    config = PolicyConfig(treatment_cost=10, benefit_ratio=1.0, budget=20)  # only afford 2 treatments
    table = individual_recommendations(cate, config)
    assert table["recommended"].sum() == 2
    # the two highest net-benefit individuals should be chosen
    assert table.loc[0, "recommended"] and table.loc[1, "recommended"]


def test_policy_summary():
    cate = np.array([50, 5, -10, 100, 0])
    config = PolicyConfig(treatment_cost=20, benefit_ratio=1.0)
    table = individual_recommendations(cate, config)
    summary = policy_summary(table)
    assert summary["n_recommended"] == 2
    assert summary["total_cost"] == 40


def test_compare_policies():
    cate = np.array([50, 5, -10, 100, 0])
    config = PolicyConfig(treatment_cost=20, benefit_ratio=1.0)
    comparison = compare_policies(cate, config)
    assert set(comparison["policy"]) == {"Treat Everyone", "Treat No One", "Model-Based Targeting"}
    treat_all_row = comparison[comparison["policy"] == "Treat Everyone"].iloc[0]
    assert treat_all_row["n_treated"] == 5


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def test_sample_counts():
    t = np.array([1, 1, 0, 0, 0])
    counts = sample_counts(t)
    assert counts == {"n_total": 5, "n_treated": 2, "n_control": 3}


def test_placebo_refutation_runs(sample_df):
    refutation = placebo_treatment_refutation(sample_df, TREATMENT, OUTCOME, FEATURES, seed=0)
    assert isinstance(refutation.original_ate, float)
    assert isinstance(refutation.refuted_ate, float)


def test_qini_curve(sample_df):
    result = estimate_causal_effect(sample_df, TREATMENT, OUTCOME, FEATURES, prefer_econml=False)
    curve_df, is_binary = qini_curve(sample_df.loc[result.row_index, OUTCOME].values,
                                      sample_df.loc[result.row_index, TREATMENT].values,
                                      result.cate)
    assert "fraction_targeted" in curve_df.columns
    assert is_binary is False  # purchase_amount is continuous


# --------------------------------------------------------------------------
# Report generation
# --------------------------------------------------------------------------
def test_build_csv_report_runs():
    csv_bytes = build_csv_report(
        {"rows": 100}, {"treatment": TREATMENT}, "Fallback AIPW", 25.0, (20.0, 30.0),
        {"n_recommended": 10}, ["assumption 1"], ["limitation 1"],
    )
    assert isinstance(csv_bytes, bytes)
    assert b"ATE" in csv_bytes


def test_build_html_report_runs():
    html = build_html_report(
        {"rows": 100}, {"treatment": TREATMENT}, "Fallback AIPW", 25.0, (20.0, 30.0),
        {"mean_cate": 25.0}, {"n_recommended": 10}, None, ["assumption 1"], ["limitation 1"], [],
    )
    assert "<html>" in html
    assert "CausaLens AI" in html


# --------------------------------------------------------------------------
# Invalid input handling
# --------------------------------------------------------------------------
def test_estimate_causal_effect_handles_constant_outcome():
    df = generate_sample_dataset(n=100, seed=3)
    df["purchase_amount"] = 50.0  # constant outcome
    report = validate_configuration(df, TREATMENT, OUTCOME, FEATURES)
    assert report.has_errors  # should be flagged before estimation is even attempted


def test_validate_configuration_missing_treatment_outcome():
    df = generate_sample_dataset(n=50, seed=4)
    report = validate_configuration(df, None, None, FEATURES)
    assert report.has_errors
