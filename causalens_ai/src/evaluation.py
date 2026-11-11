"""
evaluation.py
-------------
Diagnostics that tell us how much to TRUST a causal estimate, plus optional
uplift evaluation. Everything here is either a standard, documented method
or is explicitly skipped with a stated reason -- nothing is fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from .estimators import estimate_causal_effect


@dataclass
class RefutationResult:
    test_name: str
    original_ate: float
    refuted_ate: float
    passed: bool
    note: str


def sample_counts(treatment: np.ndarray) -> dict:
    treatment = np.asarray(treatment)
    return {
        "n_total": int(len(treatment)),
        "n_treated": int(treatment.sum()),
        "n_control": int((1 - treatment).sum()),
    }


def placebo_treatment_refutation(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: List[str],
    seed: int = 0,
) -> RefutationResult:
    """
    Placebo-treatment refutation test:
        Randomly shuffle the treatment assignment (breaking any real
        treatment-outcome relationship) and re-estimate the "effect" using
        the SAME pipeline. If the pipeline is behaving sensibly, this
        placebo effect should be close to zero and much smaller in
        magnitude than the original estimate.

    This is the same logic used by DoWhy's `placebo_treatment_refuter`;
    it is re-implemented directly here so the check works even when DoWhy
    is not installed.
    """
    original = estimate_causal_effect(df, treatment_col, outcome_col, feature_cols, prefer_econml=False)

    rng = np.random.default_rng(seed)
    shuffled_df = df.copy()
    shuffled_df[treatment_col] = rng.permutation(shuffled_df[treatment_col].values)

    placebo = estimate_causal_effect(shuffled_df, treatment_col, outcome_col, feature_cols, prefer_econml=False)

    threshold = max(abs(original.ate) * 0.5, 1e-6)
    passed = abs(placebo.ate) < threshold

    note = (
        "The placebo (randomly shuffled treatment) effect is much smaller than the original "
        "estimate, which is consistent with the pipeline detecting a genuine treatment-outcome "
        "association rather than an artifact of the modelling procedure."
        if passed
        else "The placebo effect is NOT much smaller than the original estimate. This is a "
        "warning sign: the pipeline may be picking up structure unrelated to the actual "
        "treatment (e.g. overfitting), so the original estimate should be treated with extra caution."
    )

    return RefutationResult(
        test_name="Placebo Treatment Refutation",
        original_ate=original.ate,
        refuted_ate=placebo.ate,
        passed=passed,
        note=note,
    )


def qini_curve(outcome: np.ndarray, treatment: np.ndarray, cate: np.ndarray) -> Optional[pd.DataFrame]:
    """
    Computes a simple Qini curve when, and only when, the outcome is binary
    (e.g. conversion 0/1). For continuous outcomes (like purchase_amount)
    a "cumulative gain by predicted-uplift rank" curve is returned instead,
    which is the closest well-defined analogue -- we do NOT report a
    binary-outcome Qini score on a continuous outcome, since that would be
    a fabricated metric.
    """
    outcome = np.asarray(outcome, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    cate = np.asarray(cate, dtype=float)

    order = np.argsort(-cate)
    outcome_sorted = outcome[order]
    treatment_sorted = treatment[order]

    n = len(outcome)
    cum_treated_outcome = np.cumsum(outcome_sorted * treatment_sorted)
    cum_treated_n = np.cumsum(treatment_sorted)
    cum_control_outcome = np.cumsum(outcome_sorted * (1 - treatment_sorted))
    cum_control_n = np.cumsum(1 - treatment_sorted)

    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(
            cum_control_n > 0,
            cum_treated_outcome - cum_control_outcome * (cum_treated_n / np.maximum(cum_control_n, 1)),
            np.nan,
        )

    frac = np.arange(1, n + 1) / n
    is_binary_outcome = set(np.unique(outcome)).issubset({0, 1})

    return pd.DataFrame(
        {
            "fraction_targeted": frac,
            "cumulative_gain": gain,
        }
    ), is_binary_outcome


def diagnostics_summary(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: List[str],
    propensity_auc: Optional[float],
) -> List[str]:
    notes = []
    counts = sample_counts(df[treatment_col].values)
    notes.append(f"Sample sizes -- total: {counts['n_total']}, treated: {counts['n_treated']}, control: {counts['n_control']}.")

    if propensity_auc is not None:
        if propensity_auc > 0.9:
            notes.append(
                f"Propensity model AUC is very high ({propensity_auc:.2f}), meaning treatment is "
                "highly predictable from features. This can indicate poor overlap -- check the "
                "overlap diagnostics before trusting the estimate."
            )
        else:
            notes.append(f"Propensity model AUC: {propensity_auc:.2f} (how well treatment can be predicted from features).")

    notes.append(
        "Causal inference here relies on standard assumptions (no unmeasured confounding, "
        "positivity/overlap, consistency, SUTVA) that CANNOT be fully verified from the data alone."
    )
    return notes
