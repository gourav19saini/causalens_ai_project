"""
causal_analysis.py
-------------------
Building blocks for causal reasoning that sit BEFORE effect estimation:

    1. A simple, user-configured DAG (Directed Acyclic Graph) that makes the
       causal *assumptions* explicit (confounders -> treatment, confounders
       -> outcome, treatment -> outcome). This is an assumption-based DAG,
       not something "discovered" automatically from correlations.
    2. A propensity score model P(Treatment=1 | X) used later for overlap
       diagnostics, IPW and AIPW estimation.
    3. Overlap ("common support") and covariate-balance diagnostics, which
       tell us whether the data can even support a credible causal estimate.

None of these functions "prove" causality on their own -- they support the
assumptions that the rest of the pipeline relies on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------
# DAG (assumption-based, not auto-discovered)
# --------------------------------------------------------------------------
def build_dag_edges(confounders: List[str], treatment: str, outcome: str):
    """
    Returns the edge list of the assumed causal graph:
        confounder -> treatment   (confounders influence who gets treated)
        confounder -> outcome     (confounders directly influence outcome)
        treatment  -> outcome     (the causal effect we want to estimate)

    This graph encodes the standard "backdoor" confounding assumption used
    by adjustment-based and doubly-robust estimators. It is a MODELLING
    ASSUMPTION supplied by the user/analyst, not something derived
    automatically from the data.
    """
    edges = []
    for c in confounders:
        edges.append((c, treatment))
        edges.append((c, outcome))
    edges.append((treatment, outcome))
    return edges


# --------------------------------------------------------------------------
# Propensity score model
# --------------------------------------------------------------------------
@dataclass
class PropensityResult:
    scores: np.ndarray
    model: LogisticRegression
    feature_names: List[str]
    auc: Optional[float] = None


def fit_propensity_model(df: pd.DataFrame, treatment_col: str, feature_cols: List[str]) -> PropensityResult:
    """
    Fits a logistic-regression propensity score model P(T=1 | X).

    Logistic regression is used because it is transparent, stable on small
    hackathon-sized datasets, and its coefficients/scores are easy to sanity
    check -- appropriate for an interpretable diagnostic tool. It is NOT
    claimed to be the most flexible possible propensity model.
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    X = df[feature_cols].copy()
    X = pd.get_dummies(X, drop_first=True)
    y = df[treatment_col].astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    try:
        proba_cv = cross_val_predict(model, X_scaled, y, cv=min(5, max(2, int(y.sum()))), method="predict_proba")[:, 1]
        auc = roc_auc_score(y, proba_cv) if len(np.unique(y)) == 2 else None
    except Exception:  # noqa: BLE001
        proba_cv = model.predict_proba(X_scaled)[:, 1]
        auc = None

    return PropensityResult(scores=proba_cv, model=model, feature_names=list(X.columns), auc=auc)


# --------------------------------------------------------------------------
# Overlap ("common support") diagnostics
# --------------------------------------------------------------------------
@dataclass
class OverlapDiagnostics:
    pct_treated_below_0_1: float
    pct_control_above_0_9: float
    min_treated: float
    max_control: float
    overlap_warning: Optional[str]


def assess_overlap(propensity_scores: np.ndarray, treatment: np.ndarray) -> OverlapDiagnostics:
    treated_scores = propensity_scores[treatment == 1]
    control_scores = propensity_scores[treatment == 0]

    pct_t_low = float(np.mean(treated_scores < 0.1) * 100) if len(treated_scores) else 0.0
    pct_c_high = float(np.mean(control_scores > 0.9) * 100) if len(control_scores) else 0.0

    warning = None
    if pct_t_low > 10 or pct_c_high > 10:
        warning = (
            "Poor overlap detected: a notable share of units have propensity scores near 0 or 1. "
            "This means some individuals are almost never (or almost always) treated given their "
            "characteristics, which violates the positivity/overlap assumption and can make "
            "causal estimates unreliable for those units."
        )

    return OverlapDiagnostics(
        pct_treated_below_0_1=round(pct_t_low, 2),
        pct_control_above_0_9=round(pct_c_high, 2),
        min_treated=float(treated_scores.min()) if len(treated_scores) else np.nan,
        max_control=float(control_scores.max()) if len(control_scores) else np.nan,
        overlap_warning=warning,
    )


# --------------------------------------------------------------------------
# Covariate balance diagnostics
# --------------------------------------------------------------------------
def standardized_mean_difference(df: pd.DataFrame, treatment_col: str, feature_cols: List[str]) -> pd.DataFrame:
    """
    Computes the Standardized Mean Difference (SMD) for each numeric feature
    between treated and control groups.

    SMD = (mean_treated - mean_control) / pooled_std

    A common rule of thumb is that |SMD| > 0.1 indicates meaningful imbalance
    that a naive comparison would confound. This is DESCRIPTIVE -- it tells
    us how different the groups are, not whether treatment caused the
    outcome.
    """
    rows = []
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        m_t, m_c = treated[col].mean(), control[col].mean()
        s_t, s_c = treated[col].std(), control[col].std()
        pooled_std = np.sqrt((s_t**2 + s_c**2) / 2) if (s_t > 0 or s_c > 0) else np.nan
        smd = (m_t - m_c) / pooled_std if pooled_std and pooled_std > 0 else np.nan
        rows.append(
            {
                "feature": col,
                "mean_treated": round(m_t, 3),
                "mean_control": round(m_c, 3),
                "smd": round(smd, 3) if pd.notna(smd) else np.nan,
                "imbalanced": bool(abs(smd) > 0.1) if pd.notna(smd) else False,
            }
        )
    return pd.DataFrame(rows)
