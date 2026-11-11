"""
data_processing.py
-------------------
Utilities for loading data (sample or user-uploaded CSV), and producing
descriptive summaries used throughout the CausaLens AI app.

No causal claims are made in this module -- it is purely descriptive.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

RANDOM_SEED = 42


# --------------------------------------------------------------------------
# Sample (SYNTHETIC) dataset generator
# --------------------------------------------------------------------------
def generate_sample_dataset(n: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generates a SYNTHETIC customer-discount dataset with a known,
    programmed data-generating process (DGP).

    Because we control the DGP, this dataset is useful for demonstrating
    causal inference mechanics (confounding, treatment effect estimation),
    but it is NOT real-world data and does NOT represent a validated
    real-world causal effect. This is clearly labelled everywhere it is
    used in the app.

    DGP summary (for transparency):
        - age, income, previous_purchases, website_visits are baseline
          covariates (confounders).
        - discount_received (treatment) is MORE LIKELY for customers with
          higher `previous_purchases` and `website_visits` (i.e. more
          "engaged" customers are preferentially targeted with discounts --
          this creates real confounding).
        - purchase_amount (outcome) depends on the same confounders AND on
          the treatment, with a heterogeneous treatment effect that varies
          by income and previous_purchases (so CATE is non-trivial).
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(18, 70, size=n)
    income = rng.normal(55000, 18000, size=n).clip(12000, 200000)
    previous_purchases = rng.poisson(5, size=n)
    website_visits = rng.poisson(8, size=n)
    previous_spending = (
        200 + 15 * previous_purchases + 0.002 * income + rng.normal(0, 50, size=n)
    ).clip(0, None)

    # --- Propensity to receive discount depends on confounders (real confounding) ---
    logit = (
        -1.5
        + 0.12 * previous_purchases
        + 0.05 * website_visits
        + 0.000006 * income
        - 0.01 * (age - 40)
    )
    prob_treat = 1 / (1 + np.exp(-logit))
    discount_received = rng.binomial(1, prob_treat)

    # --- Heterogeneous treatment effect (depends on income & prior purchases) ---
    base_effect = 25
    effect_heterogeneity = 0.0003 * (income - 55000) + 1.5 * previous_purchases
    individual_treatment_effect = base_effect + effect_heterogeneity

    noise = rng.normal(0, 30, size=n)
    purchase_amount = (
        50
        + 0.15 * income / 100
        + 8 * previous_purchases
        + 3 * website_visits
        + discount_received * individual_treatment_effect
        + noise
    ).clip(0, None)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST_{i:05d}" for i in range(1, n + 1)],
            "age": age,
            "income": income.round(2),
            "previous_purchases": previous_purchases,
            "website_visits": website_visits,
            "previous_spending": previous_spending.round(2),
            "discount_received": discount_received,
            "purchase_amount": purchase_amount.round(2),
        }
    )
    return df


def save_sample_dataset(path: str, n: int = 2000, seed: int = RANDOM_SEED) -> None:
    df = generate_sample_dataset(n=n, seed=seed)
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------
# CSV loading
# --------------------------------------------------------------------------
def load_csv(file_or_buffer) -> pd.DataFrame:
    """
    Loads a CSV file (path, file-like object, or Streamlit UploadedFile)
    into a pandas DataFrame. Raises a clear ValueError on failure instead
    of letting a cryptic pandas exception bubble up to the UI.
    """
    try:
        if hasattr(file_or_buffer, "read"):
            raw = file_or_buffer.read()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(raw))
        else:
            df = pd.read_csv(file_or_buffer)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not read the CSV file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded CSV file is empty.")
    if df.shape[1] < 2:
        raise ValueError("The dataset needs at least two columns.")
    return df


# --------------------------------------------------------------------------
# Descriptive summaries
# --------------------------------------------------------------------------
@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    numeric_columns: list = field(default_factory=list)
    categorical_columns: list = field(default_factory=list)
    missing_counts: pd.Series = None
    missing_pct: pd.Series = None
    describe: pd.DataFrame = None


def summarize_dataset(df: pd.DataFrame) -> DatasetSummary:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    missing_counts = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)

    try:
        describe = df.describe(include="all").transpose()
    except Exception:  # noqa: BLE001
        describe = pd.DataFrame()

    return DatasetSummary(
        n_rows=df.shape[0],
        n_cols=df.shape[1],
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        missing_counts=missing_counts,
        missing_pct=missing_pct,
        describe=describe,
    )


def basic_clean(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Very simple, transparent missing-value handling for the selected
    columns: numeric columns are median-imputed, categorical columns are
    mode-imputed. Rows are never silently dropped without being reported
    by the caller.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                mode = df[col].mode(dropna=True)
                fill_val = mode.iloc[0] if not mode.empty else "missing"
                df[col] = df[col].fillna(fill_val)
    return df
