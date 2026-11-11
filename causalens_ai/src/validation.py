"""
validation.py
--------------
Plain-language validation checks for treatment/outcome/feature selection.

Every check returns a ValidationMessage so the UI can show clear,
beginner-friendly errors/warnings instead of raw stack traces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

MIN_GROUP_SIZE = 20  # minimum rows needed per treatment arm for a stable estimate


@dataclass
class ValidationMessage:
    level: str  # "error" | "warning" | "info" | "success"
    message: str


@dataclass
class ValidationReport:
    messages: List[ValidationMessage]

    @property
    def has_errors(self) -> bool:
        return any(m.level == "error" for m in self.messages)

    @property
    def errors(self):
        return [m.message for m in self.messages if m.level == "error"]

    @property
    def warnings(self):
        return [m.message for m in self.messages if m.level == "warning"]


def _is_binary(series: pd.Series) -> bool:
    vals = series.dropna().unique()
    return len(vals) == 2 and set(np.unique(vals)).issubset({0, 1, 0.0, 1.0, True, False})


def validate_columns_exist(df: pd.DataFrame, columns: List[str]) -> List[ValidationMessage]:
    msgs = []
    for col in columns:
        if col is None:
            continue
        if col not in df.columns:
            msgs.append(ValidationMessage("error", f"Column '{col}' was not found in the dataset."))
    return msgs


def validate_treatment(df: pd.DataFrame, treatment_col: str) -> List[ValidationMessage]:
    msgs = []
    if treatment_col not in df.columns:
        return [ValidationMessage("error", f"Treatment column '{treatment_col}' not found.")]

    series = df[treatment_col]
    if series.isna().any():
        msgs.append(
            ValidationMessage(
                "warning",
                f"Treatment column '{treatment_col}' has missing values; "
                "rows with missing treatment will be excluded from estimation.",
            )
        )
    series_clean = series.dropna()
    if not _is_binary(series_clean):
        msgs.append(
            ValidationMessage(
                "error",
                f"Treatment column '{treatment_col}' must be binary (exactly two values, "
                f"e.g. 0/1). Found values: {sorted(series_clean.unique().tolist())[:10]}",
            )
        )
        return msgs

    counts = series_clean.value_counts()
    for val, cnt in counts.items():
        if cnt < MIN_GROUP_SIZE:
            msgs.append(
                ValidationMessage(
                    "warning",
                    f"Treatment group '{val}' has only {cnt} rows. Estimates for this group "
                    f"may be unstable (fewer than {MIN_GROUP_SIZE} recommended).",
                )
            )
    if len(counts) < 2:
        msgs.append(
            ValidationMessage(
                "error",
                "The treatment column only has one unique value after removing missing rows. "
                "Causal effect estimation requires both a treated and a control group.",
            )
        )
    return msgs


def validate_outcome(df: pd.DataFrame, outcome_col: str) -> List[ValidationMessage]:
    msgs = []
    if outcome_col not in df.columns:
        return [ValidationMessage("error", f"Outcome column '{outcome_col}' not found.")]

    series = df[outcome_col]
    if not pd.api.types.is_numeric_dtype(series):
        msgs.append(
            ValidationMessage(
                "error",
                f"Outcome column '{outcome_col}' must be numeric for the continuous-outcome "
                "estimators used in this app. Please choose a numeric column, or encode your "
                "outcome as numeric (e.g. 0/1) before uploading.",
            )
        )
    if series.isna().mean() > 0.3:
        msgs.append(
            ValidationMessage(
                "warning",
                f"Outcome column '{outcome_col}' has more than 30% missing values. "
                "Results may be unreliable.",
            )
        )
    if pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() == 1:
        msgs.append(
            ValidationMessage(
                "error",
                f"Outcome column '{outcome_col}' has no variation (a single constant value). "
                "A causal effect cannot be estimated on a constant outcome.",
            )
        )
    return msgs


def validate_features(df: pd.DataFrame, feature_cols: List[str], treatment_col: str, outcome_col: str) -> List[ValidationMessage]:
    msgs = []
    if not feature_cols:
        msgs.append(
            ValidationMessage(
                "warning",
                "No confounder/feature columns selected. Without confounders, the estimate "
                "reduces to a simple treated-vs-control comparison, which is only valid if "
                "treatment was randomly assigned.",
            )
        )
    for col in feature_cols:
        if col == treatment_col:
            msgs.append(ValidationMessage("error", "A feature column cannot be the same as the treatment column."))
        if col == outcome_col:
            msgs.append(ValidationMessage("error", "A feature column cannot be the same as the outcome column."))
    msgs.extend(validate_columns_exist(df, feature_cols))
    return msgs


def validate_configuration(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: List[str],
) -> ValidationReport:
    messages: List[ValidationMessage] = []

    if treatment_col is None or outcome_col is None:
        messages.append(ValidationMessage("error", "Please select both a treatment column and an outcome column."))
        return ValidationReport(messages)

    messages.extend(validate_treatment(df, treatment_col))
    messages.extend(validate_outcome(df, outcome_col))
    messages.extend(validate_features(df, feature_cols, treatment_col, outcome_col))

    if not any(m.level == "error" for m in messages):
        messages.append(ValidationMessage("success", "Configuration looks valid. You can proceed to causal analysis."))

    return ValidationReport(messages)
