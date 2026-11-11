"""
policy_optimization.py
-----------------------
Turns estimated treatment effects into concrete treat/no-treat
recommendations and compares candidate policies.

All monetary interpretation is on the OUTCOME'S OWN SCALE, as configured by
the user -- we never silently assume `purchase_amount` increase == profit.
The `benefit_ratio` lets the user say e.g. "only 20% of an extra dollar of
purchase amount is margin", so net benefit = ratio * effect - cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class PolicyConfig:
    treatment_cost: float = 0.0
    benefit_ratio: float = 1.0        # fraction of outcome-unit effect counted as "benefit"
    min_effect_threshold: float = 0.0  # only treat if estimated effect exceeds this
    budget: Optional[float] = None     # optional max total spend


def individual_recommendations(cate: np.ndarray, config: PolicyConfig) -> pd.DataFrame:
    """
    Computes a per-individual policy table:
        estimated_effect, estimated_benefit, cost, net_benefit, recommended
    `recommended` = True if net_benefit > 0 AND effect exceeds the minimum
    threshold. Budget constraints (if any) are applied afterwards by ranking
    on net benefit and taking as many as the budget allows.
    """
    effect = np.asarray(cate, dtype=float)
    benefit = effect * config.benefit_ratio
    net_benefit = benefit - config.treatment_cost

    eligible = (net_benefit > 0) & (effect >= config.min_effect_threshold)

    table = pd.DataFrame(
        {
            "estimated_effect": effect,
            "estimated_benefit": benefit,
            "treatment_cost": config.treatment_cost,
            "net_benefit": net_benefit,
            "eligible": eligible,
        }
    )

    if config.budget is not None and eligible.sum() > 0:
        table["recommended"] = False
        ranked = table[table["eligible"]].sort_values("net_benefit", ascending=False)
        n_affordable = int(config.budget // config.treatment_cost) if config.treatment_cost > 0 else len(ranked)
        chosen_idx = ranked.index[:max(n_affordable, 0)]
        table.loc[chosen_idx, "recommended"] = True
        table.attrs["budget_binding"] = n_affordable < eligible.sum()
    else:
        table["recommended"] = eligible
        table.attrs["budget_binding"] = False

    return table


def policy_summary(table: pd.DataFrame) -> dict:
    treated = table[table["recommended"]]
    return {
        "n_recommended": int(treated.shape[0]),
        "pct_recommended": round(100 * treated.shape[0] / max(len(table), 1), 2),
        "total_estimated_benefit": float(treated["estimated_benefit"].sum()),
        "total_cost": float(treated.shape[0] * table["treatment_cost"].iloc[0]) if len(table) else 0.0,
        "total_net_benefit": float(treated["net_benefit"].sum()),
        "budget_binding": bool(table.attrs.get("budget_binding", False)),
    }


# --------------------------------------------------------------------------
# Policy comparison: treat-all vs treat-none vs model-based
# --------------------------------------------------------------------------
def compare_policies(cate: np.ndarray, config: PolicyConfig) -> pd.DataFrame:
    effect = np.asarray(cate, dtype=float)
    n = len(effect)
    benefit_per_unit = effect * config.benefit_ratio

    rows = []

    # Treat everyone
    total_benefit_all = float(benefit_per_unit.sum())
    total_cost_all = float(n * config.treatment_cost)
    rows.append(
        {
            "policy": "Treat Everyone",
            "n_treated": n,
            "estimated_total_benefit": round(total_benefit_all, 2),
            "estimated_total_cost": round(total_cost_all, 2),
            "estimated_net_benefit": round(total_benefit_all - total_cost_all, 2),
            "budget_violation": bool(config.budget is not None and total_cost_all > config.budget),
        }
    )

    # Treat no one
    rows.append(
        {
            "policy": "Treat No One",
            "n_treated": 0,
            "estimated_total_benefit": 0.0,
            "estimated_total_cost": 0.0,
            "estimated_net_benefit": 0.0,
            "budget_violation": False,
        }
    )

    # Model-based targeting (respects min-effect threshold & budget)
    table = individual_recommendations(effect, config)
    summary = policy_summary(table)
    rows.append(
        {
            "policy": "Model-Based Targeting",
            "n_treated": summary["n_recommended"],
            "estimated_total_benefit": round(summary["total_estimated_benefit"], 2),
            "estimated_total_cost": round(summary["total_cost"], 2),
            "estimated_net_benefit": round(summary["total_net_benefit"], 2),
            "budget_violation": False,  # by construction, targeting respects the budget
        }
    )

    return pd.DataFrame(rows)
