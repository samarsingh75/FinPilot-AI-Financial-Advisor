"""Compare actual spending against user-defined monthly budgets."""
from __future__ import annotations

from typing import Mapping, Optional

import pandas as pd

from src.financial_calculations import number_of_months

AVERAGE_PERIOD = "average"
COLUMNS = ["category", "budget", "actual", "difference", "percent_used", "status"]


def compare_budgets(df: pd.DataFrame, budgets: Mapping[str, float], period: str = AVERAGE_PERIOD) -> pd.DataFrame:
    """Compare monthly budgets with actual expenses.

    ``period`` is either ``"average"`` (average monthly spend over all months in the
    data) or a month such as ``"2026-05"``. Categories with a budget of 0 or less are
    treated as "no budget set" and skipped.
    """
    active = {c: float(b) for c, b in budgets.items() if b is not None and float(b) > 0}
    if not active:
        return pd.DataFrame(columns=COLUMNS)

    expenses = df[df["type"] == "expense"]
    if period == AVERAGE_PERIOD:
        months = number_of_months(df) or 1
        actual = expenses.groupby("category")["amount"].sum() / months
    else:
        in_month = expenses[expenses["date"].dt.strftime("%Y-%m") == period]
        actual = in_month.groupby("category")["amount"].sum()

    rows = []
    for category, budget in active.items():
        spent = float(actual.get(category, 0.0))
        rows.append(
            {
                "category": category,
                "budget": budget,
                "actual": spent,
                "difference": budget - spent,  # negative => overspent
                "percent_used": spent / budget * 100.0,
                "status": "Over budget" if spent > budget else "Within budget",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS).sort_values("percent_used", ascending=False).reset_index(drop=True)


def overspent_categories(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rows of a comparison table where actual spending exceeds the budget."""
    if comparison.empty:
        return comparison
    return comparison[comparison["actual"] > comparison["budget"]].reset_index(drop=True)


def unbudgeted_spending(df: pd.DataFrame, budgets: Mapping[str, float]) -> Optional[pd.DataFrame]:
    """Expense categories that have no budget set (useful hint for the user)."""
    active = {c for c, b in budgets.items() if b and float(b) > 0}
    cats = sorted(set(df.loc[df["type"] == "expense", "category"]) - active)
    return pd.DataFrame({"category": cats}) if cats else None
