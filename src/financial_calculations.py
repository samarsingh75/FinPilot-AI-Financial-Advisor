"""Accurate financial calculations done in Python (never by the LLM)."""
from __future__ import annotations

from typing import Optional

import pandas as pd

import config


# ------------------------------------------------------------ formatting
def format_inr(value: float, decimals: int = 0) -> str:
    """Format a number as Indian Rupees using lakh/crore grouping, e.g. ₹1,23,456."""
    if value is None or pd.isna(value):
        return "-"
    negative = value < 0
    value = abs(float(value))
    fixed = f"{value:.{decimals}f}"
    whole, _, frac = fixed.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    text = f"{config.CURRENCY_SYMBOL}{whole}" + (f".{frac}" if frac else "")
    return f"-{text}" if negative and float(fixed) != 0 else text


# ------------------------------------------------------------ core metrics
def total_income(df: pd.DataFrame) -> float:
    return float(df.loc[df["type"] == "income", "amount"].sum())


def total_expenses(df: pd.DataFrame) -> float:
    return float(df.loc[df["type"] == "expense", "amount"].sum())


def net_savings(income: float, expenses: float) -> float:
    """Net savings = Total income - Total expenses."""
    return float(income) - float(expenses)


def savings_rate(income: float, expenses: float) -> float:
    """Savings rate (%) = (Net savings / Total income) x 100. Returns 0.0 if income is 0."""
    if not income or income <= 0:
        return 0.0
    return net_savings(income, expenses) / float(income) * 100.0


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Income, expenses, net savings and savings rate for each calendar month."""
    columns = ["month", "income", "expenses", "net_savings", "savings_rate"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    pivot = (
        df.assign(month=df["date"].dt.strftime("%Y-%m"))
        .pivot_table(index="month", columns="type", values="amount", aggfunc="sum", fill_value=0.0)
        .reindex(columns=["income", "expense"], fill_value=0.0)
    )
    out = pivot.rename(columns={"expense": "expenses"}).reset_index()
    out["net_savings"] = out["income"] - out["expenses"]
    out["savings_rate"] = [savings_rate(i, e) for i, e in zip(out["income"], out["expenses"])]
    return out[columns].sort_values("month").reset_index(drop=True)


def number_of_months(df: pd.DataFrame) -> int:
    """Number of distinct calendar months that contain at least one transaction."""
    if df.empty:
        return 0
    return int(df["date"].dt.strftime("%Y-%m").nunique())


def average_monthly_expenses(df: pd.DataFrame) -> float:
    months = number_of_months(df)
    return total_expenses(df) / months if months else 0.0


def category_totals(df: pd.DataFrame, kind: str = "expense") -> pd.DataFrame:
    """Total amount per category for one transaction type, largest first."""
    subset = df[df["type"] == kind]
    if subset.empty:
        return pd.DataFrame(columns=["category", "amount", "share_pct"])
    totals = subset.groupby("category", as_index=False)["amount"].sum()
    totals = totals.sort_values(["amount", "category"], ascending=[False, True]).reset_index(drop=True)
    grand = totals["amount"].sum()
    totals["share_pct"] = totals["amount"] / grand * 100.0 if grand else 0.0
    return totals


def top_expense_categories(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return category_totals(df, "expense").head(n).reset_index(drop=True)


def highest_spending_category(df: pd.DataFrame) -> Optional[tuple[str, float]]:
    top = top_expense_categories(df, 1)
    if top.empty:
        return None
    return str(top.loc[0, "category"]), float(top.loc[0, "amount"])


def summarize_transactions(df: pd.DataFrame) -> dict:
    """Return all headline metrics as a plain dictionary."""
    income, expenses = total_income(df), total_expenses(df)
    top = highest_spending_category(df)
    return {
        "transaction_count": int(len(df)),
        "start_date": df["date"].min().strftime("%Y-%m-%d") if len(df) else None,
        "end_date": df["date"].max().strftime("%Y-%m-%d") if len(df) else None,
        "months_covered": number_of_months(df),
        "total_income": income,
        "total_expenses": expenses,
        "net_savings": net_savings(income, expenses),
        "savings_rate_pct": savings_rate(income, expenses),
        "average_monthly_income": income / number_of_months(df) if len(df) else 0.0,
        "average_monthly_expenses": average_monthly_expenses(df),
        "average_monthly_net_savings": net_savings(income, expenses) / number_of_months(df) if len(df) else 0.0,
        "highest_spending_category": top[0] if top else None,
        "highest_spending_amount": top[1] if top else 0.0,
    }


# ------------------------------------------------------------ goals & projections
def savings_goal_feasibility(goal_amount: float, months: int, current_monthly_savings: float) -> dict:
    """Compare the monthly saving needed for a goal with the user's current monthly savings.

    This is a simple straight-line calculation with no interest.
    """
    if goal_amount is None or goal_amount <= 0:
        raise ValueError("goal_amount must be greater than zero.")
    if months is None or int(months) != months or months <= 0:
        raise ValueError("months must be a positive whole number.")
    required = float(goal_amount) / int(months)
    current = float(current_monthly_savings)
    gap = required - current
    return {
        "goal_amount": float(goal_amount),
        "months": int(months),
        "required_monthly_saving": required,
        "current_monthly_savings": current,
        "monthly_gap": max(gap, 0.0),
        "feasible_at_current_rate": current >= required,
        "months_needed_at_current_rate": (float(goal_amount) / current) if current > 0 else None,
    }


def savings_projection(
    monthly_saving: float, months: int, annual_rate_pct: float = 0.0, starting_balance: float = 0.0
) -> pd.DataFrame:
    """Hypothetical balance growth: deposit at month end, interest compounded monthly.

    ``annual_rate_pct`` is an ASSUMPTION supplied by the user, not a prediction.
    """
    if months < 0:
        raise ValueError("months must not be negative.")
    monthly_rate = annual_rate_pct / 100.0 / 12.0
    balance = float(starting_balance)
    contributed = float(starting_balance)
    rows = [{"month": 0, "balance": balance, "total_contributed": contributed, "interest_earned": 0.0}]
    for m in range(1, int(months) + 1):
        balance = balance * (1 + monthly_rate) + monthly_saving
        contributed += monthly_saving
        rows.append(
            {"month": m, "balance": balance, "total_contributed": contributed,
             "interest_earned": balance - contributed}
        )
    return pd.DataFrame(rows)


def compound_interest(principal: float, annual_rate_pct: float, years: float, compounds_per_year: int = 12) -> float:
    """Future value of a lump sum: P * (1 + r/n) ** (n * t)."""
    if compounds_per_year <= 0:
        raise ValueError("compounds_per_year must be positive.")
    rate = annual_rate_pct / 100.0
    return float(principal) * (1 + rate / compounds_per_year) ** (compounds_per_year * years)


# ------------------------------------------------------------ advisor support
def category_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Expense categories with total, share of expenses, average per month and share of income."""
    cats = category_totals(df, "expense")
    if cats.empty:
        return pd.DataFrame(columns=["category", "amount", "share_pct", "avg_monthly", "pct_of_income"])
    months = number_of_months(df) or 1
    income = total_income(df)
    out = cats.copy()
    out["avg_monthly"] = out["amount"] / months
    out["pct_of_income"] = out["amount"] / income * 100.0 if income > 0 else 0.0
    return out


def unusual_monthly_spending(
    df: pd.DataFrame, factor: float = 1.5, min_months: int = 3, min_excess: float = 500.0
) -> pd.DataFrame:
    """Months where a category's spending was well above its typical (median) month.

    A category-month is flagged when amount > factor x median AND the excess is at least
    ``min_excess``. Months without spending count as zero, so one-off large expenses are
    flagged too. Needs at least ``min_months`` calendar months of data.
    """
    columns = ["month", "category", "amount", "typical_monthly", "excess"]
    expenses = df[df["type"] == "expense"]
    if expenses.empty or number_of_months(df) < min_months:
        return pd.DataFrame(columns=columns)
    all_months = sorted(df["date"].dt.strftime("%Y-%m").unique())
    grid = (
        expenses.assign(month=expenses["date"].dt.strftime("%Y-%m"))
        .pivot_table(index="month", columns="category", values="amount", aggfunc="sum")
        .reindex(all_months)
        .fillna(0.0)
        .astype(float)
    )
    rows = []
    for category in grid.columns:
        typical = float(grid[category].median())
        for month, amount in grid[category].items():
            if amount > factor * typical and amount - typical >= min_excess:
                rows.append({"month": month, "category": category, "amount": float(amount),
                             "typical_monthly": typical, "excess": float(amount) - typical})
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values("excess", ascending=False).reset_index(drop=True)


def monthly_savings_target_gap(target_monthly: float, current_avg_monthly_savings: float) -> dict:
    """Compare a desired monthly saving amount with the user's current average monthly net savings."""
    if target_monthly is None or target_monthly <= 0:
        raise ValueError("target_monthly must be greater than zero.")
    gap = float(target_monthly) - float(current_avg_monthly_savings)
    return {
        "target_monthly": float(target_monthly),
        "current_avg_monthly_savings": float(current_avg_monthly_savings),
        "met_at_current_rate": gap <= 0,
        "extra_needed_per_month": max(gap, 0.0),
    }
