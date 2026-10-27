import pandas as pd
import pytest

import config
from src import budget_analyzer as ba
from src import financial_calculations as fc
from src.data_loader import load_sample_transactions


def make_df(rows):
    df = pd.DataFrame(rows, columns=["date", "description", "category", "type", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def df():
    return make_df([
        ("2026-01-01", "Salary", "Salary", "income", 50000),
        ("2026-01-05", "Rent", "Rent", "expense", 15000),
        ("2026-01-10", "Food", "Groceries", "expense", 5000),
        ("2026-02-01", "Salary", "Salary", "income", 50000),
        ("2026-02-05", "Rent", "Rent", "expense", 15000),
        ("2026-02-12", "Food", "Groceries", "expense", 7000),
        ("2026-02-20", "Movie", "Entertainment", "expense", 3000),
    ])


def test_total_income_and_expenses(df):
    assert fc.total_income(df) == 100000
    assert fc.total_expenses(df) == 45000


def test_net_savings(df):
    assert fc.net_savings(fc.total_income(df), fc.total_expenses(df)) == 55000
    assert fc.net_savings(1000, 1500) == -500


def test_savings_rate_normal():
    assert fc.savings_rate(100000, 45000) == pytest.approx(55.0)


def test_savings_rate_zero_income_is_safe():
    assert fc.savings_rate(0, 500) == 0.0
    assert fc.savings_rate(0, 0) == 0.0


def test_savings_rate_negative_when_overspending():
    assert fc.savings_rate(1000, 1500) == pytest.approx(-50.0)


def test_monthly_aggregation(df):
    m = fc.monthly_summary(df)
    assert list(m["month"]) == ["2026-01", "2026-02"]
    assert list(m["income"]) == [50000, 50000]
    assert list(m["expenses"]) == [20000, 25000]
    assert list(m["net_savings"]) == [30000, 25000]
    assert m["savings_rate"].iloc[0] == pytest.approx(60.0)


def test_monthly_summary_month_with_only_income():
    d = make_df([("2026-03-01", "Pay", "Salary", "income", 100)])
    m = fc.monthly_summary(d)
    assert m.loc[0, "expenses"] == 0 and m.loc[0, "savings_rate"] == pytest.approx(100.0)


def test_monthly_summary_empty():
    assert fc.monthly_summary(make_df([])).empty


def test_average_monthly_expenses(df):
    assert fc.average_monthly_expenses(df) == pytest.approx(22500)
    assert fc.average_monthly_expenses(make_df([])) == 0.0


def test_category_totals(df):
    c = fc.category_totals(df, "expense")
    assert list(c["category"]) == ["Rent", "Groceries", "Entertainment"]
    assert list(c["amount"]) == [30000, 12000, 3000]
    assert c["share_pct"].sum() == pytest.approx(100.0)


def test_highest_spending_category(df):
    assert fc.highest_spending_category(df) == ("Rent", 30000)
    assert fc.highest_spending_category(make_df([])) is None


def test_top_expense_categories_limit(df):
    assert len(fc.top_expense_categories(df, 2)) == 2


def test_summarize_transactions(df):
    s = fc.summarize_transactions(df)
    assert s["total_income"] == 100000 and s["net_savings"] == 55000
    assert s["months_covered"] == 2 and s["highest_spending_category"] == "Rent"
    assert s["average_monthly_net_savings"] == pytest.approx(27500)


def test_sample_dataset_has_enough_realistic_transactions():
    d = load_sample_transactions()
    assert len(d) >= 30 and d["month"].nunique() >= 3
    s = fc.summarize_transactions(d)
    assert s["total_income"] > s["total_expenses"] > 0


# ---- budgets
def test_budget_overspending_average(df):
    comp = ba.compare_budgets(df, {"Groceries": 5000, "Rent": 15000, "Entertainment": 0})
    row = comp.set_index("category")
    assert row.loc["Groceries", "actual"] == pytest.approx(6000)  # (5000 + 7000) / 2 months
    assert row.loc["Groceries", "status"] == "Over budget"
    assert row.loc["Rent", "status"] == "Within budget"
    assert "Entertainment" not in row.index  # budget 0 means "no budget"
    assert list(ba.overspent_categories(comp)["category"]) == ["Groceries"]


def test_budget_specific_month(df):
    comp = ba.compare_budgets(df, {"Groceries": 6000}, period="2026-02").iloc[0]
    assert comp["actual"] == 7000 and comp["difference"] == -1000
    assert comp["percent_used"] == pytest.approx(116.666, abs=0.01)


def test_budget_category_with_no_spending(df):
    comp = ba.compare_budgets(df, {"Travel": 1000})
    assert comp.loc[0, "actual"] == 0 and comp.loc[0, "status"] == "Within budget"


def test_no_budgets_returns_empty(df):
    assert ba.compare_budgets(df, {}).empty
    assert ba.overspent_categories(ba.compare_budgets(df, {})).empty


def test_sample_budgets_produce_some_warnings():
    d = load_sample_transactions()
    over = ba.overspent_categories(ba.compare_budgets(d, config.SAMPLE_BUDGETS))
    assert 1 <= len(over) < len(config.SAMPLE_BUDGETS)


# ---- goals and projections
def test_goal_feasible():
    r = fc.savings_goal_feasibility(50000, 6, 10000)
    assert r["required_monthly_saving"] == pytest.approx(8333.333, abs=0.01)
    assert r["feasible_at_current_rate"] is True and r["monthly_gap"] == 0


def test_goal_not_feasible():
    r = fc.savings_goal_feasibility(60000, 6, 4000)
    assert r["required_monthly_saving"] == 10000
    assert r["feasible_at_current_rate"] is False
    assert r["monthly_gap"] == 6000
    assert r["months_needed_at_current_rate"] == pytest.approx(15.0)


def test_goal_with_no_savings():
    r = fc.savings_goal_feasibility(10000, 5, 0)
    assert r["feasible_at_current_rate"] is False and r["months_needed_at_current_rate"] is None


@pytest.mark.parametrize("goal,months", [(0, 6), (-5, 6), (1000, 0), (1000, -2), (1000, 2.5)])
def test_goal_invalid_inputs(goal, months):
    with pytest.raises(ValueError):
        fc.savings_goal_feasibility(goal, months, 1000)


def test_projection_no_interest():
    p = fc.savings_projection(1000, 6)
    assert p["balance"].iloc[-1] == 6000 and len(p) == 7 and p["interest_earned"].iloc[-1] == 0


def test_projection_with_interest_and_start_balance():
    p = fc.savings_projection(1000, 12, annual_rate_pct=12, starting_balance=5000)
    assert p["balance"].iloc[-1] > 5000 + 12000
    assert p["total_contributed"].iloc[-1] == 17000
    assert p["interest_earned"].iloc[-1] > 0


def test_compound_interest_matches_known_value():
    assert fc.compound_interest(100000, 6, 2, 1) == pytest.approx(112360)


def test_format_inr_indian_grouping():
    assert fc.format_inr(0) == "\u20b90"
    assert fc.format_inr(999) == "\u20b9999"
    assert fc.format_inr(1000) == "\u20b91,000"
    assert fc.format_inr(123456) == "\u20b91,23,456"
    assert fc.format_inr(12345678) == "\u20b91,23,45,678"
    assert fc.format_inr(-45000) == "-\u20b945,000"
    assert fc.format_inr(1234.5, 2) == "\u20b91,234.50"


# ---- advisor support
def test_category_overview_percent_of_income_and_average(df):
    o = fc.category_overview(df).set_index("category")
    assert o.loc["Rent", "avg_monthly"] == 15000 and o.loc["Rent", "pct_of_income"] == pytest.approx(30.0)
    assert fc.category_overview(make_df([])).empty


def test_category_overview_zero_income_is_safe():
    d = make_df([("2026-01-01", "x", "Food", "expense", 100)])
    assert fc.category_overview(d).loc[0, "pct_of_income"] == 0.0


def test_unusual_spending_flags_spikes_only():
    rows = []
    for m in range(1, 7):
        rows.append((f"2026-0{m}-05", "Rent", "Rent", "expense", 10000))
        rows.append((f"2026-0{m}-06", "Pay", "Salary", "income", 50000))
    rows.append(("2026-04-10", "Gadget", "Shopping", "expense", 9000))
    u = fc.unusual_monthly_spending(make_df(rows))
    assert list(u["category"]) == ["Shopping"] and u.loc[0, "month"] == "2026-04"
    assert "Rent" not in set(u["category"])  # constant spending is not unusual


def test_unusual_spending_needs_enough_months():
    d = make_df([("2026-01-01", "a", "Food", "expense", 9000), ("2026-02-01", "b", "Food", "expense", 100)])
    assert fc.unusual_monthly_spending(d).empty and fc.unusual_monthly_spending(make_df([])).empty


def test_monthly_savings_target_gap():
    assert fc.monthly_savings_target_gap(10000, 12000)["met_at_current_rate"] is True
    r = fc.monthly_savings_target_gap(10000, 4000)
    assert r["met_at_current_rate"] is False and r["extra_needed_per_month"] == 6000
    with pytest.raises(ValueError):
        fc.monthly_savings_target_gap(0, 100)
