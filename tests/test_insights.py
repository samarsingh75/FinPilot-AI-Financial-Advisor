"""Tests for Smart Spending Insights (engine, Gemini explanation layer and guardrails)."""
import pandas as pd
import pytest

import config
from src import insights as ins
from src.gemini_client import QUOTA, GeminiError
from src.insights_ai import explain_insights, find_unsupported_numbers
from src.insights_view import _card
from src.data_loader import load_sample_transactions

COLS = ["date", "description", "category", "type", "amount"]


def mk(rows):
    df = pd.DataFrame(rows, columns=COLS)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def month(ym, spec, rent=True):
    """Rows for one month. spec = [(day, description, category, amount)], plus rent on day 28 (so months look complete)."""
    rows = [(f"{ym}-{d:02d}", desc, cat, "expense", amt) for d, desc, cat, amt in spec]
    if rent:
        rows.append((f"{ym}-28", "House rent", "Rent", "expense", 10000))
    return rows


def kinds(report):
    return [i.kind for i in report.insights]


# ---------------------------------------------------------------- trends
def test_steady_spending_produces_no_insights():
    rows = []
    for ym in ("2026-01", "2026-02", "2026-03", "2026-04"):
        rows += month(ym, [(5, "Supermarket", "Groceries", 5000), (12, "Restaurant", "Dining", 2000)])
    r = ins.analyze(mk(rows))
    assert r.insights == [] and r.overview["mom_change_pct"] == pytest.approx(0.0)
    assert r.overview["latest_month"] == "2026-04" and r.overview["previous_month"] == "2026-03"


def test_genuine_month_over_month_increase_is_exact_and_grounded():
    rows = month("2026-03", [(5, "Cafe", "Food", 5000)]) + month("2026-04", [(5, "Cafe", "Food", 6400)])
    r = ins.analyze(mk(rows))
    up = [i for i in r.insights if i.kind == "trend_increase"]
    assert len(up) == 1 and up[0].category == "Food"
    assert up[0].values["change_pct"] == pytest.approx(28.0) and up[0].values["change"] == 1400
    assert "increased by 28% compared with 2026-03" in up[0].message
    assert "\u20b95,000 to \u20b96,400" in up[0].message
    assert up[0].severity == "medium"  # 28% is below the 50% 'high' band


def test_big_percentage_but_tiny_amount_is_ignored():
    rows = month("2026-03", [(5, "Snack", "Snacks", 100)]) + month("2026-04", [(5, "Snack", "Snacks", 250)])
    assert "trend_increase" not in kinds(ins.analyze(mk(rows)))


def test_decrease_is_reported_as_good_news():
    rows = month("2026-03", [(5, "Mall", "Shopping", 8000)]) + month("2026-04", [(5, "Mall", "Shopping", 3000)])
    dec = [i for i in ins.analyze(mk(rows)).insights if i.kind == "trend_decrease"]
    assert len(dec) == 1 and dec[0].severity == "positive" and "decreased by 62%" in dec[0].message


def test_incomplete_latest_month_suppresses_decreases_and_warns():
    rows = month("2026-03", [(5, "Mall", "Shopping", 8000)]) + [("2026-04-03", "Mall", "Shopping", "expense", 1000)]
    r = ins.analyze(mk(rows))
    assert r.overview["latest_month_partial"] is True
    assert "trend_decrease" not in kinds(r) and any("incomplete" in n for n in r.notes)


def test_new_category_is_reported():
    rows = month("2026-03", [(5, "Cafe", "Food", 3000)]) + month("2026-04", [(5, "Cafe", "Food", 3000), (9, "Clinic", "Health", 2500)])
    new = [i for i in ins.analyze(mk(rows)).insights if i.kind == "new_category"]
    assert len(new) == 1 and new[0].category == "Health" and "\u20b92,500" in new[0].message


def test_missing_previous_month_gives_no_comparison_and_a_note():
    rows = month("2026-01", [(5, "Cafe", "Food", 3000)]) + month("2026-03", [(5, "Cafe", "Food", 9000)])
    r = ins.analyze(mk(rows))
    assert r.overview["comparison_available"] is False and r.insights == [] and r.category_table.empty
    assert any("no transactions in 2026-02" in n for n in r.notes)


def test_single_month_of_data_is_handled():
    r = ins.analyze(mk(month("2026-04", [(5, "Cafe", "Food", 3000)])))
    assert r.has_data and r.insights == [] and r.overview["comparison_available"] is False


def test_budget_overspend_insight_and_within_budget_case():
    rows = month("2026-03", [(5, "Cafe", "Food", 4000)]) + month("2026-04", [(5, "Cafe", "Food", 4200)])
    over = ins.analyze(mk(rows), budgets={"Food": 3000})
    o = [i for i in over.insights if i.kind == "overspend"]
    assert len(o) == 1 and o[0].values["percent_used"] == pytest.approx(140.0) and "over by \u20b91,200" in o[0].message
    assert "overspend" not in kinds(ins.analyze(mk(rows), budgets={"Food": 5000}))
    assert "overspend" not in kinds(ins.analyze(mk(rows), budgets={}))


# ---------------------------------------------------------------- unusual transactions
def _shopping_history(extra=()):
    amounts = [1000, 1100, 1200, 1300, 1400, 1050, 1150, 1250, 1350, 1450, 1000, 1100]
    rows = []
    for i, amt in enumerate(amounts):
        ym = f"2026-0{i // 2 + 1}"
        rows.append((f"{ym}-{10 + i % 2 * 5:02d}", f"Store {i}", "Shopping", "expense", amt))
    rows += [(f"2026-0{m}-28", "House rent", "Rent", "expense", 10000) for m in range(1, 7)]
    return rows + list(extra)


def test_unusually_large_transaction_is_flagged_with_a_reason():
    rows = _shopping_history([("2026-06-14", "Electronics expo", "Shopping", "expense", 9000)])
    u = ins.analyze(mk(rows)).unusual
    assert list(u["description"]) == ["Electronics expo"] and u.loc[0, "scope"] == "category"
    assert u.loc[0, "ratio"] > 5 and "median" in u.loc[0, "reason"] and "\u20b99,000" in u.loc[0, "reason"]


def test_normal_variation_and_fixed_rent_are_not_flagged():
    assert ins.analyze(mk(_shopping_history())).unusual.empty


def test_slightly_higher_than_usual_is_not_called_unusual():
    rows = _shopping_history([("2026-06-14", "Bigger shop", "Shopping", "expense", 1900)])
    assert ins.analyze(mk(rows)).unusual.empty


def test_flat_history_with_small_deviation_is_not_flagged():
    rows = [(f"2026-0{m}-0{d}", "Cafe", "Food", "expense", 1000) for m in range(1, 7) for d in (3, 8)]
    rows.append(("2026-06-20", "Cafe", "Food", "expense", 1200))
    assert ins.analyze(mk(rows)).unusual.empty


def test_insufficient_history_does_not_guess():
    rows = month("2026-03", [(5, "Cafe", "Food", 500)]) + month("2026-04", [(5, "Big buy", "Shopping", 50000)])
    r = ins.analyze(mk(rows))
    assert r.unusual.empty and "Not enough history" in r.unusual_note


def test_outlier_note_says_not_fraud():
    rows = _shopping_history([("2026-06-14", "Electronics expo", "Shopping", "expense", 9000)])
    assert "not confirmed fraud" in ins.analyze(mk(rows)).unusual_note.lower()


# ---------------------------------------------------------------- recurring payments
def _monthly(desc, amounts, days, cat="Subscriptions", start_month=1):
    return [(f"2026-{start_month + i:02d}-{d:02d}", desc, cat, "expense", a) for i, (a, d) in enumerate(zip(amounts, days))]


def test_monthly_subscription_is_detected_with_costs():
    rows = _monthly("Netflix", [499] * 6, [5, 6, 5, 5, 6, 5])
    rec = ins.analyze(mk(rows)).recurring
    assert len(rec) == 1
    r = rec.iloc[0]
    assert r["cadence"] == "monthly" and r["pattern"] == "Fixed amount" and r["occurrences"] == 6
    assert r["est_monthly"] == 499 and r["est_annual"] == 5988 and r["status"] == "Active" and r["confidence"] == "High"


def test_merchant_variants_with_reference_numbers_are_grouped():
    rows = _monthly("NETFLIX 4821", [499, 499, 499], [5, 5, 5]) + _monthly("Netflix 1234", [499] * 2, [5, 5], start_month=4)
    assert ins.analyze(mk(rows)).recurring.iloc[0]["occurrences"] == 5


def test_repeated_merchant_is_not_automatically_a_subscription():
    # Appears 5 times but at irregular intervals (7, 7, 40, 2 days).
    dates = ["2026-01-01", "2026-01-08", "2026-01-15", "2026-02-24", "2026-02-26"]
    rows = [(d, "Gym drop-in", "Fitness", "expense", 300) for d in dates]
    assert ins.analyze(mk(rows)).recurring.empty


def test_two_occurrences_are_not_enough():
    assert ins.analyze(mk(_monthly("Magazine", [200, 200], [3, 3]))).recurring.empty


def test_weekly_payment_is_detected():
    rows = [(str((pd.Timestamp("2026-01-05") + pd.Timedelta(days=7 * i)).date()), "Milk delivery", "Groceries", "expense", 120) for i in range(8)]
    r = ins.analyze(mk(rows)).recurring.iloc[0]
    assert r["cadence"] == "weekly" and r["est_annual"] == 120 * 52


def test_variable_utility_bill_is_detected_but_labelled_variable():
    rows = _monthly("Electricity bill", [1850, 2300, 2100, 2500, 2200, 2000], [8, 9, 7, 8, 8, 10], "Utilities")
    r = ins.analyze(mk(rows)).recurring.iloc[0]
    assert r["pattern"] == "Variable amount" and 2000 < r["typical_amount"] < 2300
    assert ins.active_fixed(ins.analyze(mk(rows)).recurring).empty  # not included in 'fixed commitments'


def test_wildly_varying_amounts_are_not_recurring():
    assert ins.analyze(mk(_monthly("Online order", [100, 900, 300, 2000, 150], [5] * 5, "Shopping"))).recurring.empty


def test_ended_subscription_is_marked_as_possibly_ended():
    rows = _monthly("Old magazine", [250] * 4, [5] * 4) + month("2026-08", [(5, "Cafe", "Food", 900)])
    r = ins.analyze(mk(rows)).recurring
    assert r.iloc[0]["status"] == "May have ended" and ins.active_fixed(r).empty


def test_price_change_of_fixed_subscription_is_reported():
    rows = _monthly("Streaming plan", [499, 499, 499, 499, 649], [5] * 5)
    pc = [i for i in ins.analyze(mk(rows)).insights if i.kind == "price_change"]
    assert len(pc) == 1 and pc[0].values["change_pct"] == pytest.approx(30.06, abs=0.01) and "\u20b9649" in pc[0].message


# ---------------------------------------------------------------- data quality
def test_invalid_records_are_dropped_and_reported():
    rows = month("2026-03", [(5, "Cafe", "Food", 1000)]) + month("2026-04", [(5, "Cafe", "Food", 1000)])
    rows += [("not a date", "Bad", "Food", "expense", 100), ("2026-04-10", "Zero", "Food", "expense", 0),
             ("2026-04-11", "Neg", "Food", "expense", -50), ("2026-04-12", "Odd", "Food", "transfer", 10),
             ("2026-04-13", "NaN", "Food", "expense", None)]
    r = ins.analyze(mk(rows))
    assert r.quality["invalid_dropped"] == 5 and any("5 record(s)" in n for n in r.notes)
    assert r.overview["latest_spending"] == 11000


def test_blank_and_inconsistent_categories_are_normalised():
    rows = [("2026-04-01", "a", "Groceries", "expense", 100), ("2026-04-02", "b", "groceries ", "expense", 100),
            ("2026-04-03", "c", "GROCERIES", "expense", 100), ("2026-04-04", "d", "", "expense", 700),
            ("2026-04-05", "e", None, "expense", 300)]
    r = ins.analyze(mk(rows))
    spend, _ = ins.prepare_transactions(mk(rows))
    assert set(spend["category"]) == {"Groceries", "Uncategorized"}
    assert r.quality["merged_category_labels"] and any("merged" in n for n in r.notes)


def test_exact_duplicates_are_reported_but_kept():
    row = ("2026-04-05", "Cafe", "Food", "expense", 450)
    r = ins.analyze(mk([row, row, ("2026-04-06", "Cafe", "Food", "expense", 450)]))
    assert r.quality["possible_duplicates"] == 1 and r.overview["transaction_count"] == 3
    assert any("exact duplicates" in n for n in r.notes)


def test_missing_required_column_raises_clear_error():
    with pytest.raises(ValueError, match="amount"):
        ins.analyze(mk([("2026-04-05", "a", "b", "expense", 1)]).drop(columns=["amount"]))


def test_no_transactions_and_income_only_are_handled():
    for df in (mk([]), mk([("2026-04-01", "Salary", "Salary", "income", 50000)])):
        r = ins.analyze(df)
        assert not r.has_data and r.insights == [] and r.unusual.empty and r.recurring.empty
        assert any("no expense transactions" in n.lower() for n in r.notes)
        assert "nothing to summarise" in ins.fallback_summary(r)


# ---------------------------------------------------------------- refunds and income
def test_income_is_never_counted_as_spending_and_refunds_reduce_it():
    rows = month("2026-03", [(5, "Mall", "Shopping", 5000)]) + month("2026-04", [(5, "Mall", "Shopping", 5000)])
    rows += [("2026-04-02", "Salary", "Salary", "income", 90000),
             ("2026-04-10", "Refund for returned shoes", "Shopping", "income", 1000)]
    r = ins.analyze(mk(rows))
    assert r.overview["latest_spending"] == 14000  # 5000 + 10000 rent - 1000 refund, salary ignored
    assert r.overview["gross_expenses"] == 30000 and r.overview["refunds_total"] == 1000
    assert r.category_table.set_index("category").loc["Shopping", "latest"] == 4000
    assert any("refund" in n for n in r.notes)


def test_refund_is_attributed_through_its_description():
    rows = month("2026-04", [(5, "Mall", "Shopping", 5000)]) + [("2026-04-09", "Refund - Shopping order", "Refund", "income", 700)]
    assert ins.analyze(mk(rows)).category_table.empty  # one month only, but net spending must be right:
    assert ins.analyze(mk(rows)).overview["latest_spending"] == 14300


def test_unattributed_refund_reduces_total_but_not_a_category():
    base = month("2026-03", [(5, "Mall", "Shopping", 5000)]) + month("2026-04", [(5, "Mall", "Shopping", 5000)])
    r = ins.analyze(mk(base + [("2026-04-09", "Credit card cashback", "Rewards", "income", 200)]))
    assert r.overview["latest_spending"] == 14800
    assert ins.UNATTRIBUTED not in set(r.category_table["category"])


# ---------------------------------------------------------------- filters
def test_date_and_category_filters():
    rows = month("2026-02", [(5, "Cafe", "Food", 3000)]) + month("2026-03", [(5, "Cafe", "Food", 3000)]) + month("2026-04", [(5, "Cafe", "Food", 3900)])
    full = ins.analyze(mk(rows))
    assert full.overview["months"] == 3
    ranged = ins.analyze(mk(rows), start="2026-03-01", end="2026-03-31")
    assert ranged.overview["latest_month"] == "2026-03" and ranged.overview["months"] == 1
    only_food = ins.analyze(mk(rows), categories=["Food"])
    assert set(only_food.category_table["category"]) == {"Food"} and only_food.overview["latest_spending"] == 3900
    assert not ins.analyze(mk(rows), start="2030-01-01").has_data


# ---------------------------------------------------------------- sample dataset (integration)
def test_sample_dataset_figures_match_hand_calculation():
    r = ins.analyze(load_sample_transactions(), config.SAMPLE_BUDGETS)
    t = r.category_table.set_index("category")
    assert t.loc["Dining Out", "previous"] == 4330 and t.loc["Dining Out", "latest"] == 7820
    assert t.loc["Dining Out", "change_pct"] == pytest.approx(80.6, abs=0.05)
    assert r.overview["latest_spending"] == 53686 and r.overview["previous_spending"] == 63696
    assert any(i.kind == "trend_increase" and i.category == "Dining Out" for i in r.insights)
    rent = r.recurring[r.recurring["category"] == "Rent"].iloc[0]
    assert rent["pattern"] == "Fixed amount" and rent["est_annual"] == 216000


# ---------------------------------------------------------------- text, guardrail and Gemini layer
@pytest.fixture
def report():
    rows = month("2026-03", [(5, "Swiggy order", "Food", 5000)]) + month("2026-04", [(5, "Swiggy order", "Food", 6400)])
    return ins.analyze(mk(rows))


class FakeClient:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.prompts = reply, error, []

    def generate(self, prompt, system_instruction=None, history=None, **k):
        self.prompts.append((prompt, system_instruction))
        if self.error:
            raise self.error
        return self.reply


def test_facts_hide_merchant_names_unless_requested():
    rows = _monthly("Netflix", [499] * 6, [5] * 6)
    r = ins.analyze(mk(rows))
    assert "Netflix" not in ins.facts_text(r) and "Netflix" in ins.facts_text(r, include_merchants=True)


def test_fallback_summary_is_deterministic_and_grounded(report):
    text = ins.fallback_summary(report)
    assert "28%" in text and "\u20b96,400" in text and text == ins.fallback_summary(report)


def test_guardrail_accepts_supported_numbers_and_rejects_invented_ones(report):
    facts = ins.facts_text(report)
    assert find_unsupported_numbers("Food rose 28% to \u20b96,400 from \u20b95,000.", facts) == []
    assert find_unsupported_numbers("Food rose about 28.3%.", facts) == []  # rounding tolerance for percentages
    assert find_unsupported_numbers("Food rose 45% to \u20b99,999.", facts) == ["45%", "9999"]
    assert find_unsupported_numbers("Take these 3 steps in 2026-04.", facts) == []  # small counts and dates are not figures


def test_gemini_explanation_is_used_when_it_sticks_to_the_facts(report):
    client = FakeClient("### What changed\nFood spending rose 28% (\u20b95,000 to \u20b96,400).")
    out = explain_insights(report, client)
    assert out["source"] == "gemini" and out["warning"] is None and "28%" in out["text"]
    prompt, system = client.prompts[0]
    assert "VERIFIED FACTS" in prompt and "Swiggy" not in prompt  # no merchant names by default
    assert "Never add, estimate" in system and "Never call them fraud" in system


def test_gemini_failure_falls_back_to_calculated_summary(report):
    out = explain_insights(report, FakeClient(error=GeminiError(QUOTA, "Gemini rate limit reached.")))
    assert out["source"] == "fallback" and "rate limit" in out["warning"] and "28%" in out["text"]


def test_gemini_text_with_invented_figures_is_discarded(report):
    out = explain_insights(report, FakeClient("Food spending jumped 75% to \u20b912,345."))
    assert out["source"] == "fallback" and "75%" in out["warning"] and "75%" not in out["text"]


def test_no_data_never_calls_gemini():
    client = FakeClient("should not be used")
    out = explain_insights(ins.analyze(mk([])), client)
    assert out["source"] == "fallback" and client.prompts == []


def test_missing_api_key_gives_fallback_not_a_crash(report, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    out = explain_insights(report)
    assert out["source"] == "fallback" and "No Gemini API key" in out["warning"]


def test_insight_card_escapes_user_text():
    card = _card(ins.Insight("trend_increase", "high", "<b>T</b>", "msg <script>x</script>", "do <i>it</i>",
                             details=["<img src=x onerror=alert(1)> (\u20b91 on 01 Jan)"]))
    assert "<script>" not in card and "<img" not in card and "&lt;script&gt;" in card