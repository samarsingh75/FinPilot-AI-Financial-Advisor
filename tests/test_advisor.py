import pytest

import config
from src import advisor
from src.data_loader import load_sample_transactions
from src.prompts import SYSTEM_PROMPT, build_advisor_prompt, build_rag_prompt


@pytest.mark.parametrize("question,amount,months", [
    ("How can I save \u20b950,000 in six months?", 50000, 6),
    ("I want to save 1.5 lakh in 2 years", 150000, 24),
    ("save Rs. 20,000 in 8 months", 20000, 8),
    ("Can I save 50000 in a year?", 50000, 12),
    ("my goal is \u20b92 lakh within 18 months", 200000, 18),
])
def test_parse_savings_goal(question, amount, months):
    assert advisor.parse_savings_goal(question) == {"amount": amount, "months": months}


@pytest.mark.parametrize("question", [
    "What is my savings rate?", "Explain compound interest over 10 years", "Where am I spending the most?", "", None,
    "I will save in 6 months",
])
def test_parse_savings_goal_returns_none_when_not_a_goal(question):
    assert advisor.parse_savings_goal(question) is None


def test_context_contains_python_computed_figures():
    df = load_sample_transactions()
    ctx = advisor.build_financial_context(df, config.SAMPLE_BUDGETS, "", 4.0)
    assert "Total income: \u20b94,36,795" in ctx and "Savings rate: 28.0%" in ctx
    assert "Highest spending category: Rent" in ctx
    assert "Categories over budget: Shopping, Dining Out, Utilities, Travel" in ctx
    assert "CALCULATED FIGURES" in ctx and "HYPOTHETICAL ESTIMATES" in ctx


def test_context_includes_goal_check_when_asked():
    df = load_sample_transactions()
    ctx = advisor.build_financial_context(df, {}, "How can I save \u20b950,000 in six months?", 0.0)
    assert "Required saving per month: \u20b98,333" in ctx and "Achievable at the current savings rate: yes" in ctx
    assert "Savings goal check" not in advisor.build_financial_context(df, {}, "What is my savings rate?", 0.0)


def test_context_never_leaks_transaction_descriptions():
    df = load_sample_transactions()
    ctx = advisor.build_financial_context(df, config.SAMPLE_BUDGETS, "", 4.0)
    for description in df["description"].unique():
        assert description not in ctx


def test_context_without_data_is_explicit():
    ctx = advisor.build_financial_context(None, {}, "", 0.0)
    assert "NOT loaded" in ctx and "\u20b9" not in ctx.replace("(", "")


def test_context_zero_income_is_safe():
    df = load_sample_transactions()
    df = df[df["type"] == "expense"]
    ctx = advisor.build_financial_context(df, {}, "", 0.0)
    assert "not defined (no income)" in ctx


def test_history_skips_failed_turns_and_limits_length():
    messages = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "boom", "error": True},
        {"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"},
    ]
    assert advisor.history_pairs(messages) == [("user", "q1"), ("assistant", "a1"), ("user", "q3"), ("assistant", "a3")]
    many = [m for i in range(20) for m in ({"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"})]
    assert len(advisor.history_pairs(many, max_turns=3)) == 6


def test_system_prompt_contains_required_safety_instructions():
    text = SYSTEM_PROMPT.lower()
    for phrase in ["educational", "calculated figures", "hypothetical", "never invent", "bank account",
                   "guarantee", "qualified professional", "not a licensed financial advisor", "citation"]:
        assert phrase in text, phrase


def test_prompt_builders_include_all_sections():
    p = build_advisor_prompt("Q?", "=== CALCULATED FIGURES ===\nx", "[1] Source: a.txt\nbody")
    assert p.index("CALCULATED FIGURES") < p.index("RETRIEVED EVIDENCE") < p.index("USER QUESTION")
    assert "RETRIEVED EVIDENCE" not in build_advisor_prompt("Q?", "ctx", None)
    assert "ONLY the numbered context passages" in build_rag_prompt("Q?", "[1] text")


@pytest.mark.parametrize("question,amount", [
    ("How can I save \u20b910,000 per month based on my transactions?", 10000),
    ("I want to save 5k a month", 5000),
    ("save Rs. 20,000 monthly", 20000),
    ("Help me set aside \u20b98000 each month", 8000),
])
def test_parse_monthly_savings_target(question, amount):
    assert advisor.parse_monthly_savings_target(question) == amount


@pytest.mark.parametrize("question", ["How can I save \u20b950,000 in six months?", "What is my monthly income?", "", None])
def test_parse_monthly_target_none_for_other_questions(question):
    assert advisor.parse_monthly_savings_target(question) is None


def test_needs_transaction_data_detection():
    assert advisor.needs_transaction_data("Which categories exceed my budget?")
    assert advisor.needs_transaction_data("Where am I spending the most?")
    assert not advisor.needs_transaction_data("How do I bake a chocolate cake?")
    assert not advisor.needs_transaction_data("How do I start investing?")
    assert not advisor.needs_transaction_data("What is compound interest?")
    assert not advisor.needs_transaction_data("Explain the 50/30/20 budgeting rule.")


def test_context_has_category_percent_of_income_and_monthly_averages():
    ctx = advisor.build_financial_context(load_sample_transactions(), {}, "", 0.0)
    assert "% of income, average" in ctx and "per month" in ctx
    assert "Unusual spending" in ctx and "Travel in 2026-08" in ctx


def test_context_monthly_target_check_and_shortfall():
    df = load_sample_transactions()
    ok = advisor.build_financial_context(df, {}, "save \u20b910,000 per month", 0.0)
    assert "Target met at the current rate: yes" in ok and "Savings goal check" not in ok
    short = advisor.build_financial_context(df, {}, "I want to save \u20b950,000 per month", 0.0)
    assert "Target met at the current rate: no (extra" in short


def test_system_prompt_asks_for_readable_markdown_and_separation():
    text = SYSTEM_PROMPT.lower()
    assert "markdown" in text and "bullet" in text and "separate" in text
