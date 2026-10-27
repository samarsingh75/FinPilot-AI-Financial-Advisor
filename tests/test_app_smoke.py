"""Streamlit smoke tests using Streamlit's AppTest (no browser, no network).

The Gemini client and the embedding model are replaced by test doubles ONLY inside these tests,
so the pages can be exercised end to end offline. The real app never uses these doubles.
"""
import pytest
from streamlit.testing.v1 import AppTest

import config
from src.gemini_client import QUOTA, GeminiError
from src.vector_store import VectorStore
from tests.conftest import HashingEmbedder

TIMEOUT = 60


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _dashboard():
    from src import dashboard
    dashboard.render()


def _advisor():
    from src import advisor
    advisor.render()


def _kb():
    from src import knowledge_base_page
    knowledge_base_page.render()


@pytest.fixture
def kb_env(tmp_path, monkeypatch):
    """Point the knowledge-base page at a temp index and the offline test embedder."""
    from src import knowledge_base_page as kb
    store, embedder = VectorStore(tmp_path / "vs"), HashingEmbedder()
    monkeypatch.setattr(kb, "get_store", lambda: store)
    monkeypatch.setattr(kb, "get_embedder", lambda: embedder)
    return store


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


@pytest.fixture
def with_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")


def test_main_app_starts_without_errors(no_key):
    at = AppTest.from_file(str(config.BASE_DIR / "app.py"), default_timeout=TIMEOUT).run()
    assert not at.exception
    assert [h.value for h in at.header] == ["Financial Dashboard"]
    assert any("Gemini API key not set" in w.value for w in at.warning)


def test_dashboard_empty_state_then_sample_data(no_key):
    at = AppTest.from_function(_dashboard, default_timeout=TIMEOUT).run()
    assert not at.exception and any("Get started" in i.value for i in at.info)
    _button(at, "Load sample dataset").click().run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total income"] == "\u20b94,36,795" and metrics["Savings rate"] == "28.0%"
    assert metrics["Highest spending category"] == "Rent"
    assert sum("is over budget" in w.value for w in at.warning) == 4
    assert len(at.number_input) >= 10  # budget inputs


def _load_sample(at):
    from src.data_loader import load_sample_transactions
    at.session_state["transactions"] = load_sample_transactions()
    at.session_state["budgets"] = dict(config.SAMPLE_BUDGETS)
    at.session_state["data_source"] = "Sample dataset (fictional data)"


def _ask(at, text):
    at.text_area[0].set_value(text)
    _button(at, "Ask Advisor").click().run()


class FakeGemini:
    """Test double: records what it was sent and returns a canned Markdown answer."""
    calls = []
    reply = "### Summary\n- Your savings rate is 28.0%."

    def __init__(self, *a, **k): pass

    def generate(self, prompt, system_instruction=None, history=None, **k):
        FakeGemini.calls.append({"prompt": prompt, "system": system_instruction, "history": history})
        return FakeGemini.reply


@pytest.fixture
def fake_gemini(with_key, monkeypatch):
    FakeGemini.calls = []
    FakeGemini.reply = "### Summary\n- Your savings rate is 28.0%."
    monkeypatch.setattr("src.advisor.GeminiClient", FakeGemini)
    return FakeGemini


@pytest.fixture
def empty_kb(tmp_path, monkeypatch):
    """Advisor tests that do not care about the knowledge base: point it at an empty temp index."""
    from src import knowledge_base_page as kb
    store = VectorStore(tmp_path / "advisor_vs")
    monkeypatch.setattr(kb, "get_store", lambda: store)
    monkeypatch.setattr(kb, "get_embedder", lambda: HashingEmbedder())
    return store


def test_advisor_without_key_is_disabled_and_explained(no_key, empty_kb):
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("Gemini is not configured" in w.value for w in at.warning)
    assert _button(at, "Ask Advisor").disabled is True
    assert not at.chat_message  # nothing was faked


def test_advisor_has_visible_text_area_and_ask_button(fake_gemini, empty_kb):
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT).run()
    assert len(at.text_area) == 1 and _button(at, "Ask Advisor").disabled is False


def test_custom_prompt_is_sent_with_real_figures_and_no_descriptions(fake_gemini, empty_kb):
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT)
    _load_sample(at)
    at.run()
    question = "Analyze my spending and identify where I waste the most money."
    _ask(at, question)
    assert not at.exception and len(fake_gemini.calls) == 1
    call = fake_gemini.calls[0]
    assert question in call["prompt"] and "Savings rate: 28.0%" in call["prompt"]
    assert "% of income" in call["prompt"] and "Unusual spending" in call["prompt"]
    assert "Monthly salary credit" not in call["prompt"]
    assert "markdown" in call["system"].lower()
    assert [m.markdown[0].value for m in at.chat_message][-1].startswith("### Summary")
    assert at.text_area[0].value == ""  # box cleared after a successful answer


def test_advisor_uses_currently_loaded_dataset_not_stale_data(fake_gemini, empty_kb):
    import pandas as pd
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT)
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-05", "2026-01-20"]), "description": ["x", "y"],
                       "category": ["Pay", "Gadgets"], "type": ["income", "expense"], "amount": [1000.0, 400.0]})
    df["month"] = "2026-01"
    at.session_state["transactions"] = df
    at.run()
    _ask(at, "What is my savings rate?")
    prompt = fake_gemini.calls[0]["prompt"]
    assert "Total income: \u20b91,000" in prompt and "Gadgets" in prompt and "Rent" not in prompt


def test_follow_up_questions_work_repeatedly_and_history_is_kept(fake_gemini, empty_kb):
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT)
    _load_sample(at)
    at.run()
    _ask(at, "Which categories are consuming most of my income?")
    _ask(at, "And how can I cut the biggest one?")
    _ask(at, "Create a realistic monthly budget based on my spending.")
    assert len(fake_gemini.calls) == 3
    assert fake_gemini.calls[1]["history"][0] == ("user", "Which categories are consuming most of my income?")
    assert len(fake_gemini.calls[2]["history"]) == 4
    assert len(at.chat_message) == 6  # all earlier questions and answers stay visible
    _button(at, "Reset conversation").click().run()
    assert not at.chat_message


def test_empty_prompt_is_rejected_without_calling_gemini(fake_gemini, empty_kb):
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT)
    _load_sample(at)
    at.run()
    _ask(at, "   \n  ")
    assert not at.exception and fake_gemini.calls == []
    assert any("type a question" in w.value for w in at.warning)


def test_personal_question_without_data_shows_error_and_skips_gemini(fake_gemini, empty_kb):
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT).run()
    _ask(at, "How can I save \u20b910,000 per month based on my transactions?")
    assert not at.exception and fake_gemini.calls == []
    assert any("no transaction data is loaded" in e.value for e in at.error)


def test_general_question_without_data_still_works(fake_gemini, empty_kb):
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT).run()
    _ask(at, "What is compound interest?")
    assert len(fake_gemini.calls) == 1 and "NOT loaded" in fake_gemini.calls[0]["prompt"]


def test_monthly_saving_target_is_calculated_before_prompting(fake_gemini, empty_kb):
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT)
    _load_sample(at)
    at.run()
    _ask(at, "How can I save \u20b910,000 per month based on my transactions?")
    prompt = fake_gemini.calls[0]["prompt"]
    assert "Target: \u20b910,000 per month" in prompt and "Target met at the current rate: yes" in prompt


def test_gemini_error_is_shown_and_question_is_kept(with_key, monkeypatch, empty_kb):
    class Failing:
        def __init__(self, *a, **k): pass
        def generate(self, *a, **k):
            raise GeminiError(QUOTA, "Gemini rate limit or free-tier quota reached.")

    monkeypatch.setattr("src.advisor.GeminiClient", Failing)
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT).run()
    _ask(at, "Explain compound interest.")
    assert not at.exception
    assert any("quota reached" in e.value for e in at.error)
    assert at.text_area[0].value == "Explain compound interest."  # can retry without retyping


def test_unexpected_exception_does_not_crash_the_page(with_key, monkeypatch, empty_kb):
    class Broken:
        def __init__(self, *a, **k): pass
        def generate(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr("src.advisor.GeminiClient", Broken)
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT).run()
    _ask(at, "What is compound interest?")
    assert not at.exception and any("unexpected" in e.value for e in at.error)


def test_advisor_uses_rag_context_from_existing_index(fake_gemini, kb_env):
    from src.rag_pipeline import RAGPipeline
    RAGPipeline(kb_env, HashingEmbedder()).ingest_directory(config.SAMPLE_DOCS_DIR)  # index already on disk
    FakeGemini.reply = "Aim for three to six months of essential expenses [1]."
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT)
    _load_sample(at)
    at.run()
    _ask(at, "How many months of expenses should my emergency fund cover?")
    prompt = fake_gemini.calls[0]["prompt"]
    assert "RETRIEVED EVIDENCE" in prompt and "Source: emergency_fund_guide.txt" in prompt
    assert prompt.index("CALCULATED FIGURES") < prompt.index("RETRIEVED EVIDENCE")  # kept separate
    assert any("Cited sources" in c.value and "emergency_fund_guide.txt" in c.value for c in at.caption)
    assert kb_env.document_count == 5  # nothing was re-indexed


def test_advisor_continues_when_no_passage_is_relevant(fake_gemini, kb_env):
    from src.rag_pipeline import RAGPipeline
    RAGPipeline(kb_env, HashingEmbedder()).ingest_directory(config.SAMPLE_DOCS_DIR)
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT).run()
    _ask(at, "How do I bake a chocolate cake?")
    assert len(fake_gemini.calls) == 1 and "RETRIEVED EVIDENCE" not in fake_gemini.calls[0]["prompt"]
    assert any("No relevant passages" in c.value for c in at.caption)


def test_advisor_continues_when_embedding_model_fails(fake_gemini, kb_env, monkeypatch):
    from src import knowledge_base_page as kb
    from src.rag_pipeline import RAGPipeline
    from src.vector_store import EmbeddingError
    RAGPipeline(kb_env, HashingEmbedder()).ingest_directory(config.SAMPLE_DOCS_DIR)

    class Broken:
        model_name = "test-hashing-embedder"
        def encode(self, texts):
            raise EmbeddingError("Could not load the embedding model.")

    monkeypatch.setattr(kb, "get_embedder", lambda: Broken())
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT).run()
    _ask(at, "What is compound interest?")
    assert not at.exception and len(fake_gemini.calls) == 1
    assert any("Could not load the embedding model" in c.value for c in at.caption)


def test_advisor_works_with_empty_knowledge_base(fake_gemini, empty_kb):
    at = AppTest.from_function(_advisor, default_timeout=TIMEOUT).run()
    _ask(at, "Explain the 50/30/20 budgeting rule.")
    assert not at.exception and len(fake_gemini.calls) == 1
    assert at.checkbox[0].disabled is True


def test_knowledge_base_end_to_end_with_offline_embedder(kb_env, with_key, monkeypatch):
    at = AppTest.from_function(_kb, default_timeout=TIMEOUT).run()
    assert not at.exception
    _button(at, "Index sample documents").click().run()
    assert not at.exception
    assert kb_env.document_count == 5
    assert sum("Indexed" in s.value for s in at.success) == 5
    counts = {m.label: m.value for m in at.metric}
    assert counts["Indexed documents"] == "5"

    at.text_input[0].set_value("How many months of expenses should an emergency fund cover?").run()
    _button(at, "Retrieve passages only (no Gemini)").click().run()
    assert not at.exception
    assert any("emergency_fund_guide.txt" in e.label for e in at.expander)

    class FakeGemini:
        def __init__(self, *a, **k): pass
        def generate(self, prompt, system_instruction=None, **k):
            assert "[1] Source:" in prompt
            return "Three to six months of essential expenses [1]."

    monkeypatch.setattr("src.knowledge_base_page.GeminiClient", FakeGemini)
    _button(at, "Ask Gemini").click().run()
    assert not at.exception
    assert any("Cited sources" in m.value and "emergency_fund_guide.txt" in m.value for m in at.markdown)


def test_knowledge_base_irrelevant_question_does_not_call_gemini(kb_env, with_key, monkeypatch):
    class MustNotBeCalled:
        def __init__(self, *a, **k): pass
        def generate(self, *a, **k):
            raise AssertionError("Gemini must not be called without evidence")

    monkeypatch.setattr("src.knowledge_base_page.GeminiClient", MustNotBeCalled)
    at = AppTest.from_function(_kb, default_timeout=TIMEOUT).run()
    _button(at, "Index sample documents").click().run()
    at.text_input[0].set_value("How do I bake a chocolate cake?").run()
    _button(at, "Ask Gemini").click().run()
    assert not at.exception
    assert any("could not find enough information" in m.value for m in at.markdown)


def test_knowledge_base_reports_missing_embedding_dependency_gracefully(tmp_path, monkeypatch, no_key):
    """With the real embedder and no model available, the page shows a clear error instead of crashing."""
    from src import knowledge_base_page as kb
    from src.vector_store import EmbeddingError

    class Broken:
        model_name = "broken"
        def encode(self, texts):
            raise EmbeddingError("Could not load the embedding model.")

    store = VectorStore(tmp_path / "vs")
    monkeypatch.setattr(kb, "get_store", lambda: store)
    monkeypatch.setattr(kb, "get_embedder", lambda: Broken())
    at = AppTest.from_function(_kb, default_timeout=TIMEOUT).run()
    _button(at, "Index sample documents").click().run()
    assert not at.exception
    assert any("Could not load the embedding model" in e.value for e in at.error)
    assert store.is_empty


# ---------------------------------------------------------------- Smart Spending Insights (dashboard section)
def _dashboard_with(csv_text=None):
    at = AppTest.from_function(_dashboard, default_timeout=TIMEOUT)
    if csv_text is None:
        at.run()
        _button(at, "Load sample dataset").click().run()
    else:
        from src.data_loader import load_transactions
        at.session_state["transactions"] = load_transactions(csv_text.encode())
        at.session_state["data_source"] = "test.csv"
        at.session_state["budgets"] = {}
        at.run()
    return at


def _all_markdown(at):
    return " ".join(m.value for m in at.markdown)


def test_insights_section_shows_calculated_results_for_sample_data(no_key):
    at = _dashboard_with()
    assert not at.exception
    assert any("Smart Spending Insights" in s.value for s in at.subheader)
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Spending in 2026-09"] == "\u20b953,686" and metrics["Change vs 2026-08"] == "-15.7%"
    text = _all_markdown(at)
    assert "Dining Out spending increased by 81%" in text and "Recurring payments" in text and "Unusual transactions" in text
    assert metrics["Active fixed payments"] == "6"  # rent, broadband, insurance, metro card, OTT, music
    explain = _button(at, "Explain these insights with Gemini")
    assert explain.disabled is True  # no key: the AI summary is unavailable, the calculated insights still work


def test_insights_gemini_explanation_is_shown_when_grounded(with_key, monkeypatch):
    class Grounded:
        def __init__(self, *a, **k): pass
        def generate(self, prompt, system_instruction=None, **k):
            assert "VERIFIED FACTS" in prompt and "Monthly house rent" not in prompt  # merchant names not sent by default
            return "### What changed\nSpending fell 15.7% to \u20b953,686 in 2026-09."

    monkeypatch.setattr("src.insights_ai.GeminiClient", Grounded)
    at = _dashboard_with()
    _button(at, "Explain these insights with Gemini").click().run()
    assert not at.exception
    assert "Spending fell 15.7%" in _all_markdown(at)
    assert any("AI-generated explanation" in c.value for c in at.caption)


def test_insights_gemini_failure_shows_calculated_summary(with_key, monkeypatch):
    class Failing:
        def __init__(self, *a, **k): pass
        def generate(self, *a, **k):
            raise GeminiError(QUOTA, "Gemini rate limit or free-tier quota reached.")

    monkeypatch.setattr("src.insights_ai.GeminiClient", Failing)
    at = _dashboard_with()
    _button(at, "Explain these insights with Gemini").click().run()
    assert not at.exception
    assert any("quota reached" in i.value and "calculated summary" in i.value for i in at.info)
    assert "Overview:" in _all_markdown(at)


def test_insights_gemini_invented_numbers_are_rejected(with_key, monkeypatch):
    class Inventing:
        def __init__(self, *a, **k): pass
        def generate(self, *a, **k):
            return "Spending exploded by 400% to \u20b99,99,999."

    monkeypatch.setattr("src.insights_ai.GeminiClient", Inventing)
    at = _dashboard_with()
    _button(at, "Explain these insights with Gemini").click().run()
    assert any("were not in the verified results" in i.value for i in at.info)
    assert "exploded" not in _all_markdown(at)


def test_insights_single_month_file_explains_what_is_needed(no_key):
    csv = "date,description,category,type,amount\n2026-04-01,Pay,Salary,income,50000\n2026-04-05,Cafe,Food,expense,900\n2026-04-28,Rent,Rent,expense,10000\n"
    at = _dashboard_with(csv)
    assert not at.exception
    assert any("two consecutive months" in i.value for i in at.info)
    assert {m.label: m.value for m in at.metric}["Change vs 2026-03"] == "n/a"


def test_insights_income_only_file_does_not_crash(no_key):
    at = _dashboard_with("date,description,category,type,amount\n2026-04-01,Pay,Salary,income,50000\n")
    assert not at.exception
    assert any("no expense transactions" in i.value.lower() for i in at.info)