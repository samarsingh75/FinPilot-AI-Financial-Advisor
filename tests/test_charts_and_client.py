import httpx
import pytest
from google.genai import errors

import config
from src import charts
from src import financial_calculations as fc
from src import gemini_client as gc
from src.budget_analyzer import compare_budgets
from src.data_loader import load_sample_transactions


def test_all_charts_build_from_sample_data():
    df = load_sample_transactions()
    figs = [
        charts.income_vs_expenses_chart(fc.total_income(df), fc.total_expenses(df)),
        charts.expense_donut_chart(fc.category_totals(df)),
        charts.monthly_trend_chart(fc.monthly_summary(df)),
        charts.budget_vs_actual_chart(compare_budgets(df, config.SAMPLE_BUDGETS)),
    ]
    assert all(len(f.data) >= 1 for f in figs)
    assert figs[1].data[0].hole > 0  # donut
    assert len(figs[2].data) == 2 and len(figs[3].data) == 2
    assert figs[0].data[0].text[0].startswith("\u20b9")


# ---------------------------------------------------------------- Gemini client (no network)
def test_missing_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert gc.is_configured() is False
    with pytest.raises(gc.GeminiError) as exc:
        gc.GeminiClient()
    assert exc.value.kind == gc.MISSING_KEY and "Google AI Studio" in exc.value.message


def test_placeholder_key_counts_as_missing(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_api_key_here")
    assert gc.is_configured() is False
    monkeypatch.setenv("GEMINI_API_KEY", "real-looking-key")
    assert gc.is_configured() is True


def test_model_name_is_configurable(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "my-custom-model")
    assert config.get_model_name() == "my-custom-model"
    monkeypatch.setenv("GEMINI_MODEL", "your_supported_model_name")
    assert config.get_model_name() == config.DEFAULT_GEMINI_MODEL


@pytest.mark.parametrize("code,status,kind", [
    (429, "RESOURCE_EXHAUSTED", gc.QUOTA), (403, "PERMISSION_DENIED", gc.AUTH), (401, "UNAUTHENTICATED", gc.AUTH),
    (404, "NOT_FOUND", gc.MODEL_NOT_FOUND), (503, "UNAVAILABLE", gc.UNAVAILABLE), (400, "INVALID_ARGUMENT", gc.BAD_REQUEST),
])
def test_api_errors_are_translated(code, status, kind):
    exc = errors.APIError(code, {"error": {"message": "details", "status": status, "code": code}})
    assert gc.classify_exception(exc, "some-model").kind == kind


def test_invalid_key_400_is_auth_error():
    exc = errors.ClientError(400, {"error": {"message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"}})
    assert gc.classify_exception(exc).kind == gc.AUTH


def test_quota_message_mentions_provider_limits():
    exc = errors.ClientError(429, {"error": {"message": "x", "status": "RESOURCE_EXHAUSTED"}})
    assert "quota" in gc.classify_exception(exc).message.lower() and "Google" in gc.classify_exception(exc).message


def test_network_errors_are_translated():
    assert gc.classify_exception(httpx.ConnectError("boom")).kind == gc.NETWORK
    assert gc.classify_exception(httpx.ReadTimeout("slow")).kind == gc.NETWORK
    assert gc.classify_exception(ConnectionError("down")).kind == gc.NETWORK


def test_api_key_is_never_echoed_in_error_messages():
    secret = "SECRET-KEY-123"
    exc = errors.ClientError(400, {"error": {"message": f"bad request for {secret}", "status": "INVALID_ARGUMENT"}})
    assert secret not in gc.classify_exception(exc, "m", secret).message


class _Resp:
    def __init__(self, text):
        self.text = text


class _Models:
    def __init__(self, outcome):
        self.outcome, self.kwargs = outcome, None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return _Resp(self.outcome)


class _FakeSdkClient:
    def __init__(self, outcome):
        self.models = _Models(outcome)


def test_generate_passes_model_system_prompt_and_history():
    sdk = _FakeSdkClient("  hello  ")
    client = gc.GeminiClient(api_key="k", model="m-1", client=sdk)
    out = client.generate("question", system_instruction="SYS", history=[("user", "q0"), ("assistant", "a0")])
    assert out == "hello"
    kwargs = sdk.models.kwargs
    assert kwargs["model"] == "m-1" and kwargs["config"].system_instruction == "SYS"
    assert [c.role for c in kwargs["contents"]] == ["user", "model", "user"]


def test_generate_translates_failures_and_empty_responses():
    quota = errors.ClientError(429, {"error": {"message": "q", "status": "RESOURCE_EXHAUSTED"}})
    with pytest.raises(gc.GeminiError) as exc:
        gc.GeminiClient(api_key="k", client=_FakeSdkClient(quota)).generate("q")
    assert exc.value.kind == gc.QUOTA
    with pytest.raises(gc.GeminiError) as exc:
        gc.GeminiClient(api_key="k", client=_FakeSdkClient("   ")).generate("q")
    assert exc.value.kind == gc.EMPTY


def test_timeout_has_its_own_readable_message():
    err = gc.classify_exception(httpx.ReadTimeout("slow"))
    assert err.kind == gc.NETWORK and "too long" in err.message
