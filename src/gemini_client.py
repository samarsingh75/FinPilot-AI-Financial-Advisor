"""Thin wrapper around the Google Gen AI SDK (``google-genai``).

Responsibilities: create the client, call the model, and translate SDK / network
errors into clear, user-friendly ``GeminiError`` messages. The API key is never
logged or included in error messages.
"""
from __future__ import annotations

from typing import Optional, Sequence

import config

# Error "kinds" so the UI can react differently if it wants to.
MISSING_KEY = "missing_key"
AUTH = "auth"
QUOTA = "quota"
MODEL_NOT_FOUND = "model_not_found"
UNAVAILABLE = "unavailable"
NETWORK = "network"
EMPTY = "empty_response"
BAD_REQUEST = "bad_request"
UNKNOWN = "unknown"


class GeminiError(Exception):
    """A Gemini problem with a message that is safe and helpful to show to the user."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message


def is_configured() -> bool:
    return config.get_api_key() is not None


def _scrub(text: str, api_key: Optional[str]) -> str:
    return text.replace(api_key, "***") if api_key else text


def classify_exception(exc: Exception, model: str = "", api_key: Optional[str] = None) -> GeminiError:
    """Convert an SDK/network exception into a GeminiError."""
    if isinstance(exc, GeminiError):
        return exc

    code = getattr(exc, "code", None)
    status = str(getattr(exc, "status", "") or "")
    detail = _scrub(str(getattr(exc, "message", "") or str(exc)), api_key)[:300]

    if isinstance(code, int):
        lowered = detail.lower()
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            return GeminiError(
                QUOTA,
                "Gemini rate limit or free-tier quota reached. Free-tier limits are set by Google, "
                "can change, and may vary by model. Wait a minute and try again, choose a different "
                "model in .env, or check your usage in Google AI Studio.",
            )
        if code in (401, 403) or "api key not valid" in lowered or "api_key_invalid" in lowered:
            return GeminiError(
                AUTH,
                "Gemini rejected the API key or the request was not permitted. Check that GEMINI_API_KEY "
                "in your .env file is correct and that the key is enabled in Google AI Studio.",
            )
        if code == 404:
            return GeminiError(
                MODEL_NOT_FOUND,
                f"The model '{model}' was not found or is no longer available. Google retires models "
                "regularly. Set GEMINI_MODEL in your .env file to a currently supported model "
                "(see the Google AI Studio model list).",
            )
        if code >= 500:
            return GeminiError(
                UNAVAILABLE,
                "The Gemini service is temporarily unavailable or overloaded. Please try again shortly.",
            )
        if code == 400:
            return GeminiError(BAD_REQUEST, f"Gemini could not process the request. Details: {detail}")
        return GeminiError(UNKNOWN, f"Gemini returned an error (HTTP {code}). Details: {detail}")

    name = type(exc).__name__
    if isinstance(exc, TimeoutError) or "Timeout" in name:
        return GeminiError(
            NETWORK,
            "Gemini took too long to respond (timeout). Try again, ask a shorter question, or check your connection.",
        )
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)) or any(
        marker in name for marker in ("Connect", "Timeout", "Network", "Transport", "Remote")
    ):
        return GeminiError(
            NETWORK,
            "Could not reach the Gemini API. Check your internet connection (or proxy/firewall) and try again.",
        )
    return GeminiError(UNKNOWN, f"Unexpected error while calling Gemini ({name}): {detail}")


class GeminiClient:
    """Generate text with Gemini. Raises GeminiError on any failure."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, client=None):
        self.api_key = api_key if api_key is not None else config.get_api_key()
        self.model = model or config.get_model_name()
        if client is None and not self.api_key:
            raise GeminiError(
                MISSING_KEY,
                "No Gemini API key is configured. Create a free key in Google AI Studio, put it in the "
                ".env file as GEMINI_API_KEY, and restart the app.",
            )
        self._client = client

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:  # pragma: no cover - depends on install
                raise GeminiError(
                    UNKNOWN, "The 'google-genai' package is not installed. Run: pip install -r requirements.txt"
                ) from exc
            self._client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=60_000))
        return self._client

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        history: Optional[Sequence[tuple[str, str]]] = None,
        temperature: float = config.GEMINI_TEMPERATURE,
    ) -> str:
        """Return the model's text answer.

        ``history`` is a sequence of ``(role, text)`` pairs where role is "user" or "assistant".
        """
        try:
            from google.genai import types

            contents = []
            for role, text in history or []:
                contents.append(
                    types.Content(role="user" if role == "user" else "model", parts=[types.Part(text=text)])
                )
            contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
            response = self._get_client().models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_instruction, temperature=temperature),
            )
        except GeminiError:
            raise
        except Exception as exc:  # noqa: BLE001 - translated into a friendly error
            raise classify_exception(exc, self.model, self.api_key) from None

        text = getattr(response, "text", None)
        if not text or not text.strip():
            raise GeminiError(
                EMPTY,
                "Gemini returned an empty response. It may have been blocked by a safety filter. "
                "Try rephrasing your question.",
            )
        return text.strip()

    def list_generation_models(self) -> list[str]:
        """Names of models available to this key that support text generation."""
        try:
            names = []
            for model in self._get_client().models.list():
                actions = getattr(model, "supported_actions", None) or []
                if "generateContent" in actions:
                    names.append(str(model.name).removeprefix("models/"))
            return sorted(names)
        except Exception as exc:  # noqa: BLE001
            raise classify_exception(exc, self.model, self.api_key) from None
