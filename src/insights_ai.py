"""Optional Gemini explanation of the verified insight facts, with a deterministic fallback.

Gemini only receives the calculated facts (merchant names only if the user opts in). Its answer
is checked: any figure that does not appear in the facts causes the AI text to be discarded and
replaced by the deterministic summary.
"""
from __future__ import annotations

import re
from typing import Optional

from src.gemini_client import GeminiClient, GeminiError
from src.insights import InsightReport, facts_text, fallback_summary
from src.prompts import INSIGHTS_SYSTEM_PROMPT, build_insights_prompt

_NUMBER_RE = re.compile(r"(\u20b9|rs\.?\s*)?(\d[\d,]*(?:\.\d+)?)(\s*%)?", re.IGNORECASE)
_DATE_LIKE = re.compile(r"\b\d{4}-\d{2}(?:-\d{2})?\b")


def _numbers(text: str) -> list[tuple[float, bool]]:
    """Numbers worth checking: money, percentages, or any number with 3+ digits. Returns (value, is_percent)."""
    found = []
    for money, digits, pct in _NUMBER_RE.findall(_DATE_LIKE.sub(" ", text)):
        value = float(digits.replace(",", ""))
        if money or pct or len(digits.replace(",", "").split(".")[0]) >= 3:
            found.append((value, bool(pct)))
    return found


def find_unsupported_numbers(answer: str, facts: str) -> list[str]:
    """Figures in ``answer`` that do not appear in ``facts`` (percentages may differ by rounding of 0.5)."""
    known = _numbers(facts)
    bad = []
    for value, is_pct in _numbers(answer):
        tolerance = 0.5 if is_pct else 0.0
        if not any(abs(value - k) <= tolerance or (not is_pct and abs(value - k) < 0.005) for k, _ in known):
            bad.append(f"{value:g}{'%' if is_pct else ''}")
    return bad


def explain_insights(report: InsightReport, client: Optional[GeminiClient] = None, include_merchants: bool = False) -> dict:
    """Return {"text", "source": "gemini"|"fallback", "warning", "prompt"}. Never raises for Gemini problems."""
    facts = facts_text(report, include_merchants)
    prompt = build_insights_prompt(facts)
    fallback = fallback_summary(report)
    if not report.has_data:
        return {"text": fallback, "source": "fallback", "warning": None, "prompt": prompt}
    try:
        client = client or GeminiClient()
        text = client.generate(prompt, system_instruction=INSIGHTS_SYSTEM_PROMPT, temperature=0.2)
    except GeminiError as exc:
        return {"text": fallback, "source": "fallback", "prompt": prompt,
                "warning": f"{exc.message} Showing the calculated summary instead."}
    unsupported = find_unsupported_numbers(text, facts)
    if unsupported:
        return {"text": fallback, "source": "fallback", "prompt": prompt,
                "warning": "The AI explanation mentioned figures that were not in the verified results ("
                           + ", ".join(unsupported[:5]) + "), so it was discarded. Showing the calculated summary instead."}
    return {"text": text, "source": "gemini", "warning": None, "prompt": prompt}