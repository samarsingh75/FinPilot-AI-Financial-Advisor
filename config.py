"""Central configuration for FinPilot.

All settings are read from environment variables (optionally loaded from a
local ``.env`` file). No API key is ever hardcoded here.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load variables from .env (if it exists). Real environment variables win.
load_dotenv(BASE_DIR / ".env", override=False)

# ---------------------------------------------------------------- paths
DATA_DIR = BASE_DIR / "data"
SAMPLE_CSV = DATA_DIR / "sample_transactions.csv"
KB_DIR = BASE_DIR / "knowledge_base"
SAMPLE_DOCS_DIR = KB_DIR / "documents"
VECTOR_STORE_DIR = KB_DIR / "vector_store"

# ---------------------------------------------------------------- Gemini
PLACEHOLDER_KEYS = {"", "your_api_key_here", "paste_your_key_here"}
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_TEMPERATURE = 0.3


def get_api_key() -> str | None:
    """Return the Gemini API key, or None if it is missing / still a placeholder."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return None if key in PLACEHOLDER_KEYS else key


def get_model_name() -> str:
    """Return the configured Gemini model name (configurable through .env)."""
    name = os.getenv("GEMINI_MODEL", "").strip()
    if not name or name == "your_supported_model_name":
        return DEFAULT_GEMINI_MODEL
    return name


# ---------------------------------------------------------------- RAG
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHUNK_SIZE = 800      # characters
DEFAULT_CHUNK_OVERLAP = 150   # characters
DEFAULT_TOP_K = 4
# Minimum cosine similarity for a chunk to count as "relevant evidence".
DEFAULT_MIN_SCORE = 0.25
MAX_UPLOAD_MB = 20

# ---------------------------------------------------------------- Dashboard
MAX_CSV_MB = 10
CURRENCY_SYMBOL = "\u20b9"  # Indian Rupee sign

# Illustrative monthly budgets used only with the fictional sample dataset.
SAMPLE_BUDGETS = {
    "Rent": 18000,
    "Groceries": 9500,
    "Utilities": 3500,
    "Transport": 4200,
    "Dining Out": 4500,
    "Entertainment": 2000,
    "Shopping": 3000,
    "Healthcare": 1000,
    "Education": 2000,
    "Subscriptions": 700,
    "Travel": 2500,
    "Insurance": 1250,
}
