"""Shared test fixtures.

IMPORTANT: ``HashingEmbedder`` is a deterministic, dependency-free TEST DOUBLE used only so the
unit tests run offline without downloading a neural embedding model. The real application uses
Sentence Transformers (see src/vector_store.py). Retrieval *quality* with the real model is
covered by the optional integration test in test_integration_optional.py.
"""
import re
import zlib

import numpy as np
import pytest

import config
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore

STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "is", "are", "it", "that", "this", "with",
    "as", "be", "by", "at", "from", "do", "does", "how", "what", "should", "i", "you", "your", "my", "can",
    "was", "were", "have", "has", "who", "why", "where", "which", "they", "their", "than", "then", "so",
}


class HashingEmbedder:
    """Bag-of-words hashing embedder (test double)."""

    model_name = "test-hashing-embedder"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                if word in STOPWORDS:
                    continue
                if len(word) > 3 and word.endswith("s"):
                    word = word[:-1]
                out[row, zlib.crc32(word.encode()) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


@pytest.fixture
def embedder():
    return HashingEmbedder()


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "vs")


@pytest.fixture
def pipeline(store, embedder):
    return RAGPipeline(store, embedder, chunk_size=500, overlap=80)


@pytest.fixture
def indexed_pipeline(pipeline):
    """A pipeline with the bundled sample documents indexed."""
    results = pipeline.ingest_directory(config.SAMPLE_DOCS_DIR)
    assert all(r.status == "added" for r in results), results
    return pipeline


class FakeLLM:
    """Test double for the Gemini client. Never used by the real app."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, prompt, system_instruction=None, history=None, temperature=0.0):
        self.calls.append(prompt)
        return self.reply


@pytest.fixture
def fake_llm_factory():
    return FakeLLM
