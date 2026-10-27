"""OPTIONAL integration tests. They need internet access and are skipped by default.

Enable with:   set RUN_INTEGRATION=1      (Windows cmd)   or   $env:RUN_INTEGRATION=1   (PowerShell)
Gemini test additionally needs a real GEMINI_API_KEY in .env. The embedding test downloads
the Sentence Transformers model (about 90 MB) on first run.
"""
import os

import pytest

import config
from src import rag_evaluation as ev
from src.gemini_client import GeminiClient, is_configured
from src.rag_pipeline import RAGPipeline, citation_check
from src.vector_store import SentenceTransformerEmbedder, VectorStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run integration tests"),
]


def test_real_embedding_model_retrieval_quality(tmp_path):
    pipeline = RAGPipeline(VectorStore(tmp_path / "vs"), SentenceTransformerEmbedder())
    assert all(r.status == "added" for r in pipeline.ingest_directory(config.SAMPLE_DOCS_DIR))
    result = ev.retrieval_hit_rate(pipeline, top_k=3)
    assert result["hit_rate"] >= 0.75, result["rows"]
    assert all(r["correctly_rejected"] for r in ev.irrelevant_question_check(pipeline)), "tune DEFAULT_MIN_SCORE"


@pytest.mark.skipif(not is_configured(), reason="GEMINI_API_KEY is not configured")
def test_real_gemini_answer_has_valid_citations(tmp_path):
    pipeline = RAGPipeline(VectorStore(tmp_path / "vs"), SentenceTransformerEmbedder())
    pipeline.ingest_directory(config.SAMPLE_DOCS_DIR)
    answer = pipeline.answer("How many months of expenses should an emergency fund cover?", GeminiClient())
    assert answer.llm_used and citation_check(answer.answer, answer.retrieved)["passes"]
