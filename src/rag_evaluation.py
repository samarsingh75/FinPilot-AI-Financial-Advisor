"""Small, simple RAG evaluation helpers (no API key needed).

These are basic project-evaluation checks. A high retrieval hit rate shows that the
retriever finds the expected document; it does NOT prove that generated answers are correct.
"""
from __future__ import annotations

import config
from src.rag_pipeline import RAGPipeline

# Questions about the bundled sample documents, with the document that should be retrieved.
EVAL_QUESTIONS = [
    {"question": "How many months of expenses should an emergency fund cover?", "expected_source": "emergency_fund_guide.txt"},
    {"question": "Where should I keep my emergency fund?", "expected_source": "emergency_fund_guide.txt"},
    {"question": "What is the 50/30/20 rule?", "expected_source": "fifty_thirty_twenty_rule.txt"},
    {"question": "What percentage of income goes to wants in the 50/30/20 guideline?", "expected_source": "fifty_thirty_twenty_rule.txt"},
    {"question": "What is the formula for compound interest?", "expected_source": "compound_interest_explained.txt"},
    {"question": "Why does starting early help money grow through compounding?", "expected_source": "compound_interest_explained.txt"},
    {"question": "How do I work out the monthly amount needed to reach a savings goal?", "expected_source": "savings_goals_planning.txt"},
    {"question": "What is the difference between fixed and variable expenses?", "expected_source": "budgeting_fundamentals.txt"},
]

# Questions the sample documents cannot answer; retrieval should return nothing above the threshold.
IRRELEVANT_QUESTIONS = [
    "What is the capital city of France?",
    "How do I bake a chocolate cake?",
    "Who won the football world cup?",
]


def missing_sample_documents(pipeline: RAGPipeline) -> list[str]:
    indexed = {info["source"] for info in pipeline.store.documents.values()}
    needed = {item["expected_source"] for item in EVAL_QUESTIONS}
    return sorted(needed - indexed)


def retrieval_hit_rate(pipeline: RAGPipeline, eval_set: list[dict] | None = None, top_k: int = config.DEFAULT_TOP_K) -> dict:
    """Fraction of questions whose expected source appears in the top-k retrieved chunks."""
    eval_set = eval_set or EVAL_QUESTIONS
    rows = []
    for item in eval_set:
        chunks = pipeline.retrieve(item["question"], top_k=top_k, min_score=0.0)
        sources = [c.source for c in chunks]
        rows.append(
            {
                "question": item["question"],
                "expected_source": item["expected_source"],
                "hit": item["expected_source"] in sources,
                "top_source": sources[0] if sources else None,
                "top_score": round(chunks[0].score, 3) if chunks else None,
                "has_page_or_source_metadata": all(bool(c.source) for c in chunks),
            }
        )
    hits = sum(r["hit"] for r in rows)
    return {"hits": hits, "total": len(rows), "hit_rate": hits / len(rows) if rows else 0.0, "rows": rows}


def irrelevant_question_check(
    pipeline: RAGPipeline, questions: list[str] | None = None, min_score: float = config.DEFAULT_MIN_SCORE, top_k: int = config.DEFAULT_TOP_K
) -> list[dict]:
    """For off-topic questions, check that no chunk passes the relevance threshold."""
    rows = []
    for question in questions or IRRELEVANT_QUESTIONS:
        best = pipeline.retrieve(question, top_k=1, min_score=0.0)
        passing = pipeline.retrieve(question, top_k=top_k, min_score=min_score)
        rows.append(
            {
                "question": question,
                "best_score": round(best[0].score, 3) if best else None,
                "chunks_above_threshold": len(passing),
                "correctly_rejected": len(passing) == 0,
            }
        )
    return rows
