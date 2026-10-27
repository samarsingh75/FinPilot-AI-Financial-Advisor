import numpy as np
import pymupdf
import pytest

import config
from src import document_processor as dp
from src import rag_evaluation as ev
from src.rag_pipeline import RAGPipeline
from src.vector_store import VectorStore, VectorStoreError

LONG = " ".join(f"Sentence number {i} talks about saving money wisely." for i in range(60))


def make_pdf(pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


# ---------------------------------------------------------------- chunking
def test_chunks_respect_size_limit_and_cover_text():
    chunks = dp.chunk_text(LONG, chunk_size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0].startswith("Sentence number 0")
    assert chunks[-1].endswith("wisely.")


def test_chunks_overlap():
    chunks = dp.chunk_text(LONG, chunk_size=300, overlap=100)
    shared = [set(a.split()) & set(b.split()) for a, b in zip(chunks, chunks[1:])]
    assert all(len(s) > 3 for s in shared)


def test_zero_overlap_and_small_text():
    assert dp.chunk_text("short text", 300, 0) == ["short text"]
    assert len(dp.chunk_text(LONG, 300, 0)) > 3


def test_empty_text_gives_no_chunks():
    assert dp.chunk_text("", 300, 50) == [] and dp.chunk_text("  \n\t ", 300, 50) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_chunk_settings(size, overlap):
    with pytest.raises(ValueError):
        dp.chunk_text("some text", size, overlap)


def test_text_without_spaces_still_terminates():
    chunks = dp.chunk_text("x" * 1000, 100, 20)
    assert len(chunks) >= 10 and all(len(c) <= 100 for c in chunks)


def test_build_chunks_metadata():
    pages = [dp.ExtractedPage(1, LONG), dp.ExtractedPage(2, "Page two text.")]
    chunks = dp.build_chunks("a.pdf", "abc123def456xyz", pages, 300, 50)
    assert all(c["source"] == "a.pdf" and c["doc_hash"] == "abc123def456xyz" for c in chunks)
    assert chunks[0]["page"] == 1 and chunks[-1]["page"] == 2
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


# ---------------------------------------------------------------- extraction
def test_txt_extraction_utf16_with_bom():
    assert dp.extract_pages("n.txt", "Budget plan".encode("utf-16"))[0].text == "Budget plan"


def test_txt_extraction_and_encodings():
    assert dp.extract_pages("n.txt", "Hello \u20b9100".encode("utf-8"))[0].text == "Hello \u20b9100"
    assert dp.extract_pages("n.txt", "caf\xe9".encode("cp1252"))[0].text == "caf\xe9"
    assert dp.extract_pages("n.txt", b"x")[0].page is None


def test_pdf_extraction_keeps_page_numbers_and_skips_blank_pages():
    pages = dp.extract_pages("f.pdf", make_pdf(["First page text", "", "Third page text"]))
    assert [p.page for p in pages] == [1, 3]
    assert "Third page" in pages[1].text


def test_pdf_without_text_raises_clear_error():
    with pytest.raises(dp.DocumentError, match="no extractable text"):
        dp.extract_pages("scan.pdf", make_pdf([""]))


def test_empty_and_whitespace_documents_rejected():
    with pytest.raises(dp.DocumentError, match="empty"):
        dp.extract_pages("a.txt", b"")
    with pytest.raises(dp.DocumentError, match="no extractable text"):
        dp.extract_pages("a.txt", b"   \n \n  ")


def test_corrupt_pdf_and_unsupported_type():
    with pytest.raises(dp.DocumentError):
        dp.extract_pages("bad.pdf", b"this is not a pdf")
    with pytest.raises(dp.DocumentError, match="unsupported"):
        dp.extract_pages("a.docx", b"data")


def test_oversize_upload_rejected(monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)
    with pytest.raises(dp.DocumentError, match="larger"):
        dp.extract_pages("a.txt", b"hello")


def test_safe_filename_strips_paths():
    assert dp.safe_filename("../../etc/passwd.txt") == "passwd.txt"
    assert dp.safe_filename("C:\\Users\\me\\notes.txt") == "notes.txt"


# ---------------------------------------------------------------- ingestion
def test_ingest_and_counts(pipeline):
    r = pipeline.ingest_file("a.txt", LONG.encode())
    assert r.status == "added" and r.chunk_count > 1
    assert pipeline.store.document_count == 1 and pipeline.store.chunk_count == r.chunk_count


def test_duplicate_upload_is_skipped(pipeline):
    pipeline.ingest_file("a.txt", LONG.encode())
    count = pipeline.store.chunk_count
    again = pipeline.ingest_file("a.txt", LONG.encode())
    same_content_other_name = pipeline.ingest_file("copy.txt", LONG.encode())
    assert again.status == "duplicate" and same_content_other_name.status == "duplicate"
    assert pipeline.store.chunk_count == count and pipeline.store.document_count == 1


def test_updated_file_replaces_old_version(pipeline):
    pipeline.ingest_file("a.txt", b"Old content about gardening and flowers.")
    r = pipeline.ingest_file("a.txt", b"New content about budgeting and savings.")
    assert r.status == "replaced" and pipeline.store.document_count == 1
    texts = " ".join(c["text"] for c in pipeline.store.chunks)
    assert "budgeting" in texts and "gardening" not in texts


def test_reindex_same_file_with_new_chunk_settings(pipeline):
    pipeline.ingest_file("a.txt", LONG.encode())
    before = pipeline.store.chunk_count
    pipeline.chunk_size, pipeline.overlap = 200, 20
    r = pipeline.ingest_file("a.txt", LONG.encode())
    assert r.status == "replaced" and pipeline.store.document_count == 1
    assert pipeline.store.chunk_count > before


def test_empty_document_is_reported_not_indexed(pipeline):
    r = pipeline.ingest_file("blank.txt", b"   \n  ")
    assert r.status == "empty" and "no extractable text" in r.message
    assert pipeline.store.is_empty
    assert pipeline.ingest_file("x.pdf", make_pdf([""])).status == "empty"


def test_bad_files_do_not_crash_ingestion(pipeline):
    assert pipeline.ingest_file("bad.pdf", b"garbage").status == "error"
    assert pipeline.ingest_file("a.exe", b"MZ").status == "error"
    pipeline.chunk_size, pipeline.overlap = 100, 200
    assert pipeline.ingest_file("a.txt", b"hello world").status == "error"


def test_index_persists_across_restart(tmp_path, embedder):
    first = RAGPipeline(VectorStore(tmp_path / "vs"), embedder)
    first.ingest_file("a.txt", b"Emergency funds cover three to six months of essential expenses.")
    second = RAGPipeline(VectorStore(tmp_path / "vs"), embedder)
    assert second.store.document_count == 1 and second.store.chunk_count == 1
    hits = second.retrieve("emergency fund months expenses", top_k=1)
    assert hits[0].source == "a.txt"


def test_remove_document_and_clear(indexed_pipeline):
    store = indexed_pipeline.store
    victim = store.find_by_source("emergency_fund_guide.txt")
    assert store.remove_document(victim) is True and store.remove_document(victim) is False
    assert store.document_count == 4 and store.index.ntotal == store.chunk_count
    assert all(c.source != "emergency_fund_guide.txt" for c in indexed_pipeline.retrieve("emergency fund", 10))
    reloaded = VectorStore(store.directory)
    assert reloaded.document_count == 4
    store.clear()
    assert store.is_empty and VectorStore(store.directory).is_empty


def test_corrupt_index_is_handled_gracefully(tmp_path, embedder):
    p = RAGPipeline(VectorStore(tmp_path / "vs"), embedder)
    p.ingest_file("a.txt", b"some text about money")
    (tmp_path / "vs" / "chunks.json").write_text("{ not json")
    broken = VectorStore(tmp_path / "vs")
    assert broken.is_empty and broken.load_error


def test_embedding_model_mismatch_is_blocked(tmp_path, embedder):
    class Other:
        model_name = "another-model"
        def encode(self, texts):
            return np.ones((len(texts), 8), dtype="float32")
    store = VectorStore(tmp_path / "vs")
    RAGPipeline(store, embedder).ingest_file("a.txt", b"first doc text")
    result = RAGPipeline(store, Other()).ingest_file("b.txt", b"second doc text")
    assert result.status == "error" and "another-model" in result.message
    assert store.document_count == 1


def test_store_rejects_inconsistent_vectors(store):
    with pytest.raises(VectorStoreError):
        store.add_chunks("h", "a.txt", [{"text": "x", "doc_hash": "h", "source": "a.txt", "chunk_id": "1"}],
                         np.zeros((2, 4), dtype="float32"), "m")


# ---------------------------------------------------------------- retrieval
def test_retrieval_returns_relevant_source_first(indexed_pipeline):
    top = indexed_pipeline.retrieve("What is the formula for compound interest?", top_k=3)[0]
    assert top.source == "compound_interest_explained.txt" and top.rank == 1 and top.score > 0.3


def test_retrieval_is_sorted_and_respects_top_k(indexed_pipeline):
    hits = indexed_pipeline.retrieve("emergency fund months", top_k=3)
    assert len(hits) == 3 and [h.rank for h in hits] == [1, 2, 3]
    assert hits[0].score >= hits[1].score >= hits[2].score


def test_min_score_filters_results(indexed_pipeline):
    assert indexed_pipeline.retrieve("emergency fund", top_k=5, min_score=0.99) == []


def test_irrelevant_question_finds_nothing_above_threshold(indexed_pipeline):
    assert indexed_pipeline.retrieve("How do I bake a chocolate cake?", top_k=4, min_score=config.DEFAULT_MIN_SCORE) == []


def test_retrieval_on_empty_index_and_blank_question(pipeline):
    assert pipeline.retrieve("anything at all") == []
    with pytest.raises(ValueError):
        pipeline.retrieve("   ")


def test_retrieval_hit_rate_on_sample_documents(indexed_pipeline):
    assert len(ev.EVAL_QUESTIONS) >= 5
    result = ev.retrieval_hit_rate(indexed_pipeline, top_k=3)
    assert result["total"] == len(ev.EVAL_QUESTIONS)
    assert result["hit_rate"] >= 0.75, result["rows"]


def test_irrelevant_question_check_helper(indexed_pipeline):
    rows = ev.irrelevant_question_check(indexed_pipeline)
    assert rows and all(r["correctly_rejected"] for r in rows)


def test_missing_sample_documents_helper(indexed_pipeline, tmp_path, embedder):
    empty = RAGPipeline(VectorStore(tmp_path / "other"), embedder)
    assert len(ev.missing_sample_documents(empty)) == 5
    assert ev.missing_sample_documents(indexed_pipeline) == []


def test_sample_documents_cover_required_topics():
    names = {p.name for p in config.SAMPLE_DOCS_DIR.glob("*.txt")}
    assert len(names) >= 5
    text = (config.SAMPLE_DOCS_DIR / "fifty_thirty_twenty_rule.txt").read_text(encoding="utf-8").lower()
    assert "not a universal requirement" in text  # must be labelled as a guideline
