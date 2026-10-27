
from src.prompts import INSUFFICIENT_EVIDENCE_PHRASE
from src.rag_pipeline import (RetrievedChunk, citation_check, extract_citations, format_citation, format_context,
                              is_insufficient)


def chunk(rank, source="doc.pdf", page=2):
    return RetrievedChunk(rank=rank, text=f"text {rank}", source=source, page=page, score=0.5, chunk_id=f"id-{rank}")


CHUNKS = [chunk(1, "a.pdf", 3), chunk(2, "b.txt", None), chunk(3, "c.pdf", 7)]


def test_format_citation_with_and_without_page():
    assert format_citation(CHUNKS[0]) == "a.pdf, p. 3"
    assert format_citation(CHUNKS[1]) == "b.txt"


def test_context_numbers_chunks_with_source_metadata():
    ctx = format_context(CHUNKS)
    assert "[1] Source: a.pdf, page 3" in ctx and "[2] Source: b.txt\n" in ctx and "[3] Source: c.pdf, page 7" in ctx


def test_extract_valid_markers():
    cited, invalid = extract_citations("Fact one [1]. Fact two [3][2].", CHUNKS)
    assert [c.rank for c in cited] == [1, 3, 2] and invalid == []


def test_comma_separated_markers_and_duplicates():
    cited, invalid = extract_citations("A [1, 2] and again [1].", CHUNKS)
    assert [c.rank for c in cited] == [1, 2] and invalid == []


def test_invented_markers_are_never_turned_into_citations():
    cited, invalid = extract_citations("Claim [1] and fake claim [9] and [0].", CHUNKS)
    assert [c.rank for c in cited] == [1] and invalid == [9, 0]


def test_no_markers():
    assert extract_citations("No citations here.", CHUNKS) == ([], [])
    assert extract_citations("", CHUNKS) == ([], [])
    assert extract_citations("Anything [1]", []) == ([], [1])


def test_non_citation_brackets_ignored():
    cited, invalid = extract_citations("Use the [a] option or [1a] style.", CHUNKS)
    assert cited == [] and invalid == []


def test_citation_metadata_is_complete_on_retrieved_chunks(indexed_pipeline):
    for c in indexed_pipeline.retrieve("savings goal monthly amount", top_k=4):
        assert c.source.endswith(".txt") and c.chunk_id and c.text
        assert c.page is None  # text files have no page numbers
        assert format_citation(c) == c.source


def test_pdf_chunks_carry_page_numbers(pipeline):
    import pymupdf
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Cats are unrelated to money.")
    doc.new_page().insert_text((72, 72), "Emergency funds cover essential expenses for several months.")
    data = doc.tobytes()
    pipeline.ingest_file("guide.pdf", data)
    top = pipeline.retrieve("emergency funds essential expenses", top_k=1)[0]
    assert top.page == 2 and format_citation(top) == "guide.pdf, p. 2"


def test_insufficient_evidence_detection():
    assert is_insufficient(INSUFFICIENT_EVIDENCE_PHRASE)
    assert is_insufficient("  " + INSUFFICIENT_EVIDENCE_PHRASE.upper() + "  ")
    assert not is_insufficient("Emergency funds cover 3-6 months [1].")


def test_citation_check_pass_and_fail_cases():
    assert citation_check("Statement [1].", CHUNKS)["passes"] is True
    assert citation_check("Statement without citation.", CHUNKS)["passes"] is False
    assert citation_check("Statement [7].", CHUNKS)["passes"] is False
    ok = citation_check(INSUFFICIENT_EVIDENCE_PHRASE, CHUNKS)
    assert ok["passes"] and ok["insufficient_evidence"] and not ok["has_citation"]


def test_answer_without_relevant_evidence_does_not_call_llm(indexed_pipeline, fake_llm_factory):
    llm = fake_llm_factory("This should never be used")
    result = indexed_pipeline.answer("How do I bake a chocolate cake?", llm)
    assert result.insufficient and not result.llm_used and llm.calls == []
    assert result.answer == INSUFFICIENT_EVIDENCE_PHRASE and result.retrieved == []


def test_answer_with_evidence_maps_citations(indexed_pipeline, fake_llm_factory):
    llm = fake_llm_factory("An emergency fund covers three to six months of essential expenses [1]. Also see [42].")
    result = indexed_pipeline.answer("How many months should an emergency fund cover?", llm, top_k=3, min_score=0.1)
    assert result.llm_used and len(llm.calls) == 1
    assert [c.rank for c in result.cited] == [1] and result.invalid_markers == [42]
    assert result.cited[0].source == "emergency_fund_guide.txt"
    prompt = llm.calls[0]
    assert "[1] Source:" in prompt and "How many months should an emergency fund cover?" in prompt
    assert INSUFFICIENT_EVIDENCE_PHRASE in prompt  # the model is told what to say when evidence is missing


def test_only_retrieved_chunks_are_sent_not_whole_documents(indexed_pipeline, fake_llm_factory):
    llm = fake_llm_factory("Answer [1]")
    indexed_pipeline.answer("What is the 50/30/20 rule?", llm, top_k=2, min_score=0.1)
    prompt = llm.calls[0]
    assert prompt.count("Source:") <= 2
    whole = (indexed_pipeline.store.chunks and open("knowledge_base/documents/budgeting_fundamentals.txt", encoding="utf-8").read())
    assert whole not in prompt


def test_model_declining_is_flagged_insufficient(indexed_pipeline, fake_llm_factory):
    llm = fake_llm_factory(INSUFFICIENT_EVIDENCE_PHRASE)
    result = indexed_pipeline.answer("emergency fund months", llm, min_score=0.1)
    assert result.llm_used and result.insufficient and result.cited == []
