"""Page 3: RAG Knowledge Base."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from src import rag_evaluation as ev
from src.gemini_client import GeminiClient, GeminiError, is_configured
from src.rag_pipeline import RAGPipeline, citation_check, format_citation
from src.vector_store import EmbeddingError, SentenceTransformerEmbedder, VectorStore, VectorStoreError

STATUS_ICON = {"added": "\u2705", "replaced": "\U0001f504", "duplicate": "\u2139\ufe0f", "empty": "\u26a0\ufe0f", "error": "\u274c"}


@st.cache_resource(show_spinner=False)
def get_store() -> VectorStore:
    return VectorStore(config.VECTOR_STORE_DIR)


@st.cache_resource(show_spinner=False)
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(config.EMBEDDING_MODEL)  # loads lazily on first use


def get_pipeline(chunk_size: int = config.DEFAULT_CHUNK_SIZE, overlap: int = config.DEFAULT_CHUNK_OVERLAP) -> RAGPipeline:
    return RAGPipeline(get_store(), get_embedder(), chunk_size, overlap)


def _show_results(results) -> None:
    for r in results:
        line = f"{STATUS_ICON.get(r.status, '')} **{r.filename}**: {r.message}"
        if r.status in ("added", "replaced"):
            st.success(line)
        elif r.status == "duplicate":
            st.info(line)
        elif r.status == "empty":
            st.warning(line)
        else:
            st.error(line)


# ------------------------------------------------------------------ tab 1
def _index_tab() -> None:
    store = get_store()
    if store.load_error:
        st.error(store.load_error)
    pending = st.session_state.pop("kb_results", None)
    if pending:
        _show_results(pending)
    c1, c2, c3 = st.columns(3)
    c1.metric("Indexed documents", store.document_count)
    c2.metric("Text chunks", store.chunk_count)
    c3.metric("Embedding model", (store.embedding_model or config.EMBEDDING_MODEL).split("/")[-1])
    if store.is_empty:
        st.info("The index is empty. Add the sample documents or upload your own PDF/TXT files, then click **Index**.")

    st.markdown("##### Chunking settings")
    s1, s2 = st.columns(2)
    chunk_size = s1.slider("Chunk size (characters)", 200, 2000, config.DEFAULT_CHUNK_SIZE, 50)
    overlap = s2.slider("Chunk overlap (characters)", 0, 500, config.DEFAULT_CHUNK_OVERLAP, 10)
    if overlap >= chunk_size:
        st.error("Overlap must be smaller than the chunk size.")
        return
    st.caption("Settings apply to documents indexed from now on. Uploading the same file again with new settings re-indexes it.")

    st.markdown("##### Add documents")
    files = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    b1, b2 = st.columns(2)
    index_clicked = b1.button("Index uploaded documents", type="primary", disabled=not files, width="stretch")
    sample_clicked = b2.button("Index sample documents", width="stretch")

    if index_clicked or sample_clicked:
        pipeline = get_pipeline(chunk_size, overlap)
        try:
            with st.spinner("Extracting, embedding and indexing (the first run downloads the embedding model)..."):
                if sample_clicked:
                    results = pipeline.ingest_directory(config.SAMPLE_DOCS_DIR)
                else:
                    results = [pipeline.ingest_file(f.name, f.getvalue()) for f in files]
        except Exception as exc:  # noqa: BLE001 - never crash the page
            st.error(f"Indexing failed: {exc}")
        else:
            st.session_state["kb_results"] = results
            st.rerun()  # refresh the counters above with the new index state

    if not store.is_empty:
        st.markdown("##### Indexed documents")
        rows = [
            {"File": info["source"], "Chunks": info["chunk_count"], "Pages": info.get("page_count") or "-", "Added": info["added_at"]}
            for info in store.documents.values()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        r1, r2 = st.columns(2)
        with r1:
            names = {info["source"]: h for h, info in store.documents.items()}
            choice = st.selectbox("Remove a document", ["-"] + sorted(names))
            if st.button("Remove selected document", disabled=choice == "-"):
                store.remove_document(names[choice])
                st.rerun()
        with r2:
            confirm = st.checkbox("I want to delete the whole index")
            if st.button("Clear entire index", disabled=not confirm):
                store.clear()
                st.rerun()


# ------------------------------------------------------------------ tab 2
def _render_result(result: dict) -> None:
    chunks = result["retrieved"]
    min_score = result["min_score"]
    answer = result.get("answer")
    if answer is not None:
        st.markdown("#### Answer")
        st.markdown(answer.answer)
        if answer.insufficient:
            st.info("The retrieved evidence was not sufficient, so no answer was generated from it.")
        else:
            if answer.cited:
                st.markdown("**Cited sources:** " + "; ".join(f"[{c.rank}] {format_citation(c)}" for c in answer.cited))
            else:
                st.warning("The answer contains no citation markers. Treat it with caution and check the passages below.")
            if answer.invalid_markers:
                st.warning("The answer used citation number(s) " + ", ".join(map(str, answer.invalid_markers))
                           + " that do not match any retrieved passage. Those were ignored.")
        st.caption("Educational information generated by AI from the retrieved passages. Not professional financial advice.")

    st.markdown(f"#### Retrieved passages ({len(chunks)})")
    if not chunks:
        st.info("No passages were retrieved above the relevance threshold.")
    for c in chunks:
        flag = "" if c.score >= min_score else "  (below relevance threshold, not sent to Gemini)"
        with st.expander(f"[{c.rank}] {format_citation(c)}  |  similarity {c.score:.3f}{flag}"):
            st.write(c.text)
            st.caption(f"chunk id: {c.chunk_id}")
    if answer is not None and answer.prompt:
        with st.expander("Prompt sent to Gemini"):
            st.code(answer.prompt, language="text")


def _ask_tab() -> None:
    store = get_store()
    if store.is_empty:
        st.info("Nothing to search yet. Index documents on the **Index documents** tab first.")
        return
    question = st.text_input("Ask a question about your documents", placeholder="e.g. How many months should an emergency fund cover?")
    s1, s2 = st.columns(2)
    top_k = s1.slider("Top-k passages to retrieve", 1, 10, config.DEFAULT_TOP_K)
    min_score = s2.slider("Minimum similarity to use as evidence", 0.0, 0.9, config.DEFAULT_MIN_SCORE, 0.05,
                          help="Passages scoring below this are not sent to Gemini. If none pass, no answer is generated.")
    b1, b2 = st.columns(2)
    retrieve_only = b1.button("Retrieve passages only (no Gemini)", width="stretch", disabled=not question.strip())
    ask = b2.button("Ask Gemini", type="primary", width="stretch", disabled=not question.strip() or not is_configured())
    if not is_configured():
        st.caption("Gemini answers are disabled because no API key is configured. Retrieval still works. See the README.")

    if retrieve_only or ask:
        pipeline = get_pipeline()
        try:
            with st.spinner("Searching..."):
                if ask:
                    answer = pipeline.answer(question, GeminiClient(), top_k=top_k, min_score=min_score)
                    st.session_state["kb_result"] = {"retrieved": answer.retrieved, "answer": answer, "min_score": min_score}
                else:
                    chunks = pipeline.retrieve(question, top_k=top_k, min_score=0.0)
                    st.session_state["kb_result"] = {"retrieved": chunks, "answer": None, "min_score": min_score}
        except (EmbeddingError, VectorStoreError, GeminiError) as exc:
            st.session_state.pop("kb_result", None)
            st.error(getattr(exc, "message", str(exc)))
        except ValueError as exc:
            st.warning(str(exc))
    if st.session_state.get("kb_result"):
        _render_result(st.session_state["kb_result"])


# ------------------------------------------------------------------ tab 3
def _eval_tab() -> None:
    st.markdown(
        "A small, simple check of the retrieval step using the bundled sample documents. "
        "**Retrieval hit rate** is the share of test questions whose expected document appears in the top-k results. "
        "These are basic project metrics. They do not prove that every generated answer is factually correct."
    )
    store = get_store()
    missing = ev.missing_sample_documents(get_pipeline())
    if store.is_empty or missing:
        st.info("Index the sample documents first (Index tab). Missing: " + (", ".join(missing) if missing else "all"))
        return
    top_k = st.slider("Top-k for evaluation", 1, 10, config.DEFAULT_TOP_K, key="eval_topk")
    if st.button("Run retrieval evaluation", type="primary"):
        pipeline = get_pipeline()
        try:
            with st.spinner("Running..."):
                hit = ev.retrieval_hit_rate(pipeline, top_k=top_k)
                irrelevant = ev.irrelevant_question_check(pipeline, top_k=top_k)
        except EmbeddingError as exc:
            st.error(str(exc))
            return
        m1, m2 = st.columns(2)
        m1.metric("Retrieval hit rate", f"{hit['hit_rate'] * 100:.0f}%", f"{hit['hits']} of {hit['total']} questions")
        passed = sum(r["correctly_rejected"] for r in irrelevant)
        m2.metric("Irrelevant questions rejected", f"{passed} of {len(irrelevant)}", f"threshold {config.DEFAULT_MIN_SCORE}")
        st.markdown("**Hit-rate test questions**")
        st.dataframe(pd.DataFrame(hit["rows"]).drop(columns=["has_page_or_source_metadata"]), hide_index=True, width="stretch")
        st.markdown("**Irrelevant question check** (retrieval should find nothing above the threshold)")
        st.dataframe(pd.DataFrame(irrelevant), hide_index=True, width="stretch")

    last = st.session_state.get("kb_result")
    if last and last.get("answer") is not None and last["answer"].llm_used:
        check = citation_check(last["answer"].answer, last["retrieved"])
        st.markdown("**Citation check for the latest Gemini answer**")
        st.write(f"Has valid citation: **{check['has_citation']}**, valid markers: {check['valid_markers']}, "
                 f"invalid markers: {check['invalid_markers']}, said 'insufficient evidence': {check['insufficient_evidence']}, "
                 f"overall: **{'PASS' if check['passes'] else 'CHECK'}**")


def render() -> None:
    st.markdown(
        "Index your own finance documents (PDF/TXT) and ask questions. Answers are generated by Gemini using "
        "**only the retrieved passages**, with citations. Documents and the vector index stay on your computer; "
        "only the few retrieved passages and your question are sent to Gemini when you click *Ask Gemini*."
    )
    tab1, tab2, tab3 = st.tabs(["Index documents", "Ask questions", "Evaluation"])
    with tab1:
        _index_tab()
    with tab2:
        _ask_tab()
    with tab3:
        _eval_tab()
