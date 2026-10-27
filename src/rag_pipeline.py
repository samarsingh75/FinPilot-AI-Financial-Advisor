"""The RAG pipeline: ingest -> chunk -> embed -> store -> retrieve -> generate -> cite."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
from src import document_processor as dp
from src.prompts import INSUFFICIENT_EVIDENCE_PHRASE, SYSTEM_PROMPT, build_rag_prompt
from src.vector_store import Embedder, EmbeddingError, VectorStore, VectorStoreError

# ---------------------------------------------------------------- data classes


@dataclass
class RetrievedChunk:
    rank: int  # 1-based position; this is the number the model uses as [rank]
    text: str
    source: str
    page: Optional[int]
    score: float
    chunk_id: str


@dataclass
class IngestResult:
    filename: str
    status: str  # added | replaced | duplicate | empty | error
    message: str
    chunk_count: int = 0


@dataclass
class RAGAnswer:
    question: str
    answer: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    cited: list[RetrievedChunk] = field(default_factory=list)
    invalid_markers: list[int] = field(default_factory=list)
    insufficient: bool = False
    llm_used: bool = False
    prompt: Optional[str] = None


# ---------------------------------------------------------------- citations
_MARKER_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def format_citation(chunk: RetrievedChunk) -> str:
    """Human-readable citation: filename plus page number when one is known."""
    return f"{chunk.source}, p. {chunk.page}" if chunk.page is not None else chunk.source


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Number the retrieved chunks so the model can cite them as [1], [2], ..."""
    blocks = []
    for chunk in chunks:
        header = f"[{chunk.rank}] Source: {chunk.source}" + (f", page {chunk.page}" if chunk.page is not None else "")
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> tuple[list[RetrievedChunk], list[int]]:
    """Map [n] markers in an answer to the retrieved chunks.

    Returns (cited_chunks, invalid_markers). Markers that do not correspond to a
    retrieved chunk are reported as invalid and never turned into citations.
    """
    by_rank = {c.rank: c for c in chunks}
    cited: list[RetrievedChunk] = []
    invalid: list[int] = []
    for match in _MARKER_RE.finditer(answer or ""):
        for number in (int(n) for n in match.group(1).split(",")):
            if number in by_rank:
                if by_rank[number] not in cited:
                    cited.append(by_rank[number])
            elif number not in invalid:
                invalid.append(number)
    return cited, invalid


def is_insufficient(answer: str) -> bool:
    normalise = lambda s: re.sub(r"\s+", " ", s).strip().lower().rstrip(".")  # noqa: E731
    return normalise(INSUFFICIENT_EVIDENCE_PHRASE) in normalise(answer or "")


def citation_check(answer: str, chunks: list[RetrievedChunk]) -> dict:
    """Basic citation-presence check used for project evaluation."""
    cited, invalid = extract_citations(answer, chunks)
    insufficient = is_insufficient(answer)
    return {
        "has_citation": bool(cited),
        "valid_markers": [c.rank for c in cited],
        "invalid_markers": invalid,
        "insufficient_evidence": insufficient,
        "passes": insufficient or (bool(cited) and not invalid),
    }


# ---------------------------------------------------------------- pipeline
class RAGPipeline:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        chunk_size: int = config.DEFAULT_CHUNK_SIZE,
        overlap: int = config.DEFAULT_CHUNK_OVERLAP,
    ):
        self.store = store
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------ ingestion
    def ingest_file(self, filename: str, data: bytes) -> IngestResult:
        filename = dp.safe_filename(filename)
        try:
            pages = dp.extract_pages(filename, data)
            doc_hash = dp.file_hash(data)

            existing = self.store.documents.get(doc_hash)
            if existing:
                same_settings = existing.get("chunk_size") == self.chunk_size and existing.get("overlap") == self.overlap
                if existing["source"] != filename or same_settings:
                    return IngestResult(
                        filename, "duplicate", f"Skipped: identical content is already indexed as '{existing['source']}'."
                    )

            chunks = dp.build_chunks(filename, doc_hash, pages, self.chunk_size, self.overlap)
            if not chunks:
                return IngestResult(filename, "empty", f"'{filename}' contains no extractable text.")

            if not self.store.is_empty and self.store.embedding_model != self.embedder.model_name:
                raise VectorStoreError(
                    f"The index was built with '{self.store.embedding_model}', but '{self.embedder.model_name}' "
                    "is selected. Clear the index and re-index your documents."
                )
            vectors = self.embedder.encode([c["text"] for c in chunks])  # embed first: a failure keeps the old version
            old_hash = doc_hash if existing else self.store.find_by_source(filename)
            if old_hash:
                self.store.remove_document(old_hash)
            self.store.add_chunks(
                doc_hash, filename, chunks, vectors, self.embedder.model_name,
                extra_info={"chunk_size": self.chunk_size, "overlap": self.overlap},
            )
            if old_hash:
                what = "re-indexed with the new chunk settings" if old_hash == doc_hash else "replaced the previous version"
                return IngestResult(filename, "replaced", f"Updated: {what} ({len(chunks)} chunks).", len(chunks))
            return IngestResult(filename, "added", f"Indexed {len(chunks)} chunks.", len(chunks))
        except dp.DocumentError as exc:
            status = "empty" if "no extractable text" in str(exc) else "error"
            return IngestResult(filename, status, str(exc))
        except (EmbeddingError, VectorStoreError, ValueError) as exc:
            return IngestResult(filename, "error", str(exc))

    def ingest_directory(self, directory: Path | str = config.SAMPLE_DOCS_DIR) -> list[IngestResult]:
        results = []
        for path in sorted(Path(directory).iterdir()):
            if path.suffix.lower() in dp.SUPPORTED_EXTENSIONS:
                results.append(self.ingest_file(path.name, path.read_bytes()))
        return results

    # ------------------------------------------------------------ retrieval
    def retrieve(self, question: str, top_k: int = config.DEFAULT_TOP_K, min_score: float = 0.0) -> list[RetrievedChunk]:
        question = (question or "").strip()
        if not question:
            raise ValueError("Please enter a question.")
        if self.store.is_empty:
            return []
        query_vector = self.embedder.encode([question])
        hits = [(c, s) for c, s in self.store.search(query_vector, top_k) if s >= min_score]
        return [
            RetrievedChunk(rank=i, text=c["text"], source=c["source"], page=c.get("page"), score=s, chunk_id=c["chunk_id"])
            for i, (c, s) in enumerate(hits, start=1)
        ]

    # ------------------------------------------------------------ generation
    def answer(
        self,
        question: str,
        llm,
        top_k: int = config.DEFAULT_TOP_K,
        min_score: float = config.DEFAULT_MIN_SCORE,
    ) -> RAGAnswer:
        """Retrieve evidence and ask the LLM for a grounded, cited answer.

        If nothing relevant is retrieved the LLM is NOT called, so it cannot invent an answer.
        ``llm`` must provide ``generate(prompt, system_instruction=...) -> str``.
        """
        chunks = self.retrieve(question, top_k=top_k, min_score=min_score)
        if not chunks:
            return RAGAnswer(question=question, answer=INSUFFICIENT_EVIDENCE_PHRASE, insufficient=True)
        prompt = build_rag_prompt(question, format_context(chunks))
        text = llm.generate(prompt, system_instruction=SYSTEM_PROMPT)
        cited, invalid = extract_citations(text, chunks)
        return RAGAnswer(
            question=question,
            answer=text,
            retrieved=chunks,
            cited=cited,
            invalid_markers=invalid,
            insufficient=is_insufficient(text),
            llm_used=True,
            prompt=prompt,
        )
