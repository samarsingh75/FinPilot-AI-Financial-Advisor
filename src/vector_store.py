"""Embedding models and the local FAISS vector store.

Everything is saved in one folder (default ``knowledge_base/vector_store``):
    index.faiss  - the FAISS index (vectors)
    chunks.json  - chunk text and source metadata, in the same order as the vectors
    meta.json    - embedding model name, dimension, and the list of indexed documents
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

import config


class EmbeddingError(RuntimeError):
    """The embedding model could not be loaded or used."""


class VectorStoreError(RuntimeError):
    """The vector store could not complete the requested operation."""


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised float32 vectors."""

    model_name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Sentence Transformers embeddings. The model downloads once, on first use."""

    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbeddingError(
                    "The 'sentence-transformers' package is not installed. Run: pip install -r requirements.txt"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name)
            except Exception as exc:  # network, disk, or model-name problems
                raise EmbeddingError(
                    f"Could not load the embedding model '{self.model_name}'. The first run needs an internet "
                    f"connection to download it once (about 90 MB). Details: {type(exc).__name__}"
                ) from None
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts), batch_size=32, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        )
        return np.ascontiguousarray(vectors, dtype="float32")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class VectorStore:
    INDEX_FILE = "index.faiss"
    CHUNKS_FILE = "chunks.json"
    META_FILE = "meta.json"

    def __init__(self, directory: Path | str = config.VECTOR_STORE_DIR):
        self.directory = Path(directory)
        self.index = None
        self.chunks: list[dict] = []
        self.meta: dict = {"embedding_model": None, "dimension": None, "documents": {}}
        self.load_error: Optional[str] = None
        self._load()

    # ---------------------------------------------------------- properties
    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def documents(self) -> dict:
        return self.meta["documents"]

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def embedding_model(self) -> Optional[str]:
        return self.meta.get("embedding_model")

    def has_hash(self, doc_hash: str) -> bool:
        return doc_hash in self.documents

    def find_by_source(self, source: str) -> Optional[str]:
        for doc_hash, info in self.documents.items():
            if info["source"] == source:
                return doc_hash
        return None

    # ---------------------------------------------------------- persistence
    def _reset_memory(self) -> None:
        self.index = None
        self.chunks = []
        self.meta = {"embedding_model": None, "dimension": None, "documents": {}}

    def _load(self) -> None:
        paths = [self.directory / n for n in (self.INDEX_FILE, self.CHUNKS_FILE, self.META_FILE)]
        if not any(p.exists() for p in paths):
            return
        try:
            import faiss

            if not all(p.exists() for p in paths):
                raise VectorStoreError("some index files are missing")
            index = faiss.deserialize_index(np.frombuffer(paths[0].read_bytes(), dtype="uint8"))
            chunks = json.loads(paths[1].read_text(encoding="utf-8"))
            meta = json.loads(paths[2].read_text(encoding="utf-8"))
            if index.ntotal != len(chunks):
                raise VectorStoreError("index and chunk metadata do not match")
            self.index, self.chunks, self.meta = index, chunks, meta
        except Exception as exc:  # corrupted or incomplete files
            self._reset_memory()
            self.load_error = (
                f"The saved index could not be loaded ({exc}). Clear the index and index your documents again."
            )

    def save(self) -> None:
        import faiss

        self.directory.mkdir(parents=True, exist_ok=True)
        if self.index is None or not self.chunks:
            self.delete_files()
            return
        _atomic_write_bytes(self.directory / self.INDEX_FILE, faiss.serialize_index(self.index).tobytes())
        _atomic_write_bytes(
            self.directory / self.CHUNKS_FILE, json.dumps(self.chunks, ensure_ascii=False).encode("utf-8")
        )
        _atomic_write_bytes(
            self.directory / self.META_FILE, json.dumps(self.meta, ensure_ascii=False, indent=2).encode("utf-8")
        )

    def delete_files(self) -> None:
        for name in (self.INDEX_FILE, self.CHUNKS_FILE, self.META_FILE):
            path = self.directory / name
            if path.exists():
                path.unlink()

    # ---------------------------------------------------------- modification
    def add_chunks(
        self, doc_hash: str, source: str, chunks: list[dict], vectors: np.ndarray, model_name: str,
        extra_info: Optional[dict] = None,
    ) -> None:
        import faiss

        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if len(chunks) == 0 or vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise VectorStoreError("The number of vectors does not match the number of chunks.")
        if not self.is_empty and self.embedding_model != model_name:
            raise VectorStoreError(
                f"The index was built with '{self.embedding_model}', but '{model_name}' is selected. "
                "Clear the index and re-index your documents."
            )
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine for unit vectors
            self.meta["embedding_model"] = model_name
            self.meta["dimension"] = int(vectors.shape[1])
        elif vectors.shape[1] != self.index.d:
            raise VectorStoreError("Embedding dimension does not match the existing index.")
        self.index.add(vectors)
        self.chunks.extend(chunks)
        pages = sorted({c["page"] for c in chunks if c.get("page") is not None})
        self.documents[doc_hash] = {
            "source": source,
            "chunk_count": len(chunks),
            "page_count": len(pages) or None,
            "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            **(extra_info or {}),
        }
        self.save()

    def remove_document(self, doc_hash: str) -> bool:
        import faiss

        if doc_hash not in self.documents:
            return False
        keep = [i for i, c in enumerate(self.chunks) if c["doc_hash"] != doc_hash]
        if keep and self.index is not None:
            vectors = self.index.reconstruct_n(0, self.index.ntotal)[keep]
            index = faiss.IndexFlatIP(self.index.d)
            index.add(np.ascontiguousarray(vectors, dtype="float32"))
            self.index = index
            self.chunks = [self.chunks[i] for i in keep]
            del self.documents[doc_hash]
        else:
            self._reset_memory()
        self.save()
        return True

    def clear(self) -> None:
        self._reset_memory()
        self.load_error = None
        self.delete_files()

    # ---------------------------------------------------------- search
    def search(self, query_vector: np.ndarray, top_k: int = config.DEFAULT_TOP_K) -> list[tuple[dict, float]]:
        """Return up to ``top_k`` (chunk, cosine_score) pairs, best first."""
        if self.index is None or self.is_empty or top_k <= 0:
            return []
        query = np.ascontiguousarray(np.asarray(query_vector, dtype="float32").reshape(1, -1))
        if query.shape[1] != self.index.d:
            raise VectorStoreError("Query embedding dimension does not match the index.")
        scores, ids = self.index.search(query, min(int(top_k), self.index.ntotal))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
