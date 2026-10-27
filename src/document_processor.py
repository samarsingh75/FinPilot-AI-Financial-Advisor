"""Extract text from PDF/TXT files and split it into chunks with source metadata."""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Optional

import config

SUPPORTED_EXTENSIONS = (".pdf", ".txt")
MAX_PDF_PAGES = 500


class DocumentError(ValueError):
    """Raised when a document cannot be read or contains no usable text."""


@dataclass
class ExtractedPage:
    page: Optional[int]  # 1-based page number for PDFs; None for plain text files
    text: str


def safe_filename(name: str) -> str:
    """Strip any folder part from an uploaded file name."""
    return os.path.basename(str(name).replace("\\", "/")).strip() or "unnamed"


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(text: str) -> str:
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_txt(data: bytes) -> list[ExtractedPage]:
    # Only trust UTF-16 when a byte-order mark is present; without one it can "decode" any text as garbage.
    encodings = ["utf-8-sig", "cp1252", "latin-1"]
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings.insert(0, "utf-16")
    for encoding in encodings:
        try:
            text = data.decode(encoding)
            break
        except UnicodeError:
            continue
    else:  # pragma: no cover - latin-1 never fails
        raise DocumentError("Could not decode the text file.")
    text = _clean(text)
    return [ExtractedPage(page=None, text=text)] if text else []


def extract_text_from_pdf(data: bytes) -> list[ExtractedPage]:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - depends on install
        raise DocumentError("PyMuPDF is not installed. Run: pip install -r requirements.txt") from exc
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise DocumentError("The PDF could not be opened. It may be corrupted.") from None
    try:
        if doc.needs_pass:
            raise DocumentError("The PDF is password-protected.")
        if doc.page_count > MAX_PDF_PAGES:
            raise DocumentError(f"The PDF has more than {MAX_PDF_PAGES} pages, which is not supported.")
        pages = []
        for index in range(doc.page_count):
            text = _clean(doc[index].get_text("text"))
            if text:
                pages.append(ExtractedPage(page=index + 1, text=text))
        return pages
    except DocumentError:
        raise
    except Exception:
        raise DocumentError("Text could not be extracted from the PDF.") from None
    finally:
        doc.close()


def extract_pages(filename: str, data: bytes) -> list[ExtractedPage]:
    """Return the pages of text in a file. Raises DocumentError with a clear reason."""
    filename = safe_filename(filename)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentError(f"'{filename}': unsupported file type. Upload PDF or TXT files.")
    if not data:
        raise DocumentError(f"'{filename}' is empty.")
    if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise DocumentError(f"'{filename}' is larger than {config.MAX_UPLOAD_MB} MB.")
    pages = extract_text_from_pdf(data) if extension == ".pdf" else extract_text_from_txt(data)
    if not pages:
        hint = " Scanned PDFs (images of text) need OCR, which FinPilot does not include." if extension == ".pdf" else ""
        raise DocumentError(f"'{filename}' contains no extractable text.{hint}")
    return pages


def chunk_text(text: str, chunk_size: int = config.DEFAULT_CHUNK_SIZE, overlap: int = config.DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of at most ``chunk_size`` characters.

    Consecutive chunks share about ``overlap`` characters so that sentences cut at a
    boundary still appear whole in one chunk. Breaks prefer sentence ends, then spaces.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be 0 or more and smaller than chunk_size.")
    text = _clean(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = start + int(chunk_size * 0.6)
            cut = max(text.rfind(". ", window, end), text.rfind("\n", window, end))
            if cut != -1:
                end = cut + 1
            else:
                space = text.rfind(" ", window, end)
                if space != -1:
                    end = space
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        next_start = end - overlap
        if next_start <= start:  # always make progress
            next_start = end
        # Start the next chunk at a word boundary.
        limit = min(next_start + 40, end)
        while next_start < limit and not text[next_start - 1].isspace():
            next_start += 1
        start = next_start
    return chunks


def build_chunks(
    filename: str,
    doc_hash: str,
    pages: list[ExtractedPage],
    chunk_size: int = config.DEFAULT_CHUNK_SIZE,
    overlap: int = config.DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """Chunk each page separately so every chunk keeps an accurate page number."""
    filename = safe_filename(filename)
    chunks: list[dict] = []
    for page in pages:
        for piece in chunk_text(page.text, chunk_size, overlap):
            index = len(chunks)
            chunks.append(
                {
                    "chunk_id": f"{doc_hash[:12]}-{index}",
                    "doc_hash": doc_hash,
                    "source": filename,
                    "page": page.page,
                    "chunk_index": index,
                    "text": piece,
                }
            )
    return chunks
