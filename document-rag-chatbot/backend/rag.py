import os
import io
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import pdfplumber
import docx

# ── Embedding model (runs locally, free) ──────────────────────────────────────
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM   = 384
CHUNK_SIZE  = 500   # characters per chunk
CHUNK_OVERLAP = 100 # overlap between chunks for context continuity
TOP_K       = 5     # number of chunks to retrieve per query


# ── Document parsing ──────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[str, int]]:
    """Returns list of (text, page_number) tuples."""
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((text, i))
    return pages


def extract_text_from_docx(file_bytes: bytes) -> List[Tuple[str, int]]:
    """Returns list of (paragraph_text, paragraph_number) tuples."""
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = []
    for i, para in enumerate(doc.paragraphs, start=1):
        if para.text.strip():
            paragraphs.append((para.text, i))
    return paragraphs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(pages: List[Tuple[str, int]]) -> List[dict]:
    """
    Splits page/paragraph text into overlapping chunks.
    Each chunk is a dict: { text, source, page }
    """
    chunks = []
    for text, page_num in pages:
        start = 0
        while start < len(text):
            end   = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "text":   chunk,
                    "page":   page_num,
                    "source": f"Page {page_num}" if "Page" not in str(page_num) else str(page_num),
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ── FAISS vector store ─────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self):
        self.index  = faiss.IndexFlatL2(EMBED_DIM)
        self.chunks : List[dict] = []

    def add_chunks(self, chunks: List[dict]):
        texts = [c["text"] for c in chunks]
        embeddings = EMBED_MODEL.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = TOP_K) -> List[dict]:
        query_vec = EMBED_MODEL.encode([query])
        query_vec = np.array(query_vec, dtype="float32")
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(dist)
                results.append(chunk)
        return results

    def is_empty(self) -> bool:
        return self.index.ntotal == 0


# ── Public pipeline function ───────────────────────────────────────────────────

def process_document(file_bytes: bytes, filename: str) -> VectorStore:
    """Full pipeline: parse → chunk → embed → store. Returns a ready VectorStore."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        pages = extract_text_from_pdf(file_bytes)
    elif ext in ("docx", "doc"):
        pages = extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks = chunk_text(pages)
    store  = VectorStore()
    store.add_chunks(chunks)
    return store


def retrieve_context(store: VectorStore, query: str) -> Tuple[str, List[str]]:
    """
    Retrieve top-k chunks for a query.
    Returns (formatted_context_string, list_of_source_labels).
    """
    results = store.search(query)
    context_parts = []
    sources       = []

    for i, chunk in enumerate(results, start=1):
        context_parts.append(f"[Chunk {i} — {chunk['source']}]\n{chunk['text']}")
        sources.append(chunk["source"])

    context = "\n\n".join(context_parts)
    unique_sources = list(dict.fromkeys(sources))   # deduplicated, order preserved
    return context, unique_sources
