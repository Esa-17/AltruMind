import os
import io
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
import pdfplumber
import docx
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env (mirrors gemini.py's setup so this module works standalone too)
load_dotenv(Path(__file__).parent / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# ── Embeddings via the Gemini API (no local model, no torch) ──────────────────
# NOTE: this project originally used sentence-transformers ("all-MiniLM-L6-v2")
# for local, free, offline embeddings. That pulls in PyTorch at import time —
# roughly 300-500MB before the server has even started — which reliably OOM'd
# on Render's free tier (512MB total). Switching to Gemini's embedding API
# removes that dependency entirely: no torch, no local model weights, just a
# lightweight HTTP call using the same client/key already used for chat.
_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=API_KEY)
    return _client


EMBED_MODEL_NAME = "gemini-embedding-001"
EMBED_DIM = 768   # requested via output_dimensionality below
CHUNK_SIZE = 500   # characters per chunk
CHUNK_OVERLAP = 100 # overlap between chunks for context continuity
TOP_K = 5     # number of chunks to retrieve per query

EMBED_BATCH_SIZE = 100  # keep individual API requests a reasonable size


def _embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embeds a list of texts via the Gemini embedding API.
    Returns an (n, EMBED_DIM) float32 array, batching requests so a
    large document doesn't get sent as one oversized call.
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    client = _get_client()
    all_vectors: List[List[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBED_MODEL_NAME,
            contents=batch,
            config=types.EmbedContentConfig(output_dimensionality=EMBED_DIM),
        )
        all_vectors.extend(e.values for e in response.embeddings)

    return np.array(all_vectors, dtype="float32")


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
        embeddings = _embed_texts(texts)
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query: str, top_k: int = TOP_K) -> List[dict]:
        query_vec = _embed_texts([query])
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


def retrieve_context(question, vectorstore):
    """
    Retrieves the most relevant chunks from the vector store.

    Returns:
        context (str)
        sources (list)
    """

    docs = vectorstore.search(question, top_k=TOP_K)

    context_parts = []
    sources = []

    for doc in docs:

        context_parts.append(doc["text"])

        if doc["source"] not in sources:
            sources.append(doc["source"])

    context = "\n\n".join(context_parts)

    MAX_CONTEXT = 12000

    if len(context) > MAX_CONTEXT:
        context = context[:MAX_CONTEXT]

    return context, sources
