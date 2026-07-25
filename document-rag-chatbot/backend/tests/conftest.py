"""
conftest.py
-----------
Stubs out heavy or network-touching dependencies BEFORE any test module
imports rag.py / gemini.py, so the test suite:

  - never downloads the sentence-transformers model
  - never builds a real FAISS index
  - never requires the google-genai package to actually be installed
  - never needs a real GEMINI_API_KEY / network access

Only pure logic (chunking math, mode detection, followup parsing,
content-normalization plumbing) is under test here — nothing here talks
to a real API.
"""
import os
import sys
import types

import numpy as np

# ---------------------------------------------------------------------
# Stub: faiss
# ---------------------------------------------------------------------
if "faiss" not in sys.modules:
    faiss_stub = types.ModuleType("faiss")

    class _FakeIndexFlatL2:
        def __init__(self, dim):
            self.dim = dim
            self.ntotal = 0
            self._vectors = []

        def add(self, embeddings):
            self._vectors.extend(list(embeddings))
            self.ntotal += len(embeddings)

        def search(self, query_vec, top_k):
            n = min(top_k, self.ntotal) if self.ntotal else 0
            distances = np.zeros((1, top_k), dtype="float32")
            indices = np.full((1, top_k), -1, dtype="int64")
            for i in range(n):
                indices[0][i] = i
            return distances, indices

    faiss_stub.IndexFlatL2 = _FakeIndexFlatL2
    sys.modules["faiss"] = faiss_stub

# ---------------------------------------------------------------------
# Stub: sentence_transformers
# ---------------------------------------------------------------------
if "sentence_transformers" not in sys.modules:
    st_stub = types.ModuleType("sentence_transformers")

    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, show_progress_bar=False):
            if isinstance(texts, str):
                texts = [texts]
            return np.zeros((len(texts), 384), dtype="float32")

    st_stub.SentenceTransformer = _FakeSentenceTransformer
    sys.modules["sentence_transformers"] = st_stub

# ---------------------------------------------------------------------
# Stub: google.genai (only if the real package isn't installed)
# ---------------------------------------------------------------------
try:
    import google.genai  # noqa: F401
except ImportError:
    google_pkg = sys.modules.get("google")
    if google_pkg is None:
        google_pkg = types.ModuleType("google")
        google_pkg.__path__ = []  # mark as a namespace package
        sys.modules["google"] = google_pkg

    genai_stub = types.ModuleType("google.genai")
    types_stub = types.ModuleType("google.genai.types")
    errors_stub = types.ModuleType("google.genai.errors")

    class _FakePart:
        def __init__(self, text=None, inline_data=None):
            self.text = text
            self.inline_data = inline_data

        @classmethod
        def from_text(cls, text):
            return cls(text=text)

        @classmethod
        def from_bytes(cls, data, mime_type):
            return cls(inline_data={"data": data, "mime_type": mime_type})

    class _FakeContent:
        def __init__(self, role=None, parts=None):
            self.role = role
            self.parts = parts or []

    class _FakeGenerateContentConfig:
        def __init__(self, system_instruction=None, **kwargs):
            self.system_instruction = system_instruction
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _APIError(Exception):
        def __init__(self, message="", code=None):
            super().__init__(message)
            self.message = message
            self.code = code

    class _FakeModels:
        def generate_content_stream(self, model=None, contents=None, config=None):
            return []

    class _FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.models = _FakeModels()

    types_stub.Part = _FakePart
    types_stub.Content = _FakeContent
    types_stub.GenerateContentConfig = _FakeGenerateContentConfig
    errors_stub.APIError = _APIError
    genai_stub.Client = _FakeClient
    genai_stub.types = types_stub
    genai_stub.errors = errors_stub

    sys.modules["google.genai"] = genai_stub
    sys.modules["google.genai.types"] = types_stub
    sys.modules["google.genai.errors"] = errors_stub
    google_pkg.genai = genai_stub

# gemini.py raises at import time if this isn't set — tests never hit
# the network, but the module-level check still needs *a* value.
os.environ.setdefault("GEMINI_API_KEY", "test-key-for-pytest")