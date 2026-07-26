# AltruMind — single-container image (FastAPI backend + static frontend)
FROM python:3.11-slim

WORKDIR /app

# --- Install Python dependencies first (better layer caching) ---
COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r backend/requirements.txt

# --- Copy application code ---
COPY backend backend
COPY frontend frontend

WORKDIR /app/backend

EXPOSE 8000

# Render (and most PaaS hosts) inject $PORT at runtime; default to 8000
# for plain `docker run` / local testing.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]