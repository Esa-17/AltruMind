# AltruMind 🧠
### AI-Powered Startup Assistant & Document Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-3.5%20Flash-8E75B2?logo=googlegemini&logoColor=white)
![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-3776AB)
![Tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)
![Docker](https://img.shields.io/badge/deploy-Docker%20%2B%20Render-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

> An AI startup copilot that validates ideas with **live, cited market research** — not hallucinated competitor names — and a full RAG pipeline for chatting with your own documents.

**[🔗 Live Demo](#)** &nbsp;·&nbsp; **[📖 Architecture](#-how-i-built-it--architecture)** &nbsp;·&nbsp; **[🚀 Deploy Your Own](#️-deploy-render-free-tier)**

---

## About This Project

I built AltruMind during my AI/ML internship at **Altruisty Innovation Pvt Ltd, Chennai**. The goal was an intelligent assistant that does two things well:

1. **Startup & entrepreneurship advisor** — funding, product strategy, go-to-market, business modeling
2. **Document chatbot** — upload a PDF or DOCX and ask questions grounded in the actual content

The core engineering challenge was building a full **RAG (Retrieval-Augmented Generation) pipeline** from scratch — chunking, local embeddings via Sentence-Transformers, FAISS indexing, and retrieval-grounded generation via the Gemini API. On top of that, I added **live web search grounding** so the Competitor Analysis section is built from real, cited search results instead of the model guessing company names and stats from memory — the difference between a toy LLM wrapper and something that behaves like a real research tool.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 General Chat | Ask anything about startups, funding, product, marketing, or just chitchat |
| 🌐 Live Competitor Research | When a startup idea is detected, AltruMind runs a real web search and grounds the Competitor Analysis in actual results — with clickable sources |
| 📄 Document Chat | Upload a PDF/DOCX and get answers grounded in your document(s) — supports multiple files and images per session |
| 🔀 Auto Mode Switching | Detects whether to use general or document mode automatically |
| 🔍 Source Citations | Every document answer cites the exact page it came from |
| ⚡ Streaming Responses | Answers stream word-by-word like ChatGPT using Server-Sent Events |
| 🧠 Chat Memory | Full multi-turn conversation memory per session |
| 💡 Follow-up Suggestions | 3 contextual follow-up questions after every reply |
| 👍 Feedback Buttons | Thumbs up/down on every response |
| 🗂 Category Starters | Curated starter prompts for Funding, Product, Marketing, and Analysis |

---

## 📸 Screenshots

<table>
<tr>
<td width="50%">

**Landing screen**
<img src="assets/screenshots/01-hero-landing.jpg" alt="AltruMind landing screen">

</td>
<td width="50%">

**Guided starter prompts**
<img src="assets/screenshots/02-category-starters.jpg" alt="Category starter cards">

</td>
</tr>
<tr>
<td width="50%">

**Auto-generated Startup Health Score**
<img src="assets/screenshots/04-health-score.jpg" alt="Startup Health Score breakdown">

</td>
<td width="50%">

**Competitor Analysis — grounded in live search**
<img src="assets/screenshots/05-competitor-analysis-grounded.jpg" alt="Competitor analysis grounded in real search results">

</td>
</tr>
<tr>
<td width="50%">

**Lean Business Model Canvas**
<img src="assets/screenshots/06-business-model-canvas.jpg" alt="Lean business model canvas output">

</td>
<td width="50%">

**Real, clickable sources — not invented citations**
<img src="assets/screenshots/07-action-plan-sources.jpg" alt="30-day action plan and cited sources">

</td>
</tr>
</table>

<details>
<summary><b>See the full report walkthrough (idea → health score → business model → sources)</b></summary>
<br>

**1. User submits a raw idea, AltruMind summarizes it**
<img src="assets/screenshots/03-idea-summary.jpg" alt="Idea summary step">

**2. Contextual follow-up questions keep the conversation moving**
<img src="assets/screenshots/08-followups.jpg" alt="Follow-up question suggestions">

</details>

---

## 🏗 How I Built It — Architecture

```
User (Browser)
      │
      ▼
  FastAPI Backend
      │
      ├── General Chat Mode (no document)
      │       ├── Startup idea detected?
      │       │      └── Live web search (DuckDuckGo HTML, no API key)
      │       │              → grounds Competitor Analysis in real, cited results
      │       └── Gemini 3.5 Flash (via google-genai SDK)
      │           (startup/entrepreneur persona I designed)
      │
      └── Document Chat Mode (PDF/DOCX uploaded)
              ├── pdfplumber / python-docx  →  extract text per page
              ├── Text chunker             →  500-char chunks, 100-char overlap
              ├── Sentence-Transformers    →  384-dim embeddings (runs locally)
              ├── FAISS vector index       →  store + similarity search
              └── Gemini 3.5 Flash        →  grounded answer + source citation
                          │
                          ▼
              Streamed response via Server-Sent Events
                          │
                          ▼
              ChatGPT-style HTML/JS frontend
```

### Key design decisions

- **Live search to ground competitor analysis** — the startup-mode prompt originally *told* the model never to invent statistics but had no way to enforce it. `websearch.py` runs a real search for competitors whenever a startup idea is detected and instructs the model to build the Competitor Analysis strictly from those results, with sources — falling back to an honest "insufficient data" note rather than guessing when search comes up empty. The screenshots above are from a live run — those are real invoicing-software competitors and real article URLs, not model-generated names.
- **HTML scraping over the `duckduckgo-search` package** — that package pulls in a Rust-compiled dependency that fails to build on Windows without extra toolchains. I hit DuckDuckGo's plain HTML endpoint directly with `requests` + `beautifulsoup4` instead — pure Python, no compilation, works identically cross-platform, fails soft to an empty result set if the network call ever breaks.
- **Local embeddings over API embeddings** — Sentence-Transformers (`all-MiniLM-L6-v2`) instead of an embedding API, so the pipeline has zero embedding cost and works offline.
- **500-char chunks, 100-char overlap** — best balance I found between context preservation and retrieval precision after testing.
- **Two separate system prompts** — one tuned for startup advice (higher temperature, conversational), one for document Q&A (lower temperature, citation-focused).
- **Migrated from `google-generativeai` to `google-genai`** — Google deprecated the old SDK; I ported the client, streaming, and error-handling plumbing to the current `google-genai` package with zero change to the actual prompts or business logic, and backed the migration with a dedicated test suite (see [Testing](#-testing)) so the SDK swap is verifiably safe rather than just "seems to work."

---

## 🛠 Tech Stack

| Layer | Technology | Why I chose it |
|---|---|---|
| LLM | Google Gemini 3.5 Flash (via `google-genai` SDK) | Fast, generous free tier, strong instruction following |
| Embeddings | Sentence-Transformers | Runs locally, no API cost, good quality |
| Live Web Search | `requests` + `beautifulsoup4` | No API key, no compiled deps, grounds competitor analysis in real results |
| Vector Store | FAISS (`faiss-cpu`) | Simple, fast, no external service needed |
| Backend | FastAPI + Uvicorn | Async support, great for streaming SSE |
| Frontend | Vanilla HTML/CSS/JS | No build step, easy to run anywhere |
| PDF Parsing | pdfplumber | Better text extraction than PyPDF2 |
| DOCX Parsing | python-docx | Official library, reliable |
| Testing | pytest, unittest.mock | 21 tests covering logic + mocked SDK plumbing |
| Deployment | Docker + Render | Single-container deploy, free-tier friendly |

---

## 📁 Project Structure

```
AltruMind/
├── backend/
│   ├── main.py               # FastAPI app — endpoints, session management
│   ├── rag.py                # RAG pipeline — chunking, embedding, FAISS
│   ├── websearch.py          # Live search — grounds competitor analysis
│   ├── gemini.py             # Gemini API — prompts, streaming, follow-ups
│   ├── requirements.txt      # Production dependencies
│   ├── requirements-dev.txt  # + pytest, for running tests
│   ├── .env.example          # Template for required environment variables
│   └── tests/
│       ├── conftest.py       # Stubs heavy/network deps so tests run offline
│       ├── test_gemini.py    # Mode detection + follow-up parsing
│       ├── test_gemini_sdk.py# Mocked google-genai SDK plumbing
│       └── test_rag.py       # Chunking logic
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── assets/screenshots/       # README images
├── Dockerfile                 # Single-container build (backend + frontend)
├── .dockerignore
├── render.yaml                # One-click Render Blueprint deploy config
├── .gitignore
└── README.md
```

---

## 🐳 Run with Docker

```bash
docker build -t altrumind .
docker run -p 8000:8000 --env-file backend/.env altrumind
```

Then open **http://localhost:8000**.

---

## ☁️ Deploy (Render, free tier)

1. Push this repo to GitHub.
2. On [Render](https://render.com): **New → Blueprint**, point it at your repo. Render reads `render.yaml` automatically.
3. When prompted, paste in your `GEMINI_API_KEY` (kept out of the repo, entered securely in the dashboard).
4. Wait for the build — the embedding model is pre-downloaded during the Docker build, so cold starts stay fast.
5. You get a public URL like `https://altrumind.onrender.com` — that's what goes on your resume, not the repo link.

> Free-tier services spin down after ~15 min idle and take 30–50s to wake on the next request — worth a heads-up in your portfolio link so it doesn't look broken on a cold start.

Any other Docker-friendly PaaS (Railway, Fly.io) works the same way — point it at the `Dockerfile`.

---

## 🚀 Running Locally

```bash
git clone https://github.com/YOUR_USERNAME/AltruMind.git
cd AltruMind/backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env            # then paste your key — get one free at
                                 # https://aistudio.google.com/app/apikey

uvicorn main:app --reload
```

Open **http://localhost:8000** — the backend serves the frontend directly.

---

## ✅ Testing

```bash
cd backend
pip install -r requirements-dev.txt
pytest -v
```

**21 tests, all passing** — covering:
- Startup/pitch/general mode detection and follow-up JSON parsing (`test_gemini.py`)
- Document chunking + overlap math, with FAISS/Sentence-Transformers stubbed out for fast, offline runs (`test_rag.py`)
- The `google-genai` SDK migration itself — content normalization, the streaming call shape, image-part merging, and error-handling fallbacks, mocked so no live API key or network call is needed (`test_gemini_sdk.py`)

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/session/new` | Create a new chat session |
| `POST` | `/upload?session_id=...` | Upload & index a PDF or DOCX |
| `DELETE` | `/upload/{session_id}` | Remove document, back to general mode |
| `POST` | `/chat/stream` | Unified streaming chat (SSE) |
| `POST` | `/feedback` | Submit thumbs up/down feedback |
| `GET` | `/session/{session_id}` | Get session info |
| `DELETE` | `/session/{session_id}` | Delete session |
| `GET` | `/health` | Health check |

---

## 🔮 What I'd Add Next

- Persistent session/vector storage (SQLite or Redis) — sessions currently live in an in-memory dict and are lost on server restart; fine for a demo, not production
- Pinecone or a managed vector DB for production-scale storage instead of in-process FAISS
- User authentication and saved chat history across devices
- Mobile-responsive layout improvements

---

## 🙏 Acknowledgements

Built during my internship at **Altruisty Innovation Pvt Ltd, Chennai**. Thanks to the team for the opportunity to build and ship a real AI product.

---

*Made by **Esa Sri E** — CSE student at Rajalakshmi Institute of Technology, Chennai*