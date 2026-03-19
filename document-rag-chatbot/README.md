# AltruMind 🧠
### AI-Powered Startup Assistant & Document Chatbot

---

## About This Project

I built AltruMind during my AI/ML internship at **Altruisty Innovation Pvt Ltd, Chennai**. The goal was to create an intelligent assistant that could do two things really well:

1. **Act as a startup & entrepreneurship advisor** — answering questions about funding, product building, go-to-market strategy, and general business topics
2. **Let users chat with their own documents** — upload a PDF or DOCX and ask questions about it, with answers grounded in the actual document content

The core challenge I solved was building a full **RAG (Retrieval-Augmented Generation) pipeline** from scratch — chunking documents, generating embeddings locally using Sentence-Transformers, indexing them in FAISS, and retrieving the most relevant context before passing it to Gemini API for a grounded, citation-aware response.

I also wanted the UX to feel like a real product — so I built a ChatGPT-style interface with streaming responses, follow-up suggestions, feedback buttons, and a document mode indicator.

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 General Chat | Ask anything about startups, funding, product, marketing, or just chitchat |
| 📄 Document Chat | Upload a PDF/DOCX and get answers grounded in your document |
| 🔀 Auto Mode Switching | Detects whether to use general or document mode automatically |
| 🔍 Source Citations | Every document answer cites the exact page it came from |
| ⚡ Streaming Responses | Answers stream word-by-word like ChatGPT using Server-Sent Events |
| 🧠 Chat Memory | Full multi-turn conversation memory per session |
| 💡 Follow-up Suggestions | 3 contextual follow-up questions after every reply |
| 👍 Feedback Buttons | Thumbs up/down on every response |
| 📋 Copy Button | One-click copy on any message |
| 🗂 Category Starters | Curated starter questions for Funding, Product, Marketing, and Chat |

---

## 🏗 How I Built It — Architecture

```
User (Browser)
      │
      ▼
  FastAPI Backend
      │
      ├── General Chat Mode (no document)
      │       └── Gemini 1.5 Flash
      │           (startup/entrepreneur persona I designed)
      │
      └── Document Chat Mode (PDF/DOCX uploaded)
              ├── pdfplumber / python-docx  →  extract text per page
              ├── Text chunker             →  500-char chunks, 100-char overlap
              ├── Sentence-Transformers    →  384-dim embeddings (runs locally)
              ├── FAISS vector index       →  store + similarity search
              └── Gemini 1.5 Flash        →  grounded answer + source citation
                          │
                          ▼
              Streamed response via Server-Sent Events
                          │
                          ▼
              ChatGPT-style HTML/JS frontend
```

### Key design decisions I made:

- **Local embeddings over API embeddings** — I used Sentence-Transformers (`all-MiniLM-L6-v2`) instead of an embedding API so the pipeline has zero embedding cost and works offline
- **500-char chunks with 100-char overlap** — after testing, this gave the best balance between context preservation and retrieval precision
- **Two separate system prompts** — one tuned for startup advice (higher temperature, conversational) and one for document Q&A (lower temperature, citation-focused)
- **General knowledge first** — when a document is loaded, the bot answers from general knowledge first, then adds document context. This felt more natural than forcing users into document-only mode
- **Single-file frontend** — kept the entire UI in one HTML file for simplicity and easy GitHub hosting

---

## 🛠 Tech Stack

| Layer | Technology | Why I chose it |
|---|---|---|
| LLM | Google Gemini 1.5 Flash | Fast, free tier, strong instruction following |
| Embeddings | Sentence-Transformers | Runs locally, no API cost, good quality |
| Vector Store | FAISS (faiss-cpu) | Simple, fast, no external service needed |
| Backend | FastAPI + Uvicorn | Async support, great for streaming SSE |
| Frontend | Vanilla HTML/CSS/JS | No build step, easy to run anywhere |
| PDF Parsing | pdfplumber | Better text extraction than PyPDF2 |
| DOCX Parsing | python-docx | Official library, reliable |

---

## 📁 Project Structure

```
AltruMind/
├── backend/
│   ├── main.py           # FastAPI app — all endpoints, session management
│   ├── rag.py            # RAG pipeline — chunking, embedding, FAISS
│   ├── gemini.py         # Gemini API — prompts, streaming, follow-up extraction
│   └── requirements.txt
├── frontend/
│   └── index.html        # Complete UI — single file, no build needed
├── .gitignore
└── README.md
```

---

## 🚀 Running Locally

### 1. Clone and set up

```bash
git clone https://github.com/YOUR_USERNAME/AltruMind.git
cd AltruMind/backend

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Create a `.env` file inside `backend/`:
```
GEMINI_API_KEY=your_key_here
```
Get a free key at: https://aistudio.google.com/app/apikey

### 3. Start the server

```bash
uvicorn main:app --reload
```

### 4. Open the frontend

Open `frontend/index.html` in your browser. That's it!

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

- Persistent vector store — save/load FAISS index to disk so documents survive server restarts
- Pinecone integration for production-scale vector storage
- Multi-document support — index multiple files in one session
- Docker containerisation for easy deployment
- User authentication and saved chat history
- Mobile-responsive improvements

---

## 🙏 Acknowledgements

Built during my internship at **Altruisty Innovation Pvt Ltd, Chennai**. Thanks to the team for the opportunity to build and ship a real AI product.

---

*Made by **Esa Sri E** — CSE student at Rajalakshmi Institute of Technology, Chennai*