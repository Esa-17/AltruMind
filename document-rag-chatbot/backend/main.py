import os
import uuid
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import process_document, retrieve_context, VectorStore
from gemini import stream_general, stream_document, extract_followups

load_dotenv()

app = FastAPI(title="AltruMind", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ────────────────────────────────────────────────────
sessions: dict = {}


# ── Models ─────────────────────────────────────────────────────────────────────

class GeneralChatRequest(BaseModel):
    session_id: str
    question:   str

class DocChatRequest(BaseModel):
    session_id: str
    question:   str

class FeedbackRequest(BaseModel):
    session_id:  str
    message_idx: int
    feedback:    str   # "up" or "down"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = os.path.join(os.path.dirname(__file__), "../frontend/index.html")
    with open(html_path, "r") as f:
        return f.read()


@app.post("/session/new")
async def new_session():
    """Create a new chat session (general mode, no document)."""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "store":    None,
        "filename": None,
        "history":  [],
        "feedback": {},
    }
    return {"session_id": session_id}


@app.post("/upload")
async def upload_document(session_id: str, file: UploadFile = File(...)):
    """Upload and index a document into an existing session."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    filename = file.filename or "document"
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ("pdf", "docx", "doc"):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 20MB.")

    try:
        store = process_document(file_bytes, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

    session["store"]    = store
    session["filename"] = filename

    return {
        "filename": filename,
        "chunks":   store.index.ntotal,
        "message":  f"Indexed {store.index.ntotal} chunks from {filename}.",
    }


@app.delete("/upload/{session_id}")
async def remove_document(session_id: str):
    """Remove document from session, revert to general chat mode."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session["store"]    = None
    session["filename"] = None
    return {"message": "Document removed. Back to general chat mode."}


@app.post("/chat/stream")
async def chat_stream(req: GeneralChatRequest):
    """
    Unified streaming chat endpoint.
    Auto-detects general vs document mode based on session state.
    """
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please start a new session.")

    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    has_document = session["store"] is not None and not session["store"].is_empty()

    full_answer_parts = []

    def event_stream():
        if has_document:
            # Document mode — retrieve context first
            context, sources = retrieve_context(session["store"], question)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            generator = stream_document(context, question, session["history"])
        else:
            # General mode
            yield f"data: {json.dumps({'type': 'mode', 'mode': 'general'})}\n\n"
            generator = stream_general(question, session["history"])

        for token in generator:
            full_answer_parts.append(token)
            yield f"data: {json.dumps({'type': 'token', 'text': token})}\n\n"

        # Extract follow-ups from full answer
        full_raw    = "".join(full_answer_parts)
        clean_text, followups = extract_followups(full_raw)

        # Save clean version to history
        session["history"].append({"role": "user",  "parts": [question]})
        session["history"].append({"role": "model", "parts": [clean_text]})

        # Keep last 20 turns
        if len(session["history"]) > 20:
            session["history"] = session["history"][-20:]

        # Send follow-ups and done signal
        yield f"data: {json.dumps({'type': 'followups', 'questions': followups})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'clean_text': clean_text})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Save thumbs up/down feedback for a message."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session["feedback"][req.message_idx] = req.feedback
    return {"message": "Feedback saved."}


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "has_document":  session["store"] is not None,
        "filename":      session["filename"],
        "history_turns": len(session["history"]) // 2,
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": "Session deleted."}


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}
