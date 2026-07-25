import os
import uuid
import json
from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from rag import process_document, retrieve_context, VectorStore
from gemini import stream_general, stream_document, extract_followups

load_dotenv(Path(__file__).parent / ".env")
app = FastAPI(title="AltruMind", version="2.0.0")
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount(
    "/static",
    StaticFiles(directory="../frontend"),
    name="static"
)
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
        "documents": [],
        "history": [],
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
    ext = filename.rsplit(".",1)[-1].lower()

    allowed = ("pdf","doc","docx","png","jpg","jpeg","webp")

    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type."
        )
    file_bytes = await file.read()
    if len(file_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 20MB.")
    is_document = ext in ("pdf","doc","docx")
    chunk_count = 0

    if is_document:
        store = process_document(file_bytes,filename)
        chunk_count = store.index.ntotal
        session["documents"].append({
            "type":"document",
            "filename":filename,
            "store":store
        })
    else:
        mime_map={
            "png":"image/png",
            "jpg":"image/jpeg",
            "jpeg":"image/jpeg",
            "webp":"image/webp"
        }

        session["documents"].append({
            "type":"image",
            "filename":filename,
            "bytes":file_bytes,
            "mime_type":mime_map[ext]
        })
    print("Documents:", len(session["documents"]))
    print("TOTAL DOCS:", len(session["documents"]))
    return{
        "filename":filename,
        "type":"document" if is_document else "image",
        "chunks":chunk_count,
        "total_documents":len(session["documents"])
    }
@app.delete("/upload/{session_id}")
async def remove_document(session_id: str):
    """Remove all documents from session, revert to general chat mode."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    session["documents"] = []
    return {"message": "Documents removed. Back to general chat mode."}
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
    has_document=len(session["documents"])>0
    full_answer_parts = []
    print("================================")
    print("CHAT REQUEST RECEIVED")
    print("Question:", req.question)
    print("Session:", req.session_id)
    question_lower=question.lower()

    document_keywords=[
        "document",
        "pdf",
        "file",
        "resume",
        "proposal",
        "report",
        "page",
        "contract",
        "invoice",
        "presentation",
        "pitch",
        "according",
        "uploaded",
        "this file",
        "these files"
    ]

    # NOTE: this must be computed BEFORE event_stream() runs, since it's
    # used inside the if-condition below. Previously this was assigned
    # *after* being referenced, which raised UnboundLocalError on every
    # request once a document was uploaded (Python treats any variable
    # assigned inside a function as local to that whole function).
    search_documents = has_document and any(
        word in question_lower
        for word in document_keywords
    )

    def event_stream():
        images=[]
        if search_documents:
            all_chunks=[]

            for doc in session["documents"]:

                if doc["type"]=="image":
                    images.append({
                        "mime_type":doc["mime_type"],
                        "data":doc["bytes"]
                    })
                    continue
                chunks=doc["store"].search(question)
                for chunk in chunks:
                    chunk["filename"]=doc["filename"]

                all_chunks.extend(chunks)

            all_chunks=sorted(all_chunks,key=lambda x:x["score"])[:3]

            context="\n\n".join(
                f"[{c['filename']} - Page {c['page']}]\n{c['text']}"
                for c in all_chunks
            )
            context=context[:7000]

            sources=[
                {
                    "filename":c["filename"],
                    "page":c["page"]
                }
                for c in all_chunks
            ]
            print("DOCUMENTS IN SESSION:")
            for d in session["documents"]:
                print(d["filename"])

            use_images=any(word in question_lower for word in[
                "image",
                "photo",
                "picture",
                "screenshot",
                "diagram",
                "logo",
                "ui",
                "interface",
                "design",
                "screen",
                "graph",
                "chart",
                "visual",
                "what do you see",
                "describe"
            ])

            generator=stream_document(
                context,
                question,
                session["history"],
                images if use_images else None
            )
        else:
            sources = []
            yield f"data: {json.dumps({'type':'mode','mode':'general'})}\n\n"
            generator = stream_general(question,session["history"])
        print("Document mode:", has_document)
        for token in generator:
            full_answer_parts.append(token)
            yield f"data: {json.dumps({'type':'token','text':token})}\n\n"
            print("TOKEN:", repr(token))
        full_raw = "".join(full_answer_parts)
        clean_text, followups = extract_followups(full_raw)
        session["history"].append({"role":"user","parts":[question]})
        session["history"].append({"role":"model","parts":[clean_text]})
        if len(session["history"]) > 8:
            session["history"] = session["history"][-8:]
        yield f"data: {json.dumps({'type':'replace','text':clean_text})}\n\n"
        if sources:
            yield f"data: {json.dumps({'type':'sources','sources':sources})}\n\n"
        yield f"data: {json.dumps({'type':'followups','questions':followups})}\n\n"
        yield f"data: {json.dumps({'type':'done'})}\n\n"
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
        "has_document":len(session["documents"])>0,
        "documents":[doc["filename"] for doc in session["documents"]],
        "history_turns": len(session["history"]) // 2,
    }
@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": "Session deleted."}
@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(sessions)}