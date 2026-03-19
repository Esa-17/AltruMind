import os
import json
import google.generativeai as genai
from typing import List, Generator

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-1.5-flash"

# ── System prompts ─────────────────────────────────────────────────────────────

GENERAL_SYSTEM_PROMPT = """You are AltruMind, an intelligent AI assistant built by Altruisty Innovation Pvt Ltd — a human-first technology company based in Chennai, India with presence in the US and Australia.

Your personality:
- Friendly, sharp, and encouraging — especially with entrepreneurs and startup founders
- Conversational and warm, never robotic
- You celebrate ideas, challenge assumptions constructively, and give practical, actionable advice

Your expertise:
- Startups: ideation, validation, MVPs, product-market fit, pitch decks, fundraising, VC/angel funding, accelerators
- Entrepreneurship: business models, go-to-market strategy, growth hacking, scaling, hiring, co-founders
- Business & finance: revenue models, unit economics, CAC, LTV, burn rate, bootstrapping vs funding
- Technology: AI/ML trends, tech stacks, SaaS, product development
- General knowledge and chitchat: you can talk about anything the user brings up

Response format rules:
1. Keep responses concise but complete — don't over-explain
2. Use bullet points or numbered lists only when it genuinely helps structure the answer
3. Be conversational for chitchat, structured for advice
4. Always end your response with a JSON block of 3 suggested follow-up questions on a new line, formatted EXACTLY like this (no extra text after):
{"followups": ["Question 1?", "Question 2?", "Question 3?"]}

The follow-up questions should feel natural and help the user go deeper on the topic.
"""

DOCUMENT_SYSTEM_PROMPT = """You are AltruMind, an intelligent AI assistant built by Altruisty Innovation Pvt Ltd.

The user has uploaded a document. Your job is to:
1. FIRST answer the question using your general knowledge (startup, business, tech expertise, or general knowledge)
2. THEN check if the uploaded document has anything relevant to add, and if so, include it under a clear "📄 From your document:" section with the page/source citation

Rules:
- Always answer the general knowledge part first — never make the user wait for document context
- Only add the document section if it genuinely adds value — don't force it
- If the question is clearly only about the document (e.g. "summarise this"), skip the general knowledge part
- Keep answers clear and well-structured
- Always end your response with a JSON block of 3 follow-up questions:
{"followups": ["Question 1?", "Question 2?", "Question 3?"]}
"""


# ── Streaming ──────────────────────────────────────────────────────────────────

def stream_general(question: str, history: List[dict]) -> Generator[str, None, None]:
    """Stream a general chat response (no document)."""
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=GENERAL_SYSTEM_PROMPT,
    )
    messages = list(history) + [{"role": "user", "parts": [question]}]
    response = model.generate_content(
        messages,
        stream=True,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1024,
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


def stream_document(context: str, question: str, history: List[dict]) -> Generator[str, None, None]:
    """Stream a document-aware response."""
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=DOCUMENT_SYSTEM_PROMPT,
    )
    user_message = (
        f"Document context (for reference):\n{context}\n\n"
        f"Question: {question}"
    )
    messages = list(history) + [{"role": "user", "parts": [user_message]}]
    response = model.generate_content(
        messages,
        stream=True,
        generation_config=genai.types.GenerationConfig(
            temperature=0.4,
            max_output_tokens=1024,
        ),
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text


# ── Follow-up extractor ────────────────────────────────────────────────────────

def extract_followups(full_text: str):
    """
    Extract the follow-up JSON from the end of the model response.
    Returns (clean_text, list_of_followups).
    """
    try:
        last_brace = full_text.rfind('{"followups"')
        if last_brace == -1:
            return full_text.strip(), []
        json_str   = full_text[last_brace:].strip()
        clean_text = full_text[:last_brace].strip()
        data       = json.loads(json_str)
        followups  = data.get("followups", [])[:3]
        return clean_text, followups
    except Exception:
        return full_text.strip(), []
