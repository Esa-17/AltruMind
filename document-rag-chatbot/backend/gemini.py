import os
import json
from pathlib import Path
from typing import List, Generator

from google import genai
from google.genai import types, errors
from dotenv import load_dotenv

from websearch import search_competitors, format_results_for_prompt

# Load .env
load_dotenv(Path(__file__).parent / ".env")

# -------------------------------------------------------
# Configure Gemini
# -------------------------------------------------------
# NOTE: this project was migrated from the deprecated
# `google-generativeai` SDK to the current `google-genai` SDK
# (https://github.com/googleapis/python-genai). The old package has
# stopped receiving updates/fixes from Google. Behavior is unchanged —
# same prompts, same streaming responses — only the client plumbing
# below is different.

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

# Use latest supported alias
MODEL_NAME = "gemini-3.5-flash"


# -------------------------------------------------------
# SYSTEM PROMPTS
# -------------------------------------------------------

GENERAL_SYSTEM_PROMPT = """
You are AltruMind, an AI Startup Copilot built by Altruisty Innovation Pvt Ltd.

Your goal is to help founders validate startup ideas and build businesses.

If the user's message describes or asks about a startup, business idea, SaaS product, AI application, app idea or entrepreneurial concept, automatically generate a complete Startup Blueprint.

The Startup Blueprint must contain the following sections.

# 🚀 Startup Validation

Idea Summary

Startup Health Score (0-100)

Verdict
(🟢 Strong Idea / 🟡 Needs Refinement / 🔴 High Risk)

Problem Being Solved

Target Customers

Unique Value Proposition

Market Opportunity

Potential Risks

Opportunities

------------------------------------------------

# 🏆 Competitor Analysis

Top Competitors

How They Succeed

Market Gap

Differentiation Strategy

------------------------------------------------

# 💼 Lean Business Model Canvas

Customer Segments

Value Proposition

Channels

Customer Relationships

Revenue Streams

Key Activities

Key Resources

Key Partners

Cost Structure

------------------------------------------------

# 💰 Revenue Strategy

Primary Revenue Model

Secondary Revenue Model

Pricing Suggestion

Future Expansion

------------------------------------------------

# 🛠 MVP Roadmap

Core Features

Advanced Features

Recommended AI Features

Recommended Tech Stack

Development Timeline

Estimated Team Size

------------------------------------------------

# 📈 Go-To-Market Strategy

Marketing Channels

First 100 Customers

Growth Strategy

------------------------------------------------

# ⚠ Risk Analysis

Business Risks

Technical Risks

Market Risks

------------------------------------------------

# 🎯 Next Action Plan

Provide 5 practical next steps.

Rules

• Never invent fake statistics.

• Mention assumptions when necessary.

• Use markdown.

• Be concise but practical.

• Use bullet points.

If the user's question is NOT startup related, answer normally.

Always finish with

{"followups":["Question 1?","Question 2?","Question 3?"]}
"""
DOCUMENT_SYSTEM_PROMPT = """
You are AltruMind, an AI Startup Copilot.

The user has uploaded one or more documents and/or images.

Your job is to combine:

• your own knowledge
• the uploaded documents
• uploaded images (if any)

If the uploaded content is startup-related (business plans, pitch decks, product ideas, market research, UI mockups, prototypes, requirement documents, etc.), analyze it like a startup consultant.

When appropriate include:

# 🚀 Startup Health Score

Overall Score (0-100)

Problem Validation

Market Demand

Competition

Revenue Potential

Technical Feasibility

Investment Potential

Briefly explain each score.

Then continue with sections such as:

📄 Document Insights

💡 Business Insights

⚠ Risks

🚀 Opportunities

📈 Recommendations

If the uploaded files are NOT startup-related, simply answer using the document context.

Never invent information that isn't present in the uploaded files.

Always finish with

{"followups":["Question 1?","Question 2?","Question 3?"]}
"""
# -------------------------------------------------------
# GENERAL CHAT
# -------------------------------------------------------
def detect_startup_mode(question:str):
    question=question.lower()

    startup_keywords=[
        "startup",
        "business",
        "idea",
        "app",
        "platform",
        "saas",
        "product",
        "entrepreneur",
        "company",
        "founder",
        "mvp",
        "market",
        "revenue",
        "customer",
        "pitch",
        "investor",
        "funding",
        "validate",
        "business model",
        "problem statement",
        "competitor",
        "go to market",
        "gtm"
    ]

    pitch_keywords=[
        "pitch deck",
        "presentation",
        "slides",
        "investor deck",
        "pitch presentation"
    ]

    if any(word in question for word in pitch_keywords):
        return "pitch"

    if any(word in question for word in startup_keywords):
        return "startup"

    return "general"

    return any(word in question for word in keywords)


def _normalize_parts(parts: list) -> list:
    """
    Converts a list of raw parts — plain strings, or the old SDK's
    {"mime_type": ..., "data": ...} image-blob dicts — into
    google.genai `types.Part` objects.

    This exists so session["history"] in main.py (which stores parts
    as plain strings / dicts, unchanged since the old SDK) can still
    be fed straight into the new SDK without touching main.py at all.
    """
    normalized = []
    for p in parts:
        if isinstance(p, types.Part):
            normalized.append(p)
        elif isinstance(p, str):
            normalized.append(types.Part.from_text(text=p))
        elif isinstance(p, dict) and "mime_type" in p and "data" in p:
            normalized.append(types.Part.from_bytes(data=p["data"], mime_type=p["mime_type"]))
        else:
            normalized.append(types.Part.from_text(text=str(p)))
    return normalized


def _normalize_contents(messages: List[dict]) -> List["types.Content"]:
    """Converts a list of {"role": ..., "parts": [...]} dicts (the shape
    used throughout this app's session history) into the list of
    types.Content the new SDK expects for the `contents` argument."""
    return [
        types.Content(role=m["role"], parts=_normalize_parts(m["parts"]))
        for m in messages
    ]


def _stream(system_prompt: str, contents: List["types.Content"]):
    """Runs a streaming generate_content call grounded in the given
    system_instruction, yielding text chunks as they arrive."""
    response = client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    for chunk in response:
        if getattr(chunk, "text", None):
            yield chunk.text


def handle_gemini_error(e):
    # The new SDK raises google.genai.errors.APIError (and subclasses)
    # with structured .code / .message attributes for API-side errors.
    if isinstance(e, errors.APIError):
        code = getattr(e, "code", None)
        message = str(getattr(e, "message", "") or "").lower()

        if code == 429 or "resource_exhausted" in message or "quota" in message:
            return (
                "⚠️ **Gemini API quota exceeded.**\n\n"
                "Please wait about a minute and try again."
            )
        if code == 404 or "not found" in message:
            return "❌ **Gemini model unavailable.**"
        if code in (401, 403) or "permission" in message or "api key" in message:
            return "🔑 **Invalid Gemini API Key.**"

    # Fallback: text-match on whatever the exception says, in case it's
    # a network/library error rather than a structured APIError.
    error = str(e).lower()

    if "quota" in error or "rate limit" in error or "resource_exhausted" in error:
        return (
            "⚠️ **Gemini API quota exceeded.**\n\n"
            "Please wait about a minute and try again."
        )

    if "model" in error or "not found" in error:
        return (
            "❌ **Gemini model unavailable.**"
        )

    if "api key" in error or "permission" in error:
        return (
            "🔑 **Invalid Gemini API Key.**"
        )

    return (
        "⚠️ **Something went wrong while contacting Gemini.**"
    )
def stream_general(question: str, history: List[dict]):
    print("===== STREAM_GENERAL =====")
    mode=detect_startup_mode(question)
    if mode=="startup":

        # Ground the Competitor Analysis section in a real, live web
        # search instead of letting the model invent competitor names
        # and stats from memory.
        research_results = search_competitors(question)
        research_context = format_results_for_prompt(research_results)

        prompt=f"""
    User Startup Idea

    {question}

    Live Market Research (real web search results, fetched just now):

    {research_context}

    Generate a professional Startup Evaluation Report.

    Follow this structure exactly.

    # 🚀 Startup Health Score

    Overall Score: XX/100

    Rate these individually:

    Problem Validation

    Market Demand

    Competition

    Revenue Potential

    Scalability

    Technical Feasibility

    Investment Potential

    For every score explain WHY.

    Then generate:

    ## Problem Statement

    ## Target Customers

    ## Unique Value Proposition

    ## Competitor Analysis

    Base this ONLY on the Live Market Research above. Name the actual
    competitors found there. Do NOT invent competitors, funding
    numbers, or user counts that aren't in the research results. If
    the research above is insufficient to name real competitors, say
    so honestly instead of guessing.

    ## Revenue Model

    ## Recommended MVP Features

    ## Recommended Tech Stack

    ## Biggest Risks

    ## 30-Day Action Plan

    ## 🔗 Sources
    List the source URLs from the Live Market Research above that you
    actually used. If none were used, omit this section.

    Use realistic scores.

    Don't give every idea 90+.

    Finally generate three follow-up questions.

    """
    elif mode=="pitch":

        prompt=f"""
    You are creating an investor-ready startup pitch deck.

    Startup Idea

    {question}

    Generate a professional pitch deck using markdown.

    # Slide 1
    Startup Name

    Tagline

    Vision

    ---

    # Slide 2
    Problem

    ---

    # Slide 3
    Solution

    ---

    # Slide 4
    Market Opportunity

    TAM

    SAM

    SOM

    ---

    # Slide 5
    Business Model

    Revenue Streams

    Pricing

    ---

    # Slide 6
    Competitor Analysis

    Competitor

    Strength

    Weakness

    Our Advantage

    ---

    # Slide 7
    Go-To-Market Strategy

    ---

    # Slide 8
    Product Roadmap

    MVP

    Version 2

    Future Vision

    ---

    # Slide 9
    Funding Ask

    How much funding is required.

    How it will be used.

    ---

    # Slide 10
    Closing

    One powerful investor pitch.

    Always finish with

    {"followups":["Improve this pitch?","Generate a lean canvas?","Create an MVP roadmap?"]}
    """

    else:
        prompt=question

    messages=history+[
    {
        "role":"user",
        "parts":[prompt]
    }
    ]
    try:
        print("Sending request to Gemini...")
        contents = _normalize_contents(messages)
        for chunk_text in _stream(GENERAL_SYSTEM_PROMPT, contents):
            yield chunk_text
    except Exception as e:
        print(e)
        yield handle_gemini_error(e)
# -------------------------------------------------------
# DOCUMENT CHAT
# -------------------------------------------------------
def stream_document(
    context: str,
    question: str,
    history: List[dict],
    images=None
) -> Generator[str,None,None]:
    prompt=f"""
    You have access to document context and uploaded images.

    Document Context:
    {context}

    User Question:
    {question}

    If the uploaded documents describe a startup, product, business plan, pitch deck, UI, prototype or idea, generate a Startup Health Score before answering.

    Base the scores on the uploaded files.

    Explain every score briefly.

    Then continue answering normally.

    If images are available, use them together with the documents.

    If both documents and images contain useful information, combine them into one answer.
    """
    parts=[prompt]

    if images:
        parts.extend(images)

    messages=history+[
    {
        "role":"user",
        "parts":parts
    }
    ]
    try:
        contents = _normalize_contents(messages)
        for chunk_text in _stream(DOCUMENT_SYSTEM_PROMPT, contents):
            yield chunk_text
    except Exception as e:
        print(e)
        yield handle_gemini_error(e)
# -------------------------------------------------------
# FOLLOWUPS
# -------------------------------------------------------
def extract_followups(full_text: str):
    try:
        start = full_text.rfind('{"followups"')
        if start == -1:
            return full_text.strip(), []
        json_text = full_text[start:]
        clean = full_text[:start].strip()
        data = json.loads(json_text)
        return clean, data.get("followups", [])[:3]
    except Exception:
        return full_text.strip(), []