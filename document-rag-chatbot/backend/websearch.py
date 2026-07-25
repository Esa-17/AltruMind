"""
websearch.py
------------
Lightweight live web search used to GROUND AltruMind's Competitor
Analysis in real, current results — instead of letting the LLM invent
competitor names, funding numbers, or market stats from memory.

Implementation note: this deliberately does NOT use the
`duckduckgo-search` PyPI package. Recent versions of that package pull
in `pyreqwest-impersonate`, a Rust-compiled dependency that fails to
install on Windows machines without the Visual C++ Build Tools /
Rust toolchain — a bad dependency to hand anyone reviewing this repo.

Instead this hits DuckDuckGo's plain HTML results page directly with
`requests` + `beautifulsoup4` — both pure Python, no compilation,
works identically on Windows/Mac/Linux, no API key required.

Design notes:
- Fails soft everywhere: any network/parsing error returns an empty
  list rather than raising, so chat always falls back gracefully to
  the model's own knowledge instead of crashing.
- Snippets are truncated so we don't blow up the prompt / context
  window with raw scraped text.
"""

from typing import List, Dict
from urllib.parse import urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup

MAX_RESULTS = 5
MAX_SNIPPET_CHARS = 300
SEARCH_URL = "https://html.duckduckgo.com/html/"
REQUEST_TIMEOUT = 8

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _clean_ddg_url(href: str) -> str:
    """DuckDuckGo's HTML results wrap real links as /l/?uddg=<encoded-url>&..."""
    if href.startswith("/l/?"):
        query = parse_qs(urlparse(href).query)
        real_url = query.get("uddg", [""])[0]
        return unquote(real_url)
    return href


def search_web(query: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """
    Runs a live web search and returns a clean list of results:
    [{ "title": ..., "url": ..., "snippet": ... }, ...]

    Never raises — on failure it returns [] so the caller can decide
    to fall back to the model's own knowledge.
    """
    try:
        response = requests.post(
            SEARCH_URL,
            data={"q": query},
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"[websearch] search failed for '{query}': {e}")
        return []

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        results = []

        for result in soup.select(".result")[:max_results]:
            title_tag = result.select_one(".result__a")
            snippet_tag = result.select_one(".result__snippet")

            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            url = _clean_ddg_url(title_tag.get("href", ""))
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            if len(snippet) > MAX_SNIPPET_CHARS:
                snippet = snippet[:MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + "..."

            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})

        return results
    except Exception as e:
        print(f"[websearch] failed to parse results for '{query}': {e}")
        return []


def format_results_for_prompt(results: List[Dict]) -> str:
    """Formats search results into plain text the LLM can ground answers in."""
    if not results:
        return "(No live search results were available for this query.)"

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"{i}. {r['title']}\n   {r['snippet']}\n   Source: {r['url']}")
    return "\n\n".join(lines)


def search_competitors(idea: str, max_results: int = MAX_RESULTS) -> List[Dict]:
    """Convenience wrapper: search specifically for competitors/alternatives to an idea."""
    query = f"top competitors and alternatives to {idea}"
    return search_web(query, max_results=max_results)