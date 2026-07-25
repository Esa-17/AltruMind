import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import gemini  # noqa: E402
from google.genai import types  # noqa: E402


class _FakeChunk:
    def __init__(self, text):
        self.text = text


def test_normalize_parts_converts_plain_strings():
    parts = gemini._normalize_parts(["hello world"])
    assert len(parts) == 1
    assert isinstance(parts[0], types.Part)


def test_normalize_parts_converts_image_dicts():
    parts = gemini._normalize_parts([{"mime_type": "image/png", "data": b"fake-bytes"}])
    assert len(parts) == 1
    assert isinstance(parts[0], types.Part)


def test_normalize_contents_builds_content_list_from_session_history():
    history = [
        {"role": "user", "parts": ["hi"]},
        {"role": "model", "parts": ["hello, how can I help?"]},
    ]
    contents = gemini._normalize_contents(history)
    assert len(contents) == 2
    assert contents[0].role == "user"
    assert contents[1].role == "model"


def test_stream_general_yields_tokens_from_mocked_sdk():
    fake_chunks = [_FakeChunk("Hello "), _FakeChunk("world"), _FakeChunk(None)]

    with patch.object(
        gemini.client.models, "generate_content_stream", return_value=fake_chunks
    ) as mock_stream:
        tokens = list(gemini.stream_general("hi there, just chatting", []))

    assert tokens == ["Hello ", "world"]
    assert mock_stream.called


def test_stream_general_startup_mode_calls_websearch_and_streams(monkeypatch):
    monkeypatch.setattr(
        gemini,
        "search_competitors",
        lambda question: [{"title": "T", "url": "http://x", "snippet": "s"}],
    )
    fake_chunks = [_FakeChunk("Report text")]

    with patch.object(
        gemini.client.models, "generate_content_stream", return_value=fake_chunks
    ):
        tokens = list(gemini.stream_general("startup idea for a SaaS tool", []))

    assert tokens == ["Report text"]


def test_stream_document_merges_text_and_image_parts():
    fake_chunks = [_FakeChunk("Doc answer")]
    images = [{"mime_type": "image/png", "data": b"fake-bytes"}]

    with patch.object(
        gemini.client.models, "generate_content_stream", return_value=fake_chunks
    ) as mock_stream:
        tokens = list(
            gemini.stream_document("some doc context", "what's in this?", [], images)
        )

    assert tokens == ["Doc answer"]
    # contents kwarg should include the image part alongside the prompt text
    _, kwargs = mock_stream.call_args
    last_message_parts = kwargs["contents"][-1].parts
    assert len(last_message_parts) == 2  # prompt text + 1 image


def test_stream_general_handles_sdk_exception_gracefully():
    def _raise(*args, **kwargs):
        raise RuntimeError("network exploded")

    with patch.object(gemini.client.models, "generate_content_stream", side_effect=_raise):
        tokens = list(gemini.stream_general("hi", []))

    assert len(tokens) == 1
    assert "Something went wrong" in tokens[0]


def test_handle_gemini_error_quota_message():
    msg = gemini.handle_gemini_error(Exception("Resource_exhausted: quota hit"))
    assert "quota" in msg.lower()


def test_handle_gemini_error_api_key_message():
    msg = gemini.handle_gemini_error(Exception("invalid api key"))
    assert "API Key" in msg