import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gemini import detect_startup_mode, extract_followups  # noqa: E402


# ---------------------------------------------------------------------
# detect_startup_mode
# ---------------------------------------------------------------------

def test_detect_startup_mode_general_for_plain_chitchat():
    assert detect_startup_mode("how's the weather today?") == "general"


def test_detect_startup_mode_startup_for_business_idea():
    assert detect_startup_mode("I have an idea for a SaaS platform for founders") == "startup"


def test_detect_startup_mode_pitch_takes_priority_over_startup():
    # "pitch deck" should route to pitch mode even though the sentence
    # also contains startup keywords like "startup"/"investor".
    question = "make a pitch deck for my startup for investors"
    assert detect_startup_mode(question) == "pitch"


def test_detect_startup_mode_is_case_insensitive():
    assert detect_startup_mode("Tell me about my STARTUP idea") == "startup"


# ---------------------------------------------------------------------
# extract_followups
# ---------------------------------------------------------------------

def test_extract_followups_parses_trailing_json():
    raw = (
        'Here is your answer.\n\n'
        '{"followups":["Q1?","Q2?","Q3?"]}'
    )
    clean, followups = extract_followups(raw)
    assert clean == "Here is your answer."
    assert followups == ["Q1?", "Q2?", "Q3?"]


def test_extract_followups_caps_at_three():
    raw = 'Answer text {"followups":["Q1?","Q2?","Q3?","Q4?"]}'
    _, followups = extract_followups(raw)
    assert len(followups) == 3


def test_extract_followups_missing_json_returns_empty_list():
    raw = "Just a plain answer with no followups block."
    clean, followups = extract_followups(raw)
    assert clean == raw.strip()
    assert followups == []


def test_extract_followups_malformed_json_falls_back_gracefully():
    raw = 'Answer text {"followups": [unterminated'
    clean, followups = extract_followups(raw)
    # Should not raise — falls back to returning the raw text untouched.
    assert followups == []
    assert "Answer text" in clean