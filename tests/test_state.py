from typing import get_type_hints
from twelve_angry_agents.state import DebateState


def test_debate_state_has_required_fields():
    hints = get_type_hints(DebateState)
    required = [
        "topic", "enriched_topic", "verdict_framing",
        "votes", "original_votes", "transcript",
        "summary", "round", "speaking_order",
        "current_speaker_idx", "verdict", "status",
    ]
    for field in required:
        assert field in hints, f"Missing field: {field}"


def test_debate_state_can_be_constructed():
    state: DebateState = {
        "topic": "Should I quit my job?",
        "enriched_topic": "Should I quit my job? I have 6 months runway.",
        "verdict_framing": "proceed / don't proceed",
        "votes": {"The Skeptic": "don't proceed"},
        "original_votes": {"The Skeptic": "don't proceed"},
        "transcript": [],
        "summary": "",
        "round": 0,
        "speaking_order": ["The Skeptic", "The Optimist"],
        "current_speaker_idx": 0,
        "verdict": None,
        "status": "gathering",
    }
    assert state["topic"] == "Should I quit my job?"
    assert state["status"] == "gathering"
