from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from twelve_angry_agents.memory import (
    estimate_tokens,
    needs_summarization,
    format_transcript_for_summary,
    build_summarization_messages,
)


def test_estimate_tokens_approximates_length():
    text = "a" * 400  # 400 chars ≈ 100 tokens at 4 chars/token
    assert estimate_tokens(text) == 100


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_needs_summarization_when_over_threshold():
    # 128000 context * 0.75 threshold = 96000 tokens
    assert needs_summarization(
        current_tokens=97000,
        context_window=128000,
        threshold=0.75,
    ) is True


def test_needs_summarization_when_under_threshold():
    assert needs_summarization(
        current_tokens=50000,
        context_window=128000,
        threshold=0.75,
    ) is False


def test_needs_summarization_exactly_at_threshold():
    assert needs_summarization(
        current_tokens=96000,
        context_window=128000,
        threshold=0.75,
    ) is False  # at threshold, not over


def test_format_transcript_for_summary():
    transcript = [
        HumanMessage(content="The topic is: Should I quit?"),
        AIMessage(content="VOTE: proceed\nThe opportunity is clear.", name="The Optimist"),
        AIMessage(content="VOTE: don't proceed\nThe risk is high.", name="The Skeptic"),
    ]
    result = format_transcript_for_summary(transcript)
    assert "The Optimist" in result
    assert "The Skeptic" in result
    assert "proceed" in result


def test_format_transcript_skips_system_messages():
    transcript = [
        SystemMessage(content="You are a moderator."),
        AIMessage(content="VOTE: proceed", name="The Analyst"),
    ]
    result = format_transcript_for_summary(transcript)
    assert "You are a moderator" not in result
    assert "The Analyst" in result


def test_build_summarization_messages_returns_two_messages():
    transcript = [
        AIMessage(content="VOTE: proceed\nGood idea.", name="The Optimist"),
    ]
    messages = build_summarization_messages(transcript)
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
