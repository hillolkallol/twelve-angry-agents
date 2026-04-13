from langchain_core.messages import SystemMessage, HumanMessage

from twelve_angry_agents.config import AgentPersona, AppConfig, DebateConfig, ModelConfig
from twelve_angry_agents.nodes.moderator import (
    extract_verdict_framing,
    extract_vote_options,
    build_foreman_open_messages,
    build_foreman_close_messages,
    build_foreman_probe_messages,
    build_context_check_messages,
)


def make_moderator_config() -> AppConfig:
    agents = [
        AgentPersona(name=f"Agent{i}", system_prompt=f"Prompt {i}.")
        for i in range(12)
    ]
    return AppConfig(
        model=ModelConfig(),
        debate=DebateConfig(),
        moderator=AgentPersona(name="The Foreman", system_prompt="You moderate."),
        agents=agents,
    )


def test_extract_verdict_framing_from_response():
    response = "FRAMING: proceed / don't proceed\nThis is a decision topic."
    assert extract_verdict_framing(response) == "proceed / don't proceed"


def test_extract_verdict_framing_returns_default_on_failure():
    response = "This is a decision topic."
    result = extract_verdict_framing(response)
    assert "/" in result  # must be a valid framing with two options


def test_extract_vote_options_splits_correctly():
    options = extract_vote_options("proceed / don't proceed")
    assert options == ["proceed", "don't proceed"]


def test_extract_vote_options_strips_whitespace():
    options = extract_vote_options("  sound  /  unsound  ")
    assert options == ["sound", "unsound"]


def test_build_foreman_open_messages_contains_topic():
    cfg = make_moderator_config()
    messages = build_foreman_open_messages(
        moderator=cfg.moderator,
        enriched_topic="Should I quit my job?",
    )
    combined = " ".join(m.content for m in messages)
    assert "quit my job" in combined


def test_build_foreman_close_messages_contains_votes():
    cfg = make_moderator_config()
    votes = {"Agent0": "proceed", "Agent1": "proceed"}
    transcript_text = "Agent0 argued X. Agent1 agreed."
    messages = build_foreman_close_messages(
        moderator=cfg.moderator,
        enriched_topic="Topic.",
        verdict_framing="proceed / don't proceed",
        votes=votes,
        transcript_text=transcript_text,
        hung_jury=False,
    )
    combined = " ".join(m.content for m in messages)
    assert "proceed" in combined


def test_build_context_check_messages_contains_topic():
    cfg = make_moderator_config()
    messages = build_context_check_messages(
        moderator=cfg.moderator,
        topic="Should I quit?",
    )
    combined = " ".join(m.content for m in messages)
    assert "quit" in combined


def test_build_foreman_probe_messages_targets_minority():
    cfg = make_moderator_config()
    votes = {"Agent0": "proceed", "Agent1": "proceed", "Agent2": "don't proceed"}
    messages = build_foreman_probe_messages(
        moderator=cfg.moderator,
        enriched_topic="Should I quit?",
        verdict_framing="proceed / don't proceed",
        votes=votes,
        summary="",
    )
    combined = " ".join(m.content for m in messages)
    # minority agent should be mentioned as target
    assert "Agent2" in combined
    assert "quit" in combined
