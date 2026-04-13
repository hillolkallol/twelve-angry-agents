from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from twelve_angry_agents.config import AgentPersona, AppConfig, DebateConfig, ModelConfig
from twelve_angry_agents.nodes.agent import (
    extract_vote,
    build_blind_vote_messages,
    build_clarify_vote_messages,
    build_deliberation_messages,
)


def test_extract_vote_parses_proceed():
    response = "VOTE: proceed\nThe opportunity is clear here."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "proceed"


def test_extract_vote_parses_dont_proceed():
    response = "VOTE: don't proceed\nThe risk is too high."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "don't proceed"


def test_extract_vote_case_insensitive():
    response = "VOTE: PROCEED\nSome reason."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "proceed"


def test_extract_vote_returns_undecided_on_failure():
    response = "I think this is a good idea overall."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "undecided"


def test_extract_vote_strips_markdown_bold():
    response = "VOTE: **don't proceed**\nThe risk is too high."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "don't proceed"


def test_extract_vote_bare_no_maps_to_negative_option():
    response = "VOTE: No\nThe risk is too high."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "don't proceed"


def test_extract_vote_bare_yes_maps_to_positive_option():
    response = "VOTE: Yes\nThe opportunity is clear."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "proceed"


def test_extract_vote_first_line_fallback():
    # No VOTE: prefix — agent just leads with the option
    response = "don't proceed\n\nThe risk outweighs the reward."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "don't proceed"


def test_extract_vote_first_line_no_fallback():
    response = "No\n\nThe consensus is clear — decline."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "don't proceed"


def test_extract_vote_bold_first_line_fallback():
    response = "**Don't proceed**\n\nThe risk is unacceptable."
    assert extract_vote(response, ["proceed", "don't proceed"]) == "don't proceed"


def test_build_clarify_vote_messages_appends_correction():
    prior = [
        SystemMessage(content="You are skeptical."),
        HumanMessage(content="Topic: Should I quit?"),
    ]
    messages = build_clarify_vote_messages(
        prior_messages=prior,
        prior_response="I think the risk is too high overall.",
        verdict_framing="proceed / don't proceed",
    )
    # prior messages + AI echo + clarification HumanMessage
    assert len(messages) == 4
    assert isinstance(messages[-1], HumanMessage)
    assert "VOTE: proceed" in messages[-1].content
    assert "VOTE: don't proceed" in messages[-1].content


def test_build_blind_vote_messages_contains_topic():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    messages = build_blind_vote_messages(
        agent=agent,
        enriched_topic="Should I quit my job? I have 6 months runway.",
        verdict_framing="proceed / don't proceed",
    )
    combined = " ".join(m.content for m in messages)
    assert "quit my job" in combined
    assert "proceed" in combined


def test_build_blind_vote_messages_starts_with_system():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    messages = build_blind_vote_messages(
        agent=agent,
        enriched_topic="Topic here.",
        verdict_framing="proceed / don't proceed",
    )
    assert isinstance(messages[0], SystemMessage)


def test_build_blind_vote_messages_includes_own_name():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    messages = build_blind_vote_messages(
        agent=agent,
        enriched_topic="Topic.",
        verdict_framing="proceed / don't proceed",
        all_agent_names=["The Skeptic", "The Optimist", "The Analyst"],
    )
    system_content = messages[0].content
    assert "The Skeptic" in system_content
    assert "The Optimist" in system_content
    assert "The Analyst" in system_content


def test_build_deliberation_messages_includes_transcript():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    transcript = [
        AIMessage(content="VOTE: proceed\nGood opportunity.", name="The Optimist"),
    ]
    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic="Should I quit?",
        verdict_framing="proceed / don't proceed",
        current_vote="don't proceed",
        transcript=transcript,
        summary="",
    )
    combined = " ".join(m.content for m in messages)
    assert "Good opportunity" in combined
    assert "don't proceed" in combined


def test_build_deliberation_messages_includes_summary_when_present():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic="Topic.",
        verdict_framing="proceed / don't proceed",
        current_vote="don't proceed",
        transcript=[],
        summary="Round 1 summary: The Optimist argued X.",
    )
    combined = " ".join(m.content for m in messages)
    assert "Round 1 summary" in combined


def test_build_deliberation_messages_surfaces_opponents():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    transcript = [
        AIMessage(content="VOTE: proceed\nThis is a great chance to grow.", name="The Optimist"),
    ]
    votes = {"The Skeptic": "don't proceed", "The Optimist": "proceed"}
    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic="Should I quit?",
        verdict_framing="proceed / don't proceed",
        current_vote="don't proceed",
        transcript=transcript,
        summary="",
        votes=votes,
    )
    combined = " ".join(m.content for m in messages)
    assert "The Optimist" in combined
    assert "great chance to grow" in combined


def test_build_deliberation_messages_no_opponents_when_unanimous():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    votes = {"The Skeptic": "don't proceed", "The Optimist": "don't proceed"}
    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic="Should I quit?",
        verdict_framing="proceed / don't proceed",
        current_vote="don't proceed",
        transcript=[],
        summary="",
        votes=votes,
    )
    combined = " ".join(m.content for m in messages)
    assert "opposing your position" not in combined


def test_build_deliberation_messages_no_votes_param_still_works():
    # Backwards-compatible: votes defaults to None, no opponents section
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic="Topic.",
        verdict_framing="proceed / don't proceed",
        current_vote="don't proceed",
        transcript=[],
        summary="",
    )
    combined = " ".join(m.content for m in messages)
    assert "opposing your position" not in combined


def test_build_deliberation_messages_includes_agent_name_in_system():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic="Topic.",
        verdict_framing="proceed / don't proceed",
        current_vote="don't proceed",
        transcript=[],
        summary="",
        all_agent_names=["The Skeptic", "The Optimist", "The Analyst"],
    )
    system_content = messages[0].content
    assert "The Skeptic" in system_content
    assert "The Optimist" in system_content


def test_build_deliberation_messages_includes_moderator_question():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic="Topic.",
        verdict_framing="proceed / don't proceed",
        current_vote="don't proceed",
        transcript=[],
        summary="",
        moderator_question="The Skeptic, what evidence would change your mind?",
    )
    human_content = messages[1].content
    assert "THE FOREMAN ASKS" in human_content
    assert "what evidence would change your mind" in human_content


def test_build_deliberation_messages_mind_change_instruction_present():
    agent = AgentPersona(name="The Skeptic", system_prompt="You are skeptical.")
    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic="Topic.",
        verdict_framing="proceed / don't proceed",
        current_vote="don't proceed",
        transcript=[],
        summary="",
    )
    human_content = messages[1].content
    assert "I changed my vote to" in human_content
