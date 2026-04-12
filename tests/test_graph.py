from twelve_angry_agents.graph import build_graph, route_after_consensus, route_after_agent_speak
from twelve_angry_agents.state import DebateState


def make_state(**overrides) -> DebateState:
    base: DebateState = {
        "topic": "Test topic",
        "enriched_topic": "Test topic with context",
        "verdict_framing": "proceed / don't proceed",
        "votes": {},
        "original_votes": {},
        "transcript": [],
        "summary": "",
        "round": 0,
        "speaking_order": ["Agent0", "Agent1"],
        "current_speaker_idx": 0,
        "verdict": None,
        "status": "voting",
    }
    base.update(overrides)
    return base


def test_route_after_consensus_goes_to_close_when_concluded():
    state = make_state(status="concluded")
    assert route_after_consensus(state) == "close"


def test_route_after_consensus_goes_to_deliberate_when_split():
    state = make_state(status="voting")
    assert route_after_consensus(state) == "deliberate"


def test_route_after_agent_speak_loops_when_agents_remain():
    state = make_state(
        speaking_order=["Agent0", "Agent1", "Agent2"],
        current_speaker_idx=1,  # just spoke, index now points to Agent1 (already done), Agent2 remains
    )
    assert route_after_agent_speak(state) == "next_agent"


def test_route_after_agent_speak_goes_to_memory_check_when_done():
    state = make_state(
        speaking_order=["Agent0", "Agent1"],
        current_speaker_idx=2,  # past end of list
    )
    assert route_after_agent_speak(state) == "memory_check"


def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")
