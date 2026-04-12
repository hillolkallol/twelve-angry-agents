"""
Integration smoke test — requires Ollama running with gemma4:e4b pulled.
Skip in CI: pytest -m "not integration"
"""
import pytest
from twelve_angry_agents.config import load_config
from twelve_angry_agents.graph import build_graph
from twelve_angry_agents.state import DebateState


@pytest.mark.integration
def test_full_debate_runs_to_completion():
    config = load_config()
    graph = build_graph()

    initial_state: DebateState = {
        "topic": "Should Python be the default language for all new backend projects?",
        "enriched_topic": "Should Python be the default language for all new backend projects? Team has mixed experience.",
        "verdict_framing": "proceed / don't proceed",
        "votes": {},
        "original_votes": {},
        "transcript": [],
        "summary": "",
        "round": 0,
        "speaking_order": [],
        "current_speaker_idx": 0,
        "verdict": None,
        "status": "gathering",
    }

    result = graph.invoke(
        initial_state,
        config={"configurable": {"app_config": config}},
    )

    assert result["status"] == "concluded"
    assert result["verdict"] is not None
    assert len(result["votes"]) == 12
