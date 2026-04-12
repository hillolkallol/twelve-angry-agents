from langgraph.graph import END, START, StateGraph

from twelve_angry_agents.nodes.agent import agent_speak_node, blind_vote_node, vote_again_node
from twelve_angry_agents.nodes.consensus import check_consensus, is_hung_jury
from twelve_angry_agents.nodes.moderator import (
    context_gather_node,
    memory_check_node,
    moderator_close_node,
    moderator_deliberate_node,
    moderator_open_node,
)
from twelve_angry_agents.state import DebateState


def route_after_consensus(state: DebateState) -> str:
    """Route to close if status is concluded, else start deliberation."""
    if state["status"] == "concluded":
        return "close"
    return "deliberate"


def route_after_agent_speak(state: DebateState) -> str:
    """Loop back to agent_speak while agents remain in speaking_order."""
    if state["current_speaker_idx"] < len(state["speaking_order"]):
        return "next_agent"
    return "memory_check"


def consensus_check_node(state: DebateState, config: dict) -> dict:
    """Check votes for consensus or hung jury; update status to concluded if either."""
    from twelve_angry_agents.config import AppConfig
    cfg: AppConfig = config["configurable"]["app_config"]

    if check_consensus(state["votes"]):
        return {"status": "concluded"}

    if is_hung_jury(state["round"], cfg.debate.max_rounds):
        return {"status": "concluded"}

    return {}


def build_graph():
    """Assemble and compile the full debate LangGraph."""
    graph = StateGraph(DebateState)

    # Register nodes
    graph.add_node("context_gather", context_gather_node)
    graph.add_node("moderator_open", moderator_open_node)
    graph.add_node("blind_vote", blind_vote_node)
    graph.add_node("consensus_check", consensus_check_node)
    graph.add_node("moderator_deliberate", moderator_deliberate_node)
    graph.add_node("agent_speak", agent_speak_node)
    graph.add_node("memory_check", memory_check_node)
    graph.add_node("vote_again", vote_again_node)
    graph.add_node("moderator_close", moderator_close_node)

    # Entry
    graph.add_edge(START, "context_gather")

    # Linear edges
    graph.add_edge("context_gather", "moderator_open")
    graph.add_edge("moderator_open", "blind_vote")
    graph.add_edge("blind_vote", "consensus_check")
    graph.add_edge("moderator_deliberate", "agent_speak")
    graph.add_edge("memory_check", "vote_again")
    graph.add_edge("vote_again", "consensus_check")
    graph.add_edge("moderator_close", END)

    # Conditional: after consensus check
    graph.add_conditional_edges(
        "consensus_check",
        route_after_consensus,
        {
            "close": "moderator_close",
            "deliberate": "moderator_deliberate",
        },
    )

    # Conditional: loop through agents during deliberation
    graph.add_conditional_edges(
        "agent_speak",
        route_after_agent_speak,
        {
            "next_agent": "agent_speak",
            "memory_check": "memory_check",
        },
    )

    return graph.compile()
