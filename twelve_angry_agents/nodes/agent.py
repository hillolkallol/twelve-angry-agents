import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from twelve_angry_agents.config import AgentPersona, AppConfig
from twelve_angry_agents.state import DebateState


def extract_vote(response: str, valid_options: list[str]) -> str:
    """Parse VOTE: <option> from agent response. Returns 'undecided' if not found."""
    match = re.search(r"VOTE:\s*(.+)", response, re.IGNORECASE)
    if not match:
        return "undecided"
    raw = match.group(1).strip().lower()
    # Sort longest first to avoid substring false matches (e.g. "proceed" inside "don't proceed")
    for option in sorted(valid_options, key=len, reverse=True):
        if option.lower() in raw:
            return option
    return "undecided"


def build_blind_vote_messages(
    agent: AgentPersona,
    enriched_topic: str,
    verdict_framing: str,
) -> list[BaseMessage]:
    """Build messages for the blind vote — no debate history visible."""
    options = [o.strip() for o in verdict_framing.split("/")]
    return [
        SystemMessage(content=agent.system_prompt),
        HumanMessage(content=(
            f"Topic: {enriched_topic}\n\n"
            f"Your verdict options are: {verdict_framing}\n"
            f"State your initial position.\n"
            f"Start with: VOTE: {options[0]} OR VOTE: {options[1]}\n"
            f"Then give one sentence of reasoning."
        )),
    ]


def build_deliberation_messages(
    agent: AgentPersona,
    enriched_topic: str,
    verdict_framing: str,
    current_vote: str,
    transcript: list[BaseMessage],
    summary: str,
) -> list[BaseMessage]:
    """Build messages for deliberation — summary + recent transcript visible."""
    options = [o.strip() for o in verdict_framing.split("/")]

    # Cap to the most recent 2 rounds (24 messages) to prevent OOM on small models.
    # Older context is covered by the running summary.
    recent = [m for m in transcript if not isinstance(m, SystemMessage)][-24:]

    context_parts = []
    if summary:
        context_parts.append(f"[Previous rounds summary]\n{summary}")
    if recent:
        context_parts.append("[Recent debate]\n" + "\n\n".join(
            f"{getattr(m, 'name', 'Unknown')}: {m.content}"
            for m in recent
        ))

    context = "\n\n".join(context_parts) if context_parts else "No arguments yet."

    # Anchor the topic in the system prompt so it is never crowded out by long transcripts
    system_content = (
        f"{agent.system_prompt}\n\n"
        f"DEBATE TOPIC (always keep this in mind):\n{enriched_topic}\n\n"
        f"Verdict options: {verdict_framing}\n"
        f"Every response you give must stay focused on this specific topic and question."
    )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=(
            f"Debate so far:\n{context}\n\n"
            f"Your current vote: {current_vote}\n\n"
            f"Respond now. Your response MUST start with exactly:\n"
            f"VOTE: {options[0]}  OR  VOTE: {options[1]}\n"
            f"Then give your argument. You may change your vote if genuinely persuaded, "
            f"but always stay on the original debate topic."
        )),
    ]


def blind_vote_node(state: DebateState, config: RunnableConfig) -> dict:
    """All 12 agents cast their initial blind vote. No peer arguments visible."""
    from langchain_ollama import ChatOllama
    from rich.console import Console
    from rich.rule import Rule

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature)
    console = Console()

    valid_options = [o.strip() for o in state["verdict_framing"].split("/")]
    votes = {}
    transcript = list(state["transcript"])

    console.print(Rule("[bold]BLIND VOTE[/bold]"))

    for agent in cfg.agents:
        messages = build_blind_vote_messages(
            agent=agent,
            enriched_topic=state["enriched_topic"],
            verdict_framing=state["verdict_framing"],
        )
        response = llm.invoke(messages)
        vote = extract_vote(response.content, valid_options)
        votes[agent.name] = vote
        transcript.append(AIMessage(content=response.content, name=agent.name))
        console.print(f"  {agent.name:<28} → {vote}")

    # Print tally
    tally = {opt: sum(1 for v in votes.values() if v == opt) for opt in valid_options}
    tally_str = ", ".join(f"{count} {opt}" for opt, count in tally.items())
    all_same = len(set(votes.values())) == 1
    verdict_label = "unanimous — no deliberation needed" if all_same else "deliberation begins"
    console.print(f"\n  Vote: {tally_str} — {verdict_label}\n")

    return {
        "votes": votes,
        "original_votes": dict(votes),
        "transcript": transcript,
        "status": "voting",
    }


def agent_speak_node(state: DebateState, config: RunnableConfig) -> dict:
    """Current agent speaks during deliberation. Streams output to console."""
    from langchain_ollama import ChatOllama
    from rich.console import Console
    from rich.rule import Rule

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature)
    console = Console()

    idx = state["current_speaker_idx"]
    agent_name = state["speaking_order"][idx]
    agent = next(a for a in cfg.agents if a.name == agent_name)
    current_vote = state["votes"].get(agent_name, "undecided")

    valid_options = [o.strip() for o in state["verdict_framing"].split("/")]

    console.print(Rule(f"[bold]{agent_name}[/bold]  [dim][{current_vote}][/dim]"))

    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic=state["enriched_topic"],
        verdict_framing=state["verdict_framing"],
        current_vote=current_vote,
        transcript=state["transcript"],
        summary=state["summary"],
    )

    full_response = ""
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
        full_response += chunk.content
    print("\n")

    new_vote = extract_vote(full_response, valid_options)
    votes = dict(state["votes"])
    votes[agent_name] = new_vote

    transcript = list(state["transcript"])
    transcript.append(AIMessage(content=full_response, name=agent_name))

    return {
        "votes": votes,
        "transcript": transcript,
        "current_speaker_idx": idx + 1,
    }


def vote_again_node(state: DebateState, config: RunnableConfig) -> dict:
    """Display re-vote table after a deliberation round. Votes already updated by agent_speak_node."""
    from rich.console import Console
    from rich.table import Table

    cfg: AppConfig = config["configurable"]["app_config"]
    console = Console()

    table = Table(title=f"Re-Vote (Round {state['round']})", show_header=True)
    table.add_column("Agent", style="bold")
    table.add_column("Vote")
    table.add_column("Change")

    for agent in cfg.agents:
        name = agent.name
        current = state["votes"].get(name, "undecided")
        original = state["original_votes"].get(name, "undecided")
        changed = " (changed)" if current != original else " (holds)"
        table.add_row(name, current, changed)

    console.print(table)

    return {
        "round": state["round"] + 1,
        "original_votes": dict(state["votes"]),  # reset baseline for next round's change detection
    }
