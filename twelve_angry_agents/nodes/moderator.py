# twelve_angry_agents/nodes/moderator.py
import random
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from twelve_angry_agents.config import AgentPersona, AppConfig
from twelve_angry_agents.memory import (
    build_summarization_messages,
    format_transcript_for_summary,
    needs_summarization,
    transcript_token_count,
)
from twelve_angry_agents.nodes.consensus import is_hung_jury, majority_vote
from twelve_angry_agents.state import DebateState


def extract_verdict_framing(response: str) -> str:
    """Parse FRAMING: option1 / option2 from moderator response."""
    match = re.search(r"FRAMING:\s*(.+)", response, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "proceed / don't proceed"


def extract_vote_options(verdict_framing: str) -> list[str]:
    """Split 'option1 / option2' into ['option1', 'option2']."""
    return [o.strip() for o in verdict_framing.split("/")]


def build_context_check_messages(
    moderator: AgentPersona,
    topic: str,
) -> list[BaseMessage]:
    return [
        SystemMessage(content=moderator.system_prompt),
        HumanMessage(content=(
            f"Topic submitted by user: {topic}\n\n"
            "Evaluate if this topic has enough context for 12 agents to debate meaningfully.\n"
            "If context is sufficient, respond: SUFFICIENT\n"
            "If more context is needed, respond with 2-3 clarifying questions, one per line, "
            "starting each with a number and period (e.g. '1. What is...')\n"
            "Be direct. Only ask what is truly necessary."
        )),
    ]


def build_foreman_open_messages(
    moderator: AgentPersona,
    enriched_topic: str,
) -> list[BaseMessage]:
    return [
        SystemMessage(content=moderator.system_prompt),
        HumanMessage(content=(
            f"Topic: {enriched_topic}\n\n"
            "Set the verdict framing for this debate. Choose the most appropriate binary:\n"
            "- For decisions: 'proceed / don't proceed'\n"
            "- For evaluations: 'sound / unsound'\n"
            "- For ethical questions: 'ethical / unethical'\n"
            "- For viability: 'viable / not viable'\n\n"
            "Respond with: FRAMING: <option1> / <option2>\n"
            "Then write a one-sentence opening statement for the debate."
        )),
    ]


def build_foreman_close_messages(
    moderator: AgentPersona,
    enriched_topic: str,
    verdict_framing: str,
    votes: dict[str, str],
    transcript_text: str,
    hung_jury: bool,
) -> list[BaseMessage]:
    vote_summary = "\n".join(f"- {name}: {vote}" for name, vote in votes.items())
    hung_note = "NOTE: This is a hung jury — consensus was not reached." if hung_jury else ""
    return [
        SystemMessage(content=moderator.system_prompt),
        HumanMessage(content=(
            f"Topic: {enriched_topic}\n"
            f"Verdict framing: {verdict_framing}\n"
            f"Final votes:\n{vote_summary}\n\n"
            f"Debate transcript:\n{transcript_text}\n\n"
            f"{hung_note}\n"
            "Deliver the final verdict as a clear prose paragraph. Include:\n"
            "1. The unanimous decision (or majority if hung jury)\n"
            "2. The decisive argument(s) that shifted the room\n"
            "3. The key risk or concern raised by the last holdout(s) that users should address"
        )),
    ]


def context_gather_node(state: DebateState, config: RunnableConfig) -> dict:
    """Foreman checks if topic has enough context; asks CLI questions if needed."""
    from langchain_ollama import ChatOllama
    from rich.console import Console

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature)
    console = Console()

    messages = build_context_check_messages(
        moderator=cfg.moderator,
        topic=state["topic"],
    )
    response = llm.invoke(messages)
    content = response.content.strip()

    if content.upper().startswith("SUFFICIENT"):
        return {"enriched_topic": state["topic"], "status": "voting"}

    # Parse questions
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    questions = [l for l in lines if re.match(r"^\d+\.", l)]

    if not questions:
        return {"enriched_topic": state["topic"], "status": "voting"}

    console.print("\n[bold]THE FOREMAN[/bold]")
    console.print("Before we begin, I need a bit more context:\n")

    answers = []
    for q in questions:
        console.print(f"  {q}")
        answer = input("> ").strip()
        answers.append(f"{q} {answer}")
        print()

    enriched = state["topic"] + "\n\nAdditional context:\n" + "\n".join(answers)
    return {"enriched_topic": enriched, "status": "voting"}


def moderator_open_node(state: DebateState, config: RunnableConfig) -> dict:
    """Foreman sets verdict framing and announces the debate opening."""
    from langchain_ollama import ChatOllama
    from rich.console import Console
    from rich.rule import Rule

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature)
    console = Console()

    messages = build_foreman_open_messages(
        moderator=cfg.moderator,
        enriched_topic=state["enriched_topic"],
    )
    response = llm.invoke(messages)
    verdict_framing = extract_verdict_framing(response.content)

    console.print(Rule("[bold]THE FOREMAN[/bold]"))
    console.print(response.content)
    console.print()

    return {"verdict_framing": verdict_framing, "status": "voting"}


def build_foreman_probe_messages(
    moderator: AgentPersona,
    enriched_topic: str,
    verdict_framing: str,
    votes: dict[str, str],
    summary: str,
) -> list[BaseMessage]:
    """Build messages for the Foreman to generate a targeted probe question."""
    vote_lines = "\n".join(f"- {name}: {vote}" for name, vote in votes.items())
    options = extract_vote_options(verdict_framing)
    split_agents = [name for name, vote in votes.items() if vote != max(set(votes.values()), key=list(votes.values()).count)]
    targets = ", ".join(split_agents[:3]) if split_agents else "the jury"
    context_hint = f"\n[Summary of earlier rounds]\n{summary}" if summary else ""
    return [
        SystemMessage(content=moderator.system_prompt),
        HumanMessage(content=(
            f"Topic: {enriched_topic}\n"
            f"Verdict options: {verdict_framing}\n"
            f"Current votes:\n{vote_lines}\n"
            f"{context_hint}\n\n"
            f"The jury is split. As Foreman, pose ONE short, sharp follow-up question "
            f"to move the debate forward. Address it to {targets} by name if relevant. "
            f"Examples: 'The Skeptic, what specific evidence would change your mind?' "
            f"or 'Can the Optimist and the Pessimist address the financial risk directly?'\n"
            f"Output ONLY the question — one sentence, no preamble."
        )),
    ]


def moderator_deliberate_node(state: DebateState, config: RunnableConfig) -> dict:
    """Foreman randomizes speaking order and poses a probe question for this round."""
    from langchain_ollama import ChatOllama
    from rich.console import Console

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature)
    console = Console()

    agent_names = [a.name for a in cfg.agents]
    speaking_order = random.sample(agent_names, len(agent_names))

    options = extract_vote_options(state["verdict_framing"])
    tally = {opt: sum(1 for v in state["votes"].values() if v == opt) for opt in options}
    tally_str = "  ".join(f"{opt}: {count}" for opt, count in tally.items())

    console.print(f"\n[bold]━━━ ROUND {state['round'] + 1} ━━━[/bold]  {tally_str}")

    # Generate a probe question when votes are split
    moderator_question = ""
    all_same = len(set(state["votes"].values()) - {"undecided"}) <= 1
    if not all_same and state["votes"]:
        messages = build_foreman_probe_messages(
            moderator=cfg.moderator,
            enriched_topic=state["enriched_topic"],
            verdict_framing=state["verdict_framing"],
            votes=state["votes"],
            summary=state["summary"],
        )
        response = llm.invoke(messages)
        moderator_question = response.content.strip()
        console.print(f"\n[bold]THE FOREMAN:[/bold] {moderator_question}\n")

    return {
        "speaking_order": speaking_order,
        "current_speaker_idx": 0,
        "status": "deliberating",
        "moderator_question": moderator_question,
    }


def memory_check_node(state: DebateState, config: RunnableConfig) -> dict:
    """After a full round, check if transcript needs summarization."""
    from langchain_ollama import ChatOllama

    cfg: AppConfig = config["configurable"]["app_config"]

    current_tokens = transcript_token_count(state["transcript"])
    if not needs_summarization(
        current_tokens=current_tokens,
        context_window=cfg.model.context_window,
        threshold=cfg.debate.context_summary_threshold,
    ):
        return {}

    llm = ChatOllama(model=cfg.model.name, temperature=0.3)
    summary_messages = build_summarization_messages(state["transcript"])
    response = llm.invoke(summary_messages)

    new_summary = (
        state["summary"] + "\n\n" + response.content
        if state["summary"]
        else response.content
    )

    return {
        "summary": new_summary,
        "transcript": [],  # clear transcript — summary replaces it
    }


def moderator_close_node(state: DebateState, config: RunnableConfig) -> dict:
    """Foreman delivers the final verdict with reasoning."""
    from langchain_ollama import ChatOllama
    from rich.console import Console
    from rich.rule import Rule

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature)
    console = Console()

    hung = is_hung_jury(state["round"], cfg.debate.max_rounds)
    transcript_text = format_transcript_for_summary(state["transcript"])
    if state["summary"]:
        transcript_text = (
            f"[Summary of earlier rounds]\n{state['summary']}\n\n"
            f"[Recent]\n{transcript_text}"
        )

    messages = build_foreman_close_messages(
        moderator=cfg.moderator,
        enriched_topic=state["enriched_topic"],
        verdict_framing=state["verdict_framing"],
        votes=state["votes"],
        transcript_text=transcript_text,
        hung_jury=hung,
    )

    consensus_option = majority_vote(state["votes"])
    console.print(Rule("[bold]VERDICT[/bold]"))
    if hung:
        console.print(f"[yellow]HUNG JURY — Majority: {consensus_option}[/yellow]\n")
    else:
        console.print(f"[bold green]UNANIMOUS: {consensus_option}[/bold green]\n")

    console.print("[bold]THE FOREMAN[/bold]")
    response = llm.invoke(messages)
    print(response.content)
    print()

    return {
        "verdict": consensus_option,
        "status": "concluded",
    }
