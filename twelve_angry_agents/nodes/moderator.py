# twelve_angry_agents/nodes/moderator.py
import random
import re
import sys

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


_FALLBACK_FRAMING = "proceed / don't proceed"
_INVALID_OPTION_WORDS = {"undecided", "unquantifiable", "unclear", "unknown", "tbd", "n/a"}


def _is_valid_framing(framing: str) -> bool:
    """Return False if the framing looks like nonsense (bad options, missing slash, etc.)."""
    if "/" not in framing:
        return False
    parts = [o.strip().lower() for o in framing.split("/")]
    if len(parts) != 2:
        return False
    a, b = parts
    if not a or not b:
        return False
    if a == b:
        return False
    if a in _INVALID_OPTION_WORDS or b in _INVALID_OPTION_WORDS:
        return False
    return True


def extract_verdict_framing(response: str) -> str:
    """Parse FRAMING: option1 / option2 from moderator response, with validation fallback."""
    match = re.search(r"FRAMING:\s*(.+)", response, re.IGNORECASE)
    if match:
        framing = match.group(1).strip()
        if _is_valid_framing(framing):
            return framing
    return _FALLBACK_FRAMING


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
            f"Topic: {topic}\n\n"
            "Is there enough here for 12 people to actually debate this? "
            "If yes: SUFFICIENT\n"
            "If not: ask 2-3 direct questions, one per line, numbered. "
            "Only ask what you genuinely need — don't pad it."
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
            "Pick a binary verdict framing. Use one of these unless there's a strong reason not to:\n"
            "  proceed / don't proceed  (decisions, actions)\n"
            "  viable / not viable      (plans, business ideas)\n"
            "  sound / unsound          (strategies, logic)\n"
            "  ethical / unethical      (moral questions)\n"
            "Both options must be clear action-or-judgment words — not 'undecided', "
            "'unquantifiable', or anything vague.\n\n"
            "Respond with: FRAMING: <option1> / <option2>\n"
            "Then one sentence framing what this debate is actually about. "
            "Not a reaction — nobody has said anything yet. Just set the stage."
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
    hung_note = "Hung jury — they didn't get there." if hung_jury else ""
    return [
        SystemMessage(content=moderator.system_prompt),
        HumanMessage(content=(
            f"Topic: {enriched_topic}\n"
            f"Verdict: {verdict_framing}\n"
            f"Final votes:\n{vote_summary}\n\n"
            f"Debate transcript:\n{transcript_text}\n\n"
            f"{hung_note}\n"
            "Deliver the verdict in plain prose — no headers, no numbered lists. "
            "Cover: what they decided, what actually moved the room, and the one thing "
            "the holdouts kept coming back to that the person asking this should take seriously. "
            "Be direct. Don't pad it."
        )),
    ]


def context_gather_node(state: DebateState, config: RunnableConfig) -> dict:
    """Foreman checks if topic has enough context; asks CLI questions if needed."""
    from langchain_ollama import ChatOllama
    from rich.console import Console

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature, num_ctx=cfg.model.context_window)
    console = config["configurable"].get("console") or Console()

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
        # Clear the raw input line so console.print captures it without duplication
        sys.stdout.write("\033[A\r\033[2K")
        sys.stdout.flush()
        console.print(f"  > {answer}")
        answers.append(f"{q} {answer}")
        console.print()

    enriched = state["topic"] + "\n\nAdditional context:\n" + "\n".join(answers)
    return {"enriched_topic": enriched, "status": "voting"}


def moderator_open_node(state: DebateState, config: RunnableConfig) -> dict:
    """Foreman sets verdict framing and announces the debate opening."""
    from langchain_ollama import ChatOllama
    from rich.console import Console
    from rich.rule import Rule

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature, num_ctx=cfg.model.context_window)
    console = config["configurable"].get("console") or Console()

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
    recent_arguments: str,
) -> list[BaseMessage]:
    """Build messages for the Foreman to generate a targeted probe question.

    The question is directed at the full jury — not individual agents by name —
    because the speaking order is random and named agents may not respond first.
    It is grounded in the actual arguments made so far, not just the topic.
    """
    vote_lines = "\n".join(f"- {name}: {vote}" for name, vote in votes.items())
    context_parts = []
    if summary:
        context_parts.append(f"[Summary of earlier rounds]\n{summary}")
    if recent_arguments:
        context_parts.append(f"[Recent arguments]\n{recent_arguments}")
    context = "\n\n".join(context_parts) if context_parts else "No arguments yet."
    return [
        SystemMessage(content=moderator.system_prompt),
        HumanMessage(content=(
            f"Topic: {enriched_topic}\n"
            f"Verdict options: {verdict_framing}\n"
            f"Votes:\n{vote_lines}\n\n"
            f"What's been argued:\n{context}\n\n"
            f"They're stuck. Read what's actually being said and find the real sticking point — "
            f"the specific claim or assumption where the two sides genuinely disagree. "
            f"Ask one sharp question that goes right at it. "
            f"Don't name anyone, don't restate the topic, don't add preamble. "
            f"Just the question. One sentence."
        )),
    ]


def moderator_deliberate_node(state: DebateState, config: RunnableConfig) -> dict:
    """Foreman randomizes speaking order and poses a probe question for this round."""
    from langchain_ollama import ChatOllama
    from rich.console import Console

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature, num_ctx=cfg.model.context_window)
    console = config["configurable"].get("console") or Console()

    agent_names = [a.name for a in cfg.agents]

    # Put minority-vote agents first so they speak early each round.
    # This ensures the agents with the most to defend open the round
    # rather than being crowded to the end after the majority has spoken.
    if state["votes"]:
        majority = max(set(state["votes"].values()), key=list(state["votes"].values()).count)
        minority = [n for n in agent_names if state["votes"].get(n) != majority]
        majority_names = [n for n in agent_names if state["votes"].get(n) == majority]
        speaking_order = (
            random.sample(minority, len(minority))
            + random.sample(majority_names, len(majority_names))
        )
    else:
        speaking_order = random.sample(agent_names, len(agent_names))

    options = extract_vote_options(state["verdict_framing"])
    tally = {opt: sum(1 for v in state["votes"].values() if v == opt) for opt in options}
    tally_str = "  ".join(f"{opt}: {count}" for opt, count in tally.items())

    console.print(f"\n[bold]━━━ ROUND {state['round'] + 1} ━━━[/bold]  {tally_str}")

    # Generate a probe question when the jury is split
    moderator_question = ""
    unique_votes = set(v for v in state["votes"].values() if v != "undecided")
    if len(unique_votes) > 1 and state["votes"]:
        # Summarise the most recent argument from each side (capped to avoid prompt bloat)
        recent_transcript = [
            m for m in state["transcript"]
            if isinstance(m, AIMessage)
        ][-12:]  # last 12 messages ≈ 1 full round
        recent_arguments = "\n\n".join(
            f"{getattr(m, 'name', 'Unknown')}: {m.content[:200].rstrip()}{'…' if len(m.content) > 200 else ''}"
            for m in recent_transcript
        )
        messages = build_foreman_probe_messages(
            moderator=cfg.moderator,
            enriched_topic=state["enriched_topic"],
            verdict_framing=state["verdict_framing"],
            votes=state["votes"],
            summary=state["summary"],
            recent_arguments=recent_arguments,
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

    llm = ChatOllama(model=cfg.model.name, temperature=0.3, num_ctx=cfg.model.context_window)
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
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature, num_ctx=cfg.model.context_window)
    console = config["configurable"].get("console") or Console()

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
    console.print(response.content)
    console.print()

    console.print(Rule())
    console.print(
        "[dim]The jury has spoken. But these twelve minds are not you — they don't carry "
        "your mortgage, your relationships, or your sense of what a life well-lived looks like. "
        "Take what's useful. Leave what isn't. The decision is yours.[/dim]"
    )
    console.print()

    return {
        "verdict": consensus_option,
        "status": "concluded",
    }
