import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from twelve_angry_agents.config import AgentPersona, AppConfig
from twelve_angry_agents.state import DebateState


def _strip_markdown(text: str) -> str:
    """Remove bold/italic markdown markers so **No** and *proceed* match cleanly."""
    return re.sub(r"\*+", "", text).strip()


def _match_option(raw: str, valid_options: list[str]) -> str | None:
    """Return the first matching option from raw text, longest-first to avoid substring collisions."""
    cleaned = _strip_markdown(raw).lower()
    for option in sorted(valid_options, key=len, reverse=True):
        if option.lower() in cleaned:
            return option
    return None


def _yes_no_to_option(raw: str, valid_options: list[str]) -> str | None:
    """Map bare 'yes'/'no' to the appropriate option based on which option contains a negation."""
    cleaned = _strip_markdown(raw).lower().strip(".,!? ")
    if cleaned not in ("yes", "no"):
        return None
    # The "negative" option is the one containing don't / not / un- / never
    neg_markers = ("don't", "dont", "not", "un", "never", "no ")
    negative = next(
        (o for o in valid_options if any(m in o.lower() for m in neg_markers)),
        valid_options[-1],  # fallback: treat last option as negative
    )
    positive = next(o for o in valid_options if o != negative)
    return negative if cleaned == "no" else positive


def extract_vote(response: str, valid_options: list[str]) -> str:
    """Parse the agent's vote from their response.

    Tries in order:
    1. Text after 'VOTE:' prefix (canonical format)
    2. First line of the response (agents often put the vote first without prefix)
    3. yes/no mapping based on which option contains a negation word
    Returns 'undecided' if nothing matches.
    """
    # 1. Canonical VOTE: prefix
    match = re.search(r"VOTE:\s*(.+)", response, re.IGNORECASE)
    if match:
        result = _match_option(match.group(1), valid_options)
        if result:
            return result
        # VOTE: was present but contained yes/no
        result = _yes_no_to_option(match.group(1), valid_options)
        if result:
            return result

    # 2. First non-empty line of response (agents often lead with their vote)
    first_line = next((l for l in response.splitlines() if l.strip()), "")
    if first_line:
        result = _match_option(first_line, valid_options)
        if result:
            return result
        result = _yes_no_to_option(first_line, valid_options)
        if result:
            return result

    return "undecided"


def build_clarify_vote_messages(
    prior_messages: list[BaseMessage],
    prior_response: str,
    verdict_framing: str,
) -> list[BaseMessage]:
    """Ask the agent to restate their vote in the correct format after a failed parse."""
    options = [o.strip() for o in verdict_framing.split("/")]
    return prior_messages + [
        AIMessage(content=prior_response),
        HumanMessage(content=(
            f"Your vote was not clear. Please restate it using exactly one of these:\n"
            f"VOTE: {options[0]}\n"
            f"VOTE: {options[1]}\n"
            f"Only output the VOTE line — nothing else."
        )),
    ]


def missing_change_explanation(response: str) -> bool:
    """Return True if the response changed a vote but omitted a reason."""
    return "i changed my vote" not in response.lower()


def build_clarify_change_messages(
    prior_messages: list[BaseMessage],
    prior_response: str,
    old_vote: str,
    new_vote: str,
) -> list[BaseMessage]:
    """Ask the agent to explain why they switched votes."""
    return prior_messages + [
        AIMessage(content=prior_response),
        HumanMessage(content=(
            f"You switched your vote from '{old_vote}' to '{new_vote}' without explaining why.\n"
            f"Please give ONE sentence explaining what specifically changed your mind.\n"
            f"Start with exactly: 'I changed my vote to {new_vote} because'"
        )),
    ]


def build_blind_vote_messages(
    agent: AgentPersona,
    enriched_topic: str,
    verdict_framing: str,
    all_agent_names: list[str] | None = None,
) -> list[BaseMessage]:
    """Build messages for the blind vote — no debate history visible."""
    options = [o.strip() for o in verdict_framing.split("/")]
    other_names = [n for n in (all_agent_names or []) if n != agent.name]
    jury_note = (
        f"\nYour name is {agent.name}. "
        f"The other jury members are: {', '.join(other_names)}."
        if other_names else f"\nYour name is {agent.name}."
    )
    return [
        SystemMessage(content=agent.system_prompt + jury_note),
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
    votes: dict[str, str] | None = None,
    all_agent_names: list[str] | None = None,
    moderator_question: str = "",
) -> list[BaseMessage]:
    """Build messages for deliberation — summary + recent transcript visible."""
    options = [o.strip() for o in verdict_framing.split("/")]

    # Extract this agent's own prior statements from the full transcript (not capped).
    # Shown separately so they are not crowded out by the shared debate context.
    own_history = [
        m.content for m in transcript
        if isinstance(m, AIMessage) and getattr(m, "name", "") == agent.name
    ]

    # Cap shared transcript to the most recent 2 rounds (24 messages) to prevent OOM.
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

    # Build your own history block so the agent can anchor to its own reasoning
    own_history_text = ""
    if own_history:
        entries = "\n\n".join(
            f"[Round {i + 1}] {content}" for i, content in enumerate(own_history)
        )
        own_history_text = f"Your previous arguments (for reference):\n{entries}\n\n"

    # Identify opponents — agents currently on the other side — and surface their
    # most recent argument so this agent can directly engage with them by name.
    opponents_text = ""
    if votes and current_vote and current_vote != "undecided":
        opposing_names = [
            name for name, vote in votes.items()
            if vote != current_vote and vote != "undecided" and name != agent.name
        ]
        if opposing_names:
            # For each opponent, find their most recent message in the transcript
            opponent_snippets = []
            for name in opposing_names:
                msgs = [
                    m.content for m in transcript
                    if isinstance(m, AIMessage) and getattr(m, "name", "") == name
                ]
                if msgs:
                    # Trim to 300 chars so we don't blow up the prompt
                    snippet = msgs[-1][:300].rstrip()
                    if len(msgs[-1]) > 300:
                        snippet += "…"
                    opponent_snippets.append(f"  {name}: {snippet}")
            if opponent_snippets:
                opponents_text = (
                    "Agents currently opposing your position (address them directly):\n"
                    + "\n".join(opponent_snippets)
                    + "\n\n"
                )

    # Build jury-awareness note so the agent knows their own name and recognises others
    other_names = [n for n in (all_agent_names or []) if n != agent.name]
    jury_note = (
        f"\n\nYour name is {agent.name}. "
        f"The other jury members are: {', '.join(other_names)}. "
        f"When another agent addresses you by name, acknowledge it directly in your response."
        if other_names else f"\n\nYour name is {agent.name}."
    )

    # Anchor the topic in the system prompt so it is never crowded out by long transcripts
    system_content = (
        f"{agent.system_prompt}{jury_note}\n\n"
        f"DEBATE TOPIC (always keep this in mind):\n{enriched_topic}\n\n"
        f"Verdict options: {verdict_framing}\n"
        f"Speak like a real person in a heated discussion — direct, conversational, no bullet points, "
        f"no formal preamble. Every response must stay focused on this specific topic."
    )

    engagement_instruction = (
        "Engage with the opposing agents listed above — call them out by name, push back on their "
        "specific claim, or concede a point if they've got it right."
        if opponents_text else
        "Make your point directly."
    )

    # Pull the prior round count so agents know how many times they've already spoken
    prior_rounds = len(own_history)
    if prior_rounds > 0:
        # Summarise their last argument in one line so they can reference it as a tl;dr
        last_arg_snippet = own_history[-1][:120].rstrip()
        if len(own_history[-1]) > 120:
            last_arg_snippet += "…"
        no_repeat_instruction = (
            f"You've already spoken {prior_rounds} time(s). "
            f"If you're still making the same point, open with: "
            f"'I've already said this — in short: [{last_arg_snippet}]' "
            f"then immediately move to something NEW: respond to what an opponent just said, "
            f"add a fresh angle, or concede a point. Do not re-argue what you've already stated."
        )
    else:
        no_repeat_instruction = ""

    foreman_section = (
        f"THE FOREMAN ASKS: {moderator_question}\n\n"
        if moderator_question else ""
    )

    return [
        SystemMessage(content=system_content),
        HumanMessage(content=(
            f"{own_history_text}"
            f"{opponents_text}"
            f"{foreman_section}"
            f"Debate so far:\n{context}\n\n"
            f"Your current vote: {current_vote}\n\n"
            f"Respond now. Start with:\n"
            f"VOTE: {options[0]}  OR  VOTE: {options[1]}\n"
            f"{engagement_instruction} "
            f"Hold your position unless a genuinely new argument persuades you.\n"
            f"— If you keep your vote ({current_vote}): do NOT use the phrase "
            f"'I changed my vote'. {no_repeat_instruction}\n"
            f"— If you change your vote: you MUST start with exactly: "
            f"'I changed my vote to [new option] because [specific reason].' "
            f"Do not open with anything else."
        )),
    ]


# 12 distinct Rich colors — one per jury seat (cycles if custom agents exceed 12)
_AGENT_COLORS = [
    "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
    "bright_red", "cyan", "green", "yellow", "magenta", "red",
    "bright_blue", "blue",
]


def _agent_color(cfg: "AppConfig", agent_name: str) -> str:
    names = [a.name for a in cfg.agents]
    try:
        return _AGENT_COLORS[names.index(agent_name) % len(_AGENT_COLORS)]
    except ValueError:
        return "white"


def blind_vote_node(state: DebateState, config: RunnableConfig) -> dict:
    """All 12 agents cast their initial blind vote. No peer arguments visible."""
    from langchain_ollama import ChatOllama
    from rich.console import Console
    from rich.rule import Rule

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature, num_ctx=cfg.model.context_window)
    console = config["configurable"].get("console") or Console()

    valid_options = [o.strip() for o in state["verdict_framing"].split("/")]
    votes = {}
    all_agent_names = [a.name for a in cfg.agents]

    console.print(Rule("[bold]BLIND VOTE[/bold]"))

    for agent in cfg.agents:
        messages = build_blind_vote_messages(
            agent=agent,
            enriched_topic=state["enriched_topic"],
            verdict_framing=state["verdict_framing"],
            all_agent_names=all_agent_names,
        )
        response = llm.invoke(messages)
        vote = extract_vote(response.content, valid_options)
        if vote == "undecided":
            clarify = build_clarify_vote_messages(messages, response.content, state["verdict_framing"])
            response = llm.invoke(clarify)
            vote = extract_vote(response.content, valid_options)
        votes[agent.name] = vote
        # Blind vote reasoning is intentionally NOT added to the shared transcript.
        # Agents must not be able to reference reasoning they never saw.
        color = _agent_color(cfg, agent.name)
        console.print(f"  [{color}]{agent.name:<28}[/{color}] → {vote}")

    # Print tally
    tally = {opt: sum(1 for v in votes.values() if v == opt) for opt in valid_options}
    tally_str = ", ".join(f"{count} {opt}" for opt, count in tally.items())
    all_same = len(set(votes.values())) == 1
    verdict_label = "unanimous — no deliberation needed" if all_same else "deliberation begins"
    console.print(f"\n  Vote: {tally_str} — {verdict_label}\n")

    return {
        "votes": votes,
        "original_votes": dict(votes),
        "status": "voting",
    }


def agent_speak_node(state: DebateState, config: RunnableConfig) -> dict:
    """Current agent speaks during deliberation. Streams output to console."""
    from langchain_ollama import ChatOllama
    from rich.console import Console

    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(
        model=cfg.model.name,
        temperature=cfg.model.temperature,
        num_ctx=cfg.model.context_window,
    )
    console = config["configurable"].get("console") or Console()

    idx = state["current_speaker_idx"]
    agent_name = state["speaking_order"][idx]
    agent = next(a for a in cfg.agents if a.name == agent_name)
    current_vote = state["votes"].get(agent_name, "undecided")
    all_agent_names = [a.name for a in cfg.agents]
    color = _agent_color(cfg, agent_name)

    valid_options = [o.strip() for o in state["verdict_framing"].split("/")]

    # Conversational header: colored name + current vote, no full-width rule
    console.print(f"\n[bold {color}]{agent_name}[/bold {color}] [dim][{current_vote}][/dim]")

    messages = build_deliberation_messages(
        agent=agent,
        enriched_topic=state["enriched_topic"],
        verdict_framing=state["verdict_framing"],
        current_vote=current_vote,
        transcript=state["transcript"],
        summary=state["summary"],
        votes=state["votes"],
        all_agent_names=all_agent_names,
        moderator_question=state.get("moderator_question", ""),
    )

    full_response = ""
    for chunk in llm.stream(messages):
        console.print(chunk.content, end="", highlight=False)
        full_response += chunk.content
    console.print()

    new_vote = extract_vote(full_response, valid_options)
    if new_vote == "undecided":
        clarify = build_clarify_vote_messages(messages, full_response, state["verdict_framing"])
        clarify_response = llm.invoke(clarify)
        new_vote = extract_vote(clarify_response.content, valid_options)
        if new_vote != "undecided":
            full_response = full_response + f"\n[Clarified] {clarify_response.content}"

    # If the agent changed their vote but gave no explanation, ask for one
    if new_vote != "undecided" and new_vote != current_vote and missing_change_explanation(full_response):
        clarify = build_clarify_change_messages(messages, full_response, current_vote, new_vote)
        clarify_response = llm.invoke(clarify)
        full_response = full_response + f"\n[Explanation] {clarify_response.content}"
        console.print(f"  [dim][explanation requested and received][/dim]")

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
    console = config["configurable"].get("console") or Console()

    table = Table(title=f"Re-Vote (Round {state['round'] + 1})", show_header=True)
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
