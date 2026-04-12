from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def estimate_tokens(text: str) -> int:
    """Approximate token count using 4 chars per token heuristic."""
    return len(text) // 4


def transcript_token_count(transcript: list[BaseMessage]) -> int:
    """Estimate total tokens in the transcript."""
    return sum(estimate_tokens(msg.content) for msg in transcript)


def needs_summarization(
    current_tokens: int,
    context_window: int,
    threshold: float,
) -> bool:
    """Return True if current token usage exceeds the summarization threshold."""
    return current_tokens > int(context_window * threshold)


def format_transcript_for_summary(transcript: list[BaseMessage]) -> str:
    """Format transcript into readable text for LLM summarization, skipping system messages."""
    lines = []
    for msg in transcript:
        if isinstance(msg, SystemMessage):
            continue
        speaker = getattr(msg, "name", None) or type(msg).__name__
        lines.append(f"[{speaker}]: {msg.content}")
    return "\n\n".join(lines)


def build_summarization_messages(transcript: list[BaseMessage]) -> list[BaseMessage]:
    """Build the prompt for summarizing the transcript."""
    formatted = format_transcript_for_summary(transcript)
    return [
        SystemMessage(content=(
            "You are summarizing a debate transcript. "
            "For each agent, capture: their position, their key argument, "
            "and whether they changed their vote and why. Be concise."
        )),
        HumanMessage(content=f"Summarize this debate:\n\n{formatted}"),
    ]
