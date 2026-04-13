from typing import Literal, NotRequired, TypedDict

from langchain_core.messages import BaseMessage


class DebateState(TypedDict):
    topic: str                      # original user input
    enriched_topic: str             # topic + answers to clarifying questions
    verdict_framing: str            # e.g. "proceed / don't proceed"
    votes: dict[str, str]           # agent_name → current vote
    original_votes: dict[str, str]  # blind votes, used to track flips
    transcript: list[BaseMessage]   # full debate history (shared by all agents)
    summary: str                    # compressed summary of older rounds
    round: int                      # current deliberation round number
    speaking_order: list[str]       # randomized agent order for this round
    current_speaker_idx: int        # index into speaking_order
    verdict: NotRequired[str | None]  # set when consensus is reached
    moderator_question: NotRequired[str]  # per-round probe question from the Foreman
    status: Literal["gathering", "voting", "deliberating", "concluded"]
