# Twelve Angry Agents — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CLI tool where 12 AI agents with distinct personalities debate any user topic sequentially until unanimous verdict, powered by Gemma 4 via Ollama.

**Architecture:** LangGraph state machine with a neutral Foreman orchestrating 12 debating agents. All agents share the same Ollama model instance differentiated by system prompt. Debate flows: context gathering → blind vote → deliberation rounds (sequential, streamed) → unanimous verdict.

**Tech Stack:** Python 3.11+, LangGraph, langchain-ollama, langchain-core, Click, PyYAML, Rich (terminal UI), pytest

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, `taa` entry point |
| `config/config.yaml` | Model name, debate limits, thresholds |
| `config/agents.yaml` | 12 agent personas + Foreman with full system prompts |
| `twelve_angry_agents/state.py` | `DebateState` TypedDict — single source of truth for all graph state |
| `twelve_angry_agents/config.py` | Load and validate `config.yaml` + `agents.yaml` into dataclasses |
| `twelve_angry_agents/nodes/consensus.py` | Deterministic consensus check — no LLM, pure Python |
| `twelve_angry_agents/nodes/agent.py` | `blind_vote_node`, `agent_speak_node`, `vote_again_node` |
| `twelve_angry_agents/nodes/moderator.py` | `context_gather_node`, `moderator_open_node`, `moderator_deliberate_node`, `moderator_close_node` |
| `twelve_angry_agents/memory.py` | Token counting, transcript trimming, summarization |
| `twelve_angry_agents/graph.py` | Assemble LangGraph graph from nodes + routing functions |
| `twelve_angry_agents/cli.py` | Click CLI, streaming output to terminal with Rich |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `twelve_angry_agents/__init__.py`
- Create: `twelve_angry_agents/nodes/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/nodes/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "twelve-angry-agents"
version = "0.1.0"
description = "Run your decisions through a jury of 12 AI minds before you commit."
license = {text = "CC-BY-NC-4.0"}
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2",
    "langchain-core>=0.3",
    "langchain-ollama>=0.2",
    "click>=8.0",
    "pyyaml>=6.0",
    "rich>=13.0",
]

[project.scripts]
taa = "twelve_angry_agents.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/
.pytest_cache/
*.db
```

- [ ] **Step 3: Create empty init files**

```bash
mkdir -p twelve_angry_agents/nodes tests/nodes
touch twelve_angry_agents/__init__.py
touch twelve_angry_agents/nodes/__init__.py
touch tests/__init__.py
touch tests/nodes/__init__.py
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -e ".[dev]"
```

Expected: No errors. `taa --help` will fail (no cli.py yet) — that's fine.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore twelve_angry_agents/ tests/
git commit -m "chore: project scaffold with dependencies"
```

---

## Task 2: DebateState

**Files:**
- Create: `twelve_angry_agents/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_state.py
from typing import get_type_hints
from twelve_angry_agents.state import DebateState


def test_debate_state_has_required_fields():
    hints = get_type_hints(DebateState)
    required = [
        "topic", "enriched_topic", "verdict_framing",
        "votes", "original_votes", "transcript",
        "summary", "round", "speaking_order",
        "current_speaker_idx", "verdict", "status",
    ]
    for field in required:
        assert field in hints, f"Missing field: {field}"


def test_debate_state_can_be_constructed():
    state: DebateState = {
        "topic": "Should I quit my job?",
        "enriched_topic": "Should I quit my job? I have 6 months runway.",
        "verdict_framing": "proceed / don't proceed",
        "votes": {"The Skeptic": "don't proceed"},
        "original_votes": {"The Skeptic": "don't proceed"},
        "transcript": [],
        "summary": "",
        "round": 0,
        "speaking_order": ["The Skeptic", "The Optimist"],
        "current_speaker_idx": 0,
        "verdict": None,
        "status": "gathering",
    }
    assert state["topic"] == "Should I quit my job?"
    assert state["status"] == "gathering"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_state.py -v
```

Expected: `ModuleNotFoundError: No module named 'twelve_angry_agents.state'`

- [ ] **Step 3: Implement `state.py`**

```python
# twelve_angry_agents/state.py
from typing import Literal, TypedDict

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
    verdict: str | None             # set when consensus is reached
    status: Literal["gathering", "voting", "deliberating", "concluded"]
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_state.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add twelve_angry_agents/state.py tests/test_state.py
git commit -m "feat: add DebateState TypedDict"
```

---

## Task 3: Config Layer

**Files:**
- Create: `config/config.yaml`
- Create: `config/agents.yaml`
- Create: `twelve_angry_agents/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import pytest
from pathlib import Path
from twelve_angry_agents.config import load_config, AppConfig, AgentPersona


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_load_config_returns_app_config(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    agents_yaml = tmp_path / "agents.yaml"

    config_yaml.write_text("""
model:
  name: gemma4:e4b
  temperature: 0.7
  context_window: 128000
debate:
  max_rounds: 50
  max_tokens_per_response: 250
  context_summary_threshold: 0.75
""")

    agents_yaml.write_text("""
moderator:
  name: The Foreman
  system_prompt: "You are a neutral moderator."
agents:
  - name: The Skeptic
    system_prompt: "You question everything."
  - name: The Analyst
    system_prompt: "You reason from data."
""")

    config = load_config(config_path=config_yaml, agents_path=agents_yaml)

    assert isinstance(config, AppConfig)
    assert config.model.name == "gemma4:e4b"
    assert config.model.temperature == 0.7
    assert config.debate.max_rounds == 50
    assert config.moderator.name == "The Foreman"
    assert len(config.agents) == 2
    assert config.agents[0].name == "The Skeptic"


def test_load_config_validates_agent_count(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    agents_yaml = tmp_path / "agents.yaml"

    config_yaml.write_text("""
model:
  name: gemma4:e4b
  temperature: 0.7
  context_window: 128000
debate:
  max_rounds: 50
  max_tokens_per_response: 250
  context_summary_threshold: 0.75
""")

    # Only 1 agent — should raise
    agents_yaml.write_text("""
moderator:
  name: The Foreman
  system_prompt: "You are a neutral moderator."
agents:
  - name: The Skeptic
    system_prompt: "You question everything."
""")

    with pytest.raises(ValueError, match="exactly 12"):
        load_config(config_path=config_yaml, agents_path=agents_yaml)


def test_agent_persona_has_name_and_prompt(tmp_path):
    config_yaml = tmp_path / "config.yaml"
    agents_yaml = tmp_path / "agents.yaml"

    config_yaml.write_text("""
model:
  name: gemma4:e4b
  temperature: 0.7
  context_window: 128000
debate:
  max_rounds: 50
  max_tokens_per_response: 250
  context_summary_threshold: 0.75
""")

    agents_yaml.write_text("""
moderator:
  name: The Foreman
  system_prompt: "Moderate."
agents:
""" + "\n".join(
        f"  - name: Agent{i}\n    system_prompt: \"Prompt {i}.\""
        for i in range(12)
    ))

    config = load_config(config_path=config_yaml, agents_path=agents_yaml)
    assert len(config.agents) == 12
    for agent in config.agents:
        assert isinstance(agent, AgentPersona)
        assert agent.name
        assert agent.system_prompt
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named 'twelve_angry_agents.config'`

- [ ] **Step 3: Implement `config.py`**

```python
# twelve_angry_agents/config.py
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    name: str = "gemma4:e4b"
    temperature: float = 0.7
    context_window: int = 128000


@dataclass
class DebateConfig:
    max_rounds: int = 50
    max_tokens_per_response: int = 250
    context_summary_threshold: float = 0.75


@dataclass
class AgentPersona:
    name: str
    system_prompt: str


@dataclass
class AppConfig:
    model: ModelConfig
    debate: DebateConfig
    moderator: AgentPersona
    agents: list[AgentPersona] = field(default_factory=list)


def load_config(
    config_path: Path | None = None,
    agents_path: Path | None = None,
) -> AppConfig:
    base = Path(__file__).parent.parent / "config"
    config_path = config_path or base / "config.yaml"
    agents_path = agents_path or base / "agents.yaml"

    with open(config_path) as f:
        raw_config = yaml.safe_load(f)

    with open(agents_path) as f:
        raw_agents = yaml.safe_load(f)

    model = ModelConfig(**raw_config["model"])
    debate = DebateConfig(**raw_config["debate"])

    moderator_data = raw_agents["moderator"]
    moderator = AgentPersona(
        name=moderator_data["name"],
        system_prompt=moderator_data["system_prompt"],
    )

    agents_data = raw_agents.get("agents", [])
    agents = [
        AgentPersona(name=a["name"], system_prompt=a["system_prompt"])
        for a in agents_data
    ]

    if len(agents) != 12:
        raise ValueError(
            f"agents.yaml must define exactly 12 agents, got {len(agents)}"
        )

    return AppConfig(model=model, debate=debate, moderator=moderator, agents=agents)
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_config.py -v
```

Expected: 3 passed

- [ ] **Step 5: Create `config/config.yaml`**

```bash
mkdir -p config
```

```yaml
# config/config.yaml
model:
  name: gemma4:e4b          # gemma4:e2b for lower VRAM, any ollama model works
  temperature: 0.7
  context_window: 128000

debate:
  max_rounds: 50            # safety valve — debate runs as long as needed
  max_tokens_per_response: 250
  context_summary_threshold: 0.75
```

- [ ] **Step 6: Create `config/agents.yaml`**

```yaml
# config/agents.yaml
# Edit system_prompt for any agent to customize their personality.
# Do not change the number of agents (must be exactly 12).

moderator:
  name: The Foreman
  system_prompt: |
    You are The Foreman — a neutral debate moderator. You have NO opinion on the topic.
    Your responsibilities:
    - Analyze topics to identify context gaps and ask targeted clarifying questions.
    - Determine appropriate verdict framing (e.g. "proceed / don't proceed").
    - Call and tally votes. Announce the vote distribution clearly.
    - Detect when consensus is reached or when agents are caving without argument.
    - Deliver the final verdict as a clear prose summary with decisive arguments and key risks.
    Never argue for a position. Be procedural, efficient, and fair.

agents:
  - name: The Analyst
    system_prompt: |
      You are The Analyst. You reason from data, evidence, and logic only.
      You dismiss anecdote and emotional appeals without evidence.
      You demand specifics: numbers, precedents, verifiable claims.
      When you lack data, you say so rather than guess.
      Your arguments are precise and evidence-based.
      You change your position only when presented with stronger evidence or a logical flaw.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Skeptic
    system_prompt: |
      You are The Skeptic. You question every premise and assumption.
      You never accept arguments at face value — you expose hidden assumptions and weak logic.
      You are hard to convince but not impossible.
      You change your mind only when the assumption you questioned has been genuinely addressed.
      You don't argue for the opposite — you refuse to accept the current position without justification.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Optimist
    system_prompt: |
      You are The Optimist. You focus on opportunity, potential, and best-case outcomes.
      You see what could go right and what strengths are being overlooked.
      You acknowledge risks but believe they can be managed.
      You are persuaded by strong risk arguments but return to: is the upside worth it?
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Pessimist
    system_prompt: |
      You are The Pessimist. You focus on what can go wrong.
      You assess probability of failure and identify overlooked risks.
      You believe risks are systematically underweighted by most people.
      You change your position when shown risks are genuinely manageable, not just dismissed.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Empath
    system_prompt: |
      You are The Empath. You prioritize human impact above all else.
      Who gets hurt? Who benefits? What are the emotional and social consequences?
      You believe decisions that look good on paper fail when they ignore how people actually feel.
      You are moved by ethical arguments and human stories more than data alone.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Ethicist
    system_prompt: |
      You are The Ethicist. You evaluate everything through a moral lens.
      Is this right or wrong — not just effective or profitable?
      You apply ethical frameworks (consequences, duties, fairness) and flag moral issues
      even when something is technically legal or effective.
      You change your position when a stronger moral argument is presented.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Pragmatist
    system_prompt: |
      You are The Pragmatist. You care about what actually works in the real world.
      Theory and ideals mean little if something can't be executed today.
      You focus on feasibility, resources, timelines, and practical constraints.
      You are persuaded by implementation plans and track records, not visions.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Devil's Advocate
    system_prompt: |
      You are The Devil's Advocate. When consensus forms, you argue against it —
      even if you personally might agree — to force the majority to justify themselves.
      You ask: what is being assumed? What could go wrong? Is this agreement too easy?
      You concede when the majority has genuinely earned their position through rigorous defense.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Historian
    system_prompt: |
      You are The Historian. You look for precedents, patterns, and analogies.
      "This has happened before" is your lens. You ground debate in what history shows.
      You distrust novelty claims — most situations have parallels with lessons.
      You change your position when the current situation has genuinely novel features.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Visionary
    system_prompt: |
      You are The Visionary. You think in decades, not days.
      You evaluate everything by where it leads in 10 years and what second-order effects it creates.
      You ignore short-term noise. You are persuaded by long-term structural arguments.
      You change your position when a short-term constraint has permanent long-term consequences.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Realist
    system_prompt: |
      You are The Realist. You cut through spin on both sides.
      You state what is actually true right now, without wishful thinking or catastrophizing.
      You are neither for nor against — you are accurate.
      You persuade by grounding debate in current reality when others drift into hypotheticals.
      You change your position when the facts you're working from are shown to be incomplete.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.

  - name: The Contrarian
    system_prompt: |
      You are The Contrarian. When consensus forms too quickly, you resist it.
      You are the last holdout — not because you're stubborn, but because easy agreement is lazy agreement.
      You dig in and require the most rigorous justification of anyone in the room.
      You eventually concede, but only when genuinely persuaded, not worn down.
      State your VOTE clearly at the start: VOTE: [option]. Keep your response under 200 words.
```

- [ ] **Step 7: Commit**

```bash
git add twelve_angry_agents/config.py config/ tests/test_config.py
git commit -m "feat: config layer with dataclasses and 12 agent personas"
```

---

## Task 4: Consensus Checker

**Files:**
- Create: `twelve_angry_agents/nodes/consensus.py`
- Create: `tests/nodes/test_consensus.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/nodes/test_consensus.py
from twelve_angry_agents.nodes.consensus import check_consensus, is_hung_jury


def test_unanimous_votes_return_true():
    votes = {
        "The Analyst": "proceed",
        "The Skeptic": "proceed",
        "The Optimist": "proceed",
    }
    assert check_consensus(votes) is True


def test_split_votes_return_false():
    votes = {
        "The Analyst": "proceed",
        "The Skeptic": "don't proceed",
        "The Optimist": "proceed",
    }
    assert check_consensus(votes) is False


def test_single_dissenter_not_unanimous():
    votes = {f"Agent{i}": "proceed" for i in range(11)}
    votes["The Contrarian"] = "don't proceed"
    assert check_consensus(votes) is False


def test_all_same_option_is_unanimous():
    votes = {f"Agent{i}": "don't proceed" for i in range(12)}
    assert check_consensus(votes) is True


def test_is_hung_jury_when_max_rounds_reached():
    assert is_hung_jury(current_round=50, max_rounds=50) is True


def test_is_not_hung_jury_before_max_rounds():
    assert is_hung_jury(current_round=3, max_rounds=50) is False


def test_majority_vote_calculation():
    from twelve_angry_agents.nodes.consensus import majority_vote
    votes = {
        "Agent1": "proceed",
        "Agent2": "proceed",
        "Agent3": "proceed",
        "Agent4": "don't proceed",
        "Agent5": "don't proceed",
    }
    result = majority_vote(votes)
    assert result == "proceed"


def test_majority_vote_returns_tied_option():
    from twelve_angry_agents.nodes.consensus import majority_vote
    votes = {
        "Agent1": "proceed",
        "Agent2": "don't proceed",
    }
    # Tie — either option acceptable, just must be one of the two
    result = majority_vote(votes)
    assert result in ("proceed", "don't proceed")
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/nodes/test_consensus.py -v
```

Expected: `ModuleNotFoundError: No module named 'twelve_angry_agents.nodes.consensus'`

- [ ] **Step 3: Implement `consensus.py`**

```python
# twelve_angry_agents/nodes/consensus.py
from collections import Counter


def check_consensus(votes: dict[str, str]) -> bool:
    """Return True if all agents agree on the same vote option."""
    if not votes:
        return False
    return len(set(votes.values())) == 1


def is_hung_jury(current_round: int, max_rounds: int) -> bool:
    """Return True if max deliberation rounds have been reached."""
    return current_round >= max_rounds


def majority_vote(votes: dict[str, str]) -> str:
    """Return the most common vote option. Breaks ties by picking the first."""
    counter = Counter(votes.values())
    return counter.most_common(1)[0][0]
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/nodes/test_consensus.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add twelve_angry_agents/nodes/consensus.py tests/nodes/test_consensus.py
git commit -m "feat: deterministic consensus checker"
```

---

## Task 5: Memory Management

**Files:**
- Create: `twelve_angry_agents/memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_memory.py
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from twelve_angry_agents.memory import (
    estimate_tokens,
    needs_summarization,
    format_transcript_for_summary,
)


def test_estimate_tokens_approximates_length():
    text = "a" * 400  # 400 chars ≈ 100 tokens at 4 chars/token
    assert estimate_tokens(text) == 100


def test_estimate_tokens_empty_string():
    assert estimate_tokens("") == 0


def test_needs_summarization_when_over_threshold():
    # 128000 context * 0.75 threshold = 96000 tokens
    assert needs_summarization(
        current_tokens=97000,
        context_window=128000,
        threshold=0.75,
    ) is True


def test_needs_summarization_when_under_threshold():
    assert needs_summarization(
        current_tokens=50000,
        context_window=128000,
        threshold=0.75,
    ) is False


def test_needs_summarization_exactly_at_threshold():
    assert needs_summarization(
        current_tokens=96000,
        context_window=128000,
        threshold=0.75,
    ) is False  # at threshold, not over


def test_format_transcript_for_summary():
    transcript = [
        HumanMessage(content="The topic is: Should I quit?"),
        AIMessage(content="VOTE: proceed\nThe opportunity is clear.", name="The Optimist"),
        AIMessage(content="VOTE: don't proceed\nThe risk is high.", name="The Skeptic"),
    ]
    result = format_transcript_for_summary(transcript)
    assert "The Optimist" in result
    assert "The Skeptic" in result
    assert "proceed" in result


def test_format_transcript_skips_system_messages():
    transcript = [
        SystemMessage(content="You are a moderator."),
        AIMessage(content="VOTE: proceed", name="The Analyst"),
    ]
    result = format_transcript_for_summary(transcript)
    assert "You are a moderator" not in result
    assert "The Analyst" in result
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_memory.py -v
```

Expected: `ModuleNotFoundError: No module named 'twelve_angry_agents.memory'`

- [ ] **Step 3: Implement `memory.py`**

```python
# twelve_angry_agents/memory.py
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
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_memory.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add twelve_angry_agents/memory.py tests/test_memory.py
git commit -m "feat: memory management utilities (token counting, summarization helpers)"
```

---

## Task 6: Agent Nodes

**Files:**
- Create: `twelve_angry_agents/nodes/agent.py`
- Create: `tests/nodes/test_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/nodes/test_agent.py
import pytest
from langchain_core.language_models.fake import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from twelve_angry_agents.config import AgentPersona, AppConfig, DebateConfig, ModelConfig
from twelve_angry_agents.nodes.agent import (
    extract_vote,
    build_blind_vote_messages,
    build_deliberation_messages,
)
from twelve_angry_agents.state import DebateState


def make_config(responses: list[str]) -> tuple[AppConfig, FakeListChatModel]:
    agents = [
        AgentPersona(name=f"Agent{i}", system_prompt=f"You are Agent{i}.")
        for i in range(12)
    ]
    moderator = AgentPersona(name="The Foreman", system_prompt="You moderate.")
    config = AppConfig(
        model=ModelConfig(),
        debate=DebateConfig(),
        moderator=moderator,
        agents=agents,
    )
    llm = FakeListChatModel(responses=responses)
    return config, llm


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


def test_build_deliberation_messages_includes_transcript():
    from langchain_core.messages import AIMessage
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
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/nodes/test_agent.py -v
```

Expected: `ModuleNotFoundError: No module named 'twelve_angry_agents.nodes.agent'`

- [ ] **Step 3: Implement `agent.py`**

```python
# twelve_angry_agents/nodes/agent.py
import random
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from twelve_angry_agents.config import AgentPersona, AppConfig
from twelve_angry_agents.state import DebateState


def extract_vote(response: str, valid_options: list[str]) -> str:
    """Parse VOTE: <option> from agent response. Returns 'undecided' if not found."""
    match = re.search(r"VOTE:\s*(.+)", response, re.IGNORECASE)
    if not match:
        return "undecided"
    raw = match.group(1).strip().lower()
    for option in valid_options:
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
    """Build messages for deliberation — full transcript (or summary + recent) visible."""
    options = [o.strip() for o in verdict_framing.split("/")]
    context_parts = []
    if summary:
        context_parts.append(f"[Previous rounds summary]\n{summary}")
    if transcript:
        context_parts.append("[Recent debate]\n" + "\n\n".join(
            f"{getattr(m, 'name', 'Unknown')}: {m.content}"
            for m in transcript
            if not isinstance(m, SystemMessage)
        ))

    context = "\n\n".join(context_parts) if context_parts else "No arguments yet."

    return [
        SystemMessage(content=agent.system_prompt),
        HumanMessage(content=(
            f"Topic: {enriched_topic}\n\n"
            f"Your verdict options: {verdict_framing}\n"
            f"Your current vote: {current_vote}\n\n"
            f"Debate so far:\n{context}\n\n"
            f"Respond now. Start with VOTE: {options[0]} OR VOTE: {options[1]}, "
            f"then give your argument. You may change your vote if persuaded."
        )),
    ]


def blind_vote_node(state: DebateState, config: dict) -> dict:
    """All 12 agents cast their initial blind vote. No peer arguments visible."""
    from langchain_ollama import ChatOllama
    cfg: AppConfig = config["configurable"]["app_config"]
    llm = ChatOllama(model=cfg.model.name, temperature=cfg.model.temperature)

    valid_options = [o.strip() for o in state["verdict_framing"].split("/")]
    votes = {}
    transcript = list(state["transcript"])

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

    return {
        "votes": votes,
        "original_votes": dict(votes),
        "transcript": transcript,
        "status": "voting",
    }


def agent_speak_node(state: DebateState, config: dict) -> dict:
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


def vote_again_node(state: DebateState, config: dict) -> dict:
    """All agents re-vote after a deliberation round. Uses votes already in state."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    cfg: AppConfig = config["configurable"]["app_config"]

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

    # Votes are already updated by agent_speak_node during deliberation
    return {"round": state["round"] + 1}
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/nodes/test_agent.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add twelve_angry_agents/nodes/agent.py tests/nodes/test_agent.py
git commit -m "feat: agent nodes (blind_vote, agent_speak, vote_again)"
```

---

## Task 7: Moderator Nodes

**Files:**
- Create: `twelve_angry_agents/nodes/moderator.py`
- Create: `tests/nodes/test_moderator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/nodes/test_moderator.py
import pytest
from langchain_core.language_models.fake import FakeListChatModel
from langchain_core.messages import AIMessage

from twelve_angry_agents.config import AgentPersona, AppConfig, DebateConfig, ModelConfig
from twelve_angry_agents.nodes.moderator import (
    extract_verdict_framing,
    extract_vote_options,
    build_foreman_open_messages,
    build_foreman_close_messages,
    build_context_check_messages,
)


def make_moderator_config() -> AppConfig:
    agents = [
        AgentPersona(name=f"Agent{i}", system_prompt=f"Prompt {i}.")
        for i in range(12)
    ]
    return AppConfig(
        model=ModelConfig(),
        debate=DebateConfig(),
        moderator=AgentPersona(name="The Foreman", system_prompt="You moderate."),
        agents=agents,
    )


def test_extract_verdict_framing_from_response():
    response = "FRAMING: proceed / don't proceed\nThis is a decision topic."
    assert extract_verdict_framing(response) == "proceed / don't proceed"


def test_extract_verdict_framing_returns_default_on_failure():
    response = "This is a decision topic."
    result = extract_verdict_framing(response)
    assert "/" in result  # must be a valid framing with two options


def test_extract_vote_options_splits_correctly():
    options = extract_vote_options("proceed / don't proceed")
    assert options == ["proceed", "don't proceed"]


def test_extract_vote_options_strips_whitespace():
    options = extract_vote_options("  sound  /  unsound  ")
    assert options == ["sound", "unsound"]


def test_build_foreman_open_messages_contains_topic():
    cfg = make_moderator_config()
    messages = build_foreman_open_messages(
        moderator=cfg.moderator,
        enriched_topic="Should I quit my job?",
    )
    combined = " ".join(m.content for m in messages)
    assert "quit my job" in combined


def test_build_foreman_close_messages_contains_votes():
    cfg = make_moderator_config()
    votes = {"Agent0": "proceed", "Agent1": "proceed"}
    transcript_text = "Agent0 argued X. Agent1 agreed."
    messages = build_foreman_close_messages(
        moderator=cfg.moderator,
        enriched_topic="Topic.",
        verdict_framing="proceed / don't proceed",
        votes=votes,
        transcript_text=transcript_text,
        is_hung_jury=False,
    )
    combined = " ".join(m.content for m in messages)
    assert "proceed" in combined


def test_build_context_check_messages_contains_topic():
    cfg = make_moderator_config()
    messages = build_context_check_messages(
        moderator=cfg.moderator,
        topic="Should I quit?",
    )
    combined = " ".join(m.content for m in messages)
    assert "quit" in combined
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/nodes/test_moderator.py -v
```

Expected: `ModuleNotFoundError: No module named 'twelve_angry_agents.nodes.moderator'`

- [ ] **Step 3: Implement `moderator.py`**

```python
# twelve_angry_agents/nodes/moderator.py
import random
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

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
    is_hung_jury: bool,
) -> list[BaseMessage]:
    vote_summary = "\n".join(f"- {name}: {vote}" for name, vote in votes.items())
    hung_note = "NOTE: This is a hung jury — consensus was not reached." if is_hung_jury else ""
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


def context_gather_node(state: DebateState, config: dict) -> dict:
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


def moderator_open_node(state: DebateState, config: dict) -> dict:
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


def moderator_deliberate_node(state: DebateState, config: dict) -> dict:
    """Foreman randomizes speaking order for this deliberation round."""
    from rich.console import Console

    cfg: AppConfig = config["configurable"]["app_config"]
    console = Console()

    agent_names = [a.name for a in cfg.agents]
    speaking_order = random.sample(agent_names, len(agent_names))

    options = extract_vote_options(state["verdict_framing"])
    tally = {opt: sum(1 for v in state["votes"].values() if v == opt) for opt in options}
    tally_str = "  ".join(f"{opt}: {count}" for opt, count in tally.items())

    console.print(f"\n[bold]━━━ ROUND {state['round'] + 1} ━━━[/bold]  {tally_str}\n")

    return {
        "speaking_order": speaking_order,
        "current_speaker_idx": 0,
        "status": "deliberating",
    }


def memory_check_node(state: DebateState, config: dict) -> dict:
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

    new_summary = state["summary"] + "\n\n" + response.content if state["summary"] else response.content

    return {
        "summary": new_summary,
        "transcript": [],  # clear transcript — summary replaces it
    }


def moderator_close_node(state: DebateState, config: dict) -> dict:
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
        transcript_text = f"[Summary of earlier rounds]\n{state['summary']}\n\n[Recent]\n{transcript_text}"

    messages = build_foreman_close_messages(
        moderator=cfg.moderator,
        enriched_topic=state["enriched_topic"],
        verdict_framing=state["verdict_framing"],
        votes=state["votes"],
        transcript_text=transcript_text,
        is_hung_jury=hung,
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
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/nodes/test_moderator.py -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add twelve_angry_agents/nodes/moderator.py tests/nodes/test_moderator.py
git commit -m "feat: moderator nodes (context_gather, open, deliberate, memory_check, close)"
```

---

## Task 8: LangGraph Graph Assembly

**Files:**
- Create: `twelve_angry_agents/graph.py`
- Create: `tests/test_graph.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_graph.py
from unittest.mock import patch, MagicMock
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
        current_speaker_idx=1,  # just spoke, 1 remains
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
    # Verify the graph has nodes compiled
    assert hasattr(graph, "invoke")
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_graph.py -v
```

Expected: `ModuleNotFoundError: No module named 'twelve_angry_agents.graph'`

- [ ] **Step 3: Implement `graph.py`**

```python
# twelve_angry_agents/graph.py
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
    """Route to close if unanimous or hung, else start deliberation."""
    if state["status"] == "concluded":
        return "close"
    return "deliberate"


def route_after_agent_speak(state: DebateState) -> str:
    """Loop back to agent_speak while agents remain in speaking order."""
    if state["current_speaker_idx"] < len(state["speaking_order"]):
        return "next_agent"
    return "memory_check"


def consensus_check_node(state: DebateState, config: dict) -> dict:
    """Check votes for consensus or hung jury and update status accordingly."""
    from twelve_angry_agents.config import AppConfig

    cfg: AppConfig = config["configurable"]["app_config"]

    if check_consensus(state["votes"]):
        return {"status": "concluded"}

    if is_hung_jury(state["round"], cfg.debate.max_rounds):
        return {"status": "concluded"}

    return {"status": state["status"]}


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
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_graph.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add twelve_angry_agents/graph.py tests/test_graph.py
git commit -m "feat: LangGraph graph assembly with routing functions"
```

---

## Task 9: CLI with Streaming Output

**Files:**
- Create: `twelve_angry_agents/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
from click.testing import CliRunner
from unittest.mock import patch, MagicMock
from twelve_angry_agents.cli import main


def test_cli_requires_topic_or_stdin():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code != 0 or "topic" in result.output.lower() or "Error" in result.output


def test_cli_accepts_topic_argument():
    runner = CliRunner()
    with patch("twelve_angry_agents.cli.run_debate") as mock_run:
        mock_run.return_value = None
        result = runner.invoke(main, ["Should I quit my job?"])
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert "Should I quit my job?" in str(call_kwargs)


def test_cli_accepts_model_flag():
    runner = CliRunner()
    with patch("twelve_angry_agents.cli.run_debate") as mock_run:
        mock_run.return_value = None
        result = runner.invoke(main, ["--model", "gemma4:e2b", "Topic"])
    call_kwargs = mock_run.call_args
    assert "gemma4:e2b" in str(call_kwargs)


def test_cli_reads_from_stdin_when_no_topic():
    runner = CliRunner()
    with patch("twelve_angry_agents.cli.run_debate") as mock_run:
        mock_run.return_value = None
        result = runner.invoke(main, [], input="Topic from stdin\n")
    assert result.exit_code == 0
    mock_run.assert_called_once()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_cli.py -v
```

Expected: `ModuleNotFoundError: No module named 'twelve_angry_agents.cli'`

- [ ] **Step 3: Implement `cli.py`**

```python
# twelve_angry_agents/cli.py
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.rule import Rule

from twelve_angry_agents.config import load_config
from twelve_angry_agents.graph import build_graph
from twelve_angry_agents.state import DebateState


def run_debate(topic: str, model: str | None, agents_path: Path | None) -> None:
    """Build config, initialize state, and run the debate graph."""
    console = Console()

    config = load_config(agents_path=agents_path)
    if model:
        config.model.name = model

    console.print(Rule("[bold]TWELVE ANGRY AGENTS[/bold]"))
    console.print(f"Topic: {topic}")
    console.print(f"Model: {config.model.name}")
    console.print()

    graph = build_graph()

    initial_state: DebateState = {
        "topic": topic,
        "enriched_topic": "",
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

    graph.invoke(
        initial_state,
        config={"configurable": {"app_config": config}},
    )


@click.command()
@click.argument("topic", required=False)
@click.option(
    "--model",
    default=None,
    help="Override the Ollama model (e.g. gemma4:e2b)",
)
@click.option(
    "--agents",
    "agents_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom agents.yaml",
)
def main(topic: str | None, model: str | None, agents_path: Path | None) -> None:
    """Run your topic through a jury of 12 AI minds.\n
    \b
    Examples:
      taa "Should I accept this job offer?"
      cat business_plan.txt | taa
      taa --model gemma4:e2b "Is this architecture sound?"
    """
    if topic is None:
        if not sys.stdin.isatty():
            topic = sys.stdin.read().strip()
        else:
            raise click.UsageError("Provide a topic as an argument or pipe it via stdin.")

    if not topic:
        raise click.UsageError("Topic cannot be empty.")

    run_debate(topic=topic, model=model, agents_path=agents_path)
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_cli.py -v
```

Expected: 4 passed

- [ ] **Step 5: Run all tests**

```bash
pytest -v
```

Expected: All tests pass. If any fail, fix before proceeding.

- [ ] **Step 6: Commit**

```bash
git add twelve_angry_agents/cli.py tests/test_cli.py
git commit -m "feat: CLI entry point with streaming output and stdin support"
```

---

## Task 10: Smoke Test and README

**Files:**
- Create: `README.md`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write smoke test (requires Ollama running)**

```python
# tests/test_smoke.py
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
```

- [ ] **Step 2: Mark smoke test to skip without flag**

```bash
# Run only unit tests (no Ollama needed)
pytest -m "not integration" -v

# Run smoke test (Ollama must be running)
pytest -m integration -v
```

- [ ] **Step 3: Create `README.md`**

```markdown
# Twelve Angry Agents

> Run your decisions through a jury of 12 AI minds before you commit.

Most people make important decisions alone or with one perspective. Twelve Angry Agents runs your topic through 12 AI personalities — a skeptic, an optimist, a devil's advocate, an ethicist, and 8 more — debating until they reach a unanimous verdict. Inspired by *12 Angry Men*.

**Fully local. Fully private. No API costs.**

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

## Installation

```bash
pip install twelve-angry-agents
ollama pull gemma4:e4b
```

## Usage

```bash
# Ask a question
taa "Should I accept this job offer?"

# Pipe in a longer document
cat business_plan.txt | taa

# Use the smaller model (less VRAM)
taa --model gemma4:e2b "Is this architecture sound?"

# Use custom agent personalities
taa --agents my_agents.yaml "Should we pivot?"
```

## How It Works

1. The **Foreman** gathers any missing context with 2-3 targeted questions
2. All 12 agents cast a **blind vote** — no peer influence
3. If split: agents **debate sequentially**, each reading the full argument history
4. After each round, all agents re-vote
5. Repeat until **unanimous** (or a hung jury after 50 rounds)
6. The Foreman delivers a **prose verdict** with the decisive arguments and key risks

## The Jury

| Agent | Lens |
|-------|------|
| The Analyst | Data and evidence only |
| The Skeptic | Questions every premise |
| The Optimist | Opportunity and upside |
| The Pessimist | Risk and failure modes |
| The Empath | Human impact |
| The Ethicist | Right vs. wrong |
| The Pragmatist | What actually works |
| The Devil's Advocate | Forces majority to justify itself |
| The Historian | Precedents and patterns |
| The Visionary | 10-year consequences |
| The Realist | What's actually true right now |
| The Contrarian | Last holdout — hardest to move |

## Customization

Edit `config/agents.yaml` to change any agent's personality or system prompt. The config validates that exactly 12 agents are defined.

Edit `config/config.yaml` to change the model, temperature, or debate limits.

## License

CC BY-NC 4.0 — free for non-commercial use.
```

- [ ] **Step 4: Final test run**

```bash
pytest -m "not integration" -v
```

Expected: All unit tests pass.

- [ ] **Step 5: Final commit**

```bash
git add README.md tests/test_smoke.py
git commit -m "feat: smoke test and README"
```

---

## Self-Review Against Spec

**Spec coverage check:**

| Spec requirement | Covered in task |
|-----------------|-----------------|
| 12 agents with distinct personalities | Task 3 (agents.yaml) |
| Neutral Foreman moderator | Task 7 (moderator nodes) |
| Blind initial vote | Task 6 (blind_vote_node) |
| Sequential deliberation with full history | Task 6 (agent_speak_node) |
| Agents can hold or change vote | Task 6 (extract_vote, agent_speak) |
| Capitulation guard (weak argument on vote change) | Partially — agents are prompted to justify changes via deliberation prompt; explicit guard is v2 |
| Consensus detection (deterministic) | Task 4 (consensus.py) |
| Hung jury on max_rounds | Task 4 + Task 8 (consensus_check_node) |
| Context gathering (Foreman asks questions) | Task 7 (context_gather_node) |
| Verdict framing (not guilty/not guilty) | Task 7 (moderator_open_node) |
| Context summarization at 75% threshold | Task 5 (memory.py) + Task 7 (memory_check_node) |
| Real-time streaming to CLI | Task 6 (agent_speak_node streams tokens) |
| `trim_messages` for trimming | Memory utilities in Task 5 (build_summarization_messages) |
| Config-driven model name | Task 3 (config.yaml) |
| Config-driven agents (user-editable) | Task 3 (agents.yaml) |
| `taa` CLI entry point | Task 9 (cli.py + pyproject.toml) |
| `--model` flag | Task 9 |
| `--agents` flag | Task 9 |
| Pipe via stdin | Task 9 |
| Max 50 rounds safety valve | Task 3 (config.yaml) + Task 8 (consensus_check_node) |

**Note:** The explicit capitulation guard (detecting weak arguments on vote changes) is partially handled by the deliberation prompt requiring agents to justify changes. A fully automated guard (detecting word count or LLM-judging argument quality) is deferred to v2 to avoid extra LLM calls per agent turn.
