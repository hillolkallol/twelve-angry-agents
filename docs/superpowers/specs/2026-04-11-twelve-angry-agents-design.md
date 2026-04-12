# Twelve Angry Agents — Design Spec
**Date:** 2026-04-11
**Status:** Approved

---

## Overview

Twelve Angry Agents is a local, open-source CLI tool inspired by the film *12 Angry Men*. It runs any topic — a decision, argument, ethical dilemma, or analysis — through a structured debate between 12 AI agents with distinct reasoning personalities, moderated by a neutral Foreman, until they reach a unanimous verdict. The system runs entirely on a local LLM via Ollama with no cloud dependency.

**Primary use case:** Stress-testing decisions and arguments from 12 adversarial perspectives before committing. Marketed as a decision auditor — "run your decision through a jury of 12 AI minds before you commit."

**License:** Non-commercial open source (GitHub release).

---

## Goals

- Produce a clear verdict with detailed reasoning, not just better questions
- Run fully locally — private, no API costs, no cloud dependency
- Stream debate output in real-time so users aren't staring at a spinner
- Be hackable — personas defined in config, not hardcoded
- Install and run in under 5 minutes (`pip install` + `ollama pull`)

---

## Non-Goals

- Factual question answering (not a search engine or RAG system)
- Multi-user or server deployment
- GUI or web interface (v1 is CLI only)
- Custom agent frameworks per-user (config file overrides only)

---

## Architecture

### Top-level components

```
twelve-angry-agents/
├── twelve_angry_agents/
│   ├── cli.py              ← entry point, argument parsing, streaming output
│   ├── graph.py            ← LangGraph graph definition, nodes, edges
│   ├── nodes/
│   │   ├── moderator.py    ← open, context-gather, deliberate, close nodes
│   │   ├── agent.py        ← blind vote + deliberation speak nodes
│   │   └── consensus.py    ← deterministic consensus check (no LLM)
│   ├── memory.py           ← trim_messages + LangMem summarization wiring
│   ├── state.py            ← DebateState TypedDict definition
│   └── config.py           ← loads and validates agents.yaml + config.yaml
├── config/
│   ├── agents.yaml         ← 12 agent personas + moderator (user-editable)
│   └── config.yaml         ← model, max_rounds, token limits, thresholds
├── docs/
│   └── superpowers/specs/
├── pyproject.toml
└── README.md
```

### LangGraph state

```python
class DebateState(TypedDict):
    topic: str                    # original user input
    enriched_topic: str           # topic + clarifying answers
    verdict_framing: str          # e.g. "proceed / don't proceed"
    votes: dict[str, str]         # agent_name → current vote
    original_votes: dict[str, str]  # first blind votes, for tracking flips
    transcript: list[BaseMessage] # full debate history
    summary: str                  # running summary of older rounds (LangMem)
    round: int                    # current deliberation round
    speaking_order: list[str]     # randomized each round by Foreman
    verdict: str | None           # set when consensus reached
    status: Literal["gathering", "voting", "deliberating", "concluded"]
```

---

## LangGraph Flow

```
[START]
   ↓
[context_gather]       ← Foreman analyzes topic; asks CLI clarifying questions if needed
   ↓
[moderator_open]       ← Foreman sets verdict framing, presents topic to all agents
   ↓
[blind_vote]           ← All 12 agents vote sequentially (topic only, no peer history visible)
   ↓
[consensus_check]      ← Deterministic: unanimous? → close. Split? → deliberate.
   ↓ split
[moderator_deliberate] ← Foreman randomizes speaking order for this round
   ↓
[agent_speak]          ← One agent speaks with full transcript visible
   ↓ (loops back to agent_speak until all agents in speaking_order have spoken)
[memory_check]         ← After full round: check token usage; summarize if > 75% context
   ↓
[vote_again]           ← All 12 re-vote after full round completes
   ↓
[consensus_check]      ← Unanimous → close. Still split → next round (back to moderator_deliberate).
   ↓ unanimous (or max_rounds hit)
[moderator_close]      ← Foreman delivers prose verdict + reasoning chain
   ↓
[END]
```

---

## Context Gathering

Before the debate opens, the Foreman evaluates whether the topic contains sufficient context. If not, it generates 2-3 targeted clarifying questions, asks them interactively in the CLI, and appends the answers to form an `enriched_topic`. No debate begins without sufficient context.

Example:
```
THE FOREMAN
Before we begin, I need a bit more context:

1. What's your financial runway if you quit today?
2. Do you have a specific idea or still exploring?
3. What's driving the urgency — opportunity or dissatisfaction?

> [user types answers]
```

---

## Verdict Framing

The Foreman determines the appropriate binary framing from the topic before the blind vote. Agents vote on this framing — not on "guilty/not guilty."

| Topic type | Example framing |
|------------|-----------------|
| Decision | `proceed / don't proceed` |
| Evaluation | `sound / unsound` |
| Ethical question | `ethical / unethical` |
| Viability | `viable / not viable` |

The final verdict is always prose, not just a label. The Foreman synthesizes the unanimous position into a natural language conclusion including the decisive arguments and key risks flagged by holdouts.

---

## Agent Personas

All 12 agents share the same underlying Ollama model instance, differentiated purely by system prompt. Personas are defined in `config/agents.yaml` and are fully user-editable.

**Model:** Gemma 4 is available on Ollama. The "E" in the model tags stands for "effective parameters" (edge-optimized variants):
- `gemma4:e2b` — 2B effective parameters, 4 GB VRAM minimum, 128K context
- `gemma4:e4b` — 4B effective parameters, 6 GB VRAM minimum, 128K context (recommended default)
- Instruction-tuned variants: `gemma4:e2b-it`, `gemma4:e4b-it`

| # | Name | Reasoning Lens |
|---|------|----------------|
| 1 | The Analyst | Pure logic and data. Dismisses anecdote, demands evidence. |
| 2 | The Skeptic | Questions every premise. Hard to convince. Never accepts surface arguments. |
| 3 | The Optimist | Focuses on opportunity and best-case outcome. |
| 4 | The Pessimist | Focuses on failure modes and probability of things going wrong. |
| 5 | The Empath | Human impact above all. Who gets hurt? Emotional truth matters. |
| 6 | The Ethicist | Moral framework. Right vs. wrong, not just effective vs. ineffective. |
| 7 | The Pragmatist | What actually works in the real world. Dismisses theory. |
| 8 | The Devil's Advocate | Argues against prevailing consensus. Forces the majority to justify itself. |
| 9 | The Historian | Looks for precedents and patterns. "This has happened before." |
| 10 | The Visionary | Long-term, systemic thinking. Where does this lead in 10 years? |
| 11 | The Realist | Cuts through spin on both sides. What's actually true right now. |
| 12 | The Contrarian | Disagrees when consensus forms too quickly. The last holdout. Hardest to move. |

### The Foreman (13th, neutral moderator)
Procedural only — no vote, no opinions. Manages process: gathers context, sets framing, calls votes, randomizes speaking order, detects consensus, closes debate. Kept narrow to minimize latency from extra LLM calls.

---

## Debate Mechanics

### Blind vote
Each agent receives only the `enriched_topic` and the `verdict_framing`. No peer arguments visible. Returns a structured response:
```
VOTE: proceed
REASON: one sentence
```

### Deliberation
- Speaking order is **randomized each round** (not round-robin, not smart-ordered — random is reliable and fair)
- Each speaking agent receives: their system prompt + full transcript (or summary + recent rounds if context compressed)
- Agents must state their current vote and can hold or change it
- Response capped at `max_tokens_per_response` (default: 250 tokens) to keep debate focused and context manageable

### Capitulation guard
If an agent changes their vote in a round where they produced no substantive argument, the Foreman flags it and prompts them to explicitly state what changed their mind. Prevents silent caving.

### Consensus check
Deterministic — no LLM:
```python
def check_consensus(votes: dict) -> bool:
    return len(set(votes.values())) == 1
```

### Hung jury
If `max_rounds` is reached without consensus, the Foreman announces a hung jury: majority verdict stated, dissenting positions clearly preserved. No false consensus manufactured.

---

## Memory Management

| Need | Tool |
|------|------|
| Conversation history | Plain `list[BaseMessage]` in LangGraph state |
| Token-aware trimming | `trim_messages` from `langchain_core` |
| Summarization when context fills | LangMem `SummarizationNode` |
| Resume interrupted debates | `SqliteSaver` checkpointer (optional) |

**Summarization trigger:** When transcript token count exceeds `context_summary_threshold` (default: 75% of model context window), a summarization node condenses older rounds into a `summary` string. The transcript is replaced with `[summary] + recent round messages`. Agents continue with compressed but coherent history.

---

## Configuration

### `config/config.yaml`
```yaml
model:
  name: gemma4:e4b         # gemma4:e2b (2B) or gemma4:e4b (4B) — any ollama-compatible model
  temperature: 0.7
  context_window: 128000

debate:
  max_rounds: 50           # safety valve only, not a quality limit
  max_tokens_per_response: 250
  context_summary_threshold: 0.75
```

### `config/agents.yaml`
```yaml
moderator:
  name: The Foreman
  system_prompt: >
    You are a neutral debate moderator. You have no opinion on the topic.
    Your only job is to manage the debate process fairly...

agents:
  - name: The Skeptic
    system_prompt: >
      You question every premise. You never accept arguments at face value...
  - name: The Analyst
    system_prompt: >
      You reason from data and evidence only...
```

---

## CLI Interface

### Usage
```bash
# basic
taa "Should I accept this job offer?"

# pipe long-form input
cat business_plan.txt | taa

# custom agents config
taa --agents my_agents.yaml "Is this architecture sound?"

# override model
taa --model gemma4:e2b "Is TypeScript worth adopting?"
```

### Streaming output format
```
TWELVE ANGRY AGENTS
Topic: Should I accept this job offer?
Model: gemma4:e4b
──────────────────────────────────────────────────

THE FOREMAN
Before we begin, I need a bit more context:
1. What's the role and how does it compare to your current one?
2. What's your primary concern — compensation, growth, or culture?

> Senior engineer role, 30% pay increase, early-stage startup
> Mostly worried about startup risk vs. stability

──────────────────────────────────────────────────

━━━ BLIND VOTE ━━━
THE ANALYST        → proceed
THE SKEPTIC        → don't proceed
THE OPTIMIST       → proceed
...

Vote: 8 proceed, 4 don't proceed — deliberation begins.

━━━ ROUND 1 ━━━

THE SKEPTIC  [don't proceed]
A 30% pay increase at an early-stage startup needs
to be weighed against equity dilution risk and...

THE OPTIMIST  [proceed]
The compensation signal alone suggests the company
values this role highly. Early-stage at senior level...

━━━ RE-VOTE ━━━
THE SKEPTIC    → don't proceed (holds)
THE OPTIMIST   → proceed (holds)
THE ANALYST    → proceed (holds)
...

━━━ VERDICT ━━━
UNANIMOUS: proceed

THE FOREMAN
The jury recommends proceeding. The argument that
shifted the room: the 30% increase combined with a
senior scope represents an asymmetric opportunity.
The Skeptic held longest — their concern about runway
and dilution is the critical risk to verify before signing.
```

---

## GitHub Release Narrative

**Tagline:** *Run your decisions through a jury of 12 AI minds before you commit.*

**README pitch:** Most people make important decisions alone or with one perspective. Twelve Angry Agents runs your decision through 12 adversarial AI personalities — a skeptic, an optimist, a devil's advocate, an ethicist, and 8 more — until they reach a unanimous verdict. Runs on Gemma 4 via Ollama. Fully local. Fully private. No API costs.

**Key points to lead with:**
- Works on any topic: decisions, arguments, plans, ethical dilemmas
- Runs locally on Gemma 4B via Ollama — your data never leaves your machine
- Streams the debate live so you can follow the reasoning, not just the verdict
- The verdict comes with the reasoning chain — you see *why*, not just *what*

---

## Open Questions / Future Work

- Smart speaking order (Foreman calls on specific agents based on vote split) — v2
- Web UI for richer debate visualization — v2
- Support for injecting domain-specific context (e.g., RAG over documents) — v2
- Saving debate transcripts to file — v1.1
