# Twelve Angry Agents

[![CI](https://github.com/hillolkallol/twelve-angry-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hillolkallol/twelve-angry-agents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Ollama](https://img.shields.io/badge/runs%20on-Ollama-black)](https://ollama.com)

> Run your decisions through a jury of 12 AI minds before you commit.

Most people make important decisions alone or with one perspective. Twelve Angry Agents runs your topic through 12 AI personalities — a skeptic, an optimist, a devil's advocate, an ethicist, and 8 more — debating until they reach a unanimous verdict. Inspired by *12 Angry Men*.

**Fully local. Fully private. No API costs.**

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

## Installation

**1. Install Ollama**

```bash
brew install ollama      # macOS
# or download from https://ollama.com for Linux/Windows
```

**2. Start the server and pull the model**

```bash
ollama serve             # keep this running in the background
```

Then in a separate terminal:

```bash
ollama pull gemma4:e2b   # ~5 GB — default
ollama pull gemma4:e4b   # ~9 GB — higher quality, optional
```

**3. Install Twelve Angry Agents in a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
pip install twelve-angry-agents
```

## Usage

```bash
# Career decision with context
taa "I've been offered a senior engineer role at an early-stage startup — 30% pay increase, significant equity, but no runway visibility beyond 18 months. I have a mortgage and a family. Should I take it?"

# Architecture decision
taa "We're considering breaking our Django monolith into microservices. The team is 6 engineers, we deploy twice a week, and our main pain point is that a bug in the payments module blocks unrelated features from shipping."

# Pipe in a longer document
cat business_plan.txt | taa

# Use the larger model for higher quality
taa --model gemma4:e4b "We're debating whether to rewrite our mobile app in React Native or keep separate iOS and Android codebases. We have 2 mobile engineers and ship features every 2 weeks."

# Use custom agent personalities
taa --agents my_agents.yaml "Should we pivot from B2C to B2B?"
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
