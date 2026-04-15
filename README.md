# Twelve Angry Agents

[![CI](https://github.com/hillolkallol/twelve-angry-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/hillolkallol/twelve-angry-agents/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/twelve-angry-agents)](https://pypi.org/project/twelve-angry-agents/)
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

# Save the full debate transcript to a file
taa --output debate.txt "Should I accept this acquisition offer?"

# Run longer — allow up to 10 rounds before declaring a hung jury
taa --max-rounds 10 "Should we raise a Series A now or wait 12 months?"
```

> **Tip:** The debate is only as good as the details you give it. Include your constraints, context, and what's already on your mind — the more specific, the sharper the arguments. A one-liner gets a generic debate; a paragraph gets a real one.

> **Note:** If your topic contains a `$` sign (e.g. `$40k`), use single quotes or escape it — otherwise the shell will expand it as a variable:
> ```bash
> taa 'I have $40k saved — should I invest it all?'
> # or
> taa "I have \$40k saved — should I invest it all?"
> ```

## Example

See [`examples/food-truck-debate.md`](examples/food-truck-debate.md) for a full debate transcript — a food truck business decision that ran 5 rounds and ended in a hung jury (10–2 against proceeding).

## How It Works

1. The **Foreman** gathers any missing context with 2-3 targeted questions
2. All agents cast a **blind vote** — no peer influence
3. If split: agents **debate sequentially**, each reading the full argument history
4. After each round, all agents re-vote
5. Repeat until **unanimous** (or a hung jury after the round limit)
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

## Built-in Juries

Ready-to-use agent configurations for specific decision types. Pass them with `--agents`:

| File | Best for | Agents |
|------|----------|--------|
| [`config/agents/startup.yaml`](config/agents/startup.yaml) | Fundraising, pivots, strategic decisions | CEO, CFO, CTO, Head of Sales, VC Investor, Legal Counsel, COO |
| [`config/agents/engineering.yaml`](config/agents/engineering.yaml) | Architecture, tech stack, technical tradeoffs | Backend, Frontend, DevOps/SRE, Security, Architect, Tech Lead, QA, PM, EM |
| [`config/agents/product.yaml`](config/agents/product.yaml) | Features, roadmap bets, design choices | PM, UX, Engineer, Data Scientist, CS Manager, Sales Lead, CEO, Customer Advocate, Competitor Analyst |
| [`config/agents/investment.yaml`](config/agents/investment.yaml) | Stocks, assets, portfolio allocation | Value, Growth, Risk Manager, Macro Economist, Financial Planner, Tax Strategist, Behavioral Economist, VC, Contrarian |
| [`config/agents/life.yaml`](config/agents/life.yaml) | Career changes, relocations, major life decisions | Therapist, Financial Advisor, Life Coach, Career Counselor, Pragmatist Friend, Optimist, Cynic, Mentor, Realist |

```bash
taa --agents config/agents/startup.yaml "Should we raise a Series A now or wait 12 months?"
taa --agents config/agents/engineering.yaml "Should we rewrite the auth service in Go?"
taa --agents config/agents/product.yaml "Should we build a native mobile app or stick with the web?"
taa --agents config/agents/investment.yaml 'Should I put $50k into index funds or real estate?'
taa --agents config/agents/life.yaml "Should I quit my job and move abroad?"
```

## Custom Agent Personalities

You can create your own jury with any number of agents (minimum 2) and a moderator.

Create a `my_agents.yaml` file:

```yaml
moderator:
  name: The Chair
  system_prompt: >
    You are a neutral facilitator. You have no opinion on the topic.
    Your only job is to manage the debate process fairly and ensure
    all voices are heard.

agents:
  - name: The CFO
    system_prompt: >
      You evaluate every decision through the lens of financial impact,
      unit economics, and capital efficiency. You demand clear ROI.
      State your VOTE clearly at the start: VOTE: [option].
      Keep your response under 200 words.

  - name: The CTO
    system_prompt: >
      You focus on technical feasibility, system complexity, and
      engineering risk. You push back on unrealistic timelines.
      State your VOTE clearly at the start: VOTE: [option].
      Keep your response under 200 words.

  # add as many agents as make sense for your use case
```

Then run it:

```bash
taa --agents my_agents.yaml "Should we rebuild the platform or migrate incrementally?"
```

Each agent needs a `name` and a `system_prompt`.

## Configuration

Edit `config/config.yaml` to change the model, temperature, or debate limits:

```yaml
model:
  name: gemma4:e2b      # or gemma4:e4b for higher quality
  temperature: 0.7
  context_window: 128000

debate:
  max_rounds: 5             # override with --max-rounds on the CLI
  context_summary_threshold: 0.50
```

## License

CC BY-NC 4.0 — free for non-commercial use.
