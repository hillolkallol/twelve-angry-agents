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
