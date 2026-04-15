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

    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}. "
            "Run from the project root or pass a valid --config path."
        )
    if not agents_path.exists():
        raise FileNotFoundError(
            f"agents.yaml not found at {agents_path}. "
            "Run from the project root or pass a valid --agents path."
        )

    try:
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}") from e

    try:
        with open(agents_path) as f:
            raw_agents = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {agents_path}: {e}") from e

    try:
        model = ModelConfig(**raw_config["model"])
        debate = DebateConfig(**raw_config["debate"])
    except KeyError as e:
        raise ValueError(f"Missing required section {e} in {config_path}") from e
    except TypeError as e:
        raise ValueError(f"Invalid field in {config_path}: {e}") from e

    try:
        moderator_data = raw_agents["moderator"]
        moderator = AgentPersona(
            name=moderator_data["name"],
            system_prompt=moderator_data["system_prompt"],
        )
    except KeyError as e:
        raise ValueError(f"Missing required field {e} in moderator section of {agents_path}") from e

    agents_data = raw_agents.get("agents") or []
    agents = [
        AgentPersona(name=a["name"], system_prompt=a["system_prompt"])
        for a in agents_data
    ]

    if len(agents) < 2:
        raise ValueError(
            f"agents file must define at least 2 agents, got {len(agents)}"
        )

    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate agent names found in agents.yaml: {names}")

    return AppConfig(model=model, debate=debate, moderator=moderator, agents=agents)
