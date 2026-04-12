import pytest
from pathlib import Path
from twelve_angry_agents.config import load_config, AppConfig, AgentPersona


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
