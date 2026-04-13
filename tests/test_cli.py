from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from langchain_core.messages import AIMessage, HumanMessage

from twelve_angry_agents.cli import main, save_transcript


def test_cli_requires_topic_or_stdin():
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code != 0


def test_cli_accepts_topic_argument():
    runner = CliRunner()
    with patch("twelve_angry_agents.cli.run_debate") as mock_run:
        mock_run.return_value = None
        result = runner.invoke(main, ["Should I quit my job?"])
    assert result.exit_code == 0
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert "Should I quit my job?" in str(call_kwargs)


def test_cli_accepts_model_flag():
    runner = CliRunner()
    with patch("twelve_angry_agents.cli.run_debate") as mock_run:
        mock_run.return_value = None
        result = runner.invoke(main, ["--model", "gemma4:e2b", "Topic"])
    assert result.exit_code == 0
    call_kwargs = mock_run.call_args
    assert "gemma4:e2b" in str(call_kwargs)


def test_cli_reads_from_stdin_when_no_topic():
    runner = CliRunner()
    with patch("twelve_angry_agents.cli.run_debate") as mock_run:
        mock_run.return_value = None
        result = runner.invoke(main, [], input="Topic from stdin\n")
    assert result.exit_code == 0
    mock_run.assert_called_once()


def test_cli_accepts_output_flag():
    runner = CliRunner()
    with patch("twelve_angry_agents.cli.run_debate") as mock_run:
        mock_run.return_value = None
        result = runner.invoke(main, ["--output", "debate.txt", "Topic"])
    assert result.exit_code == 0
    call_kwargs = mock_run.call_args
    assert "debate.txt" in str(call_kwargs)


def test_save_transcript_writes_file(tmp_path):
    output_path = tmp_path / "debate.txt"
    state = {
        "topic": "Should I quit?",
        "verdict": "proceed",
        "summary": "",
        "transcript": [
            HumanMessage(content="The topic is: Should I quit?"),
            AIMessage(content="VOTE: proceed\nGood idea.", name="The Optimist"),
        ],
    }
    save_transcript(output_path, state)
    content = output_path.read_text()
    assert "Should I quit?" in content
    assert "proceed" in content
    assert "The Optimist" in content


def test_save_transcript_includes_summary(tmp_path):
    output_path = tmp_path / "debate.txt"
    state = {
        "topic": "Should I quit?",
        "verdict": "proceed",
        "summary": "Earlier rounds showed strong disagreement.",
        "transcript": [],
    }
    save_transcript(output_path, state)
    content = output_path.read_text()
    assert "Earlier rounds showed strong disagreement." in content
