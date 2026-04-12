from click.testing import CliRunner
from unittest.mock import patch
from twelve_angry_agents.cli import main


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
