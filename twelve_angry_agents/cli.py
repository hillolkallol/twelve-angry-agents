import sys
from pathlib import Path

import click
from rich.console import Console
from rich.rule import Rule

from twelve_angry_agents.config import load_config
from twelve_angry_agents.graph import build_graph
from twelve_angry_agents.state import DebateState


def run_debate(
    topic: str,
    model: str | None,
    agents_path: Path | None,
    output_path: Path | None,
    max_rounds: int | None,
) -> None:
    """Build config, initialize state, and run the debate graph."""
    console = Console(record=True)

    config = load_config(agents_path=agents_path)
    if model:
        config.model.name = model
    if max_rounds is not None:
        config.debate.max_rounds = max_rounds

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
        "moderator_question": "",
        "status": "gathering",
    }

    try:
        final_state = graph.invoke(
            initial_state,
            config={"configurable": {"app_config": config, "console": console}},
        )
    except Exception as e:
        err = str(e).lower()
        if "connection refused" in err or "connect" in err or "ollama" in err:
            raise click.ClickException(
                "Cannot reach Ollama. Make sure it is running:\n\n"
                "  ollama serve\n\n"
                f"Then try again. (Original error: {e})"
            ) from e
        raise

    if output_path:
        console.save_text(str(output_path))
        console.print(f"\nTranscript saved to: {output_path}")


@click.command()
@click.argument("topic", required=False)
@click.option(
    "--model",
    default=None,
    help="Override the Ollama model (e.g. gemma4:e4b)",
)
@click.option(
    "--agents",
    "agents_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom agents.yaml",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Save full debate transcript to a file (e.g. debate.txt)",
)
@click.option(
    "--max-rounds",
    "max_rounds",
    default=None,
    type=int,
    help="Maximum deliberation rounds before a hung jury (default: 5)",
)
def main(
    topic: str | None,
    model: str | None,
    agents_path: Path | None,
    output_path: Path | None,
    max_rounds: int | None,
) -> None:
    """Run your topic through a jury of 12 AI minds.\n
    \b
    Examples:
      taa "Should I accept this job offer?"
      cat business_plan.txt | taa
      taa --model gemma4:e4b "Is this architecture sound?"
      taa --output debate.txt "Should we pivot to B2B?"
    """
    if topic is None:
        if not sys.stdin.isatty():
            topic = sys.stdin.read().strip()
        else:
            raise click.UsageError("Provide a topic as an argument or pipe it via stdin.")

    if not topic:
        raise click.UsageError("Topic cannot be empty.")

    run_debate(topic=topic, model=model, agents_path=agents_path, output_path=output_path, max_rounds=max_rounds)
