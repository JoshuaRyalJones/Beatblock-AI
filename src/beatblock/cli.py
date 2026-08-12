"""BeatBlock command line interface.

Commands are implemented per milestone:
    candidates  deterministic candidate generation, no model load (M1)
    recommend   full pipeline with the local SLM ranker (M2)
"""

import typer
from rich.console import Console
from rich.table import Table

from beatblock.domain.models import RecommendationContext
from beatblock.music.candidates import generate_candidates

app = typer.Typer(help="Local AI-assisted chord recommendation.")


@app.callback()
def cli() -> None:
    """Run BeatBlock from the command line."""


@app.command()
def candidates(
    key: str = typer.Option(..., help="Musical key, for example 'D minor'."),
    progression: str = typer.Option(..., help="Comma-separated chord symbols."),
) -> None:
    """Show deterministic candidates without loading a model."""
    context = RecommendationContext(key=key, progression=progression.split(","))
    results = generate_candidates(context)
    table = Table(title=f"BeatBlock candidates — {context.key}")
    table.add_column("Chord")
    table.add_column("Degree")
    table.add_column("Function")
    table.add_column("Rule")
    table.add_column("Score", justify="right")
    for candidate in results:
        table.add_row(
            candidate.symbol,
            candidate.degree,
            candidate.function,
            candidate.source_rule,
            f"{candidate.theory_score:.2f}",
        )
    Console().print(table)


def main() -> None:
    app()
