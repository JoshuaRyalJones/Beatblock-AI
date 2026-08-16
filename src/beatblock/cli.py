"""BeatBlock command line interface.

Commands are implemented per milestone:
    candidates  deterministic candidate generation, no model load (M1)
    recommend   full pipeline with the local SLM ranker (M2)
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from beatblock.domain.models import RecommendationContext, RecommendationResult
from beatblock.model.loader import load_inference_config, load_local_model
from beatblock.model.ranker import rank_candidates
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


@app.command()
def recommend(
    key: str = typer.Option(..., help="Musical key, for example 'D minor'."),
    progression: str = typer.Option(..., help="Comma-separated chord symbols."),
    genre: str = typer.Option(..., help="Genre or production style."),
    mood: Annotated[
        list[str] | None, typer.Option(help="Repeat for multiple mood descriptors.")
    ] = None,
    tension: float = typer.Option(0.5, min=0.0, max=1.0),
    bpm: int | None = typer.Option(None, min=20, max=300),
    section: str | None = typer.Option(None),
    config_path: Annotated[Path, typer.Option(help="Inference YAML config.")] = Path(
        "configs/model.yaml"
    ),
) -> None:
    """Generate candidates and rank them with the configured local model."""
    context = RecommendationContext(
        key=key,
        progression=progression.split(","),
        genre=genre,
        moods=mood or [],
        tension=tension,
        bpm=bpm,
        section=section,
    )
    generated = generate_candidates(context)
    config = load_inference_config(config_path)
    local_model = load_local_model(config)
    ranked = rank_candidates(
        context,
        generated,
        local_model.generate,
        enable_thinking=config.model.enable_thinking,
    )
    result = RecommendationResult(
        context=context,
        candidates_generated=len(generated),
        recommendations=ranked,
        model_id=config.model.id,
    )

    console = Console()
    console.print("[bold]BeatBlock[/bold]")
    console.print(f"Key: {context.key}")
    console.print(f"Progression: {' -> '.join(context.progression)}")
    console.print(f"Genre: {context.genre}\n")
    for recommendation in result.recommendations:
        console.print(
            f"[bold]{recommendation.rank}. {recommendation.symbol}[/bold]  "
            f"score: {recommendation.model_score:.2f}"
        )
        console.print(f"   {recommendation.reason}")


def main() -> None:
    app()
