"""BeatBlock command line interface.

Commands are implemented per milestone:
    candidates  deterministic candidate generation, no model load (M1)
    recommend   full pipeline with the local SLM ranker (M2)
"""

import typer

app = typer.Typer(help="Local AI-assisted chord recommendation.")


def main() -> None:
    app()
