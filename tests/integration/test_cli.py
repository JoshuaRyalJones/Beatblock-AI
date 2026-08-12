from typer.testing import CliRunner

from beatblock.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Local AI-assisted chord recommendation" in result.stdout


def test_candidates_command() -> None:
    result = runner.invoke(
        app, ["candidates", "--key", "D minor", "--progression", "Dm9,Gm9"]
    )

    assert result.exit_code == 0
    assert "harmonic_minor_dominant" in result.stdout
    assert "A7" in result.stdout
