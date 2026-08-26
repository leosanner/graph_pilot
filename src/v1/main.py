"""Composition root: wires dependencies and exposes the CLI."""

import typer

app = typer.Typer(no_args_is_help=True, help="RAG p1 — ingest and query")


@app.command()
def ingest(path: str) -> None:
    typer.echo(f"ingest {path}")


@app.command()
def query(question: str) -> None:
    typer.echo(f"query {question!r}")


def main() -> None:
    app()
