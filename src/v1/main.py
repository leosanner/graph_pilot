"""Composition root: wires dependencies and exposes the CLI."""

import typer

from v1.tui.app import run as run_tui

app = typer.Typer(
    no_args_is_help=False,
    help="LazyDocs — local RAG from the terminal",
)


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        run_tui()


@app.command()
def ingest(path: str) -> None:
    typer.echo(f"ingest {path}")


@app.command()
def query(question: str) -> None:
    typer.echo(f"query {question!r}")


def main() -> None:
    app()
