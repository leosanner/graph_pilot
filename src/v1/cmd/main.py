"""Composition root: monta as dependências e expõe a CLI."""

import typer

from ragp1.app.agent.config.settings import Settings

app = typer.Typer(no_args_is_help=True, help="RAG p1 — ingest and query")


@app.command()
def ingest(path: str) -> None:
    settings = Settings()
    typer.echo(f"ingest {path} -> {settings.chroma_collection}")


@app.command()
def query(question: str) -> None:
    settings = Settings()
    typer.echo(f"query {question!r} com {settings.ollama_model}")


def main() -> None:
    app()
