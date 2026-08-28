"""Composition root: wires dependencies and exposes the CLI."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from v1.app.ingestion.ingestion import DocumentIngestion
from v1.app.ingestion.repository.vector import VectorRepository
from v1.app.ingestion.vector.errors import EmbeddingError, ModelInfoError
from v1.app.ingestion.vector.processor import VectorProcessor
from v1.app.ingestion.vector.settings import RuntimeSettings, VectorSettings
from v1.app.ingestion.vector.utils.utils import load_model_metadata
from v1.infra.database import PostgresClient, PostgresSettings
from v1.tui.app import Action, run as run_tui

app = typer.Typer(
    no_args_is_help=False,
    help="LazyDocs — local RAG from the terminal",
)


def document_from_path(path: Path) -> Document:
    if not path.is_file():
        raise typer.BadParameter(f"Not a file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter(
            f"{path} is not UTF-8 text. PDF loading is not wired yet."
        ) from exc
    return Document(
        page_content=text,
        metadata={"source": str(path), "name": path.name},
    )


def vector_processor(model_name: str) -> VectorProcessor:
    metadata = load_model_metadata(model_name)
    chunk_size = max(128, metadata.context_length)
    chunk_overlap = max(32, chunk_size // 10)
    base_url = os.environ.get("OLLAMA_BASE_URL", "").strip()
    embeddings = (
        OllamaEmbeddings(model=model_name, base_url=base_url)
        if base_url
        else OllamaEmbeddings(model=model_name)
    )
    return VectorProcessor(
        VectorSettings(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model=embeddings,
            model_name=model_name,
        ),
        RuntimeSettings(),
    )


async def ingest_document(document: Document, model_name: str) -> None:
    client = PostgresClient(PostgresSettings())
    try:
        ingestion = DocumentIngestion(VectorRepository(client))
        await ingestion.ingest_vector(document, vector_processor(model_name))
    finally:
        client.close()


def run_ingest(path: Path, model_name: str) -> None:
    document = document_from_path(path)
    try:
        asyncio.run(ingest_document(document, model_name))
    except (EmbeddingError, ModelInfoError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"ingested {path} with {model_name}")


def embed_model_from_env() -> str:
    model_name = os.environ.get("OLLAMA_EMBED_MODEL", "").strip()
    if not model_name:
        raise typer.BadParameter("Pass --model or set OLLAMA_EMBED_MODEL")
    return model_name


@app.callback(invoke_without_command=True)
def default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return

    selection = run_tui()
    if selection is None:
        raise typer.Exit(code=0)

    if selection.action == Action.INGEST:
        path = Path(typer.prompt("Document path"))
        run_ingest(path, selection.model)
        return

    typer.echo(f"query {selection.model}")


@app.command()
def ingest(
    path: Path,
    model: str | None = None,
) -> None:
    run_ingest(path, model or embed_model_from_env())


@app.command()
def query(question: str) -> None:
    typer.echo(f"query {question!r}")


def main() -> None:
    app()
