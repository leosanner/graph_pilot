"""Composition root: wires dependencies and exposes the CLI."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from v1.app.ingestion.ingestion import DocumentIngestion
from v1.app.ingestion.loader.pdf_loader import PdfLoader
from v1.app.ingestion.repository.vector import VectorRepository
from v1.app.ingestion.vector.errors import EmbeddingError, ModelInfoError
from v1.app.ingestion.vector.processor import VectorProcessor
from v1.app.ingestion.vector.settings import RuntimeSettings, VectorSettings
from v1.app.ingestion.vector.utils.utils import load_model_metadata
from v1.infra.database import PostgresClient, PostgresSettings
from v1.tui.app import run as run_tui
from v1.tui.paths import FILE_TYPE

app = typer.Typer(
    no_args_is_help=False,
    help="LazyDocs — local RAG from the terminal",
)


def document_from_path(path: Path) -> Document:
    if not path.is_file():
        raise typer.BadParameter(f"Not a file: {path}")
    if path.suffix.lower() != f".{FILE_TYPE}":
        raise typer.BadParameter(f"{path} is not a .{FILE_TYPE} file")
    try:
        return PdfLoader().load(path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc


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


def ingest_path(path: Path, model_name: str) -> None:
    document = document_from_path(path)
    asyncio.run(ingest_document(document, model_name))


def run_ingest(path: Path, model_name: str) -> None:
    try:
        ingest_path(path, model_name)
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

    selection = run_tui(ingest=_tui_ingest)
    if selection is None:
        raise typer.Exit(code=0)

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


def _tui_ingest(path: str, model: str) -> None:
    ingest_path(Path(path), model)


def main() -> None:
    app()
