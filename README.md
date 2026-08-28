# LazyDocs

Local RAG from the terminal. You pick a file, a local Ollama embedding model chunks and embeds it, and Postgres (`pgvector`) stores the vectors for later retrieval.

This is a learning project: the ingest path is wired end to end. Chat and query are not — the TUI can pick a chat model, but nothing answers questions yet.

## What it does today

- Split a UTF-8 text file, embed batches with Ollama, persist document + chunk rows
- Terminal UI (`ragp1`) to choose **Ingest** and the embedding model, then a path
- CLI: `ragp1 ingest path/to/file.txt`
- HNSW index on chunk embeddings (cosine) for the retrieval step that is not wired yet

PDF loading is not wired. Files must be UTF-8 text. The default embed model is `nomic-embed-text` (768-d); another dimension needs a new migration.

How the pieces work: [`docs/README.md`](docs/README.md).

## Run

**Needs:** Python 3.13+, [uv](https://docs.astral.sh/uv/), Docker, [Ollama](https://ollama.com/) running.

```bash
make up
```

That copies `.env` if it is missing, installs deps, starts Postgres, applies migrations, pulls the embed model, and opens the TUI. How that pipeline works: [`docs/dev/project-up.md`](docs/dev/project-up.md).

Choose **Ingest**, pick the model, then enter a UTF-8 text file path.

CLI, after the stack is up:

```bash
uv run ragp1 ingest path/to/notes.txt
uv run ragp1 ingest path/to/notes.txt --model nomic-embed-text
```

`--model` overrides `OLLAMA_EMBED_MODEL` from `.env`.

```bash
uv run pytest
```
