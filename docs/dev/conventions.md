# Development conventions

Keep this short. Add a rule only when it has actually come up.

## Language

All code, comments, commit messages, and docs are written in **English**.

## Module docs

Each living module doc in this folder uses four sections, in this order:

1. **Why it exists** — the problem it prevents.
2. **What it is for** — the job it does, and what it does *not* do.
3. **How it works** — one mermaid diagram of the happy path.
4. **Example** — input JSON and the expected output JSON.

No extra sections unless the module cannot be understood without them.

## Repository layout

Paths below are relative to the repository root.

| Path | What |
|---|---|
| `src/v1/app/ingestion/vector/` | Chunking, embedding, model metadata probe. |
| `src/v1/app/agent/` | Chat model wiring and RAG stubs. |
| `src/v1/infra/` | Postgres client. |
| `tests/` | Pytest. |
| `docs/` | This documentation. |

## Secrets

Never commit secrets. Keep them in gitignored `*.env` files. `.env.example` has names only.
