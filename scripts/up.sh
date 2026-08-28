#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing $1. Install it, then run this again." >&2
    exit 1
  fi
}

need uv
need docker
need ollama

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if ! ollama list >/dev/null 2>&1; then
  echo "Start Ollama, then run this again." >&2
  exit 1
fi

uv sync
docker compose up -d --wait postgres

uv run yoyo apply --database \
  "postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}"

ollama pull "${OLLAMA_EMBED_MODEL:-nomic-embed-text}"

exec uv run ragp1
