-- Runs only on the first initialization of the volume (docker-entrypoint-initdb.d).
-- pgvector: `vector` type + HNSW/IVFFlat indexes.
CREATE EXTENSION IF NOT EXISTS vector;
-- used by gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS pgcrypto;
