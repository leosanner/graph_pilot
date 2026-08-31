from yoyo import step

__depends__ = {"001.create_documents"}

steps = [
    step(
        """
    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE chunks (
      id UUID PRIMARY KEY,
      document_id UUID NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
      content TEXT NOT NULL,
      position INTEGER NOT NULL,
      embedding vector(768) NOT NULL,
      metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
      UNIQUE (document_id, position)
    );

    CREATE INDEX chunks_embedding_hnsw
      ON chunks USING hnsw (embedding vector_cosine_ops)
    """,
        "DROP TABLE chunks",
    )
]
