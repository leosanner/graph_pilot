from yoyo import step

__depends__ = {}

steps = [
  step(
    """
    CREATE TABLE documents (
      id UUID PRIMARY KEY,
      name TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ
    )
    """,
    "DROP TABLE documents",
  )
]
