# Chunk vector index

Code: `src/v1/infra/migrations/002.create_chunks.py` → `chunks_embedding_hnsw`.

## Why it exists

A sequential scan compares the query vector to every chunk. That does not scale. The HNSW index is a proximity graph: search walks nearby neighbors instead of the full table.

The algorithm is Hierarchical Navigable Small World graphs (Malkov & Yashunin): [arXiv:1603.09320](https://arxiv.org/abs/1603.09320).

## What it is for

Approximate nearest-neighbor lookup on `chunks.embedding` with cosine distance (`vector_cosine_ops`).

It does **not** persist rows (the table does that), cluster vectors on disk, or pick the embedding model. The column is `vector(768)` to match `nomic-embed-text`.

## How it works

```mermaid
flowchart LR
  A["query embedding"] --> B["HNSW graph"]
  B --> C["nearest chunks"]
```

Each vector is a node. Edges connect nearby neighbors. Search enters at a sparse upper layer (long hops) and descends to the full bottom layer.

## Example

### Input

```json
{
  "embedding": [0.12, -0.03, 0.44],
  "k": 2
}
```

### Expected output

```json
{
  "matches": [
    { "chunk_id": "…", "position": 0, "distance": 0.04 },
    { "chunk_id": "…", "position": 3, "distance": 0.11 }
  ]
}
```

`distance` is cosine distance (`<=>`). The vectors in this example are shortened. Retrieval is not wired yet; this is the shape the index is for.
