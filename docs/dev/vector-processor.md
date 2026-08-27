# Vector processor

Code: `src/v1/app/ingestion/vector/processor.py` → `VectorProcessor.process`.

## Why it exists

A raw document is not retrievable. Retrieval needs bounded chunks, each with one embedding, plus enough metadata to know which model and splitter produced them. Without this step there is nothing to index.

## What it is for

Takes one LangChain `Document`, splits it, embeds the chunks in parallel batches, and returns a `ProcessedDocument`.

It does **not** persist vectors, choose the model, or derive `chunk_size` / `chunk_overlap`. Those come in through `VectorSettings`. Transient embed failures retry; a failed document raises `EmbeddingError` and is not partially returned.

## How it works

```mermaid
flowchart LR
  A["Document"] --> B["RecursiveCharacterTextSplitter"]
  B --> C["batches"]
  C --> D["aembed_documents"]
  D --> E["ProcessedDocument"]
```

Batches run concurrently under a semaphore. Positions stay contiguous across batches (`0 .. n-1`). Splitter settings and the source document metadata are copied onto the result.

## Example

### Input

```json
{
  "document": {
    "page_content": "hello world",
    "metadata": { "source": "doc.txt" }
  },
  "settings": {
    "chunk_size": 40,
    "chunk_overlap": 0,
    "model_name": "nomic-embed-text"
  }
}
```

### Expected output

```json
{
  "metadata": {
    "source": "doc.txt",
    "chunks_count": 1,
    "time_to_process": 0.04,
    "model": "nomic-embed-text",
    "splitter": {
      "chunk_size": 40,
      "chunk_overlap": 0
    }
  },
  "chunks": [
    {
      "position": 0,
      "content": "hello world",
      "embedding": [0.12, -0.03, 0.44]
    }
  ]
}
```

`time_to_process` is wall-clock seconds. `embedding` length must equal the model’s `embedding_dim`. The vector in this example is shortened.
