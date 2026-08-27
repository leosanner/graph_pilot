import time
import asyncio
from langchain_core.documents import Document
from v1.app.ingestion.vector.errors import EmbeddingError
from v1.app.ingestion.vector.settings import VectorSettings, RuntimeSettings
from v1.app.ingestion.vector.schemas import ProcessedDocument, Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

class VectorProcessor:
  def __init__(self, settings:VectorSettings, runtime_settings:RuntimeSettings):
    self.splitter = RecursiveCharacterTextSplitter(
      chunk_size = settings.chunk_size,
      chunk_overlap = settings.chunk_overlap
    )
    self.embeddings = settings.model
    self.runtime = runtime_settings
    self.semaphore = asyncio.Semaphore(runtime_settings.max_concurrent_batches)


  async def process(self, document:Document) -> ProcessedDocument:
    start = time.perf_counter()
    chunks = self.splitter.split_documents([document])

    batches = [
      (offset, chunks[offset: offset + self.runtime.batch_size])
      for offset in range(0, len(chunks), self.runtime.batch_size)
    ]

    try:
      async with asyncio.TaskGroup() as tg:
        tasks = [
          tg.create_task(self.process_batch(offset, batch))
          for offset, batch in batches
        ]

    except Exception as e:
      source = document.metadata.get("source", "<no_source>")
      raise EmbeddingError(f"Document {source} not processed")

    results = [chunk for task in tasks for chunk in task.result()]

    metadata = {
      "chunks_count": len(chunks),
      "time_to_process": time.perf_counter() - start,
      "model": self.embeddings.model,
      "splitter": {
        "chunk_size": self.splitter._chunk_size,
        "chunk_overlap": self.splitter._chunk_overlap,
      },
      **document.metadata
    }

    return ProcessedDocument(
      metadata = metadata,
      chunks = results
    )

  async def process_batch(self, offset:int, batch:list[Document]) -> list[Chunk]:
    texts = [document.page_content for document in batch]
    async with self.semaphore:
      vectors = await self.embeddings.aembed_documents(texts)

    if len(vectors) != len(texts):
      raise EmbeddingError(f"Expected {len(texts)}, received {len(vectors)}")

    return [
      Chunk(embedding=v, position= offset + i, content= texts[i])
      for i, v in enumerate(vectors)
    ]
