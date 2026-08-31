import asyncio

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from v1.app.ingestion.vector.errors import EmbeddingError
from v1.app.ingestion.vector.processor import VectorProcessor
from v1.app.ingestion.vector.schemas import Chunk, ProcessedDocument
from v1.app.ingestion.vector.settings import RuntimeSettings, VectorSettings

LONG_TEXT = "alpha beta gamma delta epsilon zeta eta theta. " * 20


class ScriptedEmbeddings(Embeddings):
    """Deterministic embeddings with an optional per-call script. No `.model` attr."""

    def __init__(self, dim: int = 4, script: list[str] | None = None):
        self.dim = dim
        self.script = script or []
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        index = self.calls - 1
        action = self.script[index] if index < len(self.script) else "ok"

        if action == "fail":
            raise RuntimeError("simulated embedding failure")
        if action == "timeout":
            await asyncio.sleep(10)
        if action == "bad_len":
            return []

        return [[float(i)] * self.dim for i, _ in enumerate(texts)]


def make_processor(embeddings: Embeddings, **runtime) -> VectorProcessor:
    settings = VectorSettings(
        chunk_size=40,
        chunk_overlap=0,
        model=embeddings,
        model_name="test-embed",
    )
    defaults = {
        "batch_size": 2,
        "max_concurrent_batches": 2,
        "request_timeout": 1.0,
        "max_attempts": 3,
        "retry_base_delay": 0.01,
    }
    defaults.update(runtime)
    return VectorProcessor(settings, RuntimeSettings(**defaults))


async def test_process_embeds_document_and_records_settings_model_name():
    embeddings = ScriptedEmbeddings()
    processor = make_processor(embeddings, batch_size=8)

    result = await processor.process(
        Document(page_content="hello world", metadata={"source": "doc.txt"})
    )

    assert isinstance(result, ProcessedDocument)
    assert len(result.chunks) == 1
    assert isinstance(result.chunks[0], Chunk)
    assert result.chunks[0].content == "hello world"
    assert result.chunks[0].position == 0
    assert len(result.chunks[0].embedding) == 4
    assert result.metadata["source"] == "doc.txt"
    assert result.metadata["model"] == "test-embed"
    assert result.metadata["chunks_count"] == 1
    assert embeddings.calls == 1


async def test_process_assigns_contiguous_positions_across_batches():
    embeddings = ScriptedEmbeddings()
    processor = make_processor(embeddings, batch_size=2)

    result = await processor.process(
        Document(page_content=LONG_TEXT, metadata={"source": "long.txt"})
    )

    positions = [chunk.position for chunk in result.chunks]
    assert len(result.chunks) > 2
    assert embeddings.calls > 1
    assert positions == list(range(len(positions)))


async def test_process_retries_transient_embedding_failures():
    embeddings = ScriptedEmbeddings(script=["fail", "ok"])
    processor = make_processor(embeddings, max_attempts=3)

    result = await processor.process(
        Document(page_content="retry me", metadata={"source": "retry.txt"})
    )

    assert len(result.chunks) == 1
    assert embeddings.calls == 2


async def test_process_raises_when_retries_are_exhausted():
    embeddings = ScriptedEmbeddings(script=["fail", "fail", "fail"])
    processor = make_processor(embeddings, max_attempts=2)

    with pytest.raises(EmbeddingError, match=r"fail\.txt"):
        await processor.process(
            Document(page_content="always fail", metadata={"source": "fail.txt"})
        )


async def test_process_raises_when_embedding_count_mismatches():
    embeddings = ScriptedEmbeddings(script=["bad_len"])
    processor = make_processor(embeddings, max_attempts=1)

    with pytest.raises(EmbeddingError, match="not processed"):
        await processor.process(
            Document(page_content="hello world", metadata={"source": "badlen.txt"})
        )
