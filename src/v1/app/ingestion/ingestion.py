from langchain_core.documents import Document

from v1.app.ingestion.repository.vector import (
    ChunkEntity,
    DocumentEntity,
    VectorRepository,
)
from v1.app.ingestion.vector.processor import VectorProcessor


class DocumentIngestion:
    def __init__(self, vector_repo: VectorRepository):
        self.vec_repository = vector_repo

    async def ingest_vector(self, document: Document, vec_processor: VectorProcessor):

        result = await vec_processor.process(document)
        document_entity = DocumentEntity(
            name=document.metadata.get("name", "<not_provided>")
        )
        chunks_entites: list[ChunkEntity] = [
            ChunkEntity(
                document_id=document_entity.id,
                content=chunk.content,
                position=chunk.position,
                embedding=chunk.embedding,
                metadata=result.metadata,
            )
            for chunk in result.chunks
        ]

        self.vec_repository.insert_document(document_entity)
        self.vec_repository.insert_chunks(chunks_entites)

        return
