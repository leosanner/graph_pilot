from v1.app.ingestion.vector.processor import VectorProcessor
from langchain_core.documents import Document

class DocumentIngestion:
  def __init__(self):
    pass

  async def ingest_vector(
      self,
      document: Document,
      vec_processor: VectorProcessor
  ):

    result = await vec_processor.process(document)
    

