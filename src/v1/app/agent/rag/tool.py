from langchain_core.tools import tool
from v1.app.agent.rag.retrieval import Retrieval, VectorSearch
from langchain_core.embeddings import Embeddings
from dataclasses import dataclass

@dataclass
class Config:
  top_k: int


def make_search_tool(
    retrieval: Retrieval,
    embeddings:Embeddings,
    config:Config
):

  @tool
  def search_tool(query:str) -> str:
    """Retrieve passages from the local ingested document collection.

    Use this whenever the answer may be in the user's PDFs or other
    ingested files. Search is semantic similarity over document chunks,
    not keyword matching, not SQL, and not the web. Call it before
    answering factual questions about that corpus; do not call it for
    small talk or for questions that only need the conversation so far.

    Args:
        query: A standalone natural-language search query. Restate the
            user's information need with the key entities, terms, and
            constraints. Do not pass chat filler, tool instructions, or
            the full conversation history.

    Returns:
        The closest matching chunk texts, separated by blank lines. An
        empty string means nothing was indexed or nothing matched.
    """

    emb = embeddings.embed_query(query)
    hits = retrieval.vector_search(
      VectorSearch(
        embedding = emb,
        top_k = config.top_k
      )
    )

    return "\n\n".join(h.content for h in hits)

  return search_tool
