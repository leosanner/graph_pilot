from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool

from v1.app.agent.rag.retrieval import QueryResult, Retrieval, VectorSearch


@dataclass
class Config:
    top_k: int


def make_search_tool(retrieval: Retrieval, embeddings: Embeddings, config: Config):

    @tool(response_format="content_and_artifact")
    def search_tool(query: str) -> tuple[str, list[QueryResult]]:
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
            The closest matching chunk texts, separated by blank lines. If
            nothing was indexed or nothing matched, a short message saying
            so, with permission to retry once or inform the user.
        """

        emb = embeddings.embed_query(query)
        hits = retrieval.vector_search(VectorSearch(embedding=emb, top_k=config.top_k))

        if not hits:
            return (
                "No matching passages were found in the ingested documents. "
                "Do not answer as if the corpus contained this information. "
                "You may retry once with a more specific standalone query, "
                "or inform the user.",
                [],
            )

        return "\n\n".join(h.content for h in hits), hits

    return search_tool
