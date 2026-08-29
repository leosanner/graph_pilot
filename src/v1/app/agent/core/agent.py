from v1.app.agent.rag.retrieval import Retrieval
from v1.app.agent.rag.tool import make_search_tool
from langchain_core.embeddings import Embeddings
from v1.app.agent.core.graph import build_graph
from dataclasses import dataclass


@dataclass
class AgentRetrievalConfig:
  retrieval: Retrieval
  embeddings: Embeddings


class Agent:
  def __init__(self, retrievalConfig: AgentRetrievalConfig):
    self.search_tool = make_search_tool(
      retrievalConfig.retrieval,
      retrievalConfig.embeddings
    )
    self.graph = build_graph()

