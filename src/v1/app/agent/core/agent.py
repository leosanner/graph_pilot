from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, SystemMessage

from v1.app.agent.config.settings import ModelSettings
from v1.app.agent.core.graph import build_graph, make_call_model
from v1.app.agent.core.llm import new_chat_model
from v1.app.agent.core.state import InputState, State
from v1.app.agent.rag.retrieval import Retrieval
from v1.app.agent.rag.tool import Config, make_search_tool

SYSTEM_PROMPT = (
    "You are a local assistant for the user's ingested documents. "
    "Call search_tool only when the user asks about that corpus. "
    "Greetings, thanks, and small talk get a short direct reply. "
    "Do not search and do not cite documents for those."
)


@dataclass
class AgentRetrievalConfig:
    retrieval: Retrieval
    embeddings: Embeddings
    config: Config


class Agent:
    def __init__(
        self,
        modelSettings: ModelSettings,
        retrievalConfig: AgentRetrievalConfig,
    ):
        self.search_tool = make_search_tool(
            retrievalConfig.retrieval,
            retrievalConfig.embeddings,
            retrievalConfig.config,
        )
        model = new_chat_model(modelSettings)
        call_model = make_call_model(model, self.search_tool)
        self.graph = build_graph(call_model, self.search_tool)
        self.messages = self._new_thread()

    def invoke(self, message: str) -> str:
        result = self.graph.invoke(self._input(message))
        return self._commit(result)

    async def ainvoke(self, message: str) -> str:
        result = await self.graph.ainvoke(self._input(message))
        return self._commit(result)

    def reset(self) -> None:
        self.messages = self._new_thread()

    def _new_thread(self) -> list:
        return [SystemMessage(content=SYSTEM_PROMPT)]

    def _input(self, message: str) -> InputState:
        return {"messages": [*self.messages, HumanMessage(content=message)]}

    def _commit(self, result: State) -> str:
        self.messages = list(result["messages"])
        content = result["messages"][-1].content
        return content if isinstance(content, str) else str(content)
