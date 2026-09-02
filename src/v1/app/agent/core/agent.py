from dataclasses import dataclass

from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage

from v1.app.agent.config.settings import ModelSettings
from v1.app.agent.core.graph import GraphConfig, build_graph, model_factory
from v1.app.agent.core.llm import new_chat_model
from v1.app.agent.core.state import InputState, State
from v1.app.agent.gateway.tool import make_gateway_tools
from v1.app.agent.rag.retrieval import Retrieval
from v1.app.agent.rag.tool import Config, make_search_tool

GATEWAY_PROMPT = (
    "You are the front door of a local assistant for the user's ingested "
    "documents. You do not search the corpus yourself. "
    "Call reply_to_user for greetings, thanks, small talk, or anything "
    "answerable without those documents; put the full reply in `response`. "
    "Call delegate_to_retrieval when the user asks about the ingested "
    "corpus or needs a fact that may be in their PDFs. "
    "Never answer a corpus question from memory. Never call both tools."
)

CORE_PROMPT = (
    "You are a local assistant answering from the user's ingested documents. "
    "Call search_tool to retrieve passages, then answer from those hits. "
    "If search returns nothing, say so. Do not invent document contents."
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
        gateway_tools = make_gateway_tools()
        core_tools = [
            make_search_tool(
                retrievalConfig.retrieval,
                retrievalConfig.embeddings,
                retrievalConfig.config,
            )
        ]
        model = new_chat_model(modelSettings)
        self.graph = build_graph(
            GraphConfig(
                gateway_model=model_factory(model, gateway_tools, GATEWAY_PROMPT),
                gateway_model_tools=gateway_tools,
                core_model=model_factory(model, core_tools, CORE_PROMPT),
                core_model_tools=core_tools,
            )
        )
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
        return []

    def _input(self, message: str) -> InputState:
        return {"messages": [*self.messages, HumanMessage(content=message)]}

    def _commit(self, result: State) -> str:
        self.messages = list(result["messages"])
        content = result["messages"][-1].content
        return content if isinstance(content, str) else str(content)
