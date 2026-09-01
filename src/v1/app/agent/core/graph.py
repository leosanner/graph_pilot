from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from v1.app.agent.core.state import InputState, State


@dataclass
class GraphConfig:
    core_model: Callable[[State], dict]
    tools: list[BaseTool]


def model_factory(model: BaseChatModel, tools: list[BaseTool]) -> Callable:
    llm = model.bind_tools(tools)

    def call_model(state: State) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    return call_model


def build_graph(
    graph_config: GraphConfig,
) -> CompiledStateGraph[State, None, InputState, State]:
    builder = StateGraph(
        input_schema=InputState, state_schema=State, output_schema=State
    )

    builder.add_node("model", graph_config.core_model)
    builder.add_node("tools", ToolNode(graph_config.tools))

    builder.add_edge(START, "model")
    builder.add_conditional_edges("model", tools_condition)
    builder.add_edge("tools", "model")

    return builder.compile()
