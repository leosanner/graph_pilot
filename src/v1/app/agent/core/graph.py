from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from v1.app.agent.core.state import InputState, State
from v1.app.agent.gateway.tool import DELEGATE_TO_RETRIEVAL, REPLY_TO_USER


@dataclass
class GraphConfig:
    gateway_model: Callable[[State], dict]
    gateway_model_tools: list[BaseTool]
    core_model: Callable[[State], dict]
    core_model_tools: list[BaseTool]


def model_factory(
    model: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
) -> Callable:
    llm = model.bind_tools(tools)

    def call_model(state: State) -> dict:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = llm.invoke(messages)
        return {"messages": [response]}

    return call_model


def _trailing_tool_names(state: State) -> list[str]:
    names: list[str] = []
    for message in reversed(state["messages"]):
        if not isinstance(message, ToolMessage):
            break
        names.append(message.name or "")
    names.reverse()
    return names


def gateway_tools_condition(state: State) -> str:
    names = _trailing_tool_names(state)
    if DELEGATE_TO_RETRIEVAL in names:
        return "model"
    if REPLY_TO_USER in names:
        return END
    return "gateway"


def build_graph(
    graph_config: GraphConfig,
) -> CompiledStateGraph[State, None, InputState, State]:
    builder = StateGraph(
        input_schema=InputState, state_schema=State, output_schema=State
    )

    builder.add_node("gateway", graph_config.gateway_model)
    builder.add_node("gateway_tools", ToolNode(graph_config.gateway_model_tools))
    builder.add_node("model", graph_config.core_model)
    builder.add_node("tools", ToolNode(graph_config.core_model_tools))

    builder.add_edge(START, "gateway")
    builder.add_conditional_edges(
        "gateway",
        tools_condition,
        {"tools": "gateway_tools", "__end__": END},
    )
    builder.add_conditional_edges(
        "gateway_tools",
        gateway_tools_condition,
        {"model": "model", "gateway": "gateway", "__end__": END},
    )
    builder.add_conditional_edges(
        "model",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )
    builder.add_edge("tools", "model")

    return builder.compile()
