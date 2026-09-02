from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END

from v1.app.agent.core.graph import (
    GraphConfig,
    build_graph,
    gateway_tools_condition,
)
from v1.app.agent.gateway.tool import (
    DELEGATE_TO_RETRIEVAL,
    REPLY_TO_USER,
    make_gateway_tools,
)


def _state(*messages) -> dict:
    return {"messages": list(messages)}


def test_reply_to_user_returns_the_response():
    reply = next(t for t in make_gateway_tools() if t.name == REPLY_TO_USER)
    assert reply.invoke({"response": "Hello there."}) == "Hello there."


def test_gateway_tools_end_after_a_direct_reply():
    assert (
        gateway_tools_condition(
            _state(
                HumanMessage(content="hi"),
                AIMessage(content="", tool_calls=[]),
                ToolMessage(
                    content="Hello.",
                    name=REPLY_TO_USER,
                    tool_call_id="1",
                ),
            )
        )
        == END
    )


def test_gateway_tools_hand_off_to_retrieval():
    assert (
        gateway_tools_condition(
            _state(
                HumanMessage(content="notice period?"),
                AIMessage(content="", tool_calls=[]),
                ToolMessage(
                    content="Continuing with document retrieval.",
                    name=DELEGATE_TO_RETRIEVAL,
                    tool_call_id="1",
                ),
            )
        )
        == "model"
    )


def test_gateway_tools_prefer_retrieval_if_both_ran():
    assert (
        gateway_tools_condition(
            _state(
                ToolMessage(content="hi", name=REPLY_TO_USER, tool_call_id="1"),
                ToolMessage(
                    content="go",
                    name=DELEGATE_TO_RETRIEVAL,
                    tool_call_id="2",
                ),
            )
        )
        == "model"
    )


def test_unknown_gateway_tool_loops_back():
    assert (
        gateway_tools_condition(
            _state(ToolMessage(content="ok", name="classify", tool_call_id="1"))
        )
        == "gateway"
    )


def test_compiled_graph_starts_at_the_gateway():
    @tool
    def search_tool(query: str) -> str:
        """Search the corpus."""
        return query

    graph = build_graph(
        GraphConfig(
            gateway_model=lambda state: state,
            gateway_model_tools=make_gateway_tools(),
            core_model=lambda state: state,
            core_model_tools=[search_tool],
        )
    )
    mermaid = graph.get_graph().draw_mermaid()
    assert "__start__ --> gateway;" in mermaid
    assert "gateway -. &nbsp;tools&nbsp; .-> gateway_tools;" in mermaid
    assert "gateway -.-> __end__;" in mermaid
    assert "gateway_tools -.-> model;" in mermaid
    assert "gateway_tools -.-> __end__;" in mermaid
    assert "gateway_tools -.-> gateway;" in mermaid
    assert "model -.-> tools;" in mermaid
    assert "tools --> model;" in mermaid
    assert "__start__ --> model;" not in mermaid
