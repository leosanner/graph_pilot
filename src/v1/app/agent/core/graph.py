from langgraph.graph.state import CompiledStateGraph
from v1.app.agent.core.state import State, InputState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

def make_call_model(model, search_tool):
  llm = model.bind_tools([search_tool])

  def call_model(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

  return call_model

def build_graph(call_model, search_tool) -> CompiledStateGraph[State, None, InputState, State]:
  builder = StateGraph(
    input_schema = InputState,
    state_schema = State,
    output_schema = State
  )

  builder.add_node("model", call_model)
  builder.add_node("tools", ToolNode([search_tool]))

  builder.add_edge(START, "model")
  builder.add_conditional_edges("model", tools_condition)
  builder.add_edge("tools", "model")

  return builder.compile()
