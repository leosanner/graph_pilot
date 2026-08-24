from langgraph.graph.state import CompiledStateGraph
from v1.app.agent.core.state import State, InputState
from langgraph.graph import StateGraph

def build_graph() -> CompiledStateGraph[State, None, InputState, State]:
  builder = StateGraph(
    input_schema = InputState,
    state_schema = State,
    output_schema = State
  )

  return builder.compile()
