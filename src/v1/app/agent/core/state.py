from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class InputState(TypedDict):
    messages: list[AnyMessage]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
