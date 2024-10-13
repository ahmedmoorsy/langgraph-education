from typing import Sequence, Annotated, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages

# Define the education state
class EducationState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    current_supervisor: str
    role: Literal["teacher", "student"]