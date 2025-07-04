from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages:Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a: int, b: int):
    """Add two numbers together."""
    return a + b
def subtract(a: int, b: int):
    """Subtract two numbers."""
    return a - b
def multiply(a: int, b: int):
    """Multiply two numbers."""
    return a * b
def divide(a: int, b: int):
    """Divide two numbers."""
    return a / b

tools = [add, subtract, multiply, divide]

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model=os.getenv("GOOGLE_MODEL"),
    temperature=0.2,
).bind_tools(tools)

def llm_call(state: AgentState):
    system_prompt = SystemMessage(content="You are a helpful assistant.")
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"

graph = StateGraph(AgentState)

graph.add_node("llm_call", llm_call)

tool_node = ToolNode(tools=tools)

graph.add_node("tool_node", tool_node)

graph.set_entry_point("llm_call")

graph.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "continue": "tool_node",
        "end": END,
    }
)

graph.add_edge("tool_node", "llm_call")

app = graph.compile()

# User input
user_input = {"messages": [("user","devide 10 by 2. Then multiply by 3. Then add 5. And after evaluate tell me joke")]}

# You can still get the final result with invoke() if you need it at the end
result = app.invoke(user_input)
print(result["messages"][-1].content)
