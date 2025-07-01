from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START , END
from langgraph.prebuilt import ToolNode

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import os
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a:int, b:int):
    """Add two numbers"""
    return a + b

tools = [add]

model = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-1.5-flash",
    temperature=0.2
).bind_tools(tools)


def model_call(state: AgentState):
    system_propmt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    respone = model.invoke([system_propmt] + state["messages"])
    return {"messages":[respone]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)

graph.add_node("our_model", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("our_model")
graph.add_conditional_edges(
    "our_model",
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    }
)

graph.add_edge("tool_node", "our_model")

app = graph.compile()

inputs = {"messages":[("user", "whats is 1 + 1, 78 + 87, 96 + 55")]}

result = app.invoke(inputs)

print(result["messages"][-1].content)





