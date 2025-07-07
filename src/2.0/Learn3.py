from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import os
load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str

@tool
def google_search(query: str):
    """Serach google for the query."""
    return "https://www.google.com/search?q=" + query
    
@tool
def stock_price(symbol: str):
    """Get the stock price for the symbol."""
    return 100.0

tools = [google_search, stock_price]

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model=os.getenv("GOOGLE_MODEL"),
    temperature=0.2,
).bind_tools(tools=tools)


def process_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
                                    you are a helpful assistant.
                                    you have access to the following tools:
                                    {tools}
                                  """
                                  )
    human_prompt = HumanMessage(content=state["user_input"])

    response = llm.invoke([system_prompt, human_prompt])

    return {
        "messages": state["messages"] + [human_prompt, response],
        "user_input": state["user_input"],
    }


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph_builder = StateGraph(AgentState)
graph_builder.add_node("process_node", process_node)
graph_builder.add_node("tool", ToolNode(tools=tools))

graph_builder.add_conditional_edges(
    "process_node",
    should_continue,
    {
        "continue": "tool",
        "end": END,
    }
)

graph_builder.add_edge("tool", "process_node")

graph_builder.add_edge(START, "process_node")

graph = graph_builder.compile()

user_input = input("You: ")

state = graph.invoke({"user_input": user_input})

print(state["messages"][-1].content)
