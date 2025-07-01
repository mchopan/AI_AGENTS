from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,  START, END
from typing import List, TypedDict, Union
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages:List[Union[HumanMessage, AIMessage]]

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model="gemini-1.5-flash",
    temperature=0.2
)

def process_node(state: AgentState) -> AgentState:
    """This node will solve the input provided by the user."""
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    print(f"Agent: {response.content}")
    return state

builder = StateGraph(AgentState)
builder.add_node("process_node", process_node)
builder.add_edge(START, "process_node")
builder.add_edge("process_node", END)
graph = builder.compile()

conversion_history = []

user_input = input("User: ")

while user_input != "exit":
    conversion_history.append(HumanMessage(content=user_input))
    result = graph.invoke({"messages": conversion_history}) 
    conversion_history=result["messages"]
    user_input = input("User: ")


with open("log.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    for message in conversion_history:
        if isinstance(message, HumanMessage):
            file.write(f"User: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"Agent: {message.content}\n\n")
    file.write("End of Conversation Log\n")

print("Conversation log saved to log.txt")
