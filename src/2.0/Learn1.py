from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class AgentState(TypedDict):
    name: str

def greeting_user(state: AgentState) -> AgentState:
    state["name"] = f"Hello {state['name']}!"
    return state

graph = StateGraph(AgentState)

graph.add_node("greeting_user", greeting_user)

graph.add_edge(START, "greeting_user")
graph.add_edge("greeting_user", END)


app = graph.compile()

result = app.invoke({"name":"Manzoor"})
print(result["name"])