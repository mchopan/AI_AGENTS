# -*- coding: utf-8 -*-
"""langgraph1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14nHRbKgjlbToIlEzcHNMOqlO350QUSC1
"""

from typing import Dict, TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
  message : str

def greeting_node(state:AgentState) -> AgentState:
  """Simple node that adds greeting message to the state"""

  state["message"] =  f"Hey {state['message']}, how is your day going?"

  return state

graph = StateGraph(AgentState)

graph.add_node("greeter", greeting_node)
graph.set_entry_point("greeter")
graph.set_finish_point("greeter")

app = graph.compile()

from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

result = app.invoke({"message":"Manzoor"})

result["message"]