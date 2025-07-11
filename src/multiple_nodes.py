# -*- coding: utf-8 -*-
"""multiple_nodes.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11ZwmFBbUJdPnxhZrUjqUFp_O4pePq0qQ
"""

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
  number1: str
  operation: str
  number2: str
  finalNumber: str

def adder_node(state: AgentState) -> AgentState:
  """This node will add two numbers"""
  state["finalNumber"] = state["number1"] + state["number2"]
  return state

def subtractor_node(state: AgentState) -> AgentState:
  """This node will subtract two numbers"""
  state["finalNumber"] = state["number1"] - state["number2"]
  return state

def multiply_node(state: AgentState) -> AgentState:
  """This node will multiply two numbers"""
  state["finalNumber"] = state["number1"] * state["number2"]
  return state

def divide_node(state: AgentState) -> AgentState:
  """This node will divide two numbers"""
  state["finalNumber"] = state["number1"] / state["number2"]
  return state

def unknown_node(state: AgentState) -> AgentState:
  """This node will handle unknown operations"""
  state["finalNumber"] = "Unknown operation"
  return state

def decide_operation_node(state: AgentState):
  """This node will decide which operation to perform"""

  if state["operation"] == "+":
    return "add_operation"
  elif state["operation"] == "-":
    return "subtract_operation"
  elif state["operation"] == "*":
    return "multiply_operation"
  elif state["operation"] == "/":
    return "divide_operation"
  else:
    return "unknown_operation"

builder = StateGraph(AgentState)

builder.add_node("add_node", adder_node)
builder.add_node("subtract_node", subtractor_node)
builder.add_node("multiply_node", multiply_node)
builder.add_node("divide_node", divide_node)
builder.add_node("unknown_node", unknown_node)
builder.add_node("router", lambda state:state)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    decide_operation_node,
    {
        "add_operation":"add_node",
        "subtract_operation":"subtract_node",
        "multiply_operation":"multiply_node",
        "divide_operation":"divide_node",
        "unknown_operation":"unknown_node"
    }
)

builder.add_edge("add_node", END)
builder.add_edge("subtract_node", END)


app = builder.compile()

from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

initial_state = AgentState(number1=10, operation="*", number2=5)
result = app.invoke(initial_state)
print(result['finalNumber'])