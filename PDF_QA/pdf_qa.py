from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from dotenv import load_dotenv

import os
import fitz

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    file: str | None
    user_input: str | None

@tool
def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts all text from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        str: The full extracted text from the PDF.
    """
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

tools = [extract_text_from_pdf]

llm = ChatGoogleGenerativeAI(
    api_key= os.getenv("GOOGLE_API_KEY"),
    model= os.getenv("GOOGLE_MODEL"),
    temperature=0.2
).bind_tools(tools)


def process_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
                                  You are a helpful assistant that answers questions about PDFs.Use the tools available to you to answer the user's questions.
                                  use the extract_text_from_pdf tool to extract text from the PDF. it will return the text from the PDF.
                                  """)
    user_propmt = HumanMessage(content=f"""
                               user_input = {state["user_input"]}
                               file_path = {state["file"]}
                               """)

    response  = llm.invoke([system_prompt, user_propmt])

    return {
        "messages": [system_prompt, user_propmt, response],
        "file": state["file"],
        "user_input": state["user_input"],
    }
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
     
graph = StateGraph(AgentState)

graph.add_node("process_node", process_node)
graph.add_node("toolcall", ToolNode(tools))

graph.add_conditional_edges(
    "process_node",
    should_continue,
    {
        "continue": "toolcall",
        "end": END,
    } 
)

graph.add_edge(START, "process_node")
graph.add_edge("toolcall", "process_node")

app = graph.compile()

user_input = "What is the name of the author of the book?"
file_path = "PDF_QA\\Project Phoenix.pdf"

state = app.invoke({"messages": [], "file": file_path, "user_input": user_input})
print(state["messages"][-1].content)
