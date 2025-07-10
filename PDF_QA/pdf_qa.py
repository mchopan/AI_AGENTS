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
        return f"Error reading PDF: {e}"  
    return text


@tool
def word_count(text: str) -> int:
    """
    Counts the number of words in a given text.

    Args:
        text (str): The text to count words in.

    Returns:
        int: The number of words in the text.
    """
    words = text.split()
    return len(words)

tools = [extract_text_from_pdf, word_count]

llm = ChatGoogleGenerativeAI(
    api_key= os.getenv("GOOGLE_API_KEY"),
    model= os.getenv("GOOGLE_MODEL"),
    temperature=0.2
).bind_tools(tools)


def process_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
                  You are a helpful assistant that answers questions about PDFs.
                    You have access to these tools:
                    - `extract_text_from_pdf(file_path: str) -> str`: Extracts text from a PDF file.
                    - `word_count(text: str) -> int`: Returns the number of words in a given text.

                    Use them together when necessary to answer the user's question.
                    """)
    
    # Only add system/user prompts at the 
    messages = state["messages"]
    if len(messages) == 0:
        messages = [
            system_prompt,
            HumanMessage(content=f"""
            Please answer this question: {state["user_input"]}
            Use the extract_text_from_pdf tool with this file path: {state["file"]}
            """)
        ]

    response = llm.invoke(messages)
    print("🔍 Tool Calls:", getattr(response, "tool_calls", None))
    return {
        "messages": messages + [response],
        "file": state["file"],
        "user_input": state["user_input"],
    }
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    else:
        return "end"
     
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

# user_input = input("Enter your question: ")
# file_path = "PDF_QA\\Project Phoenix.pdf"

# state = app.invoke({"messages": [], "file": file_path, "user_input": user_input})
# print(state["messages"][-1].content)


# Initial state
file_path = "PDF_QA\\Project Phoenix.pdf"
messages = []
file = file_path

while True:
    user_input = input("\nAsk a question about the PDF (or type 'exit'): ")
    if user_input.lower() == "exit":
        break

    state = app.invoke(
        {
            "messages": messages,
            "file": file,
            "user_input": user_input
        },
    )

    # print(state)
    # Get last LLM response
    last_response = state["messages"][-1]
    print(f"\n🤖 {last_response.content}")

    # Preserve the updated state
    messages = state["messages"]