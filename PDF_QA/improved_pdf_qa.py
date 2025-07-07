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
        return text if text.strip() else "No text found in the PDF file."
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

tools = [extract_text_from_pdf]

llm = ChatGoogleGenerativeAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model=os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),  # Default fallback
    temperature=0.2
).bind_tools(tools)

def process_node(state: AgentState) -> AgentState:
    """Process the user's question and determine if tools are needed."""
    system_prompt = SystemMessage(content="""
You are a helpful assistant that answers questions about PDFs. 
When given a file path and a question, use the extract_text_from_pdf tool to extract text from the PDF first.
Then answer the user's question based on the extracted text.
Be specific and cite relevant parts of the text when possible.
""")
    
    user_prompt = HumanMessage(content=f"""
Please help me answer this question about a PDF file:
Question: {state["user_input"]}
File path: {state["file"]}

First, extract the text from the PDF, then answer the question based on the content.
""")

    # Get existing messages or start fresh
    messages = state.get("messages", [])
    
    # Only add system and user messages if this is the first call
    if not messages or not any(isinstance(msg, SystemMessage) for msg in messages):
        messages_to_send = [system_prompt, user_prompt]
    else:
        # If we already have messages, just send the last few for context
        messages_to_send = messages[-5:]  # Keep last 5 messages for context
    
    response = llm.invoke(messages_to_send)
    
    # Return updated state
    return {
        "messages": messages + [response] if messages else [system_prompt, user_prompt, response],
        "file": state["file"],
        "user_input": state["user_input"],
    }

def should_continue(state: AgentState):
    """Determine if we should continue to tool execution or end."""
    messages = state["messages"]
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "continue"
    else:
        return "end"

def tool_node(state: AgentState) -> AgentState:
    """Execute tools and return results."""
    tool_executor = ToolNode(tools)
    result = tool_executor.invoke(state)
    return result

# Create the graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("process_node", process_node)
graph.add_node("toolcall", tool_node)

# Add conditional edges
graph.add_conditional_edges(
    "process_node",
    should_continue,
    {
        "continue": "toolcall",
        "end": END,
    } 
)

# Add edges
graph.add_edge(START, "process_node")
graph.add_edge("toolcall", "process_node")

# Compile the graph
app = graph.compile()

def main():
    """Main function to run the PDF Q&A agent."""
    user_input = "What is this document about?"
    file_path = "PDF_QA/Project Phoenix.pdf"  # Fixed path separator
    
    try:
        # Invoke the agent
        state = app.invoke({
            "messages": [], 
            "file": file_path, 
            "user_input": user_input
        })
        
        # Print the final response
        if state["messages"]:
            final_message = state["messages"][-1]
            print("Final Answer:")
            print(final_message.content)
        else:
            print("No response generated.")
            
    except Exception as e:
        print(f"Error running agent: {e}")

if __name__ == "__main__":
    main()