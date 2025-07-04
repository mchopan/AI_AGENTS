from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from dotenv import load_dotenv
import os

load_dotenv()

draft_content = ""

class AgentState(TypedDict):
    messages:  Annotated[Sequence[BaseMessage], add_messages]


@tool
def update_tool(content: str) -> str:
    """Updates the document with the provided content"""
    global draft_content
    draft_content = content
    return f"Document content updated successfully.\n{draft_content}"


@tool
def save_content(filename: str) -> str:
    """
    Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file
    """

    global draft_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"
    
    try:
        with open(filename, "w") as f:
            f.write(draft_content)
        return f"Document saved successfully as {filename}"
    except Exception as e:
        return f"Failed to save document: {e}"


tools = [update_tool, save_content]

llm = ChatGoogleGenerativeAI(
    api_key=os.environ["GOOGLE_API_KEY"],
    model=os.environ["GOOGLE_MODEL"],
    temperature=0.2
).bind_tools(tools=tools)


# def llm_call(state: AgentState) -> AgentState:
#     system_prompt = SystemMessage(content=f"""
#         You are a helpful AI assistant called Drafter AI.
#         Your job is to help users write clear and well-structured content such as notes, emails, blog posts, or reports.
#         Follow the user's instructions carefully and try to make the writing easy to understand.
#         Use correct grammar, organize ideas clearly, and keep the tone friendly and professional.
#         If the user doesn't give enough detail, ask questions before starting.
#         If they ask for improvements, edit the content without changing the original meaning too much.
#         Keep your answers short and focused unless the user asks for a long response.
#         the current document content is {draft_content}
#     """)

#     if not state["messages"]:
#         user_input = "i'm ready to help my you update the document. What would you like to create?"
#         user_message = HumanMessage(content=user_input)
        
#     else:
#         user_input = input("\nWhat would you like to do with the document?")
#         print(f"\n USER: {user_input}")
#         user_message = HumanMessage(content=user_input)

#     all_messages = [system_prompt] + list(state["messages"]) + [user_message]
    
#     # Pass the list of messages directly to the LLM
#     response = llm.invoke(all_messages)


#     print(f"\n AI: {response.content}")
#     if hasattr(response, "tool_calls") and response.tool_calls:
#         print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

#     return {"messages": list(state["messages"]) + [user_message, response]}

def llm_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
        You are a helpful AI assistant called Drafter AI.
        Your job is to help users write clear and well-structured content such as notes, emails, blog posts, or reports.
        Follow the user's instructions carefully and try to make the writing easy to understand.
        Use correct grammar, organize ideas clearly, and keep the tone friendly and professional.
        If the user doesn't give enough detail, ask questions before starting.
        If they ask for improvements, edit the content without changing the original meaning too much.
        Keep your answers short and focused unless the user asks for a long response.
        Save the document using the save_content tool when user asks to save the document.
        The current document content is: {draft_content}
    """)

    messages = list(state["messages"])

    # Determine if we should ask the user for input
    should_ask_user = False
    if not messages:
        should_ask_user = True
    else:
        last_message = messages[-1]
        if isinstance(last_message, ToolMessage):
            should_ask_user = True
        elif isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", []):
            should_ask_user = True

    if should_ask_user:
        user_input = input("\nWhat would you like to do with the document?\n> ")
        user_message = HumanMessage(content=user_input)
        print(f"\n USER: {user_input}")
        messages.append(user_message)

    all_messages = [system_prompt] + messages

    response = llm.invoke(all_messages)

    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": messages + [response]}




def should_continue(state:AgentState) -> str:
    """Determine if we should continue the conversation"""
    messages = state["messages"]
    if not messages:
        return "continue"
    
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return "end"
    
    return "continue"
  
def print_message(messages):
    """Function to print the messages in more readable format"""
    if not messages:
        return
    
    for message in messages:
        if isinstance(message, AIMessage):
            print(f"\n AI: {message.content}")
        elif isinstance(message, HumanMessage):
            print(f"\n USER: {message.content}")

graph  = StateGraph(AgentState)

graph.add_node("llm", llm_call)
graph.add_node("tool", ToolNode(tools=tools))

graph.set_entry_point("llm")

graph.add_edge("llm", "tool")

graph.add_conditional_edges(
    "tool",
    should_continue,
    {"continue": "llm", "end": END}
)

app = graph.compile()
def run_document_agent():
    print("===========================")
    print("Welcome to Drafter AI")
    print("===========================\n")
    
    print("Hi! Drafter AI can help you write or update documents like notes, blogs, and more.\n")

    state = {"messages":[]}
    
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_message(step["messages"])

    print("\n===========================")
    print("Thank you for using Drafter AI")
    print("===========================")



if __name__ == "__main__":
    run_document_agent()