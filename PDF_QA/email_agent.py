from typing import Annotated, Sequence, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from imaplib import IMAP4_SSL
import smtplib
from email.mime.text import MIMEText

from dotenv import load_dotenv
load_dotenv()
import os

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    last_tool_call_id: str  # NEW

@tool
def authenticate_email(username: str, password: str) -> str:
    """Authenticate to email."""
    username = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_PASSWORD")
    imap = IMAP4_SSL('imap.gmail.com')
    response = imap.login(username, password)
    if response[0] != 'OK':
        raise Exception(f"Failed to authenticate: {response[1]}")
    print(f"authenticated: {response[1]}")
    return f"Authenticated as {username}"

@tool
def get_email_list() -> str:
    """Get list of emails."""
    username = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_PASSWORD")
    imap = IMAP4_SSL('imap.gmail.com')
    imap.login(username, password)
    status, messages = imap.select('INBOX')
    print(f"select status: {status}, messages: {messages}")
    if status != 'OK':
        raise Exception(f"Failed to select inbox: {messages}")
    status, messages = imap.search(None, 'ALL')
    if status != 'OK':
        raise Exception(f"Failed to search: {messages}")
    print(f"not send to llm emails: {messages[0].decode('utf-8')}")
    return messages[0].decode('utf-8')

@tool
def get_email_content(email_id: str) -> str:
    """Fetch the subject and body of an email by its ID."""
    username = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_PASSWORD")
    imap = IMAP4_SSL('imap.gmail.com')
    imap.login(username, password)
    status, _ = imap.select('INBOX')
    print(f"select status: {status}")
    if status != 'OK':
        raise Exception(f"Failed to select inbox for content fetch.")
    typ, msg_data = imap.fetch(email_id, '(RFC822)')
    if typ != 'OK':
        raise Exception(f"Failed to fetch email: {msg_data}")
    import email
    msg = email.message_from_bytes(msg_data[0][1])
    subject = msg['subject']
    if msg.is_multipart():
        body = msg.get_payload(0).get_payload(decode=True).decode()
    else:
        body = msg.get_payload(decode=True).decode()
    print(f"not send to llm: {subject}\nBody: {body}")
    return f"Subject: {subject}\nBody: {body}"

@tool
def get_last_email() -> str:
    """Fetch the subject and body of the most recent email in the inbox."""
    username = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_PASSWORD")
    imap = IMAP4_SSL('imap.gmail.com')
    imap.login(username, password)
    status, _ = imap.select('INBOX')
    if status != 'OK':
        raise Exception("Failed to select inbox.")
    status, messages = imap.search(None, 'ALL')
    if status != 'OK':
        raise Exception(f"Failed to search: {messages}")
    email_ids = messages[0].split()
    if not email_ids:
        return "No emails found."
    last_id = email_ids[-1]
    typ, msg_data = imap.fetch(last_id, '(RFC822)')
    if typ != 'OK':
        raise Exception(f"Failed to fetch email: {msg_data}")
    import email
    msg = email.message_from_bytes(msg_data[0][1])
    subject = msg['subject']
    if msg.is_multipart():
        body = msg.get_payload(0).get_payload(decode=True).decode()
    else:
        body = msg.get_payload(decode=True).decode()
    return f"Subject: {subject}\nBody: {body}"

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient with the given subject and body."""
    username = os.getenv("EMAIL_USERNAME")
    password = os.getenv("EMAIL_PASSWORD")
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = username
    msg["To"] = recipient

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(username, password)
        server.sendmail(username, [recipient], msg.as_string())
        server.quit()
        return f"Email sent to {recipient}."
    except Exception as e:
        return f"Failed to send email: {e}"

# tools = [authenticate_email, get_email_list, get_email_content]
tools = [authenticate_email, get_email_list, get_email_content, get_last_email, send_email]

llm = ChatGoogleGenerativeAI(
    api_key= os.getenv("GOOGLE_API_KEY"),
    model= os.getenv("GOOGLE_MODEL"),
    temperature=0.2
).bind_tools(tools)

def process_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="""
        You are an intelligent email assistant.

        You have access to the following tools:

        - authenticate_email(): Authenticate to the user's email account using secure, pre-configured environment credentials. Do not ask the user for their email address or password.
        - get_email_list(): Retrieve a list of email IDs from the inbox.
        - get_email_content(email_id): Fetch the subject and body of a specific email by ID.
        - get_last_email(): Retrieve the subject and body of the most recent email.
        - send_email(recipient, subject, body): Send an email to a specified recipient. The body should be in HTML format.

        Instructions:

        1. Always begin by calling `authenticate_email()` before performing any email-related actions.
        2. Do **not** ask the user for login credentials — they are already stored securely.
        3. When the user gives you instructions like "email John confirming the meeting," you should:
        - Automatically write a clear subject line and an appropriate email body.
        - Format the email body in **HTML**.
        - Send the email via `send_email()`.
        4. If the task involves reading or responding to previous emails, use `get_last_email()` or `get_email_content(email_id)` first.
        5. Use tools **autonomously and only once per request**, unless the user explicitly asks to repeat something.
        6. After executing a tool, return a concise summary of the action taken and ask the user if they'd like to do anything else.
    """)

    user_prompt = HumanMessage(content=state["user_input"])
    messages = [system_prompt] + state["messages"] + [user_prompt]

    response = llm.invoke(messages)

    # Check if a tool was called
    tool_calls = getattr(response, "tool_calls", [])

    # Prevent duplicate tool call
    if tool_calls and state.get("last_tool_call_id") == tool_calls[0]["id"]:
        print("Skipping duplicate tool call.")
        return {
            "messages": messages + [AIMessage(content="Tool already executed. No further action needed.")],
            "user_input": state["user_input"],
            "last_tool_call_id": state.get("last_tool_call_id"),
        }

    return {
        "messages": messages + [response],
        "user_input": state["user_input"],
        "last_tool_call_id": tool_calls[0]["id"] if tool_calls else None
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
graph.add_node("tool_node", ToolNode(tools=tools))


graph.add_edge(START, "process_node")
graph.add_conditional_edges(
    "process_node",
    should_continue,
    {
        "continue": "tool_node",
        "end": END
    }
)
graph.add_edge("tool_node", "process_node")


app = graph.compile()

result = app.invoke({"user_input":"send an email to mchopan21@gmail.com, subject: hello, body: hello world snd it only once"})

# Print all messages in the conversation for debugging
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {getattr(msg, 'content', msg)}")