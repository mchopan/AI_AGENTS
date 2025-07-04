import json
from typing import TypedDict, Annotated, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph,START, END

from pydantic import BaseModel

from dotenv import load_dotenv
import os
load_dotenv()


class Superhero(BaseModel):
    code_name: str
    description: str


class AgentState(TypedDict):
    name: str
    role: str
    goal: str
    state: str
    code_name: str
    description: str



llm = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model=os.getenv("GOOGLE_MODEL"),
    temperature=1,
)

def process_node(state: AgentState) -> AgentState:
    
    system_prompt = SystemMessage(content="""
        You are a creative assistant.

        Given the user's name, role, goal, and current state,
        generate a superhero character with a codename and description.
           
        🔧 Respond in strict JSON format like this:                       
        {
            "code_name": "<superhero_codename>",
            "description": "<brief description of the superhero's powers and mission>"
        }
    """)

    user_prompt = HumanMessage(content=f"""
        Name: {state['name']}
        Role: {state['role']}
        Goal: {state['goal']}
        State: {state['state']}
    """)

    messages = [system_prompt, user_prompt]

    response = llm.invoke(messages)


    if response.content.startswith("```json"):
        response.content = response.content[7:]

    if response.content.endswith("```"):
        response.content = response.content[:-3]

    try:
        superhero = json.loads(response.content.strip())
        code_name = superhero['code_name']
        description = superhero['description']
    except Exception as e:
        print("Error parsing LLM response:", e)
        code_name = "Superman"
        description = "A superhero with superhuman strength and abilities."
    
    return {
        "name": state['name'],
        "role": state['role'],
        "goal": state['goal'],
        "state": state['state'],
        "code_name": code_name,
        "description": description
    }

graph =  StateGraph(AgentState)

graph = graph.add_node("process_node", process_node)

graph.add_edge(START, "process_node")
graph.add_edge("process_node", END)

app = graph.compile()

user_input = {
    "name": "John",
    "role": "Engineer",
    "goal": "Build a better world",
    "state": "Starting"
}

result = app.invoke(user_input)

print(result["code_name"])
print("----------------------------------------------")
print(result["description"])
