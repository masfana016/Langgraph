from langchain_google_genai import GoogleGenerativeAI
# from IPython.display import Image, display # Preview Graph
# from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph # type
from dotenv import load_dotenv
from typing import Literal
import os
import random

random.random()

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


class State(TypedDict):
    user_input: str

def node_1(state: State) -> State:
    print("---Node 1---", state)
    return {"user_input": state['user_input'] +" I am"}

def node_2(state: State) -> State:
    print("---Node 2---", state)
    return {"user_input": state['user_input'] +" happy!"}

def node_3(state: State) -> State:
    print("---Node 3---", state)
    return {"user_input": state['user_input'] +" sad!"}

def decide_mood(state: State) -> Literal["node_2", "node_3"]:
    user_input = state['user_input'] 
    if random.random() < 0.5:
        return "node_2"
    return "node_3"

# Build graph
builder: StateGraph = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph: CompiledStateGraph = builder.compile()

output = graph.invoke({"user_input" : "Hi, this is Lance."})

print(output)