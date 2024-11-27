from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
# from IPython.display import Image, display # Preview Graph
# from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph # type
from dotenv import load_dotenv
import os

load_dotenv()

llm = GoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# STATE of the GRAPH
class LearningState(TypedDict):
    prompt: str

# prompt: create an example from above LearningState
lahore_state: LearningState = LearningState(prompt= "hello from UMT Lahore")


# NODES of GRAPH
def node_1(state: LearningState) -> LearningState:
    print("---Node 1 State---", state)
    return {"prompt": state['prompt'] +" I am"}

def node_2(state: LearningState) -> LearningState:
    print("---Node 2 State---", state)
    return {"prompt": state['prompt'] +" happy!"}

# Graph Construction

# Build graph
builder: StateGraph = StateGraph(state_schema=LearningState)

# Nodes
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)

# Simples Edges Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", END)

# Add
graph: CompiledStateGraph = builder.compile()

lang = graph.invoke({"prompt" : "Hi"})

print(lang)

# print(graph)

# View
# display(Image(graph.get_graph().draw_mermaid_png()))
