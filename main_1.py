from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage
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
class FirstLLMAgentCall(TypedDict):
    prompt: str
    output: str

# NODES of GRAPH
def node_1(state: FirstLLMAgentCall):
    print("---Node 1---", state)
    prompt = state["prompt"]
    ai_msg: AIMessage = llm.invoke(prompt)
    return {"output": ai_msg}

# Graph Construction

# Build graph
builder: StateGraph = StateGraph(state_schema=FirstLLMAgentCall)

# Nodes
builder.add_node("node_1", node_1)

# Add Edges Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# Compile Graph
graph: CompiledStateGraph = builder.compile()

result = graph.invoke({"prompt" : "Motivate me to learn LangGraph"})

print(result)

# print(graph)

# View
# display(Image(graph.get_graph().draw_mermaid_png()))
