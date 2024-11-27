from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from typing_extensions import Literal
from typing import Annotated, Sequence
from pydantic import BaseModel, field_validator, ValidationError
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph # type
from dotenv import load_dotenv
import operator
import random
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)


class PydanticState(BaseModel):
    name: str
    mood: Literal["happy", "sad"]

    @field_validator('mood')
    @classmethod
    def validate_mood(cls, value):
        # Ensure the mood is either "happy" or "sad"
        if value not in ["happy", "sad"]:
            raise ValueError("Each mood must be either 'happy' or 'sad'")
        return value

# try:
#     state = PydanticState(name="John Doe", mood="happy")
# except ValidationError as e:
#     print("Validation Error:", e)

# GRAPH

def node_1(state: PydanticState):
    print("---Node 1---")
    return {"name": state.name + " is ... "}

def node_2(state: PydanticState):
    print("---Node 2---")
    return {"mood": "happy"}

def node_3(state: PydanticState):
    print("---Node 3---")
    return {"mood": "sad"}

def decide_mood(state: PydanticState) -> Literal["node_2", "node_3"]:

    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"

    # 50% of the time, we return Node 3
    return "node_3"

# Build graph
builder: StateGraph = StateGraph(PydanticState)
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


result = graph.invoke(PydanticState(name="Lance",mood="sad"))

print(result)


# A Pydantic React Agent


@tool
def search(query: str) -> str:
    """Call to surf the web."""
    # This is a placeholder for the actual implementation
    # Don't let the LLM know this though 😊
    return "The answer to your question lies within."

tools = [search]

tool_node: ToolNode = ToolNode(tools)

model: ChatGoogleGenerativeAI = llm.bind_tools(tools)

class AgentState(BaseModel):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# Define the function that determines whether to continue or not
def should_continue(state: AgentState) -> Literal["end", "continue"]:
    messages = state.messages
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state: AgentState):
    messages = state.messages
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# Define a new graph
workflow: StateGraph = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app: CompiledStateGraph = workflow.compile()

inputs = {"messages": [HumanMessage(content="what is the weather in Lahore")]}
for chunk in app.stream(inputs, stream_mode="values"):
    chunk["messages"][-1].pretty_print()