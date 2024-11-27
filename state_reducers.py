from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict
from langchain_core.messages import RemoveMessage
from typing import Annotated
from operator import add
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph # type
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

############################################################## R E DU C ER S #######################################################################

class State(TypedDict):
    foo: Annotated[list[int], add]

def node_1(state: State):
    print("---Node 1---")
    return {"foo": [state['foo'][-1] + 1]}

def node_2(state: State):
    print("---Node 2---")
    return {"foo": [state['foo'][-1] + 1]}

def node_3(state: State):
    print("---Node 3---")
    return {"foo": [state['foo'][-1] + 1]}

# Build graph
builder: StateGraph = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_1", "node_3")
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

# Add
graph: CompiledStateGraph = builder.compile()
result = graph.invoke({"foo" : [1]})
print(result)


############################################################## C U ST O M --- R E DU C ER S #######################################################################

def reduce_list(left: list | None, right: list | None) -> list:
    """Safely combine two lists, handling cases where either or both inputs might be None.

    Args:
        left (list | None): The first list to combine, or None.
        right (list | None): The second list to combine, or None.

    Returns:
        list: A new list containing all elements from both input lists.
               If an input is None, it's treated as an empty list.
    """
    if not left:
        left = []
    if not right:
        right = []
    return left + right

class DefaultState(TypedDict):
    foo: Annotated[list[int], add]

class CustomReducerState(TypedDict):
    foo: Annotated[list[int], reduce_list]

def node_1(state: DefaultState):
    print("---Node 1---")
    return {"foo": [2]}

# Build graph
builder: StateGraph = StateGraph(DefaultState)
builder.add_node("node_1", node_1)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# Add
graph: CompiledStateGraph = builder.compile()

try:
    print(graph.invoke({"foo" : None}))
except TypeError as e:
    print(f"TypeError occurred: {e}")

### The above will through an error here ###  ** TypeError occurred: can only concatenate list (not "NoneType") to list **

### Now, try with our custom reducer. We can see that no error is thrown.

def node_1(state: CustomReducerState):
    print("---Node 1---")
    return {"foo": [2]}

# Build graph
builder: StateGraph = StateGraph(CustomReducerState)
builder.add_node("node_1", node_1)

# Logic
builder.add_edge(START, "node_1")
builder.add_edge("node_1", END)

# Add
graph: CompiledStateGraph = builder.compile()


try:
    print(graph.invoke({"foo" : None}))
except TypeError as e:
    print(f"TypeError occurred: {e}")

################################################################### M E SS A GG E S #######################################################################

# Define a custom TypedDict that includes a list of messages with add_messages reducer
class CustomMessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    added_key_1: str
    added_key_2: str
    # etc

# Use MessagesState, which includes the messages key with add_messages reducer
class ExtendedMessagesState(MessagesState):
    # Add any keys needed beyond messages, which is pre-built
    added_key_1: str
    added_key_2: str
    # etc

# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance")
                   ]

# New message to add
new_message = AIMessage(content="Sure, I can help with that. What specifically are you interested in?", name="Model")

# Test
msg1 = add_messages(initial_messages , new_message)

print ("msg1 = ", msg1)



###### RE-Writing #######

# Initial state
initial_messages = [AIMessage(content="Hello! How can I assist you?", name="Model", id="1"),
                    HumanMessage(content="I'm looking for information on marine biology.", name="Lance", id="2")
                   ]

# New message to add
new_message = HumanMessage(content="I'm looking for information on whales, specifically", name="Lance", id="2")

# Test
msg2 = add_messages(initial_messages , new_message)

print ("msg2 = ", msg2)

##### REMOVAL #####

# Message list
messages = [AIMessage("Hi.", name="Bot", id="1")]
messages.append(HumanMessage("Hi.", name="Lance", id="2"))
messages.append(AIMessage("So you said you were researching ocean mammals?", name="Bot", id="3"))
messages.append(HumanMessage("Yes, I know about whales. But what others should I learn about?", name="Lance", id="4"))

# Isolate messages to delete
delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
print(delete_messages)
msg3 = add_messages(messages , delete_messages)

print("msg3 = ", msg3)




































######################################################## BE F O RE ------ R E DU C ER S #############################################################

# class State(TypedDict):
#     foo: int

# def node_1(state: State) -> dict:
#     print("---Node 1---")
#     return {"foo": state['foo'] + 1}

# def node_2(state: State) -> dict:
#     print("---Node 2---")
#     return {"foo": state['foo'] + 1}

# def node_3(state: State) -> dict:
#     print("---Node 3---")
#     return {"foo": state['foo'] + 1}

# # Build graph
# builder: StateGraph = StateGraph(State)
# builder.add_node("node_1", node_1)
# builder.add_node("node_2", node_2)
# builder.add_node("node_3", node_3)

# # Logic
# builder.add_edge(START, "node_1")
# builder.add_edge("node_1", "node_2")
# builder.add_edge("node_1", "node_3")
# builder.add_edge("node_2", END)
# builder.add_edge("node_3", END)

# # Add
# graph: CompiledStateGraph = builder.compile()

# try:
#     graph.invoke({"foo" : 1})
# except InvalidUpdateError as e:
#     print(f"InvalidUpdateError occurred: {e}")