from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START,END

# Define the state for user data
class UserState(TypedDict):
    is_premium: bool
    message: str
# Define Nodes
def greet_user(state: UserState):
    state["message"] = "Welcome "
    return state

def premium_greeting(state: UserState):
    state["message"] += "Thank you for being a premium user"
    return state
def regular_greeting(state: UserState):
    state["message"] += "Enjoy your time here"
    return state

# Define a decision node to choose the path based on user type
def check_subscription(state: UserState):
    if state["is_premium"]:
        return "premium_greeting"
    else:
        return "regular_greeting"
# Building the graph
graph_builder = StateGraph(UserState)
graph_builder.add_node("greet_user", greet_user)
graph_builder.add_node("premium_greeting", premium_greeting)
graph_builder.add_node("regular_greeting", regular_greeting)
graph_builder.add_node("check_subscription", check_subscription)

# Add edges to control the flow
graph_builder.add_edge(START, "greet_user")
graph_builder.add_conditional_edges("greet_user", check_subscription)
graph_builder.add_edge("premium_greeting", END)
graph_builder.add_edge("regular_greeting", END)

# Compile and run the graph for a premium user
graph = graph_builder.compile()
result = graph.invoke({"is_premium": True, "message": ""})
print(result)

# Compile and run the graph for a regular user
graph = graph_builder.compile()
result = graph.invoke({"is_premium": False, "message": ""})
print(result)