from langgraph.graph import START, END, MessagesState, StateGraph
from Agents.basic_ai_agent import input_message


# Define a node to simulate a weather response
def weather_node(state: MessagesState):
    weather_response = "The weather is sunny with a temperature of 25 degree C."
    return {"messages": [weather_response]}

# Define a node to handel basic arithmetic calculation
def calculator_node(state: MessagesState):
    user_query = state["messages"][-1].content.lower()
    if "add" in user_query:
        numbers = [int(s) for s in user_query.split() if s.isdigit()]
        result = sum(numbers)
        return {"messages": [f"The result addition is {result}"]}
    return {"messages": ["I can only add numbers."]}

# Define a default node to handle unrecognized queries
def default_node(state: MessagesState):
    return {"messages": ["Sorry, I'm not sure how to help with that."]}

def routing_function(state: MessagesState):
    last_message = state["messages"][-1].content.lower()
    if "weather" in last_message:
        return "weather_node"
    elif "add" in last_message or "calculate" in last_message:
        return "calculator_node"
    return "default_node"

# Building workflow graph
builder = StateGraph(MessagesState)
builder.add_node("weather_node", weather_node)
builder.add_node("calculator_node", calculator_node)
builder.add_node("default_node", default_node)
builder.add_node("routing_function", routing_function)

# setup the edges for routing
builder.add_conditional_edges(
    START,
    routing_function,
    {
        "weather_node": "weather_node",
        "calculator_node": "calculator_node",
        "default_node": "default_node"
    }
) # Route based on the function
builder.add_edge("weather_node", END)
builder.add_edge("calculator_node", END)
builder.add_edge("default_node", END)

# conpile the graph
app = builder.compile()

# Simulate interaction with the agent
def simulate_interaction():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit","quite"]:
            break
        input_message = {"messages":[("human", user_input)]}
        for result in app.stream(input_message, stream_mode="values"):
            result = result["messages"][-1].pretty_print()


# Start interacting with the agent
if __name__ == "__main__":
    simulate_interaction()

