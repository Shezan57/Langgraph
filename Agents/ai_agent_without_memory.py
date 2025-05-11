from langchain_groq import ChatGroq
from langgraph.graph import START,END,MessagesState,StateGraph
from langchain_core.messages import HumanMessage
import os
api_key = os.environ["GROQ_API_KEY"]

# Initialize the LLM model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile"
)

# Node function to handle the user query and call the LLM
def call_llm(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages[-1].content)
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(MessagesState)

# Add node to call the LLM
workflow.add_node("call_llm", call_llm)
# Define the edges (Start -> LLM -> end)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)
# Compile the graph
graph = workflow.compile()
# Function to continuously take user input
def interact_with_agent():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        input_message = {
            "messages": [HumanMessage(content=user_input)]
        }
        for chunk in graph.stream(input_message, stream_mode="values"):
            chunk["messages"][-1].pretty_print()

# Start interacting with the agent
if __name__ == "__main__":
    interact_with_agent()