from langchain_groq import ChatGroq
from langgraph.graph import START,END,MessagesState,StateGraph
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
import os

from Agents.basic_ai_agent import workflow

api_key = os.environ["GROQ_API_KEY"]

# Initialize the LLM model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile"
)

# Update the model with a memory saver
def call_llm(state: MessagesState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Define the graph
workflow = StateGraph(MessagesState)

# Add node to call the LLM
workflow.add_node("call_llm", call_llm)
# Define the edges (Start -> LLM -> end)
workflow.add_edge(START, "call_llm")
workflow.add_edge("call_llm", END)

# Initialize the memory saver
checkpointer = MemorySaver()

# Compile the graph
graph_with_memory = workflow.compile(checkpointer=checkpointer)

# Function to continuously take user input
def interact_with_agent_with_memory():
    while True:
        # Use a thread ID to simulate a continuous session
        thread_id = input("Enter a thread ID (or 'new' for new session): ")
        if thread_id.lower() in ["exit", "quit"]:
            print("Goodbye")
            break
        if thread_id.lower() == "new":
            thread_id = f'session_{os.urandom(4).hex()}'  # Generate a uid for session
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit", "end session"]:
                print(f"Exiting from the conversation session {thread_id}.")
                break
            input_message = {
                "messages": [("human",user_input)]
            }
            # invoke the graph with the short-term memory
            config = {"configurable": {"thread_id": thread_id}}
            for chunk in graph_with_memory.stream(input_message,config=config, stream_mode="values"):
                chunk["messages"][-1].pretty_print()

# Start interacting with the agent
if __name__ == "__main__":
    interact_with_agent_with_memory()