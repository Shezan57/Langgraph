from langchain_groq import ChatGroq
from langgraph.graph import START,END,MessagesState,StateGraph
# from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
import uuid
import os

api_key = os.environ["GROQ_API_KEY"]

# Initialize the LLM model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile"
)

# Initialize the memory store user information across sessions
in_memory_store = InMemoryStore()

# function to store user information across sessions
def store_user_info(state:MessagesState,config,*,store=in_memory_store):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    #create a memory based on the conversation
    memory_id = str(uuid.uuid4())
    memory = {"user_name":state["user_name"]}
    # save the memory to the in-memory store
    store.put(namespace,memory_id,memory)

# Function to retrive stored user information
def retrive_user_info(state:MessagesState,config,*,store=in_memory_store):
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    #returive the stored memeories
    memories = store.search(namespace)
    if memories:
        info = f"Hello {memories[-1].value['user_name']}, welcome back!"
    else:
        info = "I don't have any information about you yet"
    return {"messages":[info]}

# Update the model with a memory saver
def call_model(state: MessagesState, config):
    last_message = state["messages"][-1].content.lower()
    if "remember my name" in last_message:
        # store user's name in state and in memory
        user_name = last_message.split("remember my name is")[1].strip()
        state["user_name"] = user_name
        return store_user_info(state,config)
    if "what's my name" in last_message or "what is my name" in last_message:
        # retrieve the users name from memory
        return retrive_user_info(state,config)
    # default llm response for other inputs
    return {"messages": ["I don't understand your request."]}

# Define the graph
workflow = StateGraph(MessagesState)

# Add node to call the LLM
workflow.add_node("call_model", call_model)
# Define the edges (Start -> LLM -> end)
workflow.add_edge(START, "call_model")
workflow.add_edge("call_model", END)

# Initialize the memory saver
checkpointer = MemorySaver()

# Compile the graph
graph_with_memory = workflow.compile(checkpointer=checkpointer, store=in_memory_store)

# simulate sessions
def simulate_sessions():
    # first session: store user's name
    config = {"configurable":{"thread_id":"session_1","user_id":"user_123"}}
    input_message = {"messages":[{"type":"user","content":"Remember my name is Shezan"}]}

    for chunk in graph_with_memory.stream(input_message, config=config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()

    config = {"configurable":{"thread_id":"session_2","user_id":"user_123"}}
    input_message = {"messages":[{"type":"user","content":"what is my name?"}]}

    for chunk in graph_with_memory.stream(input_message, config=config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
# run the session simulations
simulate_sessions()

# Function to continuously take user input
# def interact_with_agent_with_memory():
#     while True:
#         # Use a thread ID to simulate a continuous session
#         thread_id = input("Enter a thread ID (or 'new' for new session): ")
#         if thread_id.lower() in ["exit", "quit"]:
#             print("Goodbye")
#             break
#         if thread_id.lower() == "new":
#             thread_id = f'session_{os.urandom(4).hex()}'  # Generate a uid for session
#         while True:
#             user_input = input("You: ")
#             if user_input.lower() in ["exit", "quit", "end session"]:
#                 print(f"Exiting from the conversation session {thread_id}.")
#                 break
#             input_message = {
#                 "messages": [("human",user_input)]
#             }
#             # invoke the graph with the short-term memory
#             config = {"configurable": {"thread_id": thread_id}}
#             for chunk in graph_with_memory.stream(input_message,config=config, stream_mode="values"):
#                 chunk["messages"][-1].pretty_print()
#
# # Start interacting with the agent
# if __name__ == "__main__":
#     interact_with_agent_with_memory()