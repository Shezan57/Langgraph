from typing_extensions import TypedDict
from langgraph.graph import START,END,StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
import os


os.environ["GROQ_API_KEY"]

#Initialize the llm
llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0.7
)

# Define the state
class State(TypedDict):
    input: str
    draft_content: str

# Define node functions
def create_draft(state:State):
    print("Generating draft with deepseek-r1-70b model.....")
    #prepare the prompt for generating the blog content
    prompt = f"Write a blog post on the topic: {state['input']}"
    # call the langchain chatgroq instance to generate the draft
    response = llm.invoke([{"role":"user", "content":prompt}])
    state["draft_content"] = response.content
    print(f"Generated Draft: \n{state['draft_content']}")
    return state
def review_draft(state:State):
    print("-----Reviewing Draft-----")
    print(f"Draft for review:\n{state['draft_content']}")
    return state
def publish_content(state:State):
    print("-----Publishing Content-----")
    print(f"Content for publish:\n{state['draft_content']}")
    return state

# Building the graph
builder = StateGraph(State)
builder.add_node("create_draft",create_draft)
builder.add_node("review_draft",review_draft)
builder.add_node("publish_content",publish_content)

#Define flow
builder.add_edge(START,"create_draft")
builder.add_edge("create_draft","review_draft")
builder.add_edge("review_draft","publish_content")
builder.add_edge("publish_content",END)

# Setup and breakpoints
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["publish_content"])

# Run the graph
config = {"configurable":{"thread_id":"thread_2"}}
initial_input = {"input":"The importance of AI in modern content creation"}

# Run the first part until the review step
thread = {"configurable":{"thread_id":"2"}}
for event in graph.stream(initial_input, thread, stream_mode="values"):
    print(event)
# Paursing for human review
user_approval = input("Do you approve the draft for publishing? (yes/no/modification): ")
if user_approval.lower() == "yes":
    # proceed to graph content
    for event in graph.stream(None, thread, stream_mode="values"):
        print(event)
elif user_approval.lower() == "modification":
    update_draft = input("please modify the draft content:\n")
    modified_state = {"draft_content":update_draft} # update memory with new content
    print("Draft updated by editor")
    #continue to publish with modification draft
    for event in graph.stream(modified_state, thread, stream_mode="values"):
        print(event)
else:
    print("Execution halted by user!.")