import os
from langchain_core.tools import tool
from langgraph.graph import START,END,StateGraph,MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
# from langchain_groq import ChatGroq
import finnhub
from langchain_openai import ChatOpenAI


os.environ["OPENAI_API_KEY"]
# fin_api =  os.environ["FINNHUB_API_KEY"]
fin_api = "d0jk7bhr01qjm8s1lvd0d0jk7bhr01qjm8s1lvdg"
# Initialize the Finnhub client
finnhub_client = finnhub.Client(api_key=fin_api)

# Define the tool: querying stock prices using finnhub api
@tool
def get_stock_price(symbol:str) -> str:
    """Retrive the latest stock price for the given symbol"""
    quote = finnhub_client.quote(symbol)
    return f"The current price for {symbol} is: ${quote['c']}"

# Resister tool in the tool node
tools = [get_stock_price]
tool_node = ToolNode(tools)

# Initialize the Groq client
model = ChatOpenAI(
    model = "gpt-4o-mini"
)
model = model.bind_tools(tools)

# Define the function that simulate reasoning and invokes the model
def agent_reasoning(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages":response}

# Define conditional logic to determine whether to continue or stop
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # if there are no tool call finish the response
    if not last_message.tool_calls:
        return "end"
    return "continue"

# Build the React agent using LangGraph
workflow = StateGraph(MessagesState)

# Add nodes: agent reasoning and tool invocation (stock price retrieval)
workflow.add_node("agent_reasoning", agent_reasoning)
workflow.add_node("call_tool",tool_node)

# Define the flow
workflow.add_edge(START,"agent_reasoning") #Start with reasoning
# Conditional edges: continue to tool call or end the process
workflow.add_conditional_edges(
    "agent_reasoning",should_continue,{
        "continue":"call_tool", # process to get stock price
        "end":END # end the workflow
    }
)

# Normal edge: after invoking the tool, return to agent reasoning
workflow.add_edge("call_tool","agent_reasoning")

# Set up memory for breakpoints and add a breakpoint before calling the tool
memory = MemorySaver()
app = workflow.compile(checkpointer=memory, interrupt_before=["call_tool"])

# Simulate user input for stock symbol
initial_input = {"messages":[{"role":"user","content":"Should I buy AAPL stock today?"}]}
thread = {"configurable": {"thread_id":"1"}}

# Run the agent reasoning step first
for event in app.stream(initial_input,thread,stream_mode="values"):
    print(event)

# Pausing for human approval before retrieving stock price
user_approval = input("Do you approve quering the stock price for AAPL? (yes/no): ")
if user_approval.lower() == "yes":
    # continue with tool invocation to get stock price
    for event in app.stream(None,thread,stream_mode="values"):
        print(event)
else:
    print("Execution halted by user")
