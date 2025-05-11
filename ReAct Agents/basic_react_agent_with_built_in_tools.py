from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from typing import Dict, Any
import os

api_key = os.environ["GROQ_API_KEY"]

# Tool definitions look good
def add(args: Dict[str, Any]) -> str:
    """Add two numbers together"""
    a = int(args["a"])
    b = int(args["b"])
    return f"{a+b}"  # Just return the result value

def multiply(args: Dict[str, Any]) -> str:
    """Multiply two numbers together"""
    a = int(args["a"])
    b = int(args["b"])
    return f"{a*b}"  # Just return the result value

add_tool = Tool.from_function(
    func=add,
    name="add",
    description="Add two numbers together",
    args_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The first number"},
            "b": {"type": "number", "description": "The second number"},
        },
        "required": ["a", "b"]
    }
)

multiply_tool = Tool.from_function(
    func=multiply,
    name="multiply",
    description="Multiply two numbers together",
    args_schema={
        "type": "object",
        "properties": {
            "a": {"type": "number", "description": "The first number"},
            "b": {"type": "number", "description": "The second number"},
        },
        "required": ["a", "b"]
    }
)

tools = [add_tool, multiply_tool]

# Use a supported model name
llm = ChatGroq(
    api_key=api_key,
    model="gemma2-9b-it"
)

# Create ReAct agent with recursion limit and explicit system message
graph = create_react_agent(
    model=llm,
    tools=tools,
    #system_message="""You are a helpful assistant that can perform math operations.
#When you use tools, wait for their response before using the result in another tool.
#Parse numeric results from tool outputs before using them in other calculations.""",
    #config={"recursion_limit": 10}
)

# User input (simplified)
inputs = {"messages": [("user", "First add 3 and 4, then multiply the result by 2.")]}

# Run the ReAct agent with try/except
try:
    messages = graph.invoke(inputs)
    for message in messages:
        print(message.content)
except Exception as e:
    print(f"Error: {str(e)}")