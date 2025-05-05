from langchain_groq import ChatGroq
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import Tool
import os
from typing import Union, Dict, Any, List
import json

api_key = os.environ["GROQ_API_KEY"]

# Define the weather function
def get_weather(location: str) -> str:
    """Fetch the current weather for the given location"""
    weather_data = {
        "Dhaka": "It's 40 degrees Celsius and sunny",
        "Kushtia": "It's 33 degrees Celsius and cloudy",
        "Zhengzhou": "It's 21 degrees Celsius and foggy"
    }
    return weather_data.get(location, "Weather information is unavailable for this location")

# Create the Tool object
weather_tool = Tool(
    name="get_weather",
    description="Fetch the current weather for the given location",
    func=get_weather
)

# Tools dictionary for easy access
tools_dict = {weather_tool.name: weather_tool}

# Initialize the LLM model
model = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile"
)

# Create the tool model
model_with_tools = model.bind_tools([weather_tool])

# Node to determine if tools need to be called
def determine_next_step(state):
    messages = state["messages"]
    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "call_tools"
    else:
        return "end"

# Node function to handle the user query and call the LLM
def call_llm(state):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": messages + [response]}

# Node function to call tools
def call_tools(state):
    messages = state["messages"]
    last_message = messages[-1]

    # Process each tool call
    new_messages = []
    for tool_call in last_message.tool_calls:
        # Extract tool information from dictionary
        tool_name = tool_call.get("name", "")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", "")

        # Fix for __arg1 param - extract the location value
        if "__arg1" in tool_args:
            location = tool_args["__arg1"]

            if tool_name == "get_weather":
                # Call get_weather directly with the location
                tool_result = get_weather(location)

                # Add the tool message to the list
                new_messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        name=tool_name,
                        tool_call_id=tool_id
                    )
                )

    return {"messages": messages + new_messages}

# Define the graph
workflow = StateGraph(MessagesState)

# Add nodes
workflow.add_node("call_llm", call_llm)
workflow.add_node("call_tools", call_tools)

# Define the edges
workflow.add_edge(START, "call_llm")
workflow.add_conditional_edges(
    "call_llm",
    determine_next_step,
    {
        "call_tools": "call_tools",
        "end": END
    }
)
workflow.add_edge("call_tools", "call_llm")

# Compile the graph
graph = workflow.compile()

# Function to run the agent
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat.")
            break

        input_message = {"messages": [HumanMessage(content=user_input)]}
        for chunk in graph.stream(input_message, stream_mode="values"):
            if chunk["messages"][-1].type != "tool":
                chunk["messages"][-1].pretty_print()