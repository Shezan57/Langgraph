from langchain_core.messages.tool import tool_call
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
@tool
def get_user_profile(user_id: str):
    """Retrive user profile ingormation using the user ID"""
    user_data = {
        "101": {"name": "Alice", "age": 30, "location": "New York"},
        "102": {"name": "Bob", "age": 25, "location": "San Francisco"}
    }
    return user_data.get(user_id, "User profile not found")
tools = [get_user_profile]
tool_node = ToolNode(tools)

message_with_tool_call = AIMessage(
    content="",
    tool_calls = [{
        "name":"get_user_profile",
        "args":{"user_id": "101"},
        "id":"tool_call_id",
        "type":"tool_call"
        }]
)

state = {
    "messages":[message_with_tool_call]
}

result = tool_node.invoke(state)
print(result)