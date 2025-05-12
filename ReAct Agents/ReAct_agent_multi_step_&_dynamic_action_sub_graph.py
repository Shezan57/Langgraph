from langgraph.graph import START, END, StateGraph
from typing import TypedDict

# Define shared state for the agent and the sub-graph
class ReActAgentState(TypedDict):
    message: str # Shared key between agent and sub-graph
    action: str # Sub-graph specific key (what action to take)
    sub_action: str # Additional sub-action to perform in a more complex senatrio

# Reasoning Node 1: Determine the action to take according the user query
def reasoning_node(state:ReActAgentState):
    query = state["message"]
    if "weather" in query:
        return {"action":"fetch_weather"}
    elif "news" in query:
        return {"action":"fetch_news"}
    elif "recommend" in query:
        return {"action":"recommendation", "sub_action":"book"}
    else:
        return {"action":"unknown"}

# Sub-graph for fetching weather information
def weather_subgraph_node(state:ReActAgentState):
    return {"message":"The weather is sunny today."}

# Sub-graph for fetching news information
def news_subgraph_node(state:ReActAgentState):
    return {"message":"Here are the latest news headlines."}

# Sub-graph for fetching recommendations
def recommendation_subgraph_node(state:ReActAgentState):
    if state.get("sub_action") == "book":
        return {"message":"I recommend reading 'The pragmatic programmer'."}
    else:
        return {"message":"I have no recommendations at the moment."}

# Build graph for the fetching weather information
weater_subgraph_builder = StateGraph(ReActAgentState)
weater_subgraph_builder.add_node("weather_action", weather_subgraph_node)
weater_subgraph_builder.set_entry_point("weather_action")
weater_subgraph = weater_subgraph_builder.compile()

# Build graph for the fetching news information
news_subgraph_builder = StateGraph(ReActAgentState)
news_subgraph_builder.add_node("news_action", news_subgraph_node)
news_subgraph_builder.set_entry_point("news_action")
news_subgraph = news_subgraph_builder.compile()

# Build graph for the fetching recommendations
recommendation_subgraph_builder = StateGraph(ReActAgentState)
recommendation_subgraph_builder.add_node("recommendation_action", recommendation_subgraph_node)
recommendation_subgraph_builder.set_entry_point("recommendation_action")
recommendation_subgraph = recommendation_subgraph_builder.compile()

# Define dynamic reasoning node in the parent graph to route to the corected sub-graph
def reasoning_state_manager(state:ReActAgentState):
    action = state["action"]
    if action == "fetch_weather":
        return weater_subgraph
    elif action == "fetch_news":
        return news_subgraph
    elif action == "recommendation":
        return recommendation_subgraph
    else:
        return None

# Create the parent graph
parent_builder = StateGraph(ReActAgentState)
parent_builder.add_node("reasoning", reasoning_node)
parent_builder.add_node("action_dispatch", reasoning_state_manager)

# Define the edge in the parent builder
parent_builder.add_edge(START, "reasoning")
parent_builder.add_edge("reasoning", "action_dispatch")

# Compile the parent graph
react_agent_graph = parent_builder.compile()

# Test the agent with a weather-related query
inputs_weater = {"message":"What is the weather like today?"}
result_weather = react_agent_graph.invoke(inputs_weater)
print("Weather Result:", result_weather["message"])

# Test the agent with a news-related query
inputs_news = {"message":"What is the latest news?"}
result_news = react_agent_graph.invoke(inputs_news)
print("News Result:", result_news["message"])

# Test the agent with a recommendation-related query
inputs_recommendation = {"message":"Can you recommend a book?"}
result_recommendation = react_agent_graph.invoke(inputs_recommendation)
print("Recommendation Result:", result_recommendation["message"])
