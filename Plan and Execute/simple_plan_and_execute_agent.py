import operator
import os
from langgraph.graph import StateGraph,START,END
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from typing import Annotated,List,Tuple,Union
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
import asyncio

os.environ["OPENAI_API_KEY"]
os.environ["TAVILY_API_KEY"]

# define diagonstic and action tools
tools = [TavilySearchResults(max_results=3)]

# Setup the model and agent executor
prompt = ChatPromptTemplate.from_messages([
    ("system","""You are a helpful assistant"""),
    ("placeholder","{messages}")
])
prompt.pretty_print()
llm = ChatOpenAI(model="gpt-4o-mini")
agent_executor  = create_react_agent(llm,tools,state_modifier=prompt)
# Define the plan and execution structure
class PlanExecute(TypedDict):
    input:str
    plan:List[str]
    past_step:Annotated[List[Tuple],operator.add]
    response:str
class Plan(BaseModel):
    steps:List[str] = Field(description="Numbered unique steps to follow, in order")
class Response(BaseModel):
    response:str = Field(description="Response to user")
class Act(BaseModel):
    action:Union[Response,Plan] = Field(description="Action to perform if want to respond to user, use Response."
    "If you need to further use tools to get the answer, use Plan")
#Planning Step
planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """For the given objective, come up with a simple step-by-step plan.\
        This plan should involve indidual numbered tasks, that if executed
        correctly will yeild the correct answer. Do not add any superflous steps.\
        The result of the final step should be the final answer. Make sure that
        each step has all the information needed-do not skip steps"""
    ),
    ("placeholder","{messages}")
])
planner = planner_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Plan)
# Replanning step
replanner_prompt = ChatPromptTemplate.from_template(
"""For the given objective, come up with a simple step by step numbered plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct
answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the
information needed - do not skip steps.
Your objective was this:
{input}
Your original plan was this:
{plan}
You have currently done the follow steps:
{past_steps}
Update your plan accordingly. If no more steps are needed and you can return to the user,
then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still
NEED to be done. Do not return previously done steps as part of the plan."""
)
replanner = replanner_prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(Act)
#Execute step function
def execute_step(state:PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}"for i,step in enumerate(plan))
    task = plan[0]
    task_formatted = f"For the following plan:\n\n You are tasked with executing step 1, {task}"
    agent_response = await agent_executor.ainvoke({"messages":[("user", task_formatted)]})
    return {
        "past_steps":[(task,agent_response["messages"][-1].content)]
    }
# Planning step function
def plan_step(state:PlanExecute):
    plan = await planner.ainvoke({"messages":[("user",state["input"])]})
    return {"plan":plan.steps}
# Re-Planning step function
def replan_step(state:PlanExecute):
    output = await planner.ainvoke(state)
    # If the re-planner decides to return a response, we use it as the final answer
    if isinstance(output.action, Response): # Final response provided
        return {"response":output.action.response} # Return the response to the user
    else:
        # Otherwise, we continue with the new plan (if re-planning suggests more steps)
        return {"plan":output.action.steps}
# Conditional check for ending
def should_end(state:PlanExecute):
    if "respnse" in state and state["response"]:
        return END
    else:
        return "agent"

# Building the workflow
workflow = StateGraph(PlanExecute)
workflow.add_node("planner",plan_step)
workflow.add_node("agent",execute_step)
workflow.add_node("replan",replan_step)
# Add nodes to the workflow
workflow.add_edge(START,"planner")
workflow.add_edge("planner","agent")
workflow.add_edge("agent","replan")
workflow.add_conditional_edges("replan",should_end,["agent",END])
# Compile the workflow into an executable application
app = workflow.compile()

# Example of running the agent
config = {"recursion_limit":50}
# Function to run the Plan-and-Execute agent
def run_plan_and_execute():
    # Input from the user
    inputs = {"input": "Grace weighs 125 pounds. Alex weighs 2 pounds less than 4 times what Grace weighs. What are their combined weights in pounds?"}
    # Configuration for recursion limit
    config = {"recursion_limit": 50}
    # Use a regular for loop
    for event in app.stream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)
# Run the async function
if __name__ == "__main__":
    asyncio.run(run_plan_and_execute())