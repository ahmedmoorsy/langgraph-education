# nodes.py

import functools

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

from agent.utils.tools import llm, tavily_tool, trimmer

# Base agent function
def agent_function(state, agent, name, system_prompt, tools=[]):
    """Generic agent responsible for a specific task."""
    messages = state["messages"]
    messages = [HumanMessage(content=system_prompt, role="system")] + messages
    chain = agent.bind_tools(tools)
    result = chain.invoke(messages)
    new_messages = state["messages"] + [AIMessage(content=result.content, name=name)]
    return {"messages": new_messages}

# Lesson agent
lesson_agent = functools.partial(
    agent_function,
    agent=llm,
    name="LessonAgent",
    system_prompt="Agent responsible for delivering lessons.",
    tools=[tavily_tool],
)

# Assessment agent
assessment_agent = functools.partial(
    agent_function,
    agent=llm,
    name="AssessmentAgent",
    system_prompt="Agent responsible for assessing the student's understanding.",
)

# Top-Level Supervisor function
def create_top_level_supervisor(state, llm, system_prompt, subjects):
    options = ["FINISH"] + subjects
    function_def = {
        "name": "route",
        "description": "Select the next subject supervisor to act.",
        "parameters": {
            "type": "object",
            "properties": {
                "next": {
                    "type": "string",
                    "enum": options,
                },
                "response": {
                    "type": "string",
                    "description": "Response message indicating the outcome of the action.",
                },
            },
            "required": ["next", "response"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the student's request, which subject supervisor should act next?"
                " Or should we FINISH? Select one of: {options}"
                " if it's finished, please provide a response message.",
            ),
        ]
    ).partial(options=str(options), subjects=", ".join(subjects))
    chain = (
        prompt
        | trimmer
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    result = chain.invoke({"messages": state["messages"]})
    current_supervisor = result["next"] if result["next"] != "FINISH" else ""
    return {
        "next": result["next"],
        "messages": state["messages"] + [AIMessage(content=result['response'], name="TopLevelSupervisor")],
        "current_supervisor": current_supervisor,
    }

# Subject Supervisor function (e.g., MathSupervisor)
def create_subject_supervisor(state, llm, system_prompt, agents, subject_name):
    options = ["FINISH", "ReturnToTopLevel"] + agents
    function_def = {
        "name": "route",
        "description": f"Select the next agent to act in the {subject_name} subject.",
        "parameters": {
            "type": "object",
            "properties": {
                "next": {
                    "type": "string",
                    "enum": options,
                },
                "response": {
                    "type": "string",
                    "description": "Response message indicating the outcome of the action.",
                },
            },
            "required": ["next", "response"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                f"Given the student's progress in {subject_name}, which agent should act next?"
                " Or should we FINISH or ReturnToTopLevel? Select one of: {options}"
                " if it's finished, please provide a response message.",
            ),
        ]
    ).partial(options=str(options), agents=", ".join(agents))
    chain = (
        prompt
        | trimmer
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )
    result = chain.invoke({"messages": state["messages"]})
    return {
        "next": result["next"],
        "messages": state["messages"] + [AIMessage(content=result['response'], name=subject_name)],
    }


# Function to determine next action
def should_continue(state):
    next_step = state["next"]
    current_supervisor = state["current_supervisor"]
    if next_step == "FINISH":
        return "end"
    if not current_supervisor:
        return "end"
    if current_supervisor == "MathSupervisor":
        return "math"
    elif current_supervisor == "EnglishSupervisor":
        return "english"
    return "end"