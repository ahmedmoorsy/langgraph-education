import functools
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI

from agent.utils.state import EducationState
from agent.utils.nodes import (
    lesson_agent,
    assessment_agent,
    create_subject_supervisor,
    create_top_level_supervisor,
    should_continue
)

# Instantiate the agents and graph
llm_main = ChatOpenAI(model="gpt-4o")

lesson_node = functools.partial(lesson_agent, agent=llm_main, name="LessonAgent")
assessment_node = functools.partial(assessment_agent, agent=llm_main, name="AssessmentAgent")

# Create Subject Supervisors
math_supervisor = functools.partial(
    create_subject_supervisor,
    llm=llm_main,
    system_prompt="You are the supervisor for the math subject. Guide the student through math lessons, practice, and assessment.",
    agents=["LessonAgent", "AssessmentAgent"],
    subject_name="Math",
)

english_supervisor = functools.partial(
    create_subject_supervisor,
    llm=llm_main,
    system_prompt="You are the supervisor for the English subject. Guide the student through English lessons, practice, and assessment.",
    agents=["LessonAgent", "AssessmentAgent"],
    subject_name="English",
)

# Create Top-Level Supervisor
top_level_supervisor = functools.partial(
    create_top_level_supervisor,
    llm=llm_main,
    system_prompt="You are the top-level supervisor. Decide which subject supervisor should handle the student's request.",
    subjects=["MathSupervisor", "EnglishSupervisor"],
)

# Build the graph
education_graph = StateGraph(EducationState)
education_graph.set_entry_point("TopLevelSupervisor")
education_graph.add_node("LessonAgent", lesson_node)
education_graph.add_node("AssessmentAgent", assessment_node)
education_graph.add_node("MathSupervisor", math_supervisor)
education_graph.add_node("EnglishSupervisor", english_supervisor)
education_graph.add_node("TopLevelSupervisor", top_level_supervisor)

# Define the flow between agents
# From START to Top-Level Supervisor
education_graph.add_edge(START, "TopLevelSupervisor")

# Conditional edges from Top-Level Supervisor to Subject Supervisors or FINISH
education_graph.add_conditional_edges(
    "TopLevelSupervisor",
    lambda x: x["next"],
    {
        "MathSupervisor": "MathSupervisor",
        "EnglishSupervisor": "EnglishSupervisor",
        "FINISH": END,
    },
)

# Conditional edges from MathSupervisor to Agents or Return to Top-Level Supervisor
education_graph.add_conditional_edges(
    "MathSupervisor",
    lambda x: x["next"],
    {
        "LessonAgent": "LessonAgent",
        "AssessmentAgent": "AssessmentAgent",
        "ReturnToTopLevel": "TopLevelSupervisor",
        "FINISH": END,
    },
)

# Conditional edges from EnglishSupervisor to Agents or Return to Top-Level Supervisor
education_graph.add_conditional_edges(
    "EnglishSupervisor",
    lambda x: x["next"],
    {
        "LessonAgent": "LessonAgent",
        "AssessmentAgent": "AssessmentAgent",
        "ReturnToTopLevel": "TopLevelSupervisor",
        "FINISH": END,
    },
)


education_graph.add_conditional_edges(
    "LessonAgent",
    should_continue,
    {
        "math": "MathSupervisor",
        "english": "EnglishSupervisor",
        "end": END,
    },
)

education_graph.add_conditional_edges(
    "AssessmentAgent",
    should_continue,
    {
        "math": "MathSupervisor",
        "english": "EnglishSupervisor",
        "end": END,
    },
)

# Compile the graph
graph = education_graph.compile()

# Optionally, you can run the graph with an initial state
# initial_state = EducationState(messages=[], next='', current_supervisor='', role='student')
# graph.run(initial_state)