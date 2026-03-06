from langgraph.graph import StateGraph, START, END
from app.states.agent_state import AgentState
from app.agents.context_retriver_agent import context_retriever_node
from app.agents.bug_finder import bug_finder_node
from app.agents.bug_explainer import bug_explainer_node
from app.agents.bug_verifier import bug_verifier_node, route_after_verifier
from app.agents.output_writer import output_writer_node


def create_main_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("context_retriever", context_retriever_node)
    workflow.add_node("bug_finder", bug_finder_node)
    workflow.add_node("bug_explainer", bug_explainer_node)
    workflow.add_node("bug_verifier", bug_verifier_node)
    workflow.add_node("output_writer", output_writer_node)

    # Linear path
    workflow.add_edge(START, "context_retriever")
    workflow.add_edge("context_retriever", "bug_finder")
    workflow.add_edge("bug_finder", "bug_explainer")
    workflow.add_edge("bug_explainer", "bug_verifier")

    # Conditional loop — loop back or go to output_writer
    workflow.add_conditional_edges(
        "bug_verifier",
        route_after_verifier,
        {
            "bug_finder": "bug_finder",
            "bug_explainer": "bug_explainer",
            "__end__": "output_writer",
        },
    )

    workflow.add_edge("output_writer", END)

    return workflow.compile()