from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    extract_evidence,
    generate_analysis,
    generate_answer,
)
from agent.state import AgentState


def _route_after_evidence(state: AgentState) -> str:
    if state.get("error"):
        return "end"

    return "analysis"


def _route_after_analysis(state: AgentState) -> str:
    if state.get("error"):
        return "end"

    return "answer"


def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node(
        "extract_evidence",
        extract_evidence,
    )

    graph.add_node(
        "generate_analysis",
        generate_analysis,
    )

    graph.add_node(
        "generate_answer",
        generate_answer,
    )

    graph.add_edge(
        START,
        "extract_evidence",
    )

    graph.add_conditional_edges(
        "extract_evidence",
        _route_after_evidence,
        {
            "analysis": "generate_analysis",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "generate_analysis",
        _route_after_analysis,
        {
            "answer": "generate_answer",
            "end": END,
        },
    )

    graph.add_edge(
        "generate_answer",
        END,
    )

    return graph.compile()