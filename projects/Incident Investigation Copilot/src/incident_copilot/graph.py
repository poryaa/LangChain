from langgraph.graph import StateGraph, END

from src.incident_copilot.state import IncidentState
from src.incident_copilot.nodes import (
    understand_request,
    plan_investigation,
    call_tools,
    synthesize_evidence,
    generate_report,
)


def build_graph():
    graph = StateGraph(IncidentState)

    graph.add_node("understand_request", understand_request)
    graph.add_node("plan_investigation", plan_investigation)
    graph.add_node("call_tools", call_tools)
    graph.add_node("synthesize_evidence", synthesize_evidence)
    graph.add_node("generate_report", generate_report)

    graph.set_entry_point("understand_request")
    graph.add_edge("understand_request", "plan_investigation")
    graph.add_edge("plan_investigation", "call_tools")
    graph.add_edge("call_tools", "synthesize_evidence")
    graph.add_edge("synthesize_evidence", "generate_report")
    graph.add_edge("generate_report", END)

    return graph


if __name__ == "__main__":
    app = build_graph().compile()
    initial_state: IncidentState = {
        "user_query": "Investigate a recent HDFS incident"
    }
    final_state = app.invoke(initial_state)
    print(final_state["final_answer"])