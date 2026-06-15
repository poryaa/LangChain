from ..state import IncidentState


def call_tools(state: IncidentState) -> IncidentState:
    """
    Phase 0: placeholder for tool execution.
    In Phase 1 this will call local tools over CSV/JSON.
    """
    selected = state.get("selected_incident_id")

    state["tool_results"] = [
        {
            "tool": "query_logs",
            "status": "ok" if selected else "skipped",
            "details": "log events inspected" if selected else "no incident selected",
        },
        {
            "tool": "get_metrics",
            "status": "ok" if selected else "skipped",
            "details": "metrics window inspected" if selected else "no incident selected",
        },
    ]

    return state