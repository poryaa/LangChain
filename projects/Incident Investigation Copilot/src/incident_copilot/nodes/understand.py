from ..state import IncidentState


def understand_request(state: IncidentState) -> IncidentState:
    """
    Phase 0: trivial intent + high‑level plan.
    Later you can swap this for an LLM call.
    """
    query = state.get("user_query", "")

    # For now we only support one intent
    state["intent"] = "investigate_incident"

    state["plan"] = [
        "select a relevant incident",
        "inspect its logs",
        "summarize root cause and impact",
    ]

    return state