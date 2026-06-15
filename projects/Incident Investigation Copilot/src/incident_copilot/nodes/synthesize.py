from ..state import IncidentState


def synthesize_evidence(state: IncidentState) -> IncidentState:
    """
    Phase 0: simple string synthesis from evidence + tool_results.
    Later this will be an LLM node.
    """
    evidence = state.get("evidence", [])
    tool_results = state.get("tool_results", [])

    lines = [
        "I inspected one historical incident from the synthetic dataset.",
        "Key facts:",
    ]
    lines.extend(f"- {e}" for e in evidence)

    lines.append("")  # blank line
    lines.append("Tool activity:")
    lines.extend(f"- {tr['tool']}: {tr['details']}" for tr in tool_results)

    state["confidence"] = 0.8
    state["final_answer"] = "\n".join(lines)

    return state