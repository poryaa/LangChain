# nodes/synthesize.py

from ..state import IncidentState
from ..llm import get_generation_llm
from ..prompts.synthesize import SYNTHESIZE_PROMPT


def _format_tool_result(tool_results: list, tool_name: str) -> str:
    for tr in tool_results:
        if tr["tool"] == tool_name:
            details = tr.get("details", {})
            if isinstance(details, dict):
                return json.dumps(details, indent=2)
            if isinstance(details, list):
                return json.dumps(details, indent=2)
            return str(details)
    return "No data available."


def synthesize_evidence(state: IncidentState) -> IncidentState:
    tool_results = state.get("tool_results", [])
    evidence     = state.get("evidence", [])
    incident_id  = state.get("selected_incident_id", "unknown")

    # Parse basic fields from evidence strings ("Service: hdfs-datanode" etc.)
    service  = next((e.split(": ", 1)[1] for e in evidence if e.startswith("Service")), "unknown")
    severity = next((e.split(": ", 1)[1] for e in evidence if e.startswith("Severity")), "unknown")

    prompt_input = {
        "incident_id": incident_id,
        "service":     service,
        "severity":    severity,
        "log_events":  _format_tool_result(tool_results, "query_logs"),
        "metrics":     _format_tool_result(tool_results, "get_metrics"),
        "deployments": _format_tool_result(tool_results, "list_deployments"),
        "runbook":     _format_tool_result(tool_results, "read_runbook"),
    }

    llm    = get_generation_llm()
    chain  = SYNTHESIZE_PROMPT | llm
    result = chain.invoke(prompt_input)

    state["confidence"]   = 0.85
    state["final_answer"] = result.content

    return state