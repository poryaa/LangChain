# src/incident_copilot/nodes/call_tools.py

from ..state import IncidentState
from ..tools.logs import query_logs
from ..tools.metrics import get_metrics
from ..tools.deployments import list_deployments
from ..tools.runbooks import read_runbook


def call_tools(state: IncidentState) -> IncidentState:
    """Phase 1: call real local tools from data_prep/output/."""
    incident_id = state.get("selected_incident_id")

    if not incident_id:
        state["tool_results"] = [{"tool": "all", "status": "skipped", "details": "no incident selected"}]
        return state

    logs_result        = query_logs(incident_id)
    metrics_result     = get_metrics(incident_id)
    deployments_result = list_deployments("hdfs-datanode")
    runbook_result     = read_runbook("IOException_replication")

    state["tool_results"] = [
        {"tool": "query_logs",        "status": "ok", "details": logs_result},
        {"tool": "get_metrics",       "status": "ok", "details": metrics_result},
        {"tool": "list_deployments",  "status": "ok", "details": deployments_result},
        {"tool": "read_runbook",      "status": "ok", "details": runbook_result},
    ]

    return state