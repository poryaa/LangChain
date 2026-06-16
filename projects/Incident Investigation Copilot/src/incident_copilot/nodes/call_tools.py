# src/incident_copilot/nodes/call_tools.py
from ..state import IncidentState
from ..tools.logs_api_tool import query_logs          # ← was logs.py
from ..tools.metrics_api_tool import get_metrics      # ← was metrics.py
from ..tools.deployments_api_tool import list_deployments  # ← was deployments.py
from ..tools.runbooks import read_runbook             # ← stays local (no API needed)

def call_tools(state: IncidentState) -> IncidentState:
    incident_id = state.get("selected_incident_id")
    if not incident_id:
        state["tool_results"] = [{"tool": "all", "status": "skipped", "details": "no incident selected"}]
        return state

    state["tool_results"] = [
        {"tool": "query_logs",       "status": "ok", "details": query_logs(incident_id)},
        {"tool": "get_metrics",      "status": "ok", "details": get_metrics(incident_id)},
        {"tool": "list_deployments", "status": "ok", "details": list_deployments("hdfs-datanode")},
        {"tool": "read_runbook",     "status": "ok", "details": read_runbook("IOException_replication")},
    ]
    return state