# src/incident_copilot/nodes/call_tools.py
import asyncio
from ..state import IncidentState
from ..mcp_clients.client import get_mcp_client
from ..tools.deployments_api_tool import list_deployments
from ..tools.runbooks import read_runbook


async def _run_mcp_tools(incident_id: str) -> dict:
    client = get_mcp_client()
    tools = await client.get_tools()  # loads all MCP tools as LC tools[web:79]
    tools_by_name = {t.name: t for t in tools}

    # Depending on your version, names may be just "query_incident_logs"
    # or "incident-logs:query_incident_logs". If you see a KeyError,
    # print(tools_by_name.keys()) once and adjust the keys.
    logs_tool    = tools_by_name["query_incident_logs"]
    metrics_tool = tools_by_name["get_metrics_window"]

    logs_result    = await logs_tool.ainvoke({"incident_id": incident_id})
    metrics_result = await metrics_tool.ainvoke({"incident_id": incident_id})

    return {"logs": logs_result, "metrics": metrics_result}


def call_tools(state: IncidentState) -> IncidentState:
    incident_id = state.get("selected_incident_id")
    if not incident_id:
        state["tool_results"] = [{"tool": "all", "status": "skipped", "details": "no incident selected"}]
        return state

    mcp_results = asyncio.run(_run_mcp_tools(incident_id))

    state["tool_results"] = [
        {"tool": "query_logs",       "status": "ok", "details": mcp_results["logs"]},
        {"tool": "get_metrics",      "status": "ok", "details": mcp_results["metrics"]},
        {"tool": "list_deployments", "status": "ok", "details": list_deployments("hdfs-datanode")},
        {"tool": "read_runbook",     "status": "ok", "details": read_runbook("IOException_replication")},
    ]
    return state