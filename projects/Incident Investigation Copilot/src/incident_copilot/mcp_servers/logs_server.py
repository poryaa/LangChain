# src/incident_copilot/mcp_servers/logs_server.py
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("incident-logs-server")

INCIDENTS_PATH = Path("data_prep/output/incidents.json")


@mcp.tool()
def query_incident_logs(incident_id: str) -> dict:
    """
    Return log event IDs and human-readable descriptions for a given HDFS incident block ID.
    Use this tool to inspect what happened during an incident.
    """
    with open(INCIDENTS_PATH) as f:
        incidents = json.load(f)

    for inc in incidents:
        if inc["incident_id"] == incident_id:
            return {
                "incident_id":   incident_id,
                "log_event_ids": inc.get("relevant_log_ids", []),
                "descriptions":  inc.get("log_descriptions", []),
                "severity":      inc.get("severity"),
            }

    return {"error": f"Incident {incident_id} not found"}


if __name__ == "__main__":
    mcp.run(transport="stdio")