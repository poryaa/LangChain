# src/incident_copilot/mcp_servers/metrics_server.py
import pandas as pd
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("incident-metrics-server")

METRICS_PATH = Path("data_prep/output/metrics.csv")


@mcp.tool()
def get_metrics_window(incident_id: str) -> dict:
    """
    Return CPU, memory, and error rate metrics for a given HDFS incident block ID.
    Use this tool to assess resource pressure during an incident.
    """
    df = pd.read_csv(METRICS_PATH)
    row = df[df["incident_id"] == incident_id]

    if row.empty:
        return {"error": f"No metrics found for {incident_id}"}

    r = row.iloc[0]
    return {
        "incident_id": incident_id,
        "cpu_pct":     float(r["cpu_pct"]),
        "mem_pct":     float(r["mem_pct"]),
        "error_rate":  float(r["error_rate"]),
        "timestamp":   str(r["timestamp"]),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")