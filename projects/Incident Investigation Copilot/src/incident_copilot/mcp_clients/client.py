# src/incident_copilot/mcp_clients/client.py
import sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

BASE = Path(__file__).resolve().parents[3]
LOGS_SERVER    = str(BASE / "src/incident_copilot/mcp_servers/logs_server.py")
METRICS_SERVER = str(BASE / "src/incident_copilot/mcp_servers/metrics_server.py")

def get_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "incident-logs": {
                "command": sys.executable,
                "args": [LOGS_SERVER],
                "transport": "stdio",
            },
            "incident-metrics": {
                "command": sys.executable,
                "args": [METRICS_SERVER],
                "transport": "stdio",
            },
        }
    )