# src/incident_copilot/tools/logs.py
import json
from pathlib import Path
from typing import List, Dict, Any

INCIDENTS_PATH = Path("data_prep/output/incidents.json")


def query_logs(incident_id: str) -> Dict[str, Any]:
    """Return log event IDs and human-readable descriptions for a given incident_id."""
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