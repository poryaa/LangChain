from pathlib import Path
import json
from typing import List

from ..state import IncidentState


INCIDENTS_PATH = Path("data_prep/output/incidents.json")


def _load_incidents() -> List[dict]:
    with open(INCIDENTS_PATH) as f:
        return json.load(f)


# nodes/plan.py
import re

def plan_investigation(state: IncidentState) -> IncidentState:
    incidents = _load_incidents()
    query = state.get("user_query", "")

    # Try to extract a block ID from the query (starts with blk_)
    match = re.search(r"blk_-?\d+", query)
    if match:
        target_id = match.group()
        found = next((inc for inc in incidents if inc["incident_id"] == target_id), None)
    else:
        found = None

    # Fall back to first incident if not found
    first = found or incidents[0]
    state["selected_incident_id"] = first["incident_id"]
    state["evidence"] = [
        f"Service: {first['service']}",
        f"Severity: {first['severity']}",
        f"Root cause (label): {first['root_cause']}",
    ]
    return state