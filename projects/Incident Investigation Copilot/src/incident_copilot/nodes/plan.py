# nodes/plan.py

from pathlib import Path
import json
from typing import List

from ..state import IncidentState


INCIDENTS_PATH = Path("data_prep/output/incidents.json")


def _load_incidents() -> List[dict]:
    with open(INCIDENTS_PATH) as f:
        return json.load(f)


def plan_investigation(state: IncidentState) -> IncidentState:
    """
    Phase 0: pick a single incident deterministically.
    Later this can filter by user_query or time window.
    """
    incidents = _load_incidents()
    if not incidents:
        state["evidence"] = ["No incidents available in dataset."]
        state["selected_incident_id"] = None
        return state

    first = incidents[0]
    state["selected_incident_id"] = first["incident_id"]

    # Seed basic evidence for later nodes
    state["evidence"] = [
        f"Service: {first['service']}",
        f"Severity: {first['severity']}",
        f"Root cause (label): {first['root_cause']}",
    ]

    return state