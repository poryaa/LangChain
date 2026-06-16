from fastapi import FastAPI, HTTPException
import json
from pathlib import Path

app = FastAPI(title="Logs API")
INCIDENTS_PATH = Path("data_prep/output/incidents.json")

@app.get("/logs/{incident_id}")
def get_logs(incident_id: str):
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
    raise HTTPException(status_code=404, detail=f"Incident {incident_id} not found")