from fastapi import FastAPI, HTTPException
import pandas as pd
from pathlib import Path

app = FastAPI(title="Metrics API")
METRICS_PATH = Path("data_prep/output/metrics.csv")

@app.get("/metrics/{incident_id}")
def get_metrics(incident_id: str):
    df = pd.read_csv(METRICS_PATH)
    row = df[df["incident_id"] == incident_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"No metrics for {incident_id}")
    r = row.iloc[0]
    return {
        "incident_id": incident_id,
        "cpu_pct":     float(r["cpu_pct"]),
        "mem_pct":     float(r["mem_pct"]),
        "error_rate":  float(r["error_rate"]),
        "timestamp":   str(r["timestamp"]),
    }