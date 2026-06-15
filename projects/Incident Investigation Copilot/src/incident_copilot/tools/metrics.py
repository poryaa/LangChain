# src/incident_copilot/tools/metrics.py

import pandas as pd
from pathlib import Path
from typing import Dict, Any

METRICS_PATH = Path("data_prep/output/metrics.csv")


def get_metrics(incident_id: str) -> Dict[str, Any]:
    """Return metric snapshot (cpu, mem, error_rate) for a given incident_id."""
    df = pd.read_csv(METRICS_PATH)
    row = df[df["incident_id"] == incident_id]

    if row.empty:
        return {"error": f"No metrics found for {incident_id}"}

    r = row.iloc[0]
    return {
        "incident_id": incident_id,
        "cpu_pct":     r["cpu_pct"],
        "mem_pct":     r["mem_pct"],
        "error_rate":  r["error_rate"],
        "timestamp":   r["timestamp"],
    }