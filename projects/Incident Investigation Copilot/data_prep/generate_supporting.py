# data_prep/generate_supporting

import pandas as pd
import random
import json
from pathlib import Path

OUT = Path("data_prep/output")

# Load incidents from the already-generated file
with open(OUT / "incidents.json") as f:
    incidents = json.load(f)
    
# --- metrics.csv: one row per incident window ---
metrics_rows = []
for inc in incidents:
    metrics_rows.append({
        "incident_id":   inc["incident_id"],
        "service":       inc["service"],
        "timestamp":     inc["time_window"].split("/")[0],
        "cpu_pct":       round(random.uniform(60, 99) if inc["severity"] == "high" else random.uniform(10, 60), 1),
        "mem_pct":       round(random.uniform(70, 95) if inc["severity"] == "high" else random.uniform(30, 70), 1),
        "error_rate":    round(random.uniform(0.05, 0.5) if inc["severity"] == "high" else 0.0, 3),
    })
pd.DataFrame(metrics_rows).to_csv(OUT / "metrics.csv", index=False)
print(f"✓ metrics.csv     → {len(metrics_rows)} records")

# --- deployments.csv: 5 fake deployments around the incident window ---
deployments = [
    {"deploy_id": f"deploy_{i}", "service": "hdfs-datanode",
     "version": f"2.1.{i}", "timestamp": f"2008-11-{(i+1):02d}T22:00:00", "status": "success"}
    for i in range(1, 6)
]

pd.DataFrame(deployments).to_csv(OUT / "deployments.csv", index=False)
print(f"✓ deployments.csv → {len(deployments)} records")