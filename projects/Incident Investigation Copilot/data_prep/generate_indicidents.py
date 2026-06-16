# data_prep/generate_incidents.py

import pandas as pd
import json
import random
from pathlib import Path

DATA = Path("../data/raw_data")
OUT  = Path("../data_prep/output")
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load only Event_traces.csv — it already has labels
traces = pd.read_csv(DATA / "Event_traces.csv")
templates = pd.read_csv(DATA / "HDFS.log_templates.csv")

tmpl_map = dict(zip(templates["EventId"], templates["EventTemplate"]))

# 2. Sample: 25 Fail + 75 Success
failures = traces[traces["Label"] == "Fail"].sample(25, random_state=42)
successes = traces[traces["Label"] == "Success"].sample(75, random_state=42)
subset = pd.concat([failures, successes]).reset_index(drop=True)

# 3. In the loop — fix column name and parsing
incidents = []
for _, row in subset.iterrows():
    # Features looks like "[E5,E22,E5,...]" — strip brackets and split on comma
    raw = str(row["Features"]).strip("[]")
    events = [e.strip() for e in raw.split(",")]
    readable = [tmpl_map.get(e, e) for e in events]
    root_cause = "IOException_replication" if row["Label"] == "Fail" else "none"
    last_event = events[-1] if events else "E1"

    incidents.append({
        "incident_id":             row["BlockId"],
        "service":                 "hdfs-datanode",
        "severity":                "high" if row["Label"] == "Fail" else "info",
        "root_cause":              root_cause,
        "time_window":             "2008-11-09T00:00:00/2008-11-09T01:00:00",
        "relevant_log_ids":        events,
        "log_descriptions":        readable,
        "relevant_metric_windows": [],
        "expected_tools":          ["query_logs", "read_runbook"],
        "expected_answer_keywords": [root_cause, last_event],
        "ground_truth_label":      row["Label"],
    })

with open(OUT / "incidents.json", "w") as f:
    json.dump(incidents, f, indent=2)

print(f"✓ incidents.json  → {len(incidents)} records")