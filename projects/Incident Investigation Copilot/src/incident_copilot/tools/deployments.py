# src/incident_copilot/tools/deployments.py

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

DEPLOYMENTS_PATH = Path("data_prep/output/deployments.csv")


def list_deployments(service: str) -> List[Dict[str, Any]]:
    """Return all deployments for a given service name."""
    df = pd.read_csv(DEPLOYMENTS_PATH)
    rows = df[df["service"] == service]

    if rows.empty:
        return [{"error": f"No deployments found for service '{service}'"}]

    return rows.to_dict(orient="records")