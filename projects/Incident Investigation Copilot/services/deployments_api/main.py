from fastapi import FastAPI, HTTPException
import pandas as pd
from pathlib import Path

app = FastAPI(title="Deployments API")
DEPLOYMENTS_PATH = Path("data_prep/output/deployments.csv")

@app.get("/deployments/{service}")
def list_deployments(service: str):
    df = pd.read_csv(DEPLOYMENTS_PATH)
    rows = df[df["service"] == service]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No deployments for {service}")
    return rows.to_dict(orient="records")