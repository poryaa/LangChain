import httpx
import os
from typing import Dict, Any

METRICS_API_URL = os.getenv("METRICS_API_URL", "http://127.0.0.1:8002")

def get_metrics(incident_id: str) -> Dict[str, Any]:
    try:
        resp = httpx.get(
            f"{METRICS_API_URL}/metrics/{incident_id}",
            timeout=5.0
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except httpx.TimeoutException:
        return {"error": "metrics_api timed out"}
    except httpx.ConnectError:
        return {"error": "metrics_api unreachable"}