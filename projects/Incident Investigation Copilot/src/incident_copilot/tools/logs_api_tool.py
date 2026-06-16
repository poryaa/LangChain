import httpx
import os
from typing import Dict, Any

LOGS_API_URL = os.getenv("LOGS_API_URL", "http://127.0.0.1:8001")

def query_logs(incident_id: str) -> Dict[str, Any]:
    try:
        resp = httpx.get(
            f"{LOGS_API_URL}/logs/{incident_id}",
            timeout=5.0
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except httpx.TimeoutException:
        return {"error": "logs_api timed out"}
    except httpx.ConnectError:
        return {"error": "logs_api unreachable"}