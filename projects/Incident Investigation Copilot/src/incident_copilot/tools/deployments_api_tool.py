import httpx
import os
from typing import Any, List

DEPLOYMENTS_API_URL = os.getenv("DEPLOYMENTS_API_URL", "http://127.0.0.1:8003")

def list_deployments(service: str) -> List[Any]:
    try:
        resp = httpx.get(
            f"{DEPLOYMENTS_API_URL}/deployments/{service}",
            timeout=5.0
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        return [{"error": f"HTTP {e.response.status_code}: {e.response.text}"}]
    except httpx.TimeoutException:
        return [{"error": "deployments_api timed out"}]
    except httpx.ConnectError:
        return [{"error": "deployments_api unreachable"}]