# src/incident_copilot/tools/runbooks.py
from pathlib import Path

RUNBOOKS_PATH = Path("data_prep/runbooks")


def read_runbook(root_cause: str) -> str:
    """Return the markdown content of a runbook for a given root_cause slug."""
    path = RUNBOOKS_PATH / f"{root_cause}.md"

    if not path.exists():
        available = [p.stem for p in RUNBOOKS_PATH.glob("*.md")]
        return f"No runbook found for '{root_cause}'. Available: {available}"

    return path.read_text()