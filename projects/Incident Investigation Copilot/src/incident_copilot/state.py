# src/incident_copilot/state.py
from typing import List, Optional, TypedDict, Dict, Any


class IncidentState(TypedDict, total=False):
    user_query: str
    intent: str
    plan: List[str]
    selected_incident_id: Optional[str]
    evidence: List[str]
    tool_results: List[Dict[str, Any]]
    confidence: float
    final_answer: str