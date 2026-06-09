# nodes/understand.py

import os
from typing import List, Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.recruiter_copilot.prompts.understand import UNDERSTAND_QUERY_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


class ExtractedFilters(BaseModel):
    skills: List[str] = Field(default_factory=list)
    years_min: Optional[int] = None
    location: Optional[str] = None
    languages: List[str] = Field(default_factory=list)
    role: Optional[str] = None
    candidate_ids: List[str] = Field(default_factory=list)
    candidate_files: List[str] = Field(default_factory=list)


class QueryUnderstanding(BaseModel):
    intent: str = Field(
        description="One of: candidate_search, candidate_deep_dive, candidate_compare, pool_insight"
    )
    requested_k: Optional[int] = Field(
        default=None,
        description="Requested number of candidates, if explicitly asked, e.g. top 5, top 10."
    )
    filters: ExtractedFilters = Field(default_factory=ExtractedFilters)


def get_llm() -> ChatOllama:
    model_name = os.getenv("FAST_LLM_MODEL", "gemma3:1b")
    return ChatOllama(model=model_name, temperature=0)


INTENT_TO_RESPONSE_MODE = {
    "candidate_search": "shortlist",
    "candidate_deep_dive": "profile",
    "candidate_compare": "comparison",
    "pool_insight": "aggregation",
}


def understand_query_node(state: RecruiterCopilotState) -> dict:
    user_query = state["user_query"]

    llm = get_llm()
    structured_llm = llm.with_structured_output(QueryUnderstanding)
    prompt = UNDERSTAND_QUERY_PROMPT.format(user_query=user_query)
    result: QueryUnderstanding = structured_llm.invoke(prompt)

    intent = result.intent.strip().lower()
    if intent not in INTENT_TO_RESPONSE_MODE:
        intent = "candidate_search"

    filters = result.filters.model_dump()

    requested_k = result.requested_k
    if requested_k is not None:
        try:
            requested_k = max(int(requested_k), 1)
        except (TypeError, ValueError):
            requested_k = None

    return {
        "intent": intent,
        "response_mode": INTENT_TO_RESPONSE_MODE[intent],
        "requested_k": requested_k,
        "extracted_filters": filters,
        "target_candidate_ids": filters.get("candidate_ids", []),
        "target_candidate_files": filters.get("candidate_files", []),
    }