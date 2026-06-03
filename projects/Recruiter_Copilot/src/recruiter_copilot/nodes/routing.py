# nodes/routing.py
import os
from typing import Literal

from src.recruiter_copilot.state import RecruiterCopilotState


# ── After understand_query ────────────────────────────────────────────────────

def route_after_understanding(
    state: RecruiterCopilotState,
) -> Literal["rewrite_query", "retrieve_by_ids"]:
    """
    candidate_search / pool_insight   → rewrite first, then retrieve semantically
    candidate_deep_dive / compare     → skip rewrite, retrieve by explicit ID
    """
    intent = state.get("intent", "candidate_search")
    if intent in ("candidate_deep_dive", "candidate_compare"):
        return "retrieve_by_ids"
    return "rewrite_query"


# ── After grade ───────────────────────────────────────────────────────────────

def route_after_grading(
    state: RecruiterCopilotState,
) -> Literal["generate_answer", "answer_question"]:
    if state.get("retrieval_count", 0) > 0:
        return "generate_answer"
    return "answer_question"


# ── After hallucination check ─────────────────────────────────────────────────

def route_after_hallucination_check(
    state: RecruiterCopilotState,
) -> Literal["answer_question", "increment_retry"]:
    if state.get("hallucination_ok", False):
        return "answer_question"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", int(os.getenv("MAX_RETRIES", "2")))

    if retry_count < max_retries:
        return "increment_retry"
    return "answer_question"