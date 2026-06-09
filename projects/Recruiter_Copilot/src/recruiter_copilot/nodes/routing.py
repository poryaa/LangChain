# nodes/routing.py

import os
from typing import Literal

from src.recruiter_copilot.state import RecruiterCopilotState


def route_after_understanding(
    state: RecruiterCopilotState,
) -> Literal["rewrite_query", "retrieve_by_ids"]:
    intent = state.get("intent", "candidate_search")
    if intent in ("candidate_deep_dive", "candidate_compare"):
        return "retrieve_by_ids"
    return "rewrite_query"


def route_after_grading(
    state: RecruiterCopilotState,
) -> Literal["generate_answer", "answer_question"]:
    if state.get("retrieval_count", 0) > 0:
        return "generate_answer"
    return "answer_question"


def route_after_hallucination_check(
    state: RecruiterCopilotState,
) -> Literal["answer_question", "increment_retry"]:
    grounding_ok = bool(state.get("grounding_ok", False))
    retry_count = int(state.get("retry_count", 0))
    max_retries = int(state.get("max_retries", os.getenv("MAX_RETRIES", "2")))

    if grounding_ok:
        return "answer_question"

    if retry_count >= max_retries:
        return "answer_question"

    return "increment_retry"