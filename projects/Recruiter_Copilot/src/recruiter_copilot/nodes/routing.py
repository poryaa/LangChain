import os
from typing import Literal

from src.recruiter_copilot.state import RecruiterCopilotState


def route_after_grading(state: RecruiterCopilotState) -> Literal["generate_answer", "answer_question"]:
    if state.get("retrieval_count", 0) > 0:
        return "generate_answer"
    return "answer_question"


def route_after_hallucination_check(
    state: RecruiterCopilotState,
) -> Literal["answer_question", "increment_retry"]:
    if state.get("hallucination_ok", False):
        return "answer_question"

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", int(os.getenv("MAX_RETRIES", "1")))

    if retry_count < max_retries:
        return "increment_retry"

    return "answer_question"