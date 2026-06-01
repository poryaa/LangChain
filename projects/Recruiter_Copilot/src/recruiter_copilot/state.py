from typing import Any, Optional
from typing_extensions import TypedDict


class RecruiterCopilotState(TypedDict, total=False):
    user_query: str
    rewritten_query: str

    retrieved_docs: list[Any]
    relevant_docs: list[Any]
    retrieval_count: int

    is_retrieval_relevant: bool
    generated_answer: str
    final_answer: str

    is_grounded: bool
    retry_count: int
    max_retries: int

    notes: Optional[str]