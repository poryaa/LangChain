# state.py

from typing import Any, Optional
from typing_extensions import TypedDict


class RecruiterCopilotState(TypedDict, total=False):
    # ── Input ───────────────────────────────────────────────────────
    user_query: str

    # ── Query Understanding ─────────────────────────────────────────
    intent: str
    response_mode: str
    rewritten_query: str
    extracted_filters: dict[str, Any]
    target_candidate_ids: list[str]
    target_candidate_files: list[str]
    requested_k: int

    # ── Retrieval ───────────────────────────────────────────────────
    retrieved_docs: list[dict[str, Any]]
    retrieval_count: int

    # ── Deterministic ranking / selection ───────────────────────────
    scored_candidates: list[dict[str, Any]]
    selected_candidates: list[dict[str, Any]]
    selected_count: int

    # ── Evidence assembly ───────────────────────────────────────────
    candidate_evidence: list[dict[str, Any]]

    # ── Generation ──────────────────────────────────────────────────
    generated_answer: str
    final_answer: str

    # ── Quality checks ──────────────────────────────────────────────
    grounding_ok: bool
    grounding_reason: str
    unsupported_candidate_ids: list[str]
    retry_count: int
    max_retries: int

    # ── Misc ────────────────────────────────────────────────────────
    notes: Optional[str]