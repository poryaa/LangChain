# state.py
from typing import Any, Optional
from typing_extensions import TypedDict


class RecruiterCopilotState(TypedDict, total=False):
    # ── Input ───────────────────────────────────────────────────────
    user_query: str

    # ── Query Understanding ─────────────────────────────────────────
    intent: str                        # candidate_search | candidate_deep_dive | candidate_compare | pool_insight
    response_mode: str                 # shortlist | profile | comparison | aggregation
    rewritten_query: str               # semantic search string (search intent only)
    extracted_filters: dict            # {skills, years_min, location, languages, ...}
    target_candidate_ids: list[str]    # explicit IDs mentioned in query (deep-dive / compare)
    target_candidate_files: list[str]  # explicit file names mentioned in query

    # ── Retrieval ───────────────────────────────────────────────────
    retrieved_docs: list[Any]          # raw PGVector results
    relevant_docs: list[Any]           # after grading / reranking
    retrieval_count: int

    # ── Intent-specific payloads ────────────────────────────────────
    candidate_profiles: list[dict]     # deep-dive: structured profile per candidate
    comparison_payload: dict           # compare: side-by-side evidence dict
    aggregation_result: dict           # pool_insight: counts / grouped findings

    # ── Generation ──────────────────────────────────────────────────
    generated_answer: str
    final_answer: str

    # ── Quality checks ──────────────────────────────────────────────
    hallucination_ok: bool
    hallucination_reason: str
    retry_count: int
    max_retries: int

    # ── Misc ────────────────────────────────────────────────────────
    notes: Optional[str]