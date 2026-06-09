# nodes/grounding.py

import os
import re

from src.recruiter_copilot.state import RecruiterCopilotState


CANDIDATE_ID_RE = re.compile(r"\b[a-f0-9]{32}\b")
MAX_REASON_IDS = 5


def _extract_candidate_ids(text: str) -> list[str]:
    if not text:
        return []
    return sorted(set(CANDIDATE_ID_RE.findall(text)))


def _build_short_reason(unsupported_ids: list[str]) -> str:
    if not unsupported_ids:
        return "Answer is grounded in the provided evidence."

    shown = unsupported_ids[:MAX_REASON_IDS]
    extra = len(unsupported_ids) - len(shown)
    base = f"Unsupported candidate IDs in answer: {', '.join(shown)}"
    if extra > 0:
        base += f" ... and {extra} more."
    return base


def check_hallucination_node(state: RecruiterCopilotState) -> dict:
    generated_answer = state.get("generated_answer", "") or ""
    candidate_evidence = state.get("candidate_evidence", []) or []
    selected_candidates = state.get("selected_candidates", []) or []

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", int(os.getenv("MAX_RETRIES", "2")))

    if not generated_answer or (not candidate_evidence and not selected_candidates):
        return {
            "grounding_ok": True,
            "grounding_reason": "No answer or evidence to verify.",
            "unsupported_candidate_ids": [],
            "retry_count": retry_count,
            "max_retries": max_retries,
        }

    evidence_items = candidate_evidence
    if not evidence_items:
        evidence_items = []
        for idx, doc in enumerate(selected_candidates, start=1):
            evidence_items.append(
                {
                    "rank": idx,
                    "candidate_id": doc.get("candidate_id", "unknown"),
                    "resume_file": doc.get("resume_file", doc.get("file_name", "unknown")),
                    "final_score": doc.get("_score"),
                    "score_reasons": doc.get("_score_reasons", []),
                    "content_excerpt": (doc.get("content", "") or "")[:500],
                }
            )

    evidence_candidate_ids = set()
    for item in evidence_items[:10]:
        cid = item.get("candidate_id", "unknown")
        if cid and cid != "unknown":
            evidence_candidate_ids.add(cid)

    answer_candidate_ids = set(_extract_candidate_ids(generated_answer))
    unsupported_ids = sorted(answer_candidate_ids - evidence_candidate_ids)

    grounding_ok = len(unsupported_ids) == 0
    grounding_reason = _build_short_reason(unsupported_ids)

    return {
        "grounding_ok": grounding_ok,
        "grounding_reason": grounding_reason,
        "unsupported_candidate_ids": unsupported_ids,
        "retry_count": retry_count,
        "max_retries": max_retries,
    }


def increment_retry_node(state: RecruiterCopilotState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def answer_question_node(state: RecruiterCopilotState) -> dict:
    return {"final_answer": state.get("generated_answer", "No answer generated.")}