# nodes/grade.py

import os
from src.recruiter_copilot.state import RecruiterCopilotState

def grade_retrieved_docs_node(state: RecruiterCopilotState) -> dict:
    retrieved_docs = state.get("retrieved_docs", [])
    if not retrieved_docs:
        return {"relevant_docs": [], "retrieval_count": 0}

    requested_k = state.get("requested_k") or 10
    multiplier = int(os.getenv("GRADE_MULTIPLIER", "3"))
    filters = state.get("extracted_filters", {})

    def add_filter_bonus(doc: dict) -> float:
        text = (doc.get("content", "") + " " +
                " ".join(str(v) for v in doc.get("metadata", {}).values())
               ).lower()
        bonus = 0.0

        loc = (filters.get("location") or "").lower()
        if loc and loc in text:
            bonus += 0.05

        for skill in filters.get("skills", []):
            if skill.lower() in text:
                bonus += 0.02

        for lang in filters.get("languages", []):
            if lang.lower() in text:
                bonus += 0.02

        return bonus

    # Base: distance score (lower is better)
    for d in retrieved_docs:
        base = d.get("score", 0.0)
        d["_rank_score"] = -base + add_filter_bonus(d)  # higher = better

    docs_sorted = sorted(
        retrieved_docs,
        key=lambda d: d.get("_rank_score", 0.0),
        reverse=True,
    )

    limit = max(requested_k * multiplier, requested_k)
    relevant_docs = docs_sorted[:limit]

    return {
        "relevant_docs": relevant_docs,
        "retrieval_count": len(relevant_docs),
    }