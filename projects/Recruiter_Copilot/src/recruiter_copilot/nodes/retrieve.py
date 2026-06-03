# nodes/retrieve.py
import os

from src.recruiter_copilot.retrieval import retrieve_candidates
from src.recruiter_copilot.state import RecruiterCopilotState


def retrieve_node(state: RecruiterCopilotState) -> dict:
    """Unchanged — semantic top-k retrieval for search/pool_insight."""
    top_k = int(os.getenv("TOP_K", "5"))
    rewritten_query = state.get("rewritten_query", state["user_query"])

    retrieved_docs = retrieve_candidates(query=rewritten_query, k=top_k)
    return {
        "retrieved_docs": retrieved_docs,
        "retrieval_count": len(retrieved_docs),
    }


def retrieve_by_candidate_ids_node(state: RecruiterCopilotState) -> dict:
    """
    NEW — for deep_dive and compare intents.
    Runs semantic retrieval then filters to explicitly mentioned candidate IDs.
    Falls back to top-k semantic results if no IDs are found in the results.
    """
    target_ids = state.get("target_candidate_ids", [])
    user_query = state.get("rewritten_query", state["user_query"])
    top_k = int(os.getenv("TOP_K", "5"))

    all_docs = retrieve_candidates(query=user_query, k=top_k * 2)

    if target_ids:
        filtered = [
            doc for doc in all_docs
            if doc.get("candidate_id") in target_ids
            or any(tid in doc.get("file_name", "") for tid in target_ids)
        ]
        docs = filtered if filtered else all_docs[:top_k]
    else:
        docs = all_docs[:top_k]

    return {
        "retrieved_docs": docs,
        "retrieval_count": len(docs),
    }