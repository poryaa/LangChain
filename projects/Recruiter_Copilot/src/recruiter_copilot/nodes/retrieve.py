# nodes/retrieve.py
import os

from src.recruiter_copilot.retrieval import retrieve_candidates
from src.recruiter_copilot.state import RecruiterCopilotState


def retrieve_node(state: RecruiterCopilotState) -> dict:
    """Semantic broad retrieval for shortlist generation before deterministic reranking."""
    default_top_k = int(os.getenv("TOP_K", "10"))
    requested_k = max(int(state.get("requested_k") or 5), 1)

    # Retrieve a broader pool than the final answer size so later scoring can rerank it.
    top_k = max(default_top_k, requested_k * 3)

    rewritten_query = state.get("rewritten_query", state["user_query"])
    retrieved_docs = retrieve_candidates(query=rewritten_query, k=top_k)

    return {
        "retrieved_docs": retrieved_docs,
        "retrieval_count": len(retrieved_docs),
    }


def retrieve_by_candidate_ids_node(state: RecruiterCopilotState) -> dict:
    """
    For deep-dive / compare style requests:
    retrieve semantically, then try to keep only explicitly requested candidate IDs or files.
    Falls back to the top semantic results if explicit matches are not found.
    """
    target_ids = {str(x).strip().lower() for x in state.get("target_candidate_ids", [])}
    target_files = {str(x).strip().lower() for x in state.get("target_candidate_files", [])}

    user_query = state.get("rewritten_query", state["user_query"])
    top_k = max(int(os.getenv("TOP_K", "5")), 1)

    all_docs = retrieve_candidates(query=user_query, k=top_k * 2)

    if target_ids or target_files:
        filtered = []
        for doc in all_docs:
            metadata = doc.get("metadata", {}) or {}

            candidate_id = str(
                doc.get("candidate_id")
                or metadata.get("candidate_id")
                or metadata.get("id")
                or ""
            ).strip().lower()

            file_name = str(
                doc.get("file_name")
                or doc.get("resume_file")
                or metadata.get("file_name")
                or metadata.get("resume_file")
                or metadata.get("source")
                or ""
            ).strip().lower()

            if (candidate_id and candidate_id in target_ids) or (
                file_name and file_name in target_files
            ):
                filtered.append(doc)

        docs = filtered if filtered else all_docs[:top_k]
    else:
        docs = all_docs[:top_k]

    return {
        "retrieved_docs": docs,
        "retrieval_count": len(docs),
    }