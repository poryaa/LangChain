import os

from src.recruiter_copilot.retrieval import retrieve_candidates
from src.recruiter_copilot.state import RecruiterCopilotState


def retrieve_node(state: RecruiterCopilotState) -> dict:
    top_k = int(os.getenv("TOP_K", "5"))
    rewritten_query = state.get("rewritten_query", state["user_query"])

    retrieved_docs = retrieve_candidates(query=rewritten_query, k=top_k)

    return {
        "retrieved_docs": retrieved_docs,
        "retrieval_count": len(retrieved_docs),
    }