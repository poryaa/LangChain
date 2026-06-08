# nodes/generate.py
import os

from langchain_ollama import ChatOllama

from src.recruiter_copilot.prompts.generate import GENERATE_ANSWER_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


def get_llm() -> ChatOllama:
    model_name = os.getenv("GENERATION_LLM_MODEL", "phi4-mini:latest")
    return ChatOllama(model=model_name, temperature=0)


def generate_answer_node(state: RecruiterCopilotState) -> dict:
    user_query = state["user_query"]
    rewritten_query = state.get("rewritten_query", user_query)
    response_mode = state.get("response_mode", "shortlist")
    docs = state.get("relevant_docs", state.get("retrieved_docs", []))

    if not docs:
        return {
            "generated_answer": (
                f"No relevant candidates found for: {user_query}\n"
                f"Rewritten query: {rewritten_query}"
            )
        }

    # Filter out docs with missing IDs / very weak evidence
    filtered_docs = []
    for doc in docs:
        cid = str(doc.get("candidate_id", "")).strip().lower()
        fname = str(doc.get("file_name", "")).strip().lower()
        content = doc.get("content", "") or ""

        if not content or len(content.strip()) < 80:
            continue
        if cid in ("", "unknown"):
            continue
        if fname in ("", "unknown", "unknown.pdf"):
            continue

        filtered_docs.append(doc)

    # Purely dynamic: use requested_k if present, otherwise use all filtered docs
    requested_k = state.get("requested_k")
    if requested_k is None:
        requested_k = len(filtered_docs)

    # Never exceed what you actually have
    requested_k = min(requested_k, len(filtered_docs))

    docs = filtered_docs[:requested_k]

    if not docs:
        return {
            "generated_answer": (
                f"No high-confidence candidates found for: {user_query}\n"
                f"Rewritten query: {rewritten_query}"
            )
        }

    llm = get_llm()

    evidence_blocks = []
    for doc in docs:
        evidence_blocks.append(
            "\n".join([
                f"Candidate ID: {doc.get('candidate_id', 'unknown')}",
                f"Resume file: {doc.get('file_name', 'unknown')}",
                f"Retrieval score: {doc.get('score', 0.0):.4f}",
                f"Content: {doc.get('content', '')[:400]}",
            ])
        )
    evidence = "\n\n---\n\n".join(evidence_blocks)

    prompt = GENERATE_ANSWER_PROMPT.format(
        user_query=user_query,
        rewritten_query=rewritten_query,
        response_mode=response_mode,
        evidence=evidence,
    )
    response = llm.invoke(prompt)
    return {"generated_answer": response.content}