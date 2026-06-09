# nodes/generate.py
import os

from langchain_ollama import ChatOllama

from src.recruiter_copilot.prompts.generate import GENERATE_ANSWER_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


def get_llm() -> ChatOllama:
    model_name = os.getenv("GENERATION_LLM_MODEL", "phi4-mini:latest")
    return ChatOllama(model=model_name, temperature=0)


def _safe_str(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _get_resume_file(doc: dict) -> str:
    metadata = doc.get("metadata", {}) or {}
    return (
        _safe_str(doc.get("resume_file"))
        or _safe_str(doc.get("file_name"))
        or _safe_str(metadata.get("resume_file"))
        or _safe_str(metadata.get("file_name"))
        or _safe_str(metadata.get("source"))
        or "unknown"
    )


def _get_candidate_id(doc: dict) -> str:
    metadata = doc.get("metadata", {}) or {}
    return (
        _safe_str(doc.get("candidate_id"))
        or _safe_str(metadata.get("candidate_id"))
        or _safe_str(metadata.get("id"))
        or "unknown"
    )


def generate_answer_node(state: RecruiterCopilotState) -> dict:
    user_query = state["user_query"]
    rewritten_query = state.get("rewritten_query", user_query)
    response_mode = state.get("response_mode", "shortlist")

    candidates = state.get("selected_candidates", [])
    if not candidates:
        candidates = state.get("scored_candidates", [])
    if not candidates:
        candidates = state.get("retrieved_docs", [])

    if not candidates:
        return {
            "generated_answer": (
                f"No relevant candidates found for: {user_query}\n"
                f"Rewritten query: {rewritten_query}"
            ),
            "candidate_evidence": [],
        }

    requested_k = state.get("requested_k")
    if requested_k is None:
        requested_k = len(candidates)
    requested_k = max(int(requested_k), 1)

    filtered_candidates = []
    seen_candidate_ids = set()

    for doc in candidates:
        content = _safe_str(doc.get("content"))
        candidate_id = _get_candidate_id(doc)
        resume_file = _get_resume_file(doc)

        if len(content) < 80:
            continue
        if candidate_id.lower() in {"", "unknown"}:
            continue
        if resume_file.lower() in {"", "unknown", "unknown.pdf"}:
            continue

        if candidate_id in seen_candidate_ids:
            continue

        seen_candidate_ids.add(candidate_id)
        filtered_candidates.append(doc)

    filtered_candidates = filtered_candidates[:requested_k]

    if not filtered_candidates:
        return {
            "generated_answer": (
                f"No high-confidence candidates found for: {user_query}\n"
                f"Rewritten query: {rewritten_query}"
            ),
            "candidate_evidence": [],
        }

    candidate_evidence = []
    evidence_blocks = []

    for idx, doc in enumerate(filtered_candidates, start=1):
        content = _safe_str(doc.get("content"))
        candidate_id = _get_candidate_id(doc)
        resume_file = _get_resume_file(doc)

        evidence_item = {
            "rank": idx,
            "candidate_id": candidate_id,
            "resume_file": resume_file,
            "distance": doc.get("_distance", doc.get("score")),
            "base_similarity": doc.get("_base_similarity"),
            "bonus": doc.get("_bonus"),
            "final_score": doc.get("_score"),
            "score_reasons": doc.get("_score_reasons", []),
            "content_excerpt": content[:500],
        }
        candidate_evidence.append(evidence_item)

        reasons_text = ", ".join(evidence_item["score_reasons"]) or "semantic retrieval match"

        evidence_blocks.append(
            "\n".join(
                [
                    f"Rank: {idx}",
                    f"Candidate ID: {candidate_id}",
                    f"Resume file: {resume_file}",
                    f"Final score: {evidence_item['final_score']}",
                    f"Base similarity: {evidence_item['base_similarity']}",
                    f"Bonus: {evidence_item['bonus']}",
                    f"Why selected: {reasons_text}",
                    f"Evidence excerpt: {evidence_item['content_excerpt']}",
                ]
            )
        )

    evidence = "\n\n---\n\n".join(evidence_blocks)

    prompt = GENERATE_ANSWER_PROMPT.format(
        user_query=user_query,
        rewritten_query=rewritten_query,
        response_mode=response_mode,
        evidence=evidence,
    )

    llm = get_llm()
    response = llm.invoke(prompt)

    return {
        "generated_answer": response.content,
        "candidate_evidence": candidate_evidence,
    }