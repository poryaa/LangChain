# nodes/grounding.py

import os

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.recruiter_copilot.prompts.grounding import GROUNDING_CHECK_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


class GroundingCheck(BaseModel):
    grounded: bool = Field(
        description=(
            "Whether the generated answer is fully supported by the provided evidence, "
            "does not introduce unsupported candidate facts, and follows the intended shortlist behavior."
        )
    )
    reason: str = Field(
        description="Short explanation of why the answer is grounded or not grounded."
    )


def get_llm() -> ChatOllama:
    model_name = os.getenv("FAST_LLM_MODEL", "gemma3:1b")
    return ChatOllama(model=model_name, temperature=0)


def check_hallucination_node(state: RecruiterCopilotState) -> dict:
    generated_answer = state.get("generated_answer", "")
    user_query = state.get("user_query", "")
    candidate_evidence = state.get("candidate_evidence", [])
    selected_candidates = state.get("selected_candidates", [])

    if not generated_answer or (not candidate_evidence and not selected_candidates):
        return {
            "grounding_ok": True,
            "grounding_reason": "No answer or evidence to verify.",
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

    evidence_blocks = []
    for item in evidence_items[:10]:
        evidence_blocks.append(
            "\n".join(
                [
                    f"Rank: {item.get('rank', 'unknown')}",
                    f"Candidate ID: {item.get('candidate_id', 'unknown')}",
                    f"Resume file: {item.get('resume_file', 'unknown')}",
                    f"Final score: {item.get('final_score', 'unknown')}",
                    f"Why selected: {', '.join(item.get('score_reasons', [])) or 'not specified'}",
                    f"Evidence excerpt: {item.get('content_excerpt', '')}",
                ]
            )
        )

    evidence = "\n\n---\n\n".join(evidence_blocks)

    prompt = GROUNDING_CHECK_PROMPT.format(
        user_query=user_query,
        generated_answer=generated_answer,
        evidence=evidence,
        expected_candidate_count=len(evidence_items),
    )

    llm = get_llm()
    checker = llm.with_structured_output(GroundingCheck)
    result = checker.invoke(prompt)

    return {
        "grounding_ok": result.grounded,
        "grounding_reason": result.reason,
    }


def increment_retry_node(state: RecruiterCopilotState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def answer_question_node(state: RecruiterCopilotState) -> dict:
    return {"final_answer": state.get("generated_answer", "No answer generated.")}