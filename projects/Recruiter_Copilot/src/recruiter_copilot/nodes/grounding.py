# nodes/grounding.py

import os
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.recruiter_copilot.prompts.grounding import GROUNDING_CHECK_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


class GroundingCheck(BaseModel):
    grounded: bool = Field(
        description="Whether the answer is fully supported by the retrieved evidence "
                    "AND does not introduce new, discriminatory constraints."
    )
    reason: str = Field(
        description="Short explanation of why the answer is or is not grounded and safe."
    )


def get_llm() -> ChatOllama:
    model_name = os.getenv("FAST_LLM_MODEL", "gemma3:1b")
    return ChatOllama(model=model_name, temperature=0)


def check_hallucination_node(state: RecruiterCopilotState) -> dict:
    generated_answer = state.get("generated_answer", "")
    user_query = state.get("user_query", "")
    docs = state.get("relevant_docs", state.get("retrieved_docs", []))

    if not generated_answer or not docs:
        return {
            "hallucination_ok": True,
            "hallucination_reason": "No answer or evidence to verify.",
        }

    llm = get_llm()
    checker = llm.with_structured_output(GroundingCheck)

    # Limit evidence size for speed, but cover all candidates used in generation
    max_docs = min(len(docs), 10)
    evidence_blocks = []
    for doc in docs[:max_docs]:
        evidence_blocks.append(
            "\n".join(
                [
                    f"Candidate ID: {doc.get('candidate_id', 'unknown')}",
                    f"Content: {doc.get('content', '')[:400]}",
                ]
            )
        )

    evidence = "\n\n---\n\n".join(evidence_blocks)

    prompt = GROUNDING_CHECK_PROMPT.format(
        user_query=user_query,
        generated_answer=generated_answer,
        evidence=evidence,
    )
    result = checker.invoke(prompt)

    return {
        "hallucination_ok": result.grounded,
        "hallucination_reason": result.reason,
    }


def increment_retry_node(state: RecruiterCopilotState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def answer_question_node(state: RecruiterCopilotState) -> dict:
    return {"final_answer": state.get("generated_answer", "No answer generated.")}