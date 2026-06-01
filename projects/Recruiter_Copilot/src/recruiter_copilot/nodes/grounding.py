import os

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.recruiter_copilot.prompts.grounding import GROUNDING_CHECK_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


class GroundingCheck(BaseModel):
    grounded: bool = Field(
        description="Whether the answer is fully supported by the retrieved evidence"
    )
    reason: str = Field(
        description="Short explanation of whether the answer is grounded"
    )


def get_llm() -> ChatOllama:
    model_name = os.getenv("FAST_LLM_MODEL", "qwen3:1.7b")
    return ChatOllama(model=model_name, temperature=0)


def check_hallucination_node(state: RecruiterCopilotState) -> dict:
    generated_answer = state.get("generated_answer", "")
    retrieved_docs = state.get("retrieved_docs", [])

    if not generated_answer or not retrieved_docs:
        return {
            "hallucination_ok": False,
            "hallucination_reason": "Missing answer or retrieved evidence.",
        }

    llm = get_llm()
    checker = llm.with_structured_output(GroundingCheck)

    evidence_blocks = []
    for doc in retrieved_docs:
        evidence_blocks.append(
            "\n".join(
                [
                    f"Candidate ID: {doc.get('candidate_id', 'unknown')}",
                    f"Metadata: {doc.get('metadata', {})}",
                    f"Content: {doc.get('content', '')[:1800]}",
                ]
            )
        )

    evidence = "\n\n---\n\n".join(evidence_blocks)

    prompt = GROUNDING_CHECK_PROMPT.format(
        generated_answer=generated_answer,
        evidence=evidence,
    )
    result = checker.invoke(prompt)

    return {
        "hallucination_ok": result.grounded,
        "hallucination_reason": result.reason,
    }


def increment_retry_node(state: RecruiterCopilotState) -> dict:
    retry_count = state.get("retry_count", 0)
    return {
        "retry_count": retry_count + 1
    }


def answer_question_node(state: RecruiterCopilotState) -> dict:
    final_answer = state.get("generated_answer", "No answer generated.")

    hallucination_ok = state.get("hallucination_ok")
    hallucination_reason = state.get("hallucination_reason")

    if hallucination_ok is False and hallucination_reason:
        final_answer = (
            f"{final_answer}\n\n"
            f"Groundedness check warning: {hallucination_reason}"
        )

    return {
        "final_answer": final_answer
    }