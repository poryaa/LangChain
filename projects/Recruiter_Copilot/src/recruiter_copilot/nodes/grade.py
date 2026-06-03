# nodes/grade.py
import os
from typing import List

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.recruiter_copilot.prompts.grade import GRADE_DOC_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


class DocGrade(BaseModel):
    relevant_ids: List[str] = Field(
        description="List of candidate_id values that are clearly relevant to the recruiter request"
    )


def get_llm() -> ChatOllama:
    model_name = os.getenv("FAST_LLM_MODEL", "gemma3:1b")
    return ChatOllama(model=model_name, temperature=0)


def grade_retrieved_docs_node(state: RecruiterCopilotState) -> dict:
    retrieved_docs = state.get("retrieved_docs", [])
    user_query = state["user_query"]
    rewritten_query = state.get("rewritten_query", user_query)
    extracted_filters = state.get("extracted_filters", {})  # NEW

    if not retrieved_docs:
        return {"relevant_docs": [], "retrieval_count": 0}

    candidates_text = "\n\n---\n\n".join(
        f"Candidate ID: {doc.get('candidate_id')}\n"
        f"Resume file: {doc.get('file_name')}\n"
        f"Content: {doc.get('content', '')[:800]}"
        for doc in retrieved_docs
    )

    llm = get_llm()
    grader = llm.with_structured_output(DocGrade)
    prompt = GRADE_DOC_PROMPT.format(
        user_query=user_query,
        rewritten_query=rewritten_query,
        extracted_filters=extracted_filters,  # NEW
        candidates_text=candidates_text,
    )
    result = grader.invoke(prompt)

    relevant_ids = set(result.relevant_ids)
    relevant_docs = [
        doc for doc in retrieved_docs
        if doc.get("candidate_id") in relevant_ids
    ]
    return {"relevant_docs": relevant_docs, "retrieval_count": len(relevant_docs)}