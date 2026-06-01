import os

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.recruiter_copilot.prompts.grade import GRADE_DOC_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


class DocGrade(BaseModel):
    is_relevant: bool = Field(
        description="Whether the resume chunk is relevant to the recruiter request"
    )
    reason: str = Field(
        description="Short explanation for the relevance judgment"
    )


def get_llm() -> ChatOllama:
    model_name = os.getenv("FAST_LLM_MODEL", "qwen3:1.7b")
    return ChatOllama(model=model_name, temperature=0)


def grade_retrieved_docs_node(state: RecruiterCopilotState) -> dict:
    retrieved_docs = state.get("retrieved_docs", [])
    user_query = state["user_query"]
    rewritten_query = state.get("rewritten_query", user_query)

    if not retrieved_docs:
        return {
            "retrieved_docs": [],
            "retrieval_count": 0,
        }

    llm = get_llm()
    grader = llm.with_structured_output(DocGrade)

    relevant_docs = []

    for doc in retrieved_docs:
        content = doc.get("content", "")[:1800]
        prompt = GRADE_DOC_PROMPT.format(
            user_query=user_query,
            rewritten_query=rewritten_query,
            content=content,
        )
        result = grader.invoke(prompt)

        if result.is_relevant:
            enriched_doc = dict(doc)
            enriched_doc["relevance_reason"] = result.reason
            relevant_docs.append(enriched_doc)

    return {
        "retrieved_docs": relevant_docs,
        "retrieval_count": len(relevant_docs),
    }