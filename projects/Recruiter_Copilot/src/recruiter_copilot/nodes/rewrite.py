# nodes/rewrite.py
import os

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.recruiter_copilot.prompts.rewrite import REWRITE_QUERY_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


class RewriterOutput(BaseModel):
    rewritten_query: str = Field(
        description="Short semantic search query optimized for resume retrieval"
    )


def get_llm() -> ChatOllama:
    model_name = os.getenv("FAST_LLM_MODEL", "gemma3:1b")
    return ChatOllama(model=model_name, temperature=0)


def rewrite_query_node(state: RecruiterCopilotState) -> dict:
    user_query = state["user_query"]
    extracted_filters = state.get("extracted_filters", {})  # NEW

    llm = get_llm()
    structured_llm = llm.with_structured_output(RewriterOutput)
    prompt = REWRITE_QUERY_PROMPT.format(
        user_query=user_query,
        extracted_filters=extracted_filters,  # NEW
    )
    result = structured_llm.invoke(prompt)
    return {"rewritten_query": result.rewritten_query.strip()}