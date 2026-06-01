import os
from typing import Optional

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from src.recruiter_copilot.prompts.rewrite import REWRITE_QUERY_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


class RewriterOutput(BaseModel):
    rewritten_query: str = Field(
        description="Short semantic search query optimized for resume retrieval"
    )
    city: Optional[str] = Field(
        default=None,
        description="Candidate city if explicitly mentioned in the recruiter query",
    )
    country: Optional[str] = Field(
        default=None,
        description="Candidate country if explicitly mentioned in the recruiter query",
    )


def get_llm() -> ChatOllama:
    model_name = os.getenv("FAST_LLM_MODEL", "qwen3:1.7b")
    return ChatOllama(model=model_name, temperature=0)


def rewrite_query_node(state: RecruiterCopilotState) -> dict:
    user_query = state["user_query"]

    llm = get_llm()
    structured_llm = llm.with_structured_output(RewriterOutput)

    prompt = REWRITE_QUERY_PROMPT.format(user_query=user_query)
    result = structured_llm.invoke(prompt)

    return {
        "rewritten_query": result.rewritten_query,
        "inferred_filters": {
            "city": result.city,
            "country": result.country,
        },
    }