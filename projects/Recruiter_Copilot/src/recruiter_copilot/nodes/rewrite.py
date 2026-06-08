# nodes/rewrite.py
import os
import re
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from src.recruiter_copilot.prompts.rewrite import REWRITE_QUERY_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


class RewriterOutput(BaseModel):
    rewritten_query: str = Field(
        description="Short semantic search query optimized for resume retrieval"
    )
    requested_k: int | None = Field(
        default=None,
        description=(
            "How many candidates the recruiter is asking for. "
            "Extract this from phrases like 'top 10', 'find 5', 'give me 3 candidates'. "
            "Return None if no specific number is requested."
        ),
    )


def get_llm() -> ChatOllama:
    model_name = os.getenv("FAST_LLM_MODEL", "gemma3:1b")
    return ChatOllama(model=model_name, temperature=0)


def _fallback_parse_k(user_query: str) -> int | None:
    text = user_query.lower()
    m = re.search(r"(?:top|first)\s+(\d+)", text)
    if not m:
        m = re.search(r"(\d+)\s+(?:candidates?|profiles?)", text)
    if m:
        return int(m.group(1))
    return None


def rewrite_query_node(state: RecruiterCopilotState) -> dict:
    user_query = state["user_query"]
    extracted_filters = state.get("extracted_filters", {})

    llm = get_llm()
    structured_llm = llm.with_structured_output(RewriterOutput)
    prompt = REWRITE_QUERY_PROMPT.format(
        user_query=user_query,
        extracted_filters=extracted_filters,
    )

    result: RewriterOutput = structured_llm.invoke(prompt)

    requested_k = result.requested_k
    if requested_k is None:
        requested_k = _fallback_parse_k(user_query)

    return {
        "rewritten_query": result.rewritten_query.strip(),
        "requested_k": requested_k,
    }