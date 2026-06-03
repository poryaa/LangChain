# graph.py
import os

from langgraph.graph import END, START, StateGraph

from src.recruiter_copilot.nodes.generate import generate_answer_node
from src.recruiter_copilot.nodes.grade import grade_retrieved_docs_node
from src.recruiter_copilot.nodes.grounding import (
    answer_question_node,
    check_hallucination_node,
    increment_retry_node,
)
from src.recruiter_copilot.nodes.retrieve import retrieve_node, retrieve_by_candidate_ids_node
from src.recruiter_copilot.nodes.rewrite import rewrite_query_node
from src.recruiter_copilot.nodes.understand import understand_query_node
from src.recruiter_copilot.nodes.routing import (
    route_after_understanding,
    route_after_grading,
    route_after_hallucination_check,
)
from src.recruiter_copilot.state import RecruiterCopilotState


def as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


builder = StateGraph(RecruiterCopilotState)

# ── Nodes ─────────────────────────────────────────────────────────────────────
builder.add_node("understand_query",     understand_query_node)
builder.add_node("rewrite_query",        rewrite_query_node)
builder.add_node("retrieve",             retrieve_node)
builder.add_node("retrieve_by_ids",      retrieve_by_candidate_ids_node)
builder.add_node("grade_retrieved_docs", grade_retrieved_docs_node)
builder.add_node("generate_answer",      generate_answer_node)
builder.add_node("check_hallucination",  check_hallucination_node)
builder.add_node("increment_retry",      increment_retry_node)
builder.add_node("answer_question",      answer_question_node)

# ── Entry ─────────────────────────────────────────────────────────────────────
builder.add_edge(START, "understand_query")

# ── Intent routing ────────────────────────────────────────────────────────────
builder.add_conditional_edges(
    "understand_query",
    route_after_understanding,
    {
        "rewrite_query":   "rewrite_query",
        "retrieve_by_ids": "retrieve_by_ids",
    },
)

# ── Search / pool-insight path ────────────────────────────────────────────────
builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("retrieve",      "grade_retrieved_docs")

# ── Deep-dive / compare path ──────────────────────────────────────────────────
builder.add_edge("retrieve_by_ids", "grade_retrieved_docs")

# ── After grading ─────────────────────────────────────────────────────────────
builder.add_conditional_edges(
    "grade_retrieved_docs",
    route_after_grading,
    {
        "generate_answer": "generate_answer",
        "answer_question": "answer_question",
    },
)

# ── Grounding (optional, defaults ON) ─────────────────────────────────────────
enable_groundedness = as_bool(
    os.getenv("ENABLE_GROUNDEDNESS_CHECK", "true"),
    default=True,
)

if enable_groundedness:
    builder.add_edge("generate_answer", "check_hallucination")
    builder.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination_check,
        {
            "increment_retry": "increment_retry",
            "answer_question": "answer_question",
        },
    )
    builder.add_edge("increment_retry", "generate_answer")
else:
    builder.add_edge("generate_answer", "answer_question")

builder.add_edge("answer_question", END)

graph = builder.compile()