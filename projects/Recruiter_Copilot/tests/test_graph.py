# tests/test_graph.py

from src.recruiter_copilot.graph import graph


def test_graph_returns_final_answer():
    initial_state = {
        "user_query": "Find strong Python ML candidates with RAG experience",
        "retry_count": 0,
        "max_retries": 2,
    }

    result = graph.invoke(initial_state)

    assert "final_answer" in result
    assert result["final_answer"] is not None
    assert len(result["final_answer"]) > 0


def test_graph_sets_retrieval_fields():
    initial_state = {
        "user_query": "Find data scientists with LangChain experience",
        "retry_count": 0,
        "max_retries": 2,
    }

    result = graph.invoke(initial_state)

    assert "rewritten_query" in result
    assert "retrieved_docs" in result
    assert "retrieval_count" in result
    assert result["retrieval_count"] >= 0