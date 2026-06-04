#tests/test_graph_smoke.py

from langsmith import Client
from langsmith.evaluation import evaluate
from src.recruiter_copilot.graph import graph

client = Client()

def run_graph(inputs: dict) -> dict:
    result = graph.invoke({"user_query": inputs["input"]})  # or ["user_query"] if you chose that key
    return {
        "final_answer": result.get("final_answer", ""),
        "intent":       result.get("intent", ""),
        "response_mode": result.get("response_mode", ""),
    }

def check_intent(run, example) -> dict:
    expected = example.outputs.get("expected_intent", "")
    actual   = run.outputs.get("intent", "")
    return {
        "key":   "intent_correct",
        "score": int(actual == expected),
        "comment": f"expected={expected}, got={actual}",
    }

def check_no_unknown_candidates(run, example) -> dict:
    answer = run.outputs.get("final_answer", "")
    has_unknown = "unknown" in answer.lower()
    return {
        "key":   "no_unknown_candidates",
        "score": int(not has_unknown),
    }

def check_has_answer(run, example) -> dict:
    answer = run.outputs.get("final_answer", "")
    return {
        "key":   "has_answer",
        "score": int(len(answer.strip()) > 50),
    }

# at the bottom of tests/test_graph_smoke.py
results = evaluate(
    run_graph,
    data="recruiter_copilot_smoke_tests",
    evaluators=[check_intent, check_no_unknown_candidates, check_has_answer],
    experiment_prefix="smoke-v2",
    metadata={"version": "v2"},
)

df = results.to_pandas()
print(df.columns)
print(df.head(3))