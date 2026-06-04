# tests/test_eval_preproduction.py

from dotenv import load_dotenv
load_dotenv()

from langsmith.evaluation import evaluate
from src.recruiter_copilot.graph import graph

# import your LLM-as-a-judge function from tests/llm_judge.py
from tests.llm_judge import intent_correct_llm_judge


def run_graph(inputs: dict) -> dict:
    result = graph.invoke({"user_query": inputs["input"]})
    return {
        "final_answer":  result.get("final_answer", ""),
        "intent":        result.get("intent", ""),
        "response_mode": result.get("response_mode", ""),
    }


# ── Heuristic Evaluator 1: Intent classification ──────────────────────────────
def check_intent(run, example) -> dict:
    expected = example.outputs.get("expected_intent", "")
    actual   = run.outputs.get("intent", "")
    return {
        "key":     "intent_correct",
        "score":   int(actual == expected),
        "comment": f"expected={expected}, got={actual}",
    }


# ── Heuristic Evaluator 2: Response mode ──────────────────────────────────────
def check_response_mode(run, example) -> dict:
    expected = example.outputs.get("expected_response_mode", "")
    actual   = run.outputs.get("response_mode", "")
    return {
        "key":     "response_mode_correct",
        "score":   int(actual == expected),
        "comment": f"expected={expected}, got={actual}",
    }


# ── Heuristic Evaluator 3: Expected skills mentioned in answer ────────────────
def check_skills_mentioned(run, example) -> dict:
    answer          = (run.outputs.get("final_answer") or "").lower()
    expected_skills = example.outputs.get("expected_skills_mentioned", [])

    if not expected_skills:
        return {"key": "skills_mentioned", "score": None, "comment": "no skills to check"}

    matched = [s for s in expected_skills if s.lower() in answer]
    score   = len(matched) / len(expected_skills)
    return {
        "key":     "skills_mentioned",
        "score":   round(score, 2),
        "comment": f"{len(matched)}/{len(expected_skills)} skills found: {matched}",
    }


# ── Heuristic Evaluator 4: Expected candidate IDs present in answer ───────────
def check_candidate_ids(run, example) -> dict:
    answer       = (run.outputs.get("final_answer") or "").lower()
    expected_ids = example.outputs.get("expected_candidate_ids", [])

    if not expected_ids:
        return {
            "key": "candidate_ids_present",
            "score": None,
            "comment": "pool_insight — no IDs expected",
        }

    matched = [cid for cid in expected_ids if cid.lower() in answer]
    score   = len(matched) / len(expected_ids)
    return {
        "key":     "candidate_ids_present",
        "score":   round(score, 2),
        "comment": f"{len(matched)}/{len(expected_ids)} IDs found: {matched}",
    }


# ── Heuristic Evaluator 5: Answer is non-empty and not padded ─────────────────
def check_has_answer(run, example) -> dict:
    answer      = (run.outputs.get("final_answer") or "").strip()
    has_unknown = "unknown" in answer.lower()
    return {
        "key":     "has_real_answer",
        "score":   int(len(answer) > 50 and not has_unknown),
        "comment": (
            "answer too short or contains unknown candidates"
            if not (len(answer) > 50 and not has_unknown)
            else "ok"
        ),
    }


# ── Wrapper: use intent_correct_llm_judge (inputs, outputs, reference) as evaluator ──
def check_intent_llm(run, example) -> dict:
    """
    Adapter that calls tests.llm_judge.intent_correct_llm_judge and returns
    a LangSmith evaluator dict.
    """
    score = intent_correct_llm_judge(
        inputs=example.inputs,
        outputs=run.outputs,
        reference=example.outputs,
    )
    return {
        "key":     "intent_correct_llm",
        "score":   score,
        "comment": f"LLM-judge intent score={score}",
    }


if __name__ == "__main__":
    results = evaluate(
        run_graph,
        data="recruiter_copilot_preproduction_evals",
        evaluators=[
            check_intent,
            check_response_mode,
            check_skills_mentioned,
            check_candidate_ids,
            check_has_answer,
            check_intent_llm,   # ← your LLM-as-a-judge metric
        ],
        experiment_prefix="preproduction-heuristic-v1",
        metadata={"version": "v2", "stage": "preproduction"},
        # NOTE: max_examples is not supported in your evaluate() signature, so omit it
    )

    df = results.to_pandas()
    print(
        df[
            [
                "inputs.input",
                "evals.intent_correct.score",
                "evals.response_mode_correct.score",
                "evals.skills_mentioned.score",
                "evals.candidate_ids_present.score",
                "evals.has_real_answer.score",
                "evals.intent_correct_llm.score",  # new LLM-as-judge column
            ]
        ].to_string()
    )