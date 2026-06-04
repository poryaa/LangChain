# tests/llm_judge.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from langsmith import wrappers

# Load .env so GENERATION_LLM_MODEL is available
load_dotenv()

GENERATION_LLM_MODEL = os.getenv("GENERATION_LLM_MODEL", "gemma3:4b")

# Create an OpenAI-compatible client that talks to Ollama
# Ollama's OpenAI-compatible endpoint is usually http://localhost:11434/v1
# api_key can be any non-empty string (required by OpenAI client, ignored by Ollama)
ollama_client = OpenAI(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
)

# Wrap the Ollama-backed client so calls are traced in LangSmith
oai_client = wrappers.wrap_openai(ollama_client)


def intent_correct_llm_judge(
    inputs: dict, outputs: dict, reference: dict | None = None
) -> float:
    """
    LLM-as-a-judge: does the answer correctly address the recruiter query?
    Returns a float in [0.0, 1.0].
    """
    question = inputs.get("input", "")
    # match your run_graph outputs: it returns "final_answer"
    answer = (
        outputs.get("final_answer")
        or outputs.get("output")
        or outputs.get("answer", "")
    )
    ref = (reference or {}).get("expected_answer_summary", "")

    system_prompt = (
        "You are an impartial evaluator for a recruiter copilot.\n"
        "Given the user question, the model answer, and an optional reference "
        "answer, score how well the answer addresses the question.\n\n"
        "Return ONLY a number between 0.0 and 1.0:\n"
        "- 1.0 = clearly correct and complete\n"
        "- 0.5 = partially correct or missing important parts\n"
        "- 0.0 = incorrect, irrelevant, or no answer."
    )

    user_prompt = f"""
User question:
{question}

Model answer:
{answer}

Reference answer (may be empty):
{ref}
"""

    resp = oai_client.chat.completions.create(
        model=GENERATION_LLM_MODEL,  # e.g. "phi4-mini:latest" in Ollama
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    score_str = resp.choices[0].message.content.strip()
    return float(score_str)