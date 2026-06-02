import os
from langchain_ollama import ChatOllama
from src.recruiter_copilot.prompts.generate import GENERATE_ANSWER_PROMPT
from src.recruiter_copilot.state import RecruiterCopilotState


def get_llm() -> ChatOllama:
    model_name = os.getenv("GENERATION_LLM_MODEL", "phi4-mini:latest")
    return ChatOllama(model=model_name, temperature=0)


def generate_answer_node(state: RecruiterCopilotState) -> dict:
    user_query = state["user_query"]
    rewritten_query = state.get("rewritten_query", user_query)
    docs = state.get("relevant_docs", state.get("retrieved_docs", []))

    if not docs:
        return {
            "generated_answer": (
                f"Recruiter query: {user_query}\n"
                f"Rewritten query: {rewritten_query}\n\n"
                "No relevant candidates found based on the retrieved evidence."
            )
        }

    llm = get_llm()

    evidence_blocks = []
    for doc in docs[:5]:
        evidence_blocks.append(
            "\n".join(
                [
                    f"Candidate ID: {doc.get('candidate_id', 'unknown')}",
                    f"Resume file name: {doc.get('file_name', 'unknown')}",
                    f"Retrieval score: {doc.get('score', 0.0)}",
                    f"Relevance reason: {doc.get('relevance_reason', '')}",
                    f"Content: {doc.get('content', '')[:1200]}",
                ]
            )
        )

    evidence = "\n\n---\n\n".join(evidence_blocks)

    prompt = GENERATE_ANSWER_PROMPT.format(
        user_query=user_query,
        rewritten_query=rewritten_query,
        evidence=evidence,
    )
    response = llm.invoke(prompt)

    return {
        "generated_answer": response.content,
    }