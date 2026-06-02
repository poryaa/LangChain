import os
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

load_dotenv()


def retrieve_candidates(query: str, k: int = 3) -> list[dict[str, Any]]:
    connection = os.getenv("POSTGRES_CONNECTION")
    collection_name = os.getenv("PGVECTOR_COLLECTION_NAME")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")

    if not connection:
        raise ValueError("POSTGRES_CONNECTION is not set")
    if not collection_name:
        raise ValueError("PGVECTOR_COLLECTION_NAME is not set")

    embeddings = OllamaEmbeddings(model=embedding_model_name)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    results = vector_store.similarity_search_with_score(query=query, k=k)

    retrieved_docs = []
    for i, (doc, score) in enumerate(results, start=1):
        raw_candidate_id = doc.metadata.get("candidate_id") or doc.metadata.get("source") or f"candidate_{i}"
        raw_file_name = doc.metadata.get("file_name") or doc.metadata.get("source") or "unknown"

        candidate_id = str(raw_candidate_id).strip()
        file_name = os.path.basename(str(raw_file_name).strip())

        retrieved_docs.append(
            {
                "candidate_id": candidate_id,
                "file_name": file_name,
                "content": doc.page_content,
                "score": float(score),
                "metadata": doc.metadata,
            }
        )

    return retrieved_docs