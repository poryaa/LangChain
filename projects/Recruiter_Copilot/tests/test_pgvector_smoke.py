#Test it recognize the pgvector docker container when it runs with:
#pytest tests/test_pgvector_smoke.py -v -s
# src/recruiter_copilot/retrieval.py
import os

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector


def test_pgvector_smoke_retrieval():
    load_dotenv()

    connection = os.getenv("POSTGRES_CONNECTION")
    collection_name = os.getenv("PGVECTOR_COLLECTION_NAME")
    embedding_model_name = os.getenv("EMBEDDING_MODEL")

    assert connection, "POSTGRES_CONNECTION is not set in .env"
    assert collection_name, "PGVECTOR_COLLECTION_NAME is not set in .env"

    embeddings = OllamaEmbeddings(model=embedding_model_name)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=connection,
        use_jsonb=True,
    )

    query = "Data science with RAG experience in Germany worked in Rosenxt group in lingen"
    results = vector_store.similarity_search_with_score(query=query, k=3)

    assert results is not None
    assert len(results) > 0, "No results returned from PGVector"

    print(f"\nQuery: {query}")
    print(f"Results returned: {len(results)}\n")

    for i, (doc, score) in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Score: {score}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content preview: {doc.page_content[:500]}")
        print()

    first_doc, first_score = results[0]

    assert first_doc.page_content is not None
    assert len(first_doc.page_content.strip()) > 0
    assert isinstance(first_score, float)