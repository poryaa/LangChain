from langchain_postgres.vectorstores import PGVector
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain

# --- RAG components ---

connection = "postgresql+psycopg://langchain:langchain@127.0.0.1:6024/langchain"
embedding_model = OllamaEmbeddings(model="llama3.1:latest")  # uses your running Ollama instance

# Re-use existing pgvector index (no reinsertion)
db = PGVector.from_existing_index(
    embedding=embedding_model,
    connection=connection,
)

llm_model = ChatOllama(model="llama3.1:latest")

prompt = ChatPromptTemplate.from_template(
    """Answer the question short based on the following retrieved documents.
If you don't know the answer, say "I don't know".
Context: {context}
Question: {question}
Answer:"""
)

retriever = db.as_retriever(search_kwargs={"k": 15})


@chain
def q_a(question: str):
    # fetch relevant docs
    docs = retriever.invoke(question)

    # optionally format docs (just texts)
    context = "\n\n".join(d.page_content for d in docs)

    # format prompt
    formatted = prompt.invoke({"context": context, "question": question})

    # generate answer
    answer = llm_model.invoke(formatted)
    return answer


def main():
    print("RAG CLI – type 'exit' to quit.")
    while True:
        question = input("\nYour question: ")
        if question.strip().lower() in {"exit", "quit"}:
            break

        res = q_a.invoke(question)
        # ChatOllama returns an AIMessage; get the text content
        print("\nAnswer:", getattr(res, "content", res))


if __name__ == "__main__":
    main()
