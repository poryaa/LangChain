# filename: rosenxt_bot_cli.py

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


def main():
    llm_model = ChatOllama(model="llama3.1:latest")

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Your name is Rosenxt_bot. "
            "Answer the questions in short."
        ),
        ("human", "question: {Question}"),
    ])

    chain = template | llm_model

    print("Rosenxt_bot CLI. Type 'exit' to quit.")
    while True:
        user_input = input("Your Question: ")
        if user_input.lower() in {"exit", "quit"}:
            break

        response = chain.invoke({"Question": user_input})
        # response is a BaseMessage; the text is usually in .content
        print("Rosenxt_bot:", response.content)


if __name__ == "__main__":
    main()
