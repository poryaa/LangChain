from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

llm_model = ChatOllama(model="tinyllama:latest")

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Your name is Rosenxt_bot. Answer the question."),
    ("human", "question: {Question}")
])

chain = template | llm_model

def answer_question(question: str) -> str:
    msg = chain.invoke({"Question": question})
    # msg is an AIMessage; just return its content
    return msg.content
