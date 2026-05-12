# my_agent/chains.py
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:latest", temperature=0)

rag_prompt = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you do not know the answer, just say you do not know.

Question: {question}

Context:
{context}

Answer:
"""
)

rag_chain = rag_prompt | llm | StrOutputParser()

rewrite_system = """You are a question re-writer that converts an input question to a better version optimized for web search.
Look at the input and reason about the underlying semantic intent."""

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system),
    ("human", "Here is the initial question:\n\n{question}\n\nFormulate an improved question.")
])

question_rewriter = rewrite_prompt | llm | StrOutputParser()