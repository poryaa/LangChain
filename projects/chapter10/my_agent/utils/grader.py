# my_agent/utils/grader.py
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Whether the document is relevant to the question"
    )


llm_model = ChatOllama(model="llama3.1:latest", temperature=0)
structured_llm_grader = llm_model.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question.
If the document contains keywords or semantic meaning related to the question, grade it as relevant.
Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}")
])

retrieval_grader = grade_prompt | structured_llm_grader