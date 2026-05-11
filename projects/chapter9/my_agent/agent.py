from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from langchain_ollama import ChatOllama


class State(TypedDict):
    question: str
    answer: str


llm = ChatOllama(model="qwen3:0.6b")


def answer_node(state: State) -> State:
    response = llm.invoke(state["question"])
    return {"answer": response.content}


builder = StateGraph(State)

builder.add_node("answer", answer_node)
builder.add_edge(START, "answer")
builder.add_edge("answer", END)

graph = builder.compile()