# rag_core.py
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


# 1) Vector DB connection (Docker Postgres)
CONNECTION = "postgresql+psycopg://langchain:langchain@127.0.0.1:6024/langchain"
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Reuse existing index (no reinsertion)
db = PGVector.from_existing_index(
    embedding=embedding_model,
    connection=CONNECTION,
)

retriever = db.as_retriever(search_kwargs={"k": 25})


# 2) LangGraph state + LLM
class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

llm_model = ChatOllama(model="tinyllama:latest")

prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant for questions about our colleagues.
First, use the information in CONTEXT as your primary source.
Try to answer in a short way.

CONTEXT:
{context}

Question: {question}
Answer:"""
)


def chat_bot(state: State):
    last_msg = state["messages"][-1]
    question = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)

    # RAG: retrieve docs from PGVector
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    # Build prompt string and call ChatOllama
    formatted_prompt = prompt.format(context=context, question=question)
    answer_msg = llm_model.invoke(formatted_prompt)

    return {"messages": [answer_msg]}


builder.add_node("chatbot", chat_bot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile(checkpointer=MemorySaver())

# Create a thread config (conversation memory)
THREAD_CONFIG = {"configurable": {"thread_id": "thread_1"}}


# 3) Clean function for outside world
def answer_question(question: str) -> str:
    """Run the graph for a single user question and return just the answer text."""
    # Note: for LangGraph, messages must be a list
    input_state = {"messages": [question]}
    result = graph.invoke(input_state, THREAD_CONFIG)
    # result["messages"] is a list with the history; last item is the latest AIMessage
    last_msg = result["messages"][-1]
    # last_msg is an AIMessage; return content only
    return last_msg.content
