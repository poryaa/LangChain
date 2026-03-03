# rag_core.py
from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_postgres.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

llm_model = ChatOllama(model="llama3.1:latest")


# ── Updated RAG prompt — now includes chat_history ──
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are **RoWise**, an AI assistant for questions about colleagues at ROSEN Rosenxt Group.
Always answer in a concise, friendly tone using complete sentences.

You have TWO sources of information:
1. CHAT HISTORY (previous messages in this conversation) — use this for personal/conversational questions.
2. CONTEXT (retrieved documents) — use this for knowledge questions about colleagues.

If the user asks something personal (e.g., their name, something they told you), ALWAYS answer from CHAT HISTORY.
For knowledge questions, use CONTEXT as your primary source.
If the answer is not in either source, reply: I DO NOT KNOW.

Do not fabricate names, roles, locations, or study information.

CONTEXT:
{context}"""
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


# Query rewriter prompt — contextualizes follow‑up questions
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and the latest user question, "
        "reformulate the question into a standalone question. "
        "Resolve all pronouns (he, she, him, it, they) to actual names. "
        "Do NOT answer — ONLY rewrite the question."
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


def chat_bot(state: State):
    last_msg = state["messages"][-1]
    question = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)

    # everything before last message is the history
    chat_history = state["messages"][:-1]

    # Rewrite query for retriever if there's history
    if chat_history:
        rewritten = (contextualize_q_prompt | llm_model).invoke({
            "chat_history": chat_history,
            "input": question,
        })
        search_query = rewritten.content
    else:
        search_query = question

    # RAG: retrieve docs from PGVector using rewritten (standalone) query
    docs = retriever.invoke(search_query)
    context = "\n\n".join(d.page_content for d in docs)

    # Now LLM sees BOTH chat history AND retrieved context
    answer_msg = (prompt | llm_model).invoke({
        "context": context,
        "chat_history": chat_history,
        "question": question,
    })

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
    input_state = {"messages": [question]}
    result = graph.invoke(input_state, THREAD_CONFIG)
    last_msg = result["messages"][-1]
    return last_msg.content
