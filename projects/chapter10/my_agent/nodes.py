# my_agent/nodes.py
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchResults

from my_agent.chains import question_rewriter, rag_chain
from my_agent.state import GraphState
from my_agent.utils.grader import retrieval_grader
from my_agent.utils.retriever import retriever

web_search_tool = DuckDuckGoSearchResults(output_format="list")


def retrieve(state: GraphState):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state: GraphState):
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents)
    generation = rag_chain.invoke({"context": context, "question": question})
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
    }


def grade_documents(state: GraphState):
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []

    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        if score.binary_score == "yes":
            filtered_docs.append(d)

    web_search = "Yes" if len(filtered_docs) == 0 else "No"

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search": web_search,
    }


def transform_query(state: GraphState):
    question = state["question"]
    better_question = question_rewriter.invoke({"question": question})
    return {"question": question, "search_query": better_question}


def web_search(state: GraphState):
    question = state["question"]
    search_query = state.get("search_query", question)
    documents = state["documents"]

    results = web_search_tool.invoke(search_query)
    web_results_text = "\n".join(
        f"{item.get('title', '')}\n{item.get('snippet', '')}"
        for item in results
    )
    documents.append(Document(page_content=web_results_text))

    return {
        "documents": documents,
        "question": question,
        "search_query": search_query,
    }


def decide_to_generate(state: GraphState):
    if state["web_search"] == "Yes":
        return "transform_query"
    return "generate"