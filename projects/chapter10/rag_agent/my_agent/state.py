from typing import List, TypedDict
from langchain_core.documents import Document

class InputState(TypedDict):
    question: str

class GraphState(TypedDict):
    question: str
    search_query: str
    generation: str
    web_search: str
    documents: List[Document]