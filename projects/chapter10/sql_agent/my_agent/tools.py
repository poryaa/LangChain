from langchain_core.tools import tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from my_agent.prompts import QUERY_CHECK_SYSTEM, QUERY_RESULT_CHECK_SYSTEM

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = ChatOllama(model="llama3.1", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()

query_check_prompt = ChatPromptTemplate.from_messages(
    [("system", QUERY_CHECK_SYSTEM), ("user", "{query}")]
)
query_check = query_check_prompt | llm


@tool
def check_query_tool(query: str) -> str:
    """Use this tool to double check if your query is correct before executing it."""
    return query_check.invoke({"query": query}).content


query_result_check_prompt = ChatPromptTemplate.from_messages(
    [("system", QUERY_RESULT_CHECK_SYSTEM), ("user", "{query_result}")]
)
query_result_check = query_result_check_prompt | llm


@tool
def check_result(query_result: str) -> str:
    """Use this tool to check the query result from the database to confirm it is not empty and is relevant."""
    return query_result_check.invoke({"query_result": query_result}).content


tools.append(check_query_tool)
tools.append(check_result)