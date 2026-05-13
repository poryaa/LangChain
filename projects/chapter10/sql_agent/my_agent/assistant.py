from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.prebuilt import ToolNode

from my_agent.prompts import QUERY_GEN_SYSTEM
from my_agent.state import State
from my_agent.tools import llm, tools


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools_: list):
    return ToolNode(tools_).with_fallbacks(
        [RunnableLambda(handle_tool_error)],
        exception_key="error",
    )


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            state = {**state}
            result = self.runnable.invoke(state)
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list) and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


query_gen_prompt = ChatPromptTemplate.from_messages(
    [("system", QUERY_GEN_SYSTEM), ("placeholder", "{messages}")]
)

assistant_runnable = query_gen_prompt | llm.bind_tools(tools)