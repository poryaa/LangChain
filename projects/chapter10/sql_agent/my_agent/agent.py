from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition

from my_agent.assistant import Assistant, assistant_runnable, create_tool_node_with_fallback
from my_agent.state import State
from my_agent.tools import tools

builder = StateGraph(State)

builder.add_node("assistant", Assistant(assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(tools))

builder.set_entry_point("assistant")

builder.add_conditional_edges(
    "assistant",
    tools_condition,
    {"tools": "tools", END: END},
)

builder.add_edge("tools", "assistant")

app = builder.compile()