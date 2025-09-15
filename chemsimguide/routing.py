# src/chemsimguide/routing.py
"""
Conditional-edge helpers that tell the LangGraph
engine where to go next after the chatbot or human node.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END  # special sentinel
from chemsimguide.state import ChemSimState


# ──────────────────────────────────────────────────────────────────────────
def maybe_exit_human_node(state: ChemSimState) -> Literal["chatbot", "__end__"]:
    """
    If the user typed a quit command, return END; otherwise loop back
    to the chatbot node.
    """
    #print("\n↪️  Routing decision from HUMAN node")
    if state.get("finished_guidance", False):
        #print("   → end graph")
        return END
    #print("   → chatbot")
    return "chatbot"


# ──────────────────────────────────────────────────────────────────────────
def should_route_from_chatbot(
    state: ChemSimState,
) -> Literal["rag_tool_node", "code_gen_tool_node", "human"]:
    """
    Inspect the last AIMessage for a `tool_calls` field.

    Returns
    -------
    str
        • "rag_tool_node"         – if LLM asked for search_cantera_docs  
        • "code_gen_tool_node"    – if LLM asked for generate_cantera_code  
        • "human"                 – otherwise
    """
    #print("\n↪️  Routing decision from CHATBOT node")

    last_msg = state["messages"][-1]

    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        tool_name = last_msg.tool_calls[0].get("name")
        #print(f"   LLM requested tool: {tool_name}")

        if tool_name == "search_cantera_docs":
            return "rag_tool_node"
        if tool_name == "generate_cantera_code":
            return "code_gen_tool_node"

        # Unknown tool – fall back to user
        #print("   (unknown tool) → human")
        return "human"

    # No tool calls
    #print("   → human")
    return "human"
