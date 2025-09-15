# src/chemsimguide/graph.py
"""
Factory that assembles and compiles the LangGraph for ChemSimGuide.

Usage
-----
>>> from chemsimguide.graph import build_chem_sim_graph
>>> graph = build_chem_sim_graph(llm_with_tools)
>>> final_state = graph.invoke({"messages": []})
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from chemsimguide.state import ChemSimState
from chemsimguide import nodes as csg_nodes
from chemsimguide.routing import (
    should_route_from_chatbot,
    maybe_exit_human_node,
)


def build_chem_sim_graph(llm_with_tools):
    """
    Create and compile the ChemSimGuide LangGraph.

    Parameters
    ----------
    llm_with_tools : langchain_core.LanguageModel
        Your Gemini (or other) LLM instance that already has
        `search_cantera_docs` and `generate_cantera_code` bound to it.

    Returns
    -------
    langgraph.graph.StateGraph
        A compiled graph ready for `.invoke()` / `.ainvoke()`.
    """

    # Inject the LLM into the chatbot node module once
    csg_nodes.llm_with_tools = llm_with_tools

    print("Building ChemSimGuide LangGraph …")

    g = StateGraph(ChemSimState)

    # ── Nodes ────────────────────────────────────────────────────────────
    g.add_node("chatbot", csg_nodes.chatbot_node)
    g.add_node("human", csg_nodes.human_node)
    g.add_node("rag_tool_node", csg_nodes.rag_tool_node)
    g.add_node("code_gen_tool_node", csg_nodes.code_gen_tool_node)

    # ── Edges ────────────────────────────────────────────────────────────
    g.add_edge(START, "chatbot")

    g.add_conditional_edges(
        "chatbot",
        should_route_from_chatbot,
        {
            "rag_tool_node": "rag_tool_node",
            "code_gen_tool_node": "code_gen_tool_node",
            "human": "human",
        },
    )

    g.add_conditional_edges(
        "human",
        maybe_exit_human_node,
        {"chatbot": "chatbot", END: END},
    )

    g.add_edge("rag_tool_node", "chatbot")
    g.add_edge("code_gen_tool_node", "chatbot")

    # ── Compile ──────────────────────────────────────────────────────────
    compiled = g.compile()
    print("ChemSimGuide graph compiled successfully.")
    return compiled
