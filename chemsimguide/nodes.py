# ───────────────────────────────────────────────────────────────────────────
# src/chemsimguide/nodes.py
# Defines Chatbot + Human nodes and wraps the two LangChain ToolNodes.
# ───────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import time
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from chemsimguide.state import ChemSimState
from chemsimguide.tools import search_cantera_docs, generate_cantera_code
from chemsimguide.config import CHEMSIM_SYSINT, WELCOME_MSG

# llm_with_tools will be injected by the runner (global)
llm_with_tools = None  # type: ignore


# ─────────────────── Chatbot Node ─────────────────────────────────────────
def chatbot_node(state: ChemSimState) -> Dict[str, List[BaseMessage]]:
    """
    Builds the LLM prompt, calls Gemini (with bound tools), and
    returns the AIMessage to append to the state.
    """
    history = state["messages"]

    # First turn → send welcome
    if not history:
        return {"messages": [AIMessage(content=WELCOME_MSG)]}

    prompt_msgs = [CHEMSIM_SYSINT] + history
    ai_response = llm_with_tools.invoke(prompt_msgs)  # could include tool_calls

    # merge back into state
    return state | {"messages": [ai_response]}


# ─────────────────── Human Node ───────────────────────────────────────────
def human_node(state: ChemSimState) -> Dict:
    """
    Prints the last AI response, collects console input, and decides
    whether to continue or exit.
    """
    last_ai = state["messages"][-1]

    print("\nChemSimGuide:")
    if isinstance(last_ai, AIMessage):
        print(last_ai.content)
    else:
        print(last_ai)

    user_input = input("\nYou: ").strip()
    if user_input.lower() in {"quit", "exit", "bye", "q"}:
        return {"finished_guidance": True}

    return {"messages": [HumanMessage(content=user_input)]}


rag_tool_node = ToolNode([search_cantera_docs])
code_gen_tool_node = ToolNode([generate_cantera_code])