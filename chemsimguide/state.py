# src/chemsimguide/state.py
"""
Shared type definitions for ChemSimGuide’s LangGraph agent.

These types are used by the chatbot, human, and tool nodes to exchange state.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class GuidanceStep(TypedDict):
    """Represents one step in the guidance plan shown to the user."""
    step_number: int
    explanation: str
    code_snippet: Optional[str]  # may be None until code is generated


class ChemSimState(TypedDict):
    """
    Full state object that flows through the LangGraph execution.
    """

    # ───── Core conversation ────────────────────────────────────────────
    messages: Annotated[List[BaseMessage], add_messages]
    finished_guidance: bool

    # ───── Simulation goal & context ────────────────────────────────────
    simulation_type: Optional[str]
    simulation_parameters: Dict[str, Any]
    last_retrieved_context: Optional[str]

    # ───── Guidance progress ────────────────────────────────────────────
    guidance_steps_provided: List[GuidanceStep]
    current_step_index: Optional[int]

    # ───── Interaction state ────────────────────────────────────────────
    needs_clarification: bool
    clarification_question: Optional[str]

    # ───── Error handling ───────────────────────────────────────────────
    last_action_error: Optional[str]
