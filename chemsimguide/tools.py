# src/chemsimguide/tools.py
from __future__ import annotations
from typing import List

# These globals will be assigned from the notebook:
db = None          # type: ignore
embed_fn = None    # type: ignore

@tool
def search_cantera_docs(
    query: str,
    n_results: int = 10,
) -> str:
    """
    Return up to *n_results* relevant Cantera doc chunks for *query*.

    Parameters
    ----------
    query : str
        Natural-language question.
    n_results : int, default 10
        How many passages to return.

    Notes
    -----
    Relies on two globals that **must** be set by the application:
        • chemsimguide.tools.db        – ChromaDB collection
        • chemsimguide.tools.embed_fn  – GeminiEmbeddingFunction
    """

    # --- Safety checks ---------------------------------------------------
    if db is None or embed_fn is None:
        return "Error: database or embedding function has not been initialised."
    if db.count() == 0:
        return "Error: document database is empty."

    # --- Switch embedder to query mode -----------------------------------
    original_mode = getattr(embed_fn, "document_mode", True)
    embed_fn.document_mode = False

    try:
        res = db.query(query_texts=[query], n_results=n_results)
    except Exception as exc:  # noqa: BLE001
        return f"Error during documentation search: {exc}"
    finally:
        embed_fn.document_mode = original_mode  # always restore

    chunks: List[str] = res.get("documents", [[]])[0]
    if not chunks:
        return "No relevant documentation found."

    return "\n".join(f"- {c.replace(chr(10), ' ')}" for c in chunks)


# --- OPTIONAL: LangChain wrapper -----------------------------------------
# Uncomment the next six lines only if you need an LC tool for your agent.
#
# from langchain_core.tools import tool
#
# @tool
# def lc_search_cantera_docs(query: str) -> str:  # single-arg wrapper
#     return search_cantera_docs(query)           # uses default n_results=10
#
# -------------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────
# Cantera full-script generation tool
# ──────────────────────────────────────────────────────────────────────────
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List

# Globals injected by the notebook / runner
db = None          # type: ignore
embed_fn = None    # type: ignore
llm = None         # type: ignore

# ------------------------------------------------------------------------
FULL_SCRIPT_GENERATION_PROMPT = SystemMessage(
    content=(
        "You are an expert Cantera Python code-generation assistant.\n"
        "Given a description of a desired Cantera simulation and relevant "
        "documentation context, generate a complete, runnable Python "
        "script. Include necessary imports (e.g., `import cantera as ct`). "
        "Structure the code logically, add explanatory comments, and output "
        "ONLY the code inside a ```python ...``` block."
    )
)


@tool
def generate_cantera_code(
    simulation_goal: str,
    n_rag_results: int = 10,
) -> str:
    """
    RAG → LLM chain that returns a full Cantera script.

    Parameters
    ----------
    simulation_goal : str
        Plain-language description of the task
        (e.g. 'calculate adiabatic flame temperature of CH₄/Air at 1 atm').
    n_rag_results : int, default 10
        How many doc chunks to fuse into the prompt.

    Notes
    -----
    Depends on global `db`, `embed_fn`, and `llm` handles set by the runner.
    When the function is called as a *LangChain Tool*, only the first
    positional argument (`simulation_goal`) will be passed automatically.
    """

    # ── 1. Retrieve context chunks ──────────────────────────────────────
    rag_context = "No relevant documentation found."
    try:
        embed_fn.document_mode = False  # query vector
        res = db.query(
            query_texts=[simulation_goal],
            n_results=n_rag_results,
            include=["documents"],
        )
        embed_fn.document_mode = True
        chunks: List[str] = res.get("documents", [[]])[0]

        if chunks:
            rag_context = "\n".join(
                f"--- Passage {i+1} ---\n{c}\n" for i, c in enumerate(chunks)
            )

    except Exception as exc:  # noqa: BLE001
        embed_fn.document_mode = True
        rag_context = f"Error retrieving context: {exc}"

    # ── 2. Ask the LLM for code ─────────────────────────────────────────
    try:
        prompt_msgs = [
            FULL_SCRIPT_GENERATION_PROMPT,
            HumanMessage(
                content=(
                    f"Generate a complete Cantera Python script for:\n"
                    f"{simulation_goal}\n\n"
                    "Relevant documentation context:\n"
                    f"{rag_context}"
                )
            ),
        ]
        response = llm.invoke(prompt_msgs)
        return response.content

    except Exception as exc:  # noqa: BLE001
        return f"Error generating script: {exc}"
