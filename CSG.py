import io
import os
import re
import time
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Optional,
)
from typing_extensions import TypedDict

# Third-party imports
from google import genai 
from google.api_core import retry
from google.genai import types

# ChromaDB
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings

# LangChain & LangGraph
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Application-local imports
from chemsimguide.config import (
    DB_NAME,
    DB_PATH,
    EMBEDDING_MODEL_NAME,
    FULL_SCRIPT_GENERATION_PROMPT,
    GOOGLE_API_KEY,
    LLM_MODEL_NAME,
    N_RAG_RESULTS_DEFAULT,
    WELCOME_MSG,
    CHEMSIM_SYSINT
)

# Retry wrapper for Google GenAI
is_retriable = (lambda e: isinstance(e, genai.errors.APIError) # type: ignore
    and e.code in {429, 503}
)

genai.models.Models.generate_content = retry.Retry( predicate=is_retriable)(genai.models.Models.generate_content) # type: ignore
client = genai.Client(api_key=GOOGLE_API_KEY)


# Embeddings & Vector Store 
class GeminiEmbeddingFunction(EmbeddingFunction):
    document_mode = True

    @retry.Retry(predicate=is_retriable)
    def __call__(self, input: Documents) -> Embeddings:
        if self.document_mode:
            embedding_task = "retrieval_document"
        else:
            embedding_task = "retrieval_query"

        response = client.models.embed_content(
            model="models/text-embedding-004",
            contents=input,
            config=types.EmbedContentConfig(
                task_type=embedding_task,
            ),
        )
        return [e.values for e in response.embeddings]
    
embed_fn = GeminiEmbeddingFunction()
embed_fn.document_mode = True

chroma_client = chromadb.PersistentClient(path=DB_PATH)
db = chroma_client.get_or_create_collection(
    name=DB_NAME,
    embedding_function= GeminiEmbeddingFunction(client, document_mode=True), # type: ignore
)


# Define the tools
## Search Cantera documentation Tool
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
        • chemsimguide.tools.db         ChromaDB collection
        • chemsimguide.tools.embed_fn   GeminiEmbeddingFunction
    """

    # Safety checks 
    if db is None or embed_fn is None:
        return "Error: database or embedding function has not been initialised."
    if db.count() == 0:
        return "Error: document database is empty."

    # Switch embedder to query mode 
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

    # Retrieve context chunks 
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

    #  Ask the LLM for code
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


## Define the Agent State
class GuidanceStep(TypedDict):
    """Represents one step in the guidance plan shown to the user."""
    step_number: int
    explanation: str
    code_snippet: Optional[str]  


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



llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash',google_api_key=GOOGLE_API_KEY) # type: ignore
tools = [ search_cantera_docs, generate_cantera_code ]
llm_with_tools = llm.bind_tools(tools)

## nodes
def chatbot_node(state: ChemSimState) -> Dict[str, List[BaseMessage]]:
    """
    Builds the LLM prompt, calls Gemini (with bound tools), and
    returns the AIMessage to append to the state.
    """
    history = state["messages"]
    if not history:
        return {"messages": [AIMessage(content=WELCOME_MSG)]}

    prompt_msgs = [CHEMSIM_SYSINT] + history
    ai_response = llm_with_tools.invoke(prompt_msgs)  # could include tool_calls

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


## Graph Edges
def maybe_exit_human_node(state: ChemSimState) -> Literal["chatbot", "__end__"]:
    """
    If the user typed a quit command, return END; otherwise loop back
    to the chatbot node.
    """
    #print("\n Routing decision from HUMAN node")
    if state.get("finished_guidance", False):
        #print("   → end graph")
        return END
    #print("   → chatbot")
    return "chatbot"

def should_route_from_chatbot(
    state: ChemSimState,
) -> Literal["rag_tool_node", "code_gen_tool_node", "human"]:
    """
    Inspect the last AIMessage for a `tool_calls` field.

    Returns
    -------
    str
        • "rag_tool_node"         -- if LLM asked for search_cantera_docs  
        • "code_gen_tool_node"     -- if LLM asked for generate_cantera_code
        • "human"                  -- otherwise
    """
    print("\n Routing decision from CHATBOT node")

    last_msg = state["messages"][-1]

    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        tool_name = last_msg.tool_calls[0].get("name")
        print(f"   LLM requested tool: {tool_name}")

        if tool_name == "search_cantera_docs":
            return "rag_tool_node"
        if tool_name == "generate_cantera_code":
            return "code_gen_tool_node"

        # Unknown tool – fall back to user
        print("   (unknown tool) → human")
        return "human"

    # No tool calls
    print("   → human")
    return "human"



rag_tool_node = ToolNode([search_cantera_docs])
code_gen_tool_node = ToolNode([generate_cantera_code])



## Build the Graph
graph_builder = StateGraph(ChemSimState)

# Add ALL nodes
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("human", human_node)
graph_builder.add_node("rag_tool_node", rag_tool_node)
graph_builder.add_node("code_gen_tool_node", code_gen_tool_node) # Add the new node

# Define the entry point
graph_builder.add_edge(START, "chatbot")

# Define transitions using conditional edges
graph_builder.add_conditional_edges(
    "chatbot", # Source node
    should_route_from_chatbot, 
    {
        "rag_tool_node": "rag_tool_node",         # Route to RAG node
        "code_gen_tool_node": "code_gen_tool_node", # Route to Code Gen node
        "human": "human",                       # Route to Human node
    }
)

graph_builder.add_conditional_edges(
    "human",
    maybe_exit_human_node,
    {
        "chatbot": "chatbot",
        END: END
    }
)

# Define edges BACK from tool nodes to the chatbot
graph_builder.add_edge("rag_tool_node", "chatbot")
graph_builder.add_edge("code_gen_tool_node", "chatbot") # Add edge for the new tool node

# Compile the graph

chem_sim_graph= graph_builder.compile()
def run_cli():
    state: Dict[str, Any] = {"messages": []}
    print("ChemSimGuide chat — type 'quit' to exit\n")

    while True:
        state = chem_sim_graph.invoke(state)
        last_ai = state["messages"][-1]
        print(f"\nChemSimGuide: {last_ai.content}")

        user = input("\nYou: ").strip()
        if user.lower() in {"quit", "exit", "q", "bye"}:
            break
        state["messages"].append(HumanMessage(content=user))

if __name__ == "__main__":
    run_cli()