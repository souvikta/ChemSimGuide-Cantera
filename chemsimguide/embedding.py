from typing import List, Sequence

from google import genai
from google.api_core import retry
from google.genai import types

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


def _is_retriable(exc) -> bool:  # noqa: ANN001
    """True for transient 429/503 API errors."""
    import google.api_core.exceptions as gce

    return isinstance(exc, gce.GoogleAPICallError) and exc.code in {429, 503}


class GeminiEmbeddingFunction:
    """
    Callable object compatible with ChromaDB's `EmbeddingFunction` protocol.

    Parameters
    ----------
    client : genai.Client
        Authenticated Google GenerativeAI client.
    model : str, optional
        Full resource name for the embedding model
        (default: ``models/text-embedding-004``).
    document_mode : bool, optional
        *True* → use task type ``retrieval_document`` (for adding docs).  
        *False* → use task type ``retrieval_query`` (for search queries).
    """

    def __init__(self, client: genai.Client, model: str = "models/text-embedding-004",document_mode: bool = True,) -> None:
        self.client = client
        self.model = model
        self.document_mode = document_mode
        
        self._embed = retry.Retry(predicate=_is_retriable)(
            self.client.models.embed_content
        )

    # --------------------------------------------------------------------- #
    def __call__(self, input: Documents) -> Embeddings:
        """Return a list of vectors for the input strings."""
        task = "retrieval_document" if self.document_mode else "retrieval_query"

        response = self._embed(
            model=self.model,
            contents=input,
            config=types.EmbedContentConfig(task_type=task),
        )
        # Gemini returns an Embedding object per input string
        return [emb.values for emb in response.embeddings]