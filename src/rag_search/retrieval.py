from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

DEFAULT_TOP_K: Final[int] = 5
DEFAULT_EMBEDDING_MODEL: Final[str] = ("sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_SEARCH_TYPE: Final[str] = "similarity"

def _validate_top_k(top_k: int) -> int:
    """Validate a retrieval result limit."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError(f"top_k must be an integer")
    if top_k <= 0:
        raise ValueError(f"top_k must be greater than zero")
    return top_k

class BM25Retriever:
    """BM25 lexical retriever for LangChain Documents."""

    def __init__(
            self,
            documents: Sequence[Document],
            *,
            top_k: int = DEFAULT_TOP_K,
            epsilon: float = 0.25,
    ) -> None:

        if not documents:
            raise ValueError("documents must not be empty")

        self.documents = documents
        self.top_k = _validate_top_k(top_k)

        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.epsilon = epsilon

        tokenized_documents = [
            self._tokenize(document.page_content) for document in self.documents
        ]

        self._bm25 = BM25Okapi(tokenized_documents, epsilon=self.epsilon)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 matching."""
        return text.lower().split()

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[Document]:
        """Return the highest-ranked documents for a query."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        requested_top_k = (
            self.top_k if top_k is None else _validate_top_k(top_k)
        )

        tokenized_query = self._tokenize(query)

        return self._bm25.get_top_n(
            tokenized_query, self.documents, n=min(requested_top_k, len(self.documents))
        )

class VectorRetriever:
    """Vector-based semantic retriever for LangChain Documents."""

    def __init__(
            self,
            documents: Sequence[Document],
            embedding_model: Embeddings | None = None,
            *,
            top_k: int = DEFAULT_TOP_K, 
            search_type: str = DEFAULT_SEARCH_TYPE,
    ) -> None:

        if not documents:
            raise ValueError("documents must not be empty")

        self.documents = list(documents)
        self.top_k = _validate_top_k(top_k)

        if not isinstance(search_type, str):
            raise TypeError("search_type must be a string")

        if not search_type.strip():
            raise ValueError("search_type must not be empty")
        
        self.search_type = search_type

        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else HuggingFaceEmbeddings(model_name=DEFAULT_EMBEDDING_MODEL)
        )

        self._vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embedding_model,
        )

    def retrieve(
            self, 
            query: str,
            *, 
            top_k: int | None = None, 
    ) -> list[Document]:
        """Return the highest-ranked documents for a query."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if not query.strip():
            raise ValueError("query must not be empty")

        requested_top_k = (
            self.top_k if top_k is None else _validate_top_k(top_k)
        )

        retriever = self._vector_store.as_retriever(
            search_type=self.search_type,
            search_kwargs={"k": requested_top_k },
        )

        return retriever.invoke(query)
    