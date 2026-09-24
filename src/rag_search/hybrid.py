from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

from langchain_core.documents import Document
from ranx import Run, fuse
from sentence_transformers import CrossEncoder

from .retrieval import BM25Retriever, VectorRetriever, _validate_top_k

DEFAULT_RANK_CONSTANT = 60
DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"

class RAGSearch:
    """Hybrid RAG search using BM25, vector search, RRF, and reranking."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        bm25_retriever: BM25Retriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        rank_constant: int = DEFAULT_RANK_CONSTANT,
        reranker: str = DEFAULT_RERANKER,
    ) -> None:

        if isinstance(rank_constant, bool) or not isinstance(rank_constant, int):
            raise TypeError("rank_constant must be an integer.")

        if rank_constant <= 0:
            raise ValueError("rank_constant must be greater than zero.")

        self.bm25_retriever = bm25_retriever or BM25Retriever(documents)
        self.vector_retriever = vector_retriever or VectorRetriever(documents)

        self.rank_constant = rank_constant
        self.reranker = CrossEncoder(reranker)

    def search(
            self,
            query: str,
            *,
            top_k: int | None = None,
    ) -> list[Document]:

        if top_k is not None:
            _validate_top_k(top_k)

        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(self.bm25_retriever.retrieve, query)
            vector_future = executor.submit(self.vector_retriever.retrieve, query)

            bm25_results = bm25_future.result()
            vector_results = vector_future.result()   

        results = self._reciprocal_rank_fusion(bm25_results, vector_results)

        results = self._rerank(query, results)

        if top_k is not None:
            results = results[:top_k]

        return results

    def _reciprocal_rank_fusion(
            self,
            bm25_results: list[Document],
            vector_results: list[Document],
    ) -> list[Document]:

        documents: dict[str, Document] = {}
        bm25_ids: list[str] = []
        vector_ids: list[str] = []

        for document in bm25_results:
            document_id = str(len(documents))
            documents[document_id] = document
            bm25_ids.append(document_id)

        for document in vector_results:
            document_id = next(
                (
                    doc_id
                    for doc_id, exiting_documents in documents.items()
                    if exiting_documents == document
                ),
                None,
            )

            if document_id is None:
                document_id = str(len(documents))
                documents[document_id] = document

            vector_ids.append(document_id)

        bm25_run = Run(
            {
                "query": {
                    document_id: len(bm25_ids) - rank
                    for rank, document_id in enumerate(bm25_ids)
                }
            }
        )  

        vector_run = Run(
            {
                "query": {
                    document_id: len(vector_ids) - rank
                    for rank, document_id in enumerate(vector_ids)
                }
            }
        )

        fused_run = fuse(
            [bm25_run, vector_run],
            method="rrf",
            params={"k": self.rank_constant},
        )  

        return [
            documents[document_id]
            for document_id in fused_run["query"]
        ]

    def _rerank(
            self,
            query: str,
            documents: list[Document]
    ) -> list[Document]:

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        scores = self.reranker.predict(pairs)

        return [

            document
            for _, document in sorted(
                zip(scores, documents),
                key=lambda item: item[0],
                reverse=True
            )
        ]