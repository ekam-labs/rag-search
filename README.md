# RAG Search

A hybrid retrieval component for Retrieval-Augmented Generation (RAG) pipelines.

RAG Search combines lexical retrieval and semantic vector retrieval, fuses their results using Reciprocal Rank Fusion (RRF), and applies a CrossEncoder reranker to produce a final ranked set of documents.

The package works with LangChain `Document` objects and is designed as a modular retrieval component that can be used independently or as part of a larger RAG pipeline.

## Features

* BM25 lexical retrieval
* Vector-based semantic retrieval
* Local Chroma vector store
* Configurable embedding model
* Reciprocal Rank Fusion (RRF)
* CrossEncoder reranking
* Parallel BM25 and vector retrieval
* Configurable result count
* Configurable RRF rank constant
* Configurable reranker
* Support for custom BM25 and vector retrievers

## Retrieval Pipeline

The retrieval process follows this flow:

```text
                         Query
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
        BM25 Retrieval          Vector Retrieval
              │                         │
              └────────────┬────────────┘
                           ▼
                Reciprocal Rank Fusion
                           │
                           ▼
                    CrossEncoder
                     Reranking
                           │
                           ▼
                    Top-K Results
```

BM25 provides lexical matching, while vector retrieval provides semantic matching. Their results are combined using Reciprocal Rank Fusion before the final CrossEncoder reranking stage.

## Installation

```bash
pip install rag-search
```

## Basic Usage

```python
from rag_search import RAGSearch

search = RAGSearch(documents)

results = search.search(
    "What is artificial intelligence?",
    top_k=5,
)
```

`documents` should be a sequence of LangChain `Document` objects.

The `search()` method returns a list of ranked `Document` objects.

## BM25 Retriever

The package provides a standalone BM25 retriever:

```python
from rag_search import BM25Retriever

retriever = BM25Retriever(
    documents,
    top_k=5,
)

results = retriever.retrieve(
    "What is artificial intelligence?"
)
```

The BM25 retriever performs lexical matching using tokenized document text.

## Vector Retriever

The package also provides a standalone vector retriever:

```python
from rag_search import VectorRetriever

retriever = VectorRetriever(
    documents,
    top_k=5,
)

results = retriever.retrieve(
    "What is artificial intelligence?"
)
```

By default, the vector retriever uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and stores the document embeddings in a local Chroma vector store.

A custom LangChain-compatible embedding model can also be supplied:

```python
retriever = VectorRetriever(
    documents,
    embedding_model=custom_embedding_model,
)
```

## Hybrid Search

The `RAGSearch` class combines both retrieval approaches:

```python
from rag_search import RAGSearch

search = RAGSearch(documents)

results = search.search(
    "What is artificial intelligence?",
    top_k=5,
)
```

Internally, BM25 and vector retrieval are executed concurrently. Their ranked results are then combined using Reciprocal Rank Fusion (RRF), followed by CrossEncoder reranking.

## Configuration

### Result Count

The default retrieval limit is 5.

```python
search = RAGSearch(documents)

results = search.search(
    "What is artificial intelligence?",
    top_k=10,
)
```

### RRF Rank Constant

The default RRF rank constant is `60`.

```python
search = RAGSearch(
    documents,
    rank_constant=60,
)
```

### Reranker

The default CrossEncoder reranker is:

```text
BAAI/bge-reranker-v2-m3
```

A different CrossEncoder model can be supplied:

```python
search = RAGSearch(
    documents,
    reranker="your-reranker-model",
)
```

## Custom Retrievers

`RAGSearch` also allows custom BM25 and vector retriever instances:

```python
search = RAGSearch(
    documents,
    bm25_retriever=custom_bm25,
    vector_retriever=custom_vector,
)
```

This allows the hybrid search layer to be composed with customized retrieval implementations.

## Components

The package exposes three main components:

| Component         | Purpose                                     |
| ----------------- | ------------------------------------------- |
| `BM25Retriever`   | Lexical retrieval                           |
| `VectorRetriever` | Semantic vector retrieval                   |
| `RAGSearch`       | Hybrid retrieval, RRF fusion, and reranking |

## Dependencies

The retrieval implementation uses components from the LangChain ecosystem together with:

* Chroma
* Hugging Face embeddings
* BM25
* Ranx for Reciprocal Rank Fusion
* Sentence Transformers for CrossEncoder reranking

Package dependencies are defined in `pyproject.toml`.

## Role in a RAG Pipeline

RAG Search is intended to operate after document ingestion and chunking:

```text
Documents
    ↓
Document Ingestion
    ↓
Cleaning & Chunking
    ↓
RAG Search
    ↓
Retrieved Context
    ↓
LLM
```

It focuses specifically on retrieval and ranking. Document loading and chunking are handled separately.

## Status

This is the initial version of the RAG Search component, developed as a modular retrieval building block for a larger RAG system.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
