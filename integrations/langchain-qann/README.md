# langchain-qann

A LangChain vector store backed by [qann](https://github.com/maxtmiller/qann): HNSW, IVF with fast-scan PQ and re-ranking, or exact search, with metadata filters and hybrid (dense + BM25 or SPLADE) search. Texts, ids and metadata are stored in the qann index, so a store saves to one file.

This is a community integration maintained with qann, not by LangChain.

```bash
pip install langchain-qann
```

## Quick start

```python
from langchain_qann import QANNVectorStore

store = QANNVectorStore.from_texts(texts, embeddings, metadatas=metadatas)
docs = store.similarity_search("how do bonds work?", k=10)
retriever = store.as_retriever(search_kwargs={"k": 10})

store.save("store.qann")
store = QANNVectorStore.load("store.qann", embeddings, mmap=True)
```

## Choosing the index

Three levels, from simplest to full control.

**1. Presets.** Pass `index=` by name; parameters follow qann's [configuration guide](https://github.com/maxtmiller/qann#choosing-a-configuration).

| `index=` | What it builds | When |
|---|---|---|
| `"hnsw"` (default) | `HNSWIndex(M=16, ef_construction=200)` | adds at any time; low latency |
| `"ivf-pq"` | IVF, fast-scan PQ (`dim / 2` subspaces), `RefineIndex(k_factor=4)`, `nlist = max(100, √n)` | least RAM with `load(..., mmap=True)`; trains on the first batch (at least 1,000 texts), so build it with `from_texts` on the whole corpus |
| `"flat"` | `FlatIndex` | exact; small corpora |

All presets use cosine distance; `metric="l2"` or `"ip"` changes that.

**2. Any qann index.** Build it with the parameters you want and pass it in:

```python
import qann

base = qann.IVFIndex(384, nlist=2048, metric=qann.Metric.Cosine)
base.enable_pq(192, fast_scan=True, rotate=True)
base.train(sample_vectors)
store = QANNVectorStore(embeddings, index=qann.RefineIndex(base, k_factor=8))

store = QANNVectorStore(embeddings, index=qann.HNSWIndex(384, M=32, ef_construction=400, metric=qann.Metric.Cosine))
```

`store.index` is the underlying qann index, for any other setting.

**3. Search settings.** `nprobe`, `ef_search` and `k_factor` are set on the store for every query, or per query:

```python
store.search_params = {"nprobe": 32}
docs = store.similarity_search("query", k=10, nprobe=64)
retriever = store.as_retriever(search_kwargs={"k": 10, "ef_search": 128})
```

## Hybrid search

`sparse=` adds a sparse side, searched together with the dense one and fused (qann's `HybridIndex`):

```python
store = QANNVectorStore.from_texts(texts, embeddings, sparse="bm25")
docs = store.similarity_search("form 10-K filing deadline", k=10)
```

- **`sparse="bm25"`** scores the texts with BM25: lowercase words and numbers, English stopwords removed, no stemming. Each query term's IDF comes from the current document counts, so adds and deletes update it without re-encoding stored documents. The counts are rebuilt from the stored texts on `load` (about 2 s for 58k documents).
- **A custom encoder**, e.g. SPLADE, is any object with `encode_documents(texts)` and `encode_query(text)` returning `scipy.sparse` rows with term ids below 2^31. Pass it again to `load(..., sparse=encoder)`, like the embeddings.
- **Weights.** `weights=(dense, sparse)` sets the fusion, default `(1, 1)`. Set it in `search_params` or per query: `weights=(1, 0)` is dense only and `(0, 1)` sparse only. `method="rrf"` switches from linear fusion to reciprocal rank fusion.
- **Tuning.** `store.tune_hybrid_weights(queries, relevant)` picks the weights from labeled queries and keeps them, and they're saved with the store. `relevant` holds the relevant document ids per query, as a set or `{id: grade}`. Tune on queries you won't evaluate on.
- **Text vs vector.** Searches by text use both sides; searches by vector (`similarity_search_by_vector`) have no text for the sparse side, so they use the dense side.
- **Scores.** `similarity_search_with_score` returns the fused score (higher is better) instead of a distance.

On BEIR FiQA (58k passages, bge-small, `index="ivf-pq"`), nDCG@10:

| | nDCG@10 |
|---|---|
| dense only | 0.384 |
| BM25 only | 0.238 |
| hybrid, default (1, 1) | 0.373 |
| hybrid, tuned on half the queries (1, 0.5), scored on the other half | 0.389 (default: 0.368) |

Where BM25 is much weaker than the embeddings, as here, equal weights hurt and tuning matters; on SciFact, where BM25 is strong, the default already beats dense alone.

## Filters

`filter=` takes any of:

```python
store.similarity_search("q", filter={"source": "10-K", "year": [2023, 2024]})   # exact matches; a list means any of
store.similarity_search("q", filter="year >= 2020 AND lang = 'en'")            # SQL WHERE
store.similarity_search("q", filter=("year >= ? AND lang = ?", [2020, "en"]))  # with parameters
store.similarity_search("q", filter=qann.field("year") >= 2020)                 # qann.Expr
```

Metadata strings and numbers become filterable columns (numbers as floats, booleans as 0 / 1). Other values, such as lists and dicts, are returned with the document but can't be filtered on. A key whose values change type (a string, then a number) keeps the first type for filtering.

## Also supported

- `add_texts` / `add_documents` with ids; adding an existing id replaces that document.
- `add_embeddings(texts, vectors, metadatas, ids=...)` for precomputed embeddings.
- `get_by_ids`, `delete(ids)` (`delete()` removes everything).
- `similarity_search_with_score` (qann distance, lower is closer; the fused score with hybrid search), `similarity_search_with_relevance_scores`, search by vector.
- `max_marginal_relevance_search`.
- Async methods (LangChain's defaults, run in a thread; qann releases the GIL during queries and adds).
