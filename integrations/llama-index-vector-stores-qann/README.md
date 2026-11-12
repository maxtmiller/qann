# llama-index-vector-stores-qann

A LlamaIndex vector store backed by [qann](https://github.com/maxtmiller/qann): HNSW, IVF with fast-scan PQ and re-ranking, or exact search, with metadata filters and hybrid (dense + BM25 or SPLADE) search. Node text, metadata and relationships are stored in the qann index, so a store persists to one file.

This is a community integration maintained with qann, not by LlamaIndex.

```bash
pip install --pre llama-index-vector-stores-qann
```

## Quick start

```python
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qann import QANNVectorStore

store = QANNVectorStore(index="ivf-pq", sparse="bm25")
index = VectorStoreIndex.from_documents(documents, storage_context=StorageContext.from_defaults(vector_store=store))

retriever = index.as_retriever(similarity_top_k=10, vector_store_query_mode="hybrid", alpha=0.6)
nodes = retriever.retrieve("how do bonds work?")

store.persist("store.qann")
store = QANNVectorStore.from_persist_path("store.qann", mmap=True)
index = VectorStoreIndex.from_vector_store(store)
```

## Choosing the index

**1. Presets.** Pass `index=` by name. The parameters follow qann's [configuration guide](https://github.com/maxtmiller/qann#choosing-a-configuration).

| `index=` | What it builds | When |
|---|---|---|
| `"hnsw"` (default) | `HNSWIndex(M=16, ef_construction=200)` | Adds at any time; low latency. |
| `"ivf-pq"` | IVF, fast-scan PQ (`dim / 2` subspaces), `RefineIndex(k_factor=4)`, `nlist = max(100, √n)` | Least RAM with `from_persist_path(..., mmap=True)`. Trains on the first batch of nodes (at least 1,000), so build it from the whole corpus at once. |
| `"flat"` | `FlatIndex` | Exact; small corpora. |

All presets use cosine distance; `metric="l2"` or `"ip"` changes that.

**2. Any qann index.** Build one with the parameters you want and pass it in, e.g. `QANNVectorStore(index=qann.RefineIndex(base, k_factor=8))`. `store.client` is the underlying qann index, for any other setting.

**3. Search settings.** `nprobe`, `ef_search` and `k_factor` can be set for every query, or per query through the retriever:

```python
store = QANNVectorStore(index="ivf-pq", search_params={"nprobe": 32})
retriever = index.as_retriever(vector_store_kwargs={"nprobe": 64})
```

## Hybrid search

`sparse="bm25"` adds BM25 over the node texts. A custom encoder works too, such as SPLADE: any object with `encode_documents(texts)` and `encode_query(text)` that returns `scipy.sparse` rows with term ids below 2^31. The retriever's query mode picks the side:

| `vector_store_query_mode` | Searches |
|---|---|
| `"default"` | dense side |
| `"sparse"` / `"text_search"` | sparse side |
| `"hybrid"` | both, fused; `alpha` weights dense against sparse as `(alpha, 1 - alpha)` |
| `"mmr"` | dense side, then maximal marginal relevance |

- **BM25 keeps up with the corpus.** Each query term's IDF comes from the current document counts, so adds and deletes stay exact without re-encoding stored nodes. It is `qann.BM25`, so the counts are rebuilt from the stored texts in C++ when the store is opened (0.15 s for 58k nodes, about 5 s for 1M).
- **Default weights.** Without `alpha`, hybrid mode uses the store's weights, `(1, 1)` unless tuned.
- **Tuning.** `store.tune_hybrid_weights(query_embeddings, query_strs, relevant)` picks the weights from labeled queries. `relevant` holds node ids per query, as a set or `{node id: grade}`. The weights are saved with the store. Tune on queries you won't evaluate on.
- **Custom encoders on reopen.** Pass a custom encoder again when opening the store, like `from_persist_path(..., sparse=encoder)`.

## Filters

LlamaIndex `MetadataFilters` are translated to qann filters and evaluated inside the search. These are supported:
- `==`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `nin` and `is_empty`
- the `and`, `or` and `not` conditions, including nested filter groups

A node without the filtered key matches only `!=`, `nin` and `is_empty`, the same as LlamaIndex's own stores. A test checks 150 random filter combinations against LlamaIndex's `SimpleVectorStore`.

Not supported, because they need list or substring matching that qann's filter columns don't have: `text_match`, `text_match_insensitive`, `contains`, `any` and `all`. These raise `NotImplementedError`.

Strings and numbers in node metadata become filterable columns; other values are kept on the node but can't be filtered on.

## Also supported

- `delete(ref_doc_id)` removes all nodes of a source document, which is what `index.delete_ref_doc` uses. There's also `delete_nodes(node_ids, filters)`, `get_nodes(node_ids, filters)` and `clear()`; their filters run in qann like a query's, except `text_match` and `contains`, which are checked in Python. Adding an existing node id replaces that node.
- Queries restricted to `node_ids` or `doc_ids`.
- Similarities where higher is closer: cosine similarity, the inner product, or minus the squared L2 distance; the fused score with hybrid search.
- Async methods (LlamaIndex's defaults; qann releases the GIL during queries and adds).
- `persist(path)` writes one qann file, so `StorageContext.persist(persist_dir)` writes it as `default__vector_store.json`. Open it with `QANNVectorStore.from_persist_path`. Only local paths are supported.
