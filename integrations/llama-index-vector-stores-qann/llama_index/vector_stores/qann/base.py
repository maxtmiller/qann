"""LlamaIndex vector store backed by a qann index."""
from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import qann
import scipy.sparse as sp
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.core.vector_stores.utils import build_metadata_filter_fn, metadata_dict_to_node, node_to_metadata_dict

from llama_index.vector_stores.qann.sparse import BM25, VOCAB, SparseEncoder

# Columns the store keeps next to the nodes' own metadata: text, node id, source
# document id, and LlamaIndex's serialized node (relationships, metadata, type)
# as JSON. Everything lives in the qann index file, so persist() writes one file.
TEXT, NODE, DOC, META = "_li_text", "_li_node_id", "_li_doc_id", "_li_meta"
RESERVED = (TEXT, NODE, DOC, META)
METRICS = {"cosine": qann.Metric.Cosine, "l2": qann.Metric.L2, "ip": qann.Metric.InnerProduct}
PRESETS = ("hnsw", "ivf-pq", "flat")
SEARCH_SETTINGS = ("nprobe", "ef_search", "k_factor", "weights", "method", "depth_factor", "rrf_k")
MIN_IVF_TRAIN = 1000
SPARSE_MODES = (VectorStoreQueryMode.SPARSE, VectorStoreQueryMode.TEXT_SEARCH)


class QANNVectorStore(BasePydanticVectorStore):
    """A LlamaIndex vector store on a qann index.

    index is a preset name or any qann.DenseIndex you built yourself:
      - "hnsw" (default): HNSWIndex(M=16, ef_construction=200); adds at any time.
      - "ivf-pq": IVF with fast-scan PQ (dim / 2 subspaces) under RefineIndex(k_factor=4),
        nlist = max(100, sqrt(n)); trains on the first batch of nodes added (at least 1,000).
      - "flat": exact search.
    Presets are created on the first add, once the embedding size is known.

    sparse turns on hybrid search: "bm25" for built-in BM25 on the node texts, or an
    object with encode_documents(texts) and encode_query(text) returning scipy.sparse
    rows (e.g. SPLADE). Query modes: "default" searches the dense side, "sparse" /
    "text_search" the sparse side, and "hybrid" both, fused with weights
    (alpha, 1 - alpha) when the query sets alpha.

    search_params sets search options for every query: nprobe, ef_search, k_factor and,
    with hybrid search, weights, method, depth_factor and rrf_k. Queries take the same
    names as keyword arguments (vector_store_kwargs on a retriever).
    """

    stores_text: bool = True
    is_embedding_query: bool = True
    flat_metadata: bool = False

    _index: Any = PrivateAttr(default=None)
    _preset: str | None = PrivateAttr(default=None)
    _metric: str = PrivateAttr(default="cosine")
    _sparse: Any = PrivateAttr(default=None)
    _rows: dict = PrivateAttr(default_factory=dict)   # node id -> qann id
    _docs: dict = PrivateAttr(default_factory=dict)   # ref doc id -> set of qann ids
    _keep: list = PrivateAttr(default_factory=list)   # wrapped indexes kept alive
    search_params: dict = {}

    def __init__(
        self,
        index: str | qann.IndexBase = "hnsw",
        *,
        metric: str = "cosine",
        search_params: dict[str, Any] | None = None,
        sparse: str | SparseEncoder | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(search_params=dict(search_params or {}), **kwargs)
        if isinstance(index, str) and index not in PRESETS:
            raise ValueError(f"index must be one of {PRESETS} or a qann index, got {index!r}")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {tuple(METRICS)}, got {metric!r}")
        if isinstance(index, qann.HybridIndex) and sparse is None:
            sparse = "bm25"
        if sparse == "bm25":
            self._sparse = BM25()
        elif sparse is None or (hasattr(sparse, "encode_documents") and hasattr(sparse, "encode_query")):
            self._sparse = sparse
        else:
            raise ValueError('sparse must be "bm25", None or an object with encode_documents and encode_query')
        self._metric = metric
        if isinstance(index, str):
            self._preset = index
            return
        if index.size() and NODE not in index.metadata_schema:
            raise ValueError("the index already holds vectors added outside a QANNVectorStore; "
                             "pass an empty index or one saved by QANNVectorStore.persist")
        if self._sparse is not None and not isinstance(index, qann.HybridIndex):
            if index.size():
                raise ValueError("hybrid search needs an empty dense index to wrap, or a saved hybrid store")
            index = self._hybrid(index)
        self._index = index
        self._keep += [d for d in (self._dense, getattr(self._dense, "base", None)) if d is not None]
        self._metric = {v: k for k, v in METRICS.items()}[self._dense.metric]
        self._load_ids()

    @classmethod
    def class_name(cls) -> str:
        return "QANNVectorStore"

    @property
    def client(self) -> qann.IndexBase | None:
        """The underlying qann index."""
        return self._index

    @property
    def _dense(self):
        return self._index.dense if isinstance(self._index, qann.HybridIndex) else self._index

    def _hybrid(self, dense):
        self._keep.append(dense)
        return qann.HybridIndex(dense, qann.InvertedIndex())

    # ---- adding ----

    def add(self, nodes: Sequence[BaseNode], **add_kwargs: Any) -> list[str]:
        nodes = list(nodes)
        if not nodes:
            return []
        ids = [n.node_id for n in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError("node ids must be distinct")
        vectors = np.ascontiguousarray(np.asarray([n.get_embedding() for n in nodes], dtype=np.float32))
        if self._index is None:
            self._index = self._create(vectors)
        # adding an existing node id replaces that node
        self.delete_nodes([i for i in ids if i in self._rows])
        texts = [n.get_content() for n in nodes]
        columns = self._columns(nodes, texts)
        if self._sparse is None:
            new = self._index.add(vectors, metadata=columns)
        else:
            new = self._index.add(vectors, self._sparse.encode_documents(texts), metadata=columns)
        for node, row in zip(nodes, (int(r) for r in new)):
            self._rows[node.node_id] = row
            self._docs.setdefault(node.ref_doc_id or "None", set()).add(row)
        return ids

    def _create(self, vectors: np.ndarray):
        n, dim = vectors.shape
        metric = METRICS[self._metric]
        if self._preset == "flat":
            dense = qann.FlatIndex(dim, metric=metric)
        elif self._preset == "hnsw":
            dense = qann.HNSWIndex(dim, M=16, ef_construction=200, metric=metric)
        else:
            if n < MIN_IVF_TRAIN:
                raise ValueError(f'index="ivf-pq" trains on the first batch of nodes added, which needs at least '
                                 f"{MIN_IVF_TRAIN} (got {n}); add the corpus at once, or use \"hnsw\"")
            if dim % 2:
                raise ValueError(f'index="ivf-pq" needs an even embedding size (got {dim})')
            base = qann.IVFIndex(dim, max(100, round(math.sqrt(n))), metric=metric)
            base.enable_pq(dim // 2, fast_scan=True)
            base.train(vectors)
            self._keep.append(base)
            dense = qann.RefineIndex(base, k_factor=4)
        return dense if self._sparse is None else self._hybrid(dense)

    def _columns(self, nodes: list[BaseNode], texts: list[str]) -> dict[str, list]:
        schema = self._index.metadata_schema
        meta = [node_to_metadata_dict(n, remove_text=True, flat_metadata=False) for n in nodes]
        columns = {TEXT: texts, NODE: [n.node_id for n in nodes], DOC: [n.ref_doc_id or "None" for n in nodes],
                   META: [json.dumps(m) for m in meta]}
        keys = {k for n in nodes for k in n.metadata if k not in RESERVED}
        for key in keys:
            values = [_filterable(n.metadata.get(key), schema.get(key)) for n in nodes]
            kind = schema.get(key) or _kind(values)
            if kind is not None:
                columns[key] = [v if _fits(v, kind) else None for v in values]
        return columns

    # ---- reading and deleting ----

    def get_nodes(self, node_ids: list[str] | None = None, filters: MetadataFilters | None = None) -> list[BaseNode]:
        return self._nodes(self._select(node_ids, filters))

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """Delete every node of a source document."""
        self._remove(sorted(self._docs.get(ref_doc_id, ())))

    def delete_nodes(self, node_ids: list[str] | None = None, filters: MetadataFilters | None = None,
                     **delete_kwargs: Any) -> None:
        if node_ids is None and filters is None:
            return
        self._remove(self._select(node_ids, filters))

    def clear(self) -> None:
        self._remove(sorted(self._rows.values()))

    def _select(self, node_ids, filters) -> list[int]:
        """qann ids of the given nodes (all if None) that match filters."""
        if self._index is None:
            return []
        rows = sorted(self._rows.values()) if node_ids is None else [self._rows[i] for i in node_ids if i in self._rows]
        if filters is None or not rows:
            return rows
        # qann has no filter-only call, so these (rare) operations check each node's
        # metadata in Python, with LlamaIndex's own filter semantics
        metas = self._index.get_metadata(rows, [META])[META]
        return [r for r, m in zip(rows, metas) if _matches(filters, json.loads(m))]

    def _remove(self, rows: list[int]) -> None:
        if not rows:
            return
        cols = self._index.get_metadata(rows, [NODE, DOC, TEXT])
        for node_id, doc_id, row in zip(cols[NODE], cols[DOC], rows):
            self._rows.pop(node_id, None)
            self._docs.get(doc_id, set()).discard(row)
        if isinstance(self._sparse, BM25):
            self._sparse.remove(cols[TEXT])
        self._index.remove(rows)

    def _load_ids(self) -> None:
        live = self._index.ids()
        if not len(live):
            return
        cols = self._index.get_metadata(live, [NODE, DOC, TEXT])
        for node_id, doc_id, row in zip(cols[NODE], cols[DOC], (int(r) for r in live)):
            self._rows[node_id] = row
            self._docs.setdefault(doc_id, set()).add(row)
        if isinstance(self._sparse, BM25):
            self._sparse.rebuild(cols[TEXT])

    def _nodes(self, rows: Sequence[int]) -> list[BaseNode]:
        if not len(rows):
            return []
        cols = self._index.get_metadata(list(rows), [TEXT, META])
        return [metadata_dict_to_node(json.loads(m), text=t) for t, m in zip(cols[TEXT], cols[META])]

    # ---- querying ----

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        if self._index is None or not self._rows:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        mode = query.mode
        hybrid = isinstance(self._index, qann.HybridIndex)
        if mode not in (VectorStoreQueryMode.DEFAULT, VectorStoreQueryMode.HYBRID, VectorStoreQueryMode.MMR, *SPARSE_MODES):
            raise NotImplementedError(f"QANNVectorStore does not support query mode {mode.value!r}")
        if mode in (VectorStoreQueryMode.HYBRID, *SPARSE_MODES) and not hybrid:
            raise ValueError(f'query mode {mode.value!r} needs hybrid search: create the store with sparse="bm25" '
                             "or a sparse encoder")
        if mode in (VectorStoreQueryMode.HYBRID, *SPARSE_MODES) and not query.query_str:
            raise ValueError(f"query mode {mode.value!r} needs query_str for the sparse side")

        params = {**self.search_params, **{k: v for k, v in kwargs.items() if k in SEARCH_SETTINGS}}
        if hybrid:
            if mode in SPARSE_MODES:
                params["weights"] = (0.0, 1.0)
            elif mode == VectorStoreQueryMode.HYBRID:
                if query.alpha is not None:
                    params["weights"] = (query.alpha, 1.0 - query.alpha)
            else:
                params["weights"] = (1.0, 0.0)   # default and MMR modes rank by the dense side
        k = query.similarity_top_k
        if mode == VectorStoreQueryMode.HYBRID and query.hybrid_top_k:
            k = query.hybrid_top_k
        elif mode in SPARSE_MODES and query.sparse_top_k:
            k = query.sparse_top_k
        fetch_k = max(k, kwargs.get("mmr_prefetch_k") or 4 * k) if mode == VectorStoreQueryMode.MMR else k

        where = _where(query.filters, self._index.metadata_schema) if query.filters else None
        # retrievers pass node_ids=[] for "no restriction"
        allowed = None
        if query.node_ids or query.doc_ids:
            allowed = {self._rows[i] for i in query.node_ids if i in self._rows} if query.node_ids else None
            if query.doc_ids:
                in_docs = set().union(*(self._docs.get(d, set()) for d in query.doc_ids))
                allowed = in_docs if allowed is None else allowed & in_docs
            if not allowed:
                return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
            allowed = sorted(allowed)

        if query.query_embedding is not None:
            dense = np.asarray(query.query_embedding, dtype=np.float32)
        elif hybrid and mode in SPARSE_MODES:
            # sparse-only queries come without an embedding; the dense side has weight 0
            dense = np.full(self._dense.dim(), self._dense.dim() ** -0.5, dtype=np.float32)
        else:
            raise ValueError(f"query mode {mode.value!r} needs query_embedding")
        with _settings(self._index, params):
            if hybrid:
                sq = (self._sparse.encode_query(query.query_str) if query.query_str
                      else sp.csr_matrix((1, VOCAB), dtype=np.float32))
                rows, scores = self._index.query(dense, sq, fetch_k, allowed=allowed, where=where)
            else:
                rows, scores = self._index.query(dense, fetch_k, allowed=allowed, where=where)
        keep = rows >= 0
        rows = [int(r) for r in rows[keep]]
        sims = [float(s) for s in scores[keep]] if hybrid else [self._similarity(float(d)) for d in scores[keep]]

        if mode == VectorStoreQueryMode.MMR and rows:
            from llama_index.core.indices.query.embedding_utils import get_top_k_mmr_embeddings
            embeddings = [list(map(float, v)) for v in self._dense.get_vectors(rows)]
            sims, picked = get_top_k_mmr_embeddings(
                list(map(float, dense)), embeddings, similarity_top_k=k, embedding_ids=rows,
                mmr_threshold=query.mmr_threshold)
            rows = list(picked)

        nodes = self._nodes(rows)
        return VectorStoreQueryResult(nodes=nodes, similarities=list(sims), ids=[n.node_id for n in nodes])

    def _similarity(self, distance: float) -> float:
        """Higher is closer: cosine similarity for cosine, the inner product for ip, -distance for l2."""
        return 1.0 - distance if self._metric == "cosine" else -distance

    def tune_hybrid_weights(
        self,
        query_embeddings: Sequence[Sequence[float]],
        query_strs: Sequence[str],
        relevant: Sequence[set[str] | dict[str, float]],
        k: int = 10,
        *,
        metric: str = "ndcg",
        grid: Sequence[tuple[float, float]] | None = None,
    ) -> dict[tuple[float, float], float]:
        """Pick the hybrid (dense, sparse) weights from labeled queries; they apply in "hybrid" mode
        when a query sets no alpha, and are saved with the store.

        relevant holds, per query, the relevant node ids as a set or as {node id: grade}.
        Tune on queries you won't evaluate on, or the result is optimistic.
        """
        if not isinstance(self._index, qann.HybridIndex):
            raise ValueError('tune_hybrid_weights needs hybrid search (sparse="bm25" or a sparse encoder)')
        dense = np.asarray(query_embeddings, dtype=np.float32)
        sparse = sp.vstack([self._sparse.encode_query(q) for q in query_strs]).tocsr()
        rows = [({self._rows[i] for i in r if i in self._rows} if isinstance(r, (set, frozenset, list, tuple))
                 else {self._rows[i]: g for i, g in r.items() if i in self._rows}) for r in relevant]
        return self._index.tune_weights(dense, sparse, rows, k, metric=metric, grid=grid)

    # ---- persisting ----

    def persist(self, persist_path: str, fs: Any = None) -> None:
        """Write the index, texts and nodes to one qann file at persist_path (local files only)."""
        if fs is not None and getattr(fs, "protocol", "file") not in ("file", ("file", "local")):
            raise NotImplementedError("QANNVectorStore persists to local files only")
        if self._index is None:
            raise ValueError("nothing to persist: the store is empty")
        os.makedirs(os.path.dirname(os.path.abspath(persist_path)), exist_ok=True)
        self._index.save(persist_path)

    @classmethod
    def from_persist_path(
        cls,
        persist_path: str,
        *,
        mmap: bool = False,
        search_params: dict[str, Any] | None = None,
        sparse: str | SparseEncoder | None = None,
    ) -> QANNVectorStore:
        """Open a store written by persist(). mmap=True leaves a RefineIndex's full vectors and the
        BM25 posting lists on disk. A hybrid store uses BM25 unless sparse passes the custom
        encoder it was built with."""
        return cls(qann.load(persist_path, mmap=mmap), search_params=search_params, sparse=sparse)


def _where(filters: MetadataFilters | None, schema: dict):
    if filters is None:
        return None
    if not isinstance(filters, MetadataFilters):
        raise TypeError(f"filters must be MetadataFilters, got {type(filters)}")
    terms = [_where(f, schema) if isinstance(f, MetadataFilters) else _leaf(f, schema) for f in filters.filters]
    if not terms:
        return None
    condition = filters.condition or FilterCondition.AND
    if condition == FilterCondition.AND:
        return qann.all_of(terms)
    if condition == FilterCondition.OR:
        return qann.any_of(terms)
    if condition == FilterCondition.NOT:
        return ~qann.any_of(terms)   # none of the filters match, as in LlamaIndex's own stores
    raise ValueError(f"unknown filter condition {condition}")


def _true():
    return ~qann.field(NODE).is_null()   # every stored node has an id


def _false():
    return qann.field(NODE).is_null()


def _matches(filters: MetadataFilters, metadata: dict) -> bool:
    """LlamaIndex's reference semantics, extended to nested MetadataFilters."""
    hits = [_matches(f, metadata) if isinstance(f, MetadataFilters)
            else build_metadata_filter_fn(lambda _: metadata, MetadataFilters(filters=[f]))("") for f in filters.filters]
    condition = filters.condition or FilterCondition.AND
    if condition == FilterCondition.AND:
        return all(hits)
    if condition == FilterCondition.OR:
        return any(hits)
    return not any(hits)


def _leaf(f: MetadataFilter, schema: dict):
    """One LlamaIndex filter as a qann Expr, with LlamaIndex's semantics for missing values:
    a node without the key matches only != , not in and is_empty."""
    op, value = f.operator, f.value
    unsupported = (FilterOperator.TEXT_MATCH, FilterOperator.TEXT_MATCH_INSENSITIVE,
                   FilterOperator.CONTAINS, FilterOperator.ANY, FilterOperator.ALL)
    if op in unsupported:
        raise NotImplementedError(f"QANNVectorStore does not support the {op.value!r} filter operator "
                                  "(metadata lists and substring matches aren't filter columns)")
    if f.key not in schema:   # no node has this key
        return _true() if op in (FilterOperator.NE, FilterOperator.NIN, FilterOperator.IS_EMPTY) else _false()
    field = qann.field(f.key)
    if op == FilterOperator.IS_EMPTY:
        return field.is_null() | (field == "") if schema[f.key] == "str" else field.is_null()
    if op in (FilterOperator.IN, FilterOperator.NIN):
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if not values:
            return _false() if op == FilterOperator.IN else _true()
        hit = field.isin(values)
        return hit if op == FilterOperator.IN else ~hit | field.is_null()
    if op == FilterOperator.NE:
        return (field != value) | field.is_null()
    compare = {FilterOperator.EQ: field.__eq__, FilterOperator.GT: field.__gt__, FilterOperator.GTE: field.__ge__,
               FilterOperator.LT: field.__lt__, FilterOperator.LTE: field.__le__}
    return compare[op](value)


def _filterable(value, kind):
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return int(value) if kind == "int" and float(value).is_integer() else float(value)
    return None


def _kind(values) -> str | None:
    for v in values:
        if v is not None:
            return "str" if isinstance(v, str) else "float"
    return None


def _fits(value, kind: str) -> bool:
    if value is None:
        return True
    if kind == "str":
        return isinstance(value, str)
    if kind == "float":
        return isinstance(value, (int, float))
    return isinstance(value, int)


class _settings:
    """Sets search settings on the index or what it wraps for one query, then restores them."""

    def __init__(self, index, params: dict):
        self.changes = []
        dense = getattr(index, "dense", None)
        targets = [t for t in (index, dense, getattr(dense or index, "base", None)) if t is not None]
        for name, value in params.items():
            if name not in SEARCH_SETTINGS:
                raise ValueError(f"unknown search setting {name!r}; expected one of {SEARCH_SETTINGS}")
            target = next((t for t in targets if hasattr(t, name)), None)
            if target is None:
                raise ValueError(f"{type(index).__name__} has no search setting {name!r}")
            self.changes.append((target, name, tuple(value) if name == "weights" else value))

    def __enter__(self):
        self.saved = [(t, n, getattr(t, n)) for t, n, _ in self.changes]
        for t, n, v in self.changes:
            setattr(t, n, v)

    def __exit__(self, *exc):
        for t, n, v in self.saved:
            setattr(t, n, v)
