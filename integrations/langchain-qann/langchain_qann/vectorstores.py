"""LangChain vector store backed by a qann index."""
from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
import qann
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

# Columns the store keeps next to the user's metadata. Text, id and the full
# metadata (as JSON, so nested values round-trip) live in the index file
# itself, so save() writes one file.
TEXT, ID, META = "_lc_text", "_lc_id", "_lc_meta"
RESERVED = (TEXT, ID, META)
METRICS = {"cosine": qann.Metric.Cosine, "l2": qann.Metric.L2, "ip": qann.Metric.InnerProduct}
PRESETS = ("hnsw", "ivf-pq", "flat")
SEARCH_SETTINGS = ("nprobe", "ef_search", "k_factor")
# IVF needs at least this many vectors in its first batch to train on
MIN_IVF_TRAIN = 1000

Filter = dict[str, Any] | str | qann.Expr | tuple | None


class QANNVectorStore(VectorStore):
    """A LangChain VectorStore on a qann index.

    index is a preset name or any qann.DenseIndex you built yourself:
      - "hnsw" (default): HNSWIndex(M=16, ef_construction=200); adds at any time.
      - "ivf-pq": IVF with fast-scan PQ (dim / 2 subspaces) under RefineIndex(k_factor=4),
        nlist = max(100, sqrt(n)); trains on the first batch added (at least 1,000 texts),
        so build it with from_texts on the whole corpus.
      - "flat": exact search.
    Presets are created on the first add, once the embedding size is known.

    search_params sets search options for every query: nprobe, ef_search or k_factor,
    applied to the index or its RefineIndex base. Searches take the same names as
    keyword arguments for one query.

    filter takes a dict of exact matches ({"source": "10-K", "year": [2023, 2024]}, a
    list meaning any of), a SQL WHERE string ("year >= 2020 AND lang = 'en'"), a
    (sql, params) tuple or a qann.Expr. Metadata strings and numbers are filterable;
    other values (lists, dicts) are kept with the document but not filterable.
    """

    def __init__(
        self,
        embedding: Embeddings,
        index: str | qann.DenseIndex = "hnsw",
        *,
        metric: str = "cosine",
        search_params: dict[str, int] | None = None,
    ) -> None:
        if isinstance(index, str) and index not in PRESETS:
            raise ValueError(f"index must be one of {PRESETS} or a qann.DenseIndex, got {index!r}")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {tuple(METRICS)}, got {metric!r}")
        self._embedding = embedding
        self._preset = index if isinstance(index, str) else None
        self._metric = metric
        self.index: qann.DenseIndex | None = None if isinstance(index, str) else index
        self._base = getattr(self.index, "base", None)   # a RefineIndex needs its base kept alive
        self.search_params = dict(search_params or {})
        self._rows: dict[str, int] = {}   # document id -> qann id
        if self.index is not None:
            self._metric = {v: k for k, v in METRICS.items()}[self.index.metric]
            if self.index.size() and ID not in self.index.metadata_schema:
                raise ValueError("the index already holds vectors added outside a QANNVectorStore; "
                                 "pass an empty index or one saved by QANNVectorStore.save")
            self._load_ids()

    @property
    def embeddings(self) -> Embeddings:
        return self._embedding

    # ---- adding ----

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        texts = list(texts)
        if not texts:
            return []
        vectors = np.asarray(self._embedding.embed_documents(texts), dtype=np.float32)
        return self._add(texts, vectors, metadatas, ids)

    def add_embeddings(
        self,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]] | np.ndarray,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add texts with precomputed embeddings."""
        return self._add(list(texts), np.asarray(embeddings, dtype=np.float32), metadatas, ids)

    def _add(self, texts, vectors, metadatas, ids) -> list[str]:
        n = len(texts)
        metadatas = metadatas or [{} for _ in range(n)]
        ids = [str(i) if i is not None else str(uuid.uuid4()) for i in (ids or [None] * n)]
        if not (len(metadatas) == len(ids) == len(vectors) == n):
            raise ValueError("texts, embeddings, metadatas and ids must have the same length")
        if len(set(ids)) != n:
            raise ValueError("ids must be distinct")
        if self.index is None:
            self.index = self._create(vectors)
        # adding an existing id replaces that document
        self.delete([i for i in ids if i in self._rows])
        new = self.index.add(np.ascontiguousarray(vectors), metadata=self._columns(texts, metadatas, ids))
        self._rows.update(zip(ids, (int(r) for r in new)))
        return ids

    def _create(self, vectors: np.ndarray) -> qann.DenseIndex:
        n, dim = vectors.shape
        metric = METRICS[self._metric]
        if self._preset == "flat":
            return qann.FlatIndex(dim, metric=metric)
        if self._preset == "hnsw":
            return qann.HNSWIndex(dim, M=16, ef_construction=200, metric=metric)
        if n < MIN_IVF_TRAIN:
            raise ValueError(f'index="ivf-pq" trains on the first batch added, which needs at least '
                             f"{MIN_IVF_TRAIN} texts (got {n}); add the corpus at once, or use \"hnsw\"")
        if dim % 2:
            raise ValueError(f'index="ivf-pq" needs an even embedding size (got {dim})')
        base = qann.IVFIndex(dim, max(100, round(math.sqrt(n))), metric=metric)
        base.enable_pq(dim // 2, fast_scan=True)
        base.train(np.ascontiguousarray(vectors))
        self._base = base
        return qann.RefineIndex(base, k_factor=4)

    def _columns(self, texts, metadatas, ids) -> dict[str, list]:
        schema = self.index.metadata_schema
        columns = {TEXT: list(texts), ID: list(ids), META: [json.dumps(m, default=str) for m in metadatas]}
        keys = {k for m in metadatas for k in m if k not in RESERVED}
        for key in keys:
            values = [_filterable(m.get(key), schema.get(key)) for m in metadatas]
            kind = schema.get(key) or _kind(values)
            if kind is not None:
                columns[key] = [v if _fits(v, kind) else None for v in values]
        return columns

    # ---- reading and deleting ----

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        found = [(i, self._rows[i]) for i in ids if i in self._rows]
        return self._documents([r for _, r in found])

    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        if self.index is None:
            return True
        if ids is None:
            ids = list(self._rows)
        rows = [self._rows.pop(i) for i in ids if i in self._rows]
        if rows:
            self.index.remove(rows)
        return True

    def _load_ids(self) -> None:
        live = self.index.ids()
        if len(live) and ID in self.index.metadata_schema:
            self._rows = dict(zip(self.index.get_metadata(live, [ID])[ID], (int(r) for r in live)))

    def _documents(self, rows: Sequence[int]) -> list[Document]:
        if not rows:
            return []
        cols = self.index.get_metadata(rows, [TEXT, ID, META])
        return [Document(page_content=t, metadata=json.loads(m), id=i)
                for t, i, m in zip(cols[TEXT], cols[ID], cols[META])]

    # ---- searching ----

    def similarity_search(self, query: str, k: int = 4, filter: Filter = None, **kwargs: Any) -> list[Document]:
        return [d for d, _ in self.similarity_search_with_score(query, k, filter, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Filter = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Documents with their qann distance (lower is closer)."""
        return self.similarity_search_with_score_by_vector(self._embedding.embed_query(query), k, filter, **kwargs)

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, filter: Filter = None, **kwargs: Any
    ) -> list[Document]:
        return [d for d, _ in self.similarity_search_with_score_by_vector(embedding, k, filter, **kwargs)]

    def similarity_search_with_score_by_vector(
        self, embedding: list[float], k: int = 4, filter: Filter = None, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        rows, dists = self._search(embedding, k, filter, kwargs)
        return list(zip(self._documents(rows), (float(d) for d in dists)))

    def max_marginal_relevance_search(
        self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5,
        filter: Filter = None, **kwargs: Any,
    ) -> list[Document]:
        return self.max_marginal_relevance_search_by_vector(
            self._embedding.embed_query(query), k, fetch_k, lambda_mult, filter, **kwargs)

    def max_marginal_relevance_search_by_vector(
        self, embedding: list[float], k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5,
        filter: Filter = None, **kwargs: Any,
    ) -> list[Document]:
        rows, _ = self._search(embedding, fetch_k, filter, kwargs)
        if not rows:
            return []
        chosen = maximal_marginal_relevance(np.asarray(embedding, dtype=np.float32),
                                            self.index.get_vectors(rows), lambda_mult=lambda_mult, k=k)
        return self._documents([rows[i] for i in chosen])

    def _search(self, embedding, k: int, filter: Filter, params: dict) -> tuple[list[int], list[float]]:
        if self.index is None or not self._rows:
            return [], []
        query = np.asarray(embedding, dtype=np.float32)
        with _settings(self.index, {**self.search_params, **params}):
            rows, dists = self.index.query(query, k, where=_where(filter))
        keep = rows >= 0
        return [int(r) for r in rows[keep]], list(dists[keep])

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        if self._metric == "cosine":
            return self._cosine_relevance_score_fn
        if self._metric == "l2":   # qann's L2 distance is squared
            return lambda d: self._euclidean_relevance_score_fn(math.sqrt(max(d, 0.0)))
        return self._max_inner_product_relevance_score_fn

    # ---- construction, saving and loading ----

    @classmethod
    def from_texts(
        cls,
        texts: list[str],
        embedding: Embeddings,
        metadatas: list[dict] | None = None,
        *,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> QANNVectorStore:
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    def save(self, path: str | os.PathLike) -> None:
        """Write the index, texts and metadata to one qann file."""
        if self.index is None:
            raise ValueError("nothing to save: the store is empty")
        self.index.save(path)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        embedding: Embeddings,
        *,
        mmap: bool = False,
        search_params: dict[str, int] | None = None,
    ) -> QANNVectorStore:
        """Open a store written by save(). mmap=True leaves a RefineIndex's full vectors on disk."""
        index = qann.load(path, mmap=mmap)
        return cls(embedding, index, search_params=search_params)


def _where(filter: Filter):
    if filter is None or isinstance(filter, (str, tuple, qann.Expr)):
        return filter
    if not isinstance(filter, dict):
        raise TypeError(f"filter must be a dict, SQL string, (sql, params) tuple or qann.Expr, got {type(filter)}")
    terms = []
    for key, value in filter.items():
        f = qann.field(key)
        terms.append(f.isin(list(value)) if isinstance(value, (list, tuple, set)) else f == value)
    return qann.all_of(terms) if terms else None


def _filterable(value, kind):
    """value as a metadata column value: str or float (bools and ints as numbers), else None."""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return int(value) if kind == "int" and float(value).is_integer() else float(value)
    return None


def _kind(values) -> str | None:
    """The column type a new key gets from its first filterable value."""
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
    """Sets nprobe / ef_search / k_factor on the index or its base for one query, then restores them."""

    def __init__(self, index, params: dict):
        self.changes = []
        for name, value in params.items():
            if name not in SEARCH_SETTINGS:
                raise ValueError(f"unknown search setting {name!r}; expected one of {SEARCH_SETTINGS}")
            target = next((t for t in (index, getattr(index, "base", None)) if t is not None and hasattr(t, name)), None)
            if target is None:
                raise ValueError(f"{type(index).__name__} has no search setting {name!r}")
            self.changes.append((target, name, value))

    def __enter__(self):
        self.saved = [(t, n, getattr(t, n)) for t, n, _ in self.changes]
        for t, n, v in self.changes:
            setattr(t, n, v)

    def __exit__(self, *exc):
        for t, n, v in self.saved:
            setattr(t, n, v)
