"""Sparse encoders for hybrid search: built-in BM25, or any object with the same two methods."""
from __future__ import annotations

import collections
import math
import re
import zlib
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import scipy.sparse as sp

# Term ids are hashed, so nothing but the stored texts is needed to rebuild the
# statistics; 31 bits keep scipy's indices in int32 range.
VOCAB = 2**31
STOPWORDS = frozenset("a an and are as at be but by for if in into is it no not of on or such that the their then "
                      "there these they this to was will with".split())


class SparseEncoder(Protocol):
    """What a custom sparse encoder (e.g. SPLADE) provides; term ids must be in [0, 2^31)."""

    def encode_documents(self, texts: Sequence[str]) -> sp.csr_matrix:
        """One row per text."""

    def encode_query(self, text: str) -> sp.csr_matrix:
        """One row."""


class BM25:
    """BM25 that can grow and shrink with the corpus.

    Documents store only their saturated term frequencies,
    tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgdl)), and each query term
    carries its IDF, ln(1 + (N - df + 0.5) / (df + 0.5)), from the current
    document counts. So adding or deleting documents updates IDF for every
    later query without re-encoding stored documents. avgdl is the average
    length when a document is added; later documents use the updated average.
    Tokens: lowercase letters and digits, Lucene's English stopwords, no stemming.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.df: collections.Counter[int] = collections.Counter()
        self.n_docs = 0
        self.total_len = 0

    @staticmethod
    def terms(text: str) -> list[int]:
        return [zlib.crc32(t.encode()) & (VOCAB - 1)
                for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]

    def encode_documents(self, texts: Sequence[str]) -> sp.csr_matrix:
        """Count texts into the statistics, then encode them."""
        counts = [collections.Counter(self.terms(t)) for t in texts]
        for c in counts:
            self.df.update(c.keys())
            self.total_len += sum(c.values())
        self.n_docs += len(counts)
        avgdl = self.total_len / max(self.n_docs, 1) or 1.0
        rows, cols, vals = [], [], []
        for i, c in enumerate(counts):
            norm = self.k1 * (1 - self.b + self.b * sum(c.values()) / avgdl)
            for term, tf in c.items():
                rows.append(i)
                cols.append(term)
                vals.append(tf * (self.k1 + 1) / (tf + norm))
        return sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)), shape=(len(counts), VOCAB))

    def encode_query(self, text: str) -> sp.csr_matrix:
        terms = [t for t in set(self.terms(text)) if self.df[t]]
        vals = [math.log(1 + (self.n_docs - self.df[t] + 0.5) / (self.df[t] + 0.5)) for t in terms]
        return sp.csr_matrix((np.array(vals, dtype=np.float32), ([0] * len(terms), terms)), shape=(1, VOCAB))

    def remove(self, texts: Sequence[str]) -> None:
        """Take deleted texts out of the statistics."""
        for text in texts:
            c = collections.Counter(self.terms(text))
            self.df.subtract(c.keys())
            self.total_len -= sum(c.values())
            self.n_docs -= 1
        self.df = +self.df   # drop terms no document has any more

    def rebuild(self, texts: Sequence[str]) -> None:
        """Recount the statistics from every stored text (after loading a saved store)."""
        self.__init__(self.k1, self.b)
        for text in texts:
            c = collections.Counter(self.terms(text))
            self.df.update(c.keys())
            self.total_len += sum(c.values())
        self.n_docs = len(texts)
