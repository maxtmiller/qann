"""Sparse encoders for hybrid search: qann's BM25, or any object with the same two methods."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import qann
import scipy.sparse as sp

# qann.BM25 (C++, since qann 0.1.0a29) gives the same term ids and weights as the
# Python encoder this package shipped in 0.1.0a1, so stores saved with it still load.
BM25 = qann.BM25
VOCAB = qann.BM25.vocab


class SparseEncoder(Protocol):
    """What a custom sparse encoder (e.g. SPLADE) provides; term ids must be in [0, 2^31)."""

    def encode_documents(self, texts: Sequence[str]) -> sp.csr_matrix:
        """One row per text."""

    def encode_query(self, text: str) -> sp.csr_matrix:
        """One row."""
