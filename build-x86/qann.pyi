"""
qann: Quantized Approximate Nearest Neighbors (Flat, IVF, PQ, re-ranking)
"""

from collections.abc import Iterable, Mapping, Sequence
import enum
import os
from typing import Annotated, TypeAlias

import numpy
from numpy.typing import NDArray


def l2_distance(a: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], b: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')]) -> NDArray[numpy.float32]:
    """Squared L2 distance between rows of two (M, D) float32 arrays."""

def cosine_distance(a: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], b: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')]) -> NDArray[numpy.float32]:
    """Cosine distance between rows of two (M, D) float32 arrays."""

def set_num_threads(n: int) -> None:
    """
    Threads for batch queries, batch adds and PQ training; 0 = one per core (default). Does not limit BLAS's own threads (VECLIB_MAXIMUM_THREADS / OPENBLAS_NUM_THREADS).
    """

def num_threads() -> int:
    """
    Thread count parallel work will use (resolves the 0 default to the core count).
    """

class Metric(enum.Enum):
    L2 = 0
    """Squared Euclidean distance"""

    Cosine = 1
    """
    1 - cosine similarity (0 to 2); vectors are scaled to unit length on add and query
    """

    InnerProduct = 2
    """
    Negative inner product -(a . b), so the largest inner product comes first
    """

class Expr:
    """
    A metadata filter for where=; build one with qann.field() or qann.where()
    """

    def __and__(self, other: Expr) -> Expr: ...

    def __or__(self, other: Expr) -> Expr: ...

    def __invert__(self) -> Expr: ...

    def __bool__(self) -> bool: ...

    def __repr__(self) -> str: ...

class Field:
    """A metadata column to compare in a where= filter; see qann.field()"""

    @property
    def name(self) -> str: ...

    def __eq__(self, value: int | float | str) -> Expr: ...

    def __ne__(self, value: int | float | str) -> Expr: ...

    def __lt__(self, value: int | float | str) -> Expr: ...

    def __le__(self, value: int | float | str) -> Expr: ...

    def __gt__(self, value: int | float | str) -> Expr: ...

    def __ge__(self, value: int | float | str) -> Expr: ...

    def isin(self, values: Sequence[int | float | str]) -> Expr:
        """Matches rows whose value is one of values"""

    def is_null(self) -> Expr:
        """Matches rows with no value in this column"""

    def between(self, lo: int | float | str, hi: int | float | str) -> Expr:
        """
        Matches rows with lo <= value <= hi, like SQL's BETWEEN: (field >= lo) & (field <= hi)
        """

    def __repr__(self) -> str: ...

def field(name: str) -> Field:
    """
    A metadata column, for filters such as (qann.field('price') < 50) & (qann.field('color') == 'red')
    """

def where(sql: str, params: Sequence[int | float | str] | Mapping[str, int | float | str] | None = None) -> Expr:
    """
    A filter from a SQL WHERE clause, e.g. "category = 'shoes' AND price < ?" with params [50]: comparisons (= != <> < <= > >=), [NOT] IN (...), IS [NOT] NULL, [NOT] BETWEEN x AND y, AND, OR, NOT and parentheses. Placeholders take values from params (a list for ?, a dict for :name), so user input never changes the filter's structure. NOT follows SQL: NOT (price < 50) excludes rows with no price, unlike ~. Combines with qann.field filters through & | ~; raises ValueError naming the column of a syntax error
    """

def any_of(filters: Sequence[Expr]) -> Expr:
    """
    Matches rows matching any of filters: f1 | f2 | ..., built in one step, so a long chain costs O(n) where adding terms one | at a time costs O(n^2)
    """

def all_of(filters: Sequence[Expr]) -> Expr:
    """
    Matches rows matching all of filters: f1 & f2 & ..., built in one step, like any_of
    """

def get_max_filter_conditions() -> int:
    """
    The most conditions a SQL where= string may have (default 1024); see set_max_filter_conditions
    """

def set_max_filter_conditions(n: int) -> None:
    """
    Set the most conditions (comparisons, IN, IS NULL, BETWEEN; an IN list counts once) a SQL where= string may have, for the whole process. Longer strings raise ValueError, so user input can't make a query scan its columns thousands of times. Filters built with qann.field aren't limited
    """

class IndexBase:
    """What every index has: ids, deletion, metadata and saving"""

    def save(self, path: str | os.PathLike) -> None:
        """Save the index to a file; load it back with qann.load(path)"""

    def update_metadata(self, ids: Iterable[int], metadata: dict) -> None:
        """
        Overwrite metadata in place: metadata is {name: one value per id}, None sets a value to null, and columns left out keep their values. Unknown, deleted or repeated ids, and columns the index doesn't have (see add_metadata_column), raise ValueError; a failed call changes nothing
        """

    def get_metadata(self, ids: Iterable[int], columns: Sequence[str] | None = None) -> dict:
        """
        Read metadata back: {name: one value per id} for the columns named (all of them, in the order they were added, when columns is None). int and float columns are numpy.ma masked arrays (int64 / float64) masked where a value is null; str columns are lists with None for null. Unknown, deleted or repeated ids, and columns the index doesn't have or named twice, raise ValueError
        """

    def add_metadata_column(self, name: str, type: str, *, default: int | float | str | None = None, ids: Iterable[int] | None = None, values: object | None = None) -> None:
        """
        Add a metadata column of type 'int', 'float' or 'str'. With default, every vector gets that value; with ids and values (one per id, None for null), those vectors get them and the rest are null; with neither, every vector is null. default can't be combined with ids and values (TypeError). Later adds that leave the column out store null. Raises ValueError, changing nothing, if the name is taken, a value doesn't fit the type, or an id is unknown, deleted or repeated
        """

    def drop_metadata_column(self, name: str) -> None:
        """
        Remove a metadata column and its values from every vector; raises ValueError if there is none
        """

    def remove(self, ids: Iterable[int]) -> int:
        """
        Delete the vectors with these ids; unknown ids are skipped. Returns how many were deleted
        """

    def compact(self) -> int:
        """
        Free the memory of deleted vectors; ids and results are unchanged. Returns how many were freed
        """

    @property
    def metadata_schema(self) -> dict:
        """
        Each metadata column's name and type ('int', 'float' or 'str'), in the order they were added
        """

    def size(self) -> int:
        """Get number of indexed vectors"""

class DenseIndex(IndexBase):
    """An index of dense float vectors"""

    def add(self, data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], ids: Iterable[int] | None = None, *, metadata: dict | None = None) -> NDArray[numpy.int64]:
        """
        Add an (n, dim) array of vectors. Pass ids (n int64 values >= 0) if the index was created with custom_ids=True; otherwise ids continue after the highest id ever assigned. metadata is {name: n values} (ints, floats or strings) to filter queries on with where=; a new column is null for earlier vectors, and a column left out of an add is null for its vectors. None in a column's values is a null. Returns the vectors' ids (int64), the given ones or the assigned ones
        """

    def query(self, query: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu')], k: int, *, allowed: Iterable[int] | None = None, where: Expr | str | tuple[str, Sequence[int | float | str] | Mapping[str, int | float | str]] | None = None) -> tuple[NDArray[numpy.int64], NDArray[numpy.float32]]:
        """
        Query k-NN for a (dim,) vector; returns 1D (indices, distances). With allowed (ids), only those vectors can be returned; unknown and deleted ids in it are ignored. With where (a qann.field filter, a SQL WHERE string as in qann.where, or (string, params)), only vectors whose metadata matches; with both, only those that pass both
        """

    def batch_query(self, query: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], k: int, *, allowed: Iterable[int] | None = None, where: Expr | str | tuple[str, Sequence[int | float | str] | Mapping[str, int | float | str]] | None = None) -> tuple[NDArray[numpy.int64], NDArray[numpy.float32]]:
        """
        Query k-NN for each row; returns (indices, distances) arrays of shape (num_queries, k), padded with -1 / inf. allowed (ids) and where (a filter, SQL string or (string, params)) restrict the results as in query(), for every row
        """

    @property
    def metric(self) -> Metric:
        """How vectors are compared (fixed at construction)"""

    def dim(self) -> int:
        """Get vector dimension"""

Index: TypeAlias = DenseIndex

class FlatIndex(DenseIndex):
    def __init__(self, dim: int, *, custom_ids: bool = False, metric: Metric = Metric.L2) -> None:
        """
        Construct FlatIndex with vector dimension; custom_ids=True makes add() take your own ids
        """

class PQDistance(enum.Enum):
    ADC = 0

    SDC = 1

class IVFIndex(DenseIndex):
    def __init__(self, dim: int, nlist: int, nprobe: int = 10, *, custom_ids: bool = False, metric: Metric = Metric.L2) -> None:
        """
        Construct IVFIndex with vector dimension, number of coarse clusters, and search breadth
        """

    def train(self, data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], max_iters: int = 25, seed: int | None = None) -> None:
        """Run k-means over sample vectors to compute coarse centroids"""

    def enable_pq(self, num_subspaces: int, centroids_per_subspace: int | None = None, *, fast_scan: bool = False, rotate: bool = False) -> None:
        """
        Enable PQ compression of residuals. Call before train()/add(). centroids_per_subspace defaults to 256, or 16 with fast_scan=True, which stores 4-bit codes and scores 32 vectors per SIMD instruction (much faster; distances are approximate, so pair it with a RefineIndex). fast_scan needs an even num_subspaces <= 1024 and ADC. rotate=True makes train() also learn a rotation (PCA, with the variance spread evenly over the subspaces) applied to every vector first: lower PQ error when dimensions are correlated (e.g. SIFT), no effect on near-isotropic embeddings
        """

    @property
    def pq_rotate(self) -> bool:
        """Whether enable_pq(..., rotate=True) was used"""

    def enable_sq(self, bits: int = 8) -> None:
        """
        Store vectors as scalar codes of 8 bits (4x smaller than floats) or 4 bits (8x smaller) per dimension instead of raw floats. Call before train()/add(); not with enable_pq(). Distances become approximate; 8 bits loses little recall, 4 bits should be wrapped in a RefineIndex
        """

    @property
    def sq_bits(self) -> int:
        """Bits per dimension set by enable_sq() (4 or 8), or 0 without SQ"""

    @property
    def fast_scan(self) -> bool:
        """Whether enable_pq(..., fast_scan=True) was used"""

    @property
    def nprobe(self) -> int:
        """Number of clusters scanned per query; can be changed after training"""

    @nprobe.setter
    def nprobe(self, arg: int, /) -> None: ...

    @property
    def pq_distance(self) -> PQDistance:
        """
        How queries score PQ codes: PQDistance.ADC (default, more accurate) or PQDistance.SDC
        """

    @pq_distance.setter
    def pq_distance(self, arg: PQDistance, /) -> None: ...

    @property
    def precomputed_tables(self) -> bool:
        """
        Whether ADC queries use per-list tables precomputed at train time (on by default). True only once they exist: PQ enabled, trained, and within the 256 MB cap
        """

    @precomputed_tables.setter
    def precomputed_tables(self, arg: bool, /) -> None: ...

    @property
    def filter_probe_scaling(self) -> bool:
        """
        Whether filtered queries probe more lists as the filter narrows, so recall stays near an unfiltered query's (on by default). Off is faster but loses recall under narrow filters. Not saved: a loaded index has it on
        """

    @filter_probe_scaling.setter
    def filter_probe_scaling(self, arg: bool, /) -> None: ...

class HNSWIndex(DenseIndex):
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, *, custom_ids: bool = False, seed: int | None = None, metric: Metric = Metric.L2) -> None:
        """
        Construct an HNSW graph index. add() builds the graph on all cores; no training unless enable_sq() is used. M is the neighbors per node (2 * M on the bottom level); ef_construction the search width while inserting. A seed makes the build reproducible when qann.set_num_threads(1) is set
        """

    def enable_sq(self, bits: int = 8) -> None:
        """
        Store vectors as scalar codes of 8 bits (4x smaller than floats) or 4 bits (8x smaller) per dimension. Call before train()/add(). Distances become approximate; wrap the index in a RefineIndex to re-rank with exact ones
        """

    def train(self, data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')]) -> None:
        """
        Learn each dimension's SQ range from sample vectors; required after enable_sq(), before add(). Values outside the sample's range are clamped, so use a representative sample
        """

    @property
    def sq_bits(self) -> int:
        """
        Bits per dimension set by enable_sq() (4 or 8), or 0 when vectors are stored as floats
        """

    @property
    def ef_search(self) -> int:
        """
        Search width per query (queries use max(ef_search, k)); higher is more accurate and slower
        """

    @ef_search.setter
    def ef_search(self, arg: int, /) -> None: ...

    @property
    def filter_ef_scaling(self) -> bool:
        """
        Whether filtered queries narrow the search width as the filter narrows, to max(k, ef_search * sqrt(fraction allowed)), keeping recall near an unfiltered query's at a fraction of the cost (on by default). Off uses the full ef_search: slower, higher recall. Not saved: a loaded index has it on
        """

    @filter_ef_scaling.setter
    def filter_ef_scaling(self, arg: bool, /) -> None: ...

    @property
    def M(self) -> int:
        """Neighbors per node (fixed at construction)"""

    @property
    def ef_construction(self) -> int:
        """Search width used while inserting (fixed at construction)"""

class RefineIndex(DenseIndex):
    def __init__(self, base: DenseIndex, k_factor: int = 10, *, custom_ids: bool = False) -> None:
        """
        Re-rank an empty, trained approximate index with exact distances. Add vectors through this wrapper.
        """

    @property
    def k_factor(self) -> int:
        """Candidates fetched from base per result (k * k_factor)"""

    @k_factor.setter
    def k_factor(self, arg: int, /) -> None: ...

    @property
    def base(self) -> DenseIndex:
        """
        The wrapped approximate index, e.g. to change nprobe on a loaded RefineIndex
        """

class SparseIndex(IndexBase):
    """
    An index of sparse vectors (BM25, SPLADE), scored by dot product with distance -(q . d); see InvertedIndex
    """

    def add(self, data: object, ids: Iterable[int] | None = None, *, metadata: dict | None = None) -> NDArray[numpy.int64]:
        """
        Add the rows of a scipy.sparse matrix (CSR, or converted with tocsr()); a column is a term id in [0, 2^32), values become float32, repeated entries in a row are summed and zeros dropped. ids and metadata as in DenseIndex.add. Returns the vectors' ids
        """

    def query(self, query: object, k: int, *, allowed: Iterable[int] | None = None, where: Expr | str | tuple[str, Sequence[int | float | str] | Mapping[str, int | float | str]] | None = None) -> tuple[NDArray[numpy.int64], NDArray[numpy.float32]]:
        """
        The k vectors with the largest dot product with a one-row sparse matrix, largest first; returns 1D (ids, distances) with distance -(q . d). Only vectors sharing a nonzero term with the query are returned. allowed and where filter as in DenseIndex.query
        """

    def batch_query(self, queries: object, k: int, *, allowed: Iterable[int] | None = None, where: Expr | str | tuple[str, Sequence[int | float | str] | Mapping[str, int | float | str]] | None = None) -> tuple[NDArray[numpy.int64], NDArray[numpy.float32]]:
        """
        query() for each row of a sparse matrix, in parallel; returns (ids, distances) arrays of shape (num_queries, k), padded with -1 / inf
        """

class InvertedIndex(SparseIndex):
    """
    Exact search over sparse vectors: a posting list per term, with MaxScore pruning
    """

    def __init__(self, *, custom_ids: bool = False) -> None:
        """
        An empty InvertedIndex. custom_ids=True makes add() take ids, as for the dense indexes
        """

    @property
    def pruning(self) -> bool:
        """
        Whether queries skip vectors that can't reach the top k (MaxScore; default True). Results are the same either way, up to which of several vectors tied at the k-th distance is kept. Not saved
        """

    @pruning.setter
    def pruning(self, arg: bool, /) -> None: ...

    @property
    def terms(self) -> int:
        """Distinct terms with a posting list"""

    @property
    def postings(self) -> int:
        """
        Stored (term, vector) pairs, including deleted vectors' until compact()
        """

class SparseApproxIndex(SparseIndex):
    """
    Approximate search over sparse vectors (SPLADE): the query's heaviest terms and each list's heaviest entries find candidates, re-scored exactly from a forward index
    """

    def __init__(self, *, custom_ids: bool = False, query_terms: int = 10, list_depth: int = 10000, k_factor: int = 10) -> None:
        """
        An empty SparseApproxIndex. custom_ids=True makes add() take ids. The search settings are the properties of the same names. It also keeps each vector's own terms, doubling the memory
        """

    @property
    def query_terms(self) -> int:
        """
        How many of the query's largest values to use (0: all; default 10). Saved with the index
        """

    @query_terms.setter
    def query_terms(self, arg: int, /) -> None: ...

    @property
    def list_depth(self) -> int:
        """
        How many of each query term's heaviest list entries to read, plus any added since the list was last sorted (0: all; default 10000). Saved with the index
        """

    @list_depth.setter
    def list_depth(self, arg: int, /) -> None: ...

    @property
    def k_factor(self) -> int:
        """
        The k * k_factor best partial scores are re-scored exactly with the whole query (default 10; at least 1). Saved with the index
        """

    @k_factor.setter
    def k_factor(self, arg: int, /) -> None: ...

    @property
    def terms(self) -> int:
        """Distinct terms with a posting list"""

    @property
    def postings(self) -> int:
        """
        Stored (term, vector) pairs, including deleted vectors' until compact()
        """

class IndexType(enum.Enum):
    Flat = 0

    IVF = 1

    HNSW = 2

    Inverted = 3

    SparseApprox = 4

class IndexOptions:
    def __init__(self) -> None: ...

    @property
    def capacity(self) -> int: ...

    @capacity.setter
    def capacity(self, arg: int, /) -> None: ...

    @property
    def nlist(self) -> int: ...

    @nlist.setter
    def nlist(self, arg: int, /) -> None: ...

    @property
    def nprobe(self) -> int: ...

    @nprobe.setter
    def nprobe(self, arg: int, /) -> None: ...

    @property
    def pq_subspaces(self) -> int: ...

    @pq_subspaces.setter
    def pq_subspaces(self, arg: int, /) -> None: ...

    @property
    def pq_centroids(self) -> int: ...

    @pq_centroids.setter
    def pq_centroids(self, arg: int, /) -> None: ...

    @property
    def ivf_sq_bits(self) -> int: ...

    @ivf_sq_bits.setter
    def ivf_sq_bits(self, arg: int, /) -> None: ...

    @property
    def pq_rotate(self) -> bool: ...

    @pq_rotate.setter
    def pq_rotate(self, arg: bool, /) -> None: ...

    @property
    def hnsw_m(self) -> int: ...

    @hnsw_m.setter
    def hnsw_m(self, arg: int, /) -> None: ...

    @property
    def ef_construction(self) -> int: ...

    @ef_construction.setter
    def ef_construction(self, arg: int, /) -> None: ...

    @property
    def ef_search(self) -> int: ...

    @ef_search.setter
    def ef_search(self, arg: int, /) -> None: ...

    @property
    def hnsw_sq_bits(self) -> int: ...

    @hnsw_sq_bits.setter
    def hnsw_sq_bits(self, arg: int, /) -> None: ...

    @property
    def pruning(self) -> bool: ...

    @pruning.setter
    def pruning(self, arg: bool, /) -> None: ...

    @property
    def query_terms(self) -> int: ...

    @query_terms.setter
    def query_terms(self, arg: int, /) -> None: ...

    @property
    def list_depth(self) -> int: ...

    @list_depth.setter
    def list_depth(self, arg: int, /) -> None: ...

    @property
    def k_factor(self) -> int: ...

    @k_factor.setter
    def k_factor(self, arg: int, /) -> None: ...

    @property
    def custom_ids(self) -> bool: ...

    @custom_ids.setter
    def custom_ids(self, arg: bool, /) -> None: ...

    @property
    def metric(self) -> Metric: ...

    @metric.setter
    def metric(self, arg: Metric, /) -> None: ...

class PQCodebook:
    def __init__(self, dim: int, num_subspaces: int, centroids_per_subspace: int = 256) -> None:
        """Construct a PQ codebook; dim must be divisible by num_subspaces"""

    def train(self, data: Annotated[NDArray[numpy.float32], dict(shape=(None, None), order='C', device='cpu')], max_iters: int = 25, seed: int | None = None) -> None:
        """
        Run per-subspace k-means over (N, dim) float32 vectors and build the SDC table
        """

    def encode(self, vec: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu')]) -> NDArray[numpy.uint8]:
        """Encode a (dim,) vector into num_subspaces uint8 codes"""

    def compute_adc_table(self, query: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu')]) -> NDArray[numpy.float32]:
        """
        Distances from each query slice to every centroid; reuse across many codes
        """

    def distance_adc(self, table: Annotated[NDArray[numpy.float32], dict(shape=(None,), order='C', device='cpu')], code: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu')]) -> float:
        """Approximate squared L2 between the table's query and a code"""

    def distance_sdc(self, query_code: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu')], code: Annotated[NDArray[numpy.uint8], dict(shape=(None,), order='C', device='cpu')]) -> float:
        """Approximate squared L2 between two codes via the precomputed SDC table"""

    def dim(self) -> int: ...

    def num_subspaces(self) -> int: ...

    def centroids_per_subspace(self) -> int: ...

def make_index(type: IndexType, dim: int, opts: IndexOptions = ...) -> IndexBase:
    """
    Construct a FlatIndex, IVFIndex, HNSWIndex, InvertedIndex or SparseApproxIndex from an IndexType and IndexOptions (dim is unused for the sparse ones)
    """

def load(path: str | os.PathLike) -> IndexBase:
    """
    Load an index saved with save(path); returns a FlatIndex, IVFIndex, HNSWIndex, RefineIndex, InvertedIndex or SparseApproxIndex
    """
