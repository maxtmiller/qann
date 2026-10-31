#pragma once

#include "vecengine/index.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace vecengine {

using std::size_t;
using std::span;
using std::vector;

// A batch of sparse vectors in CSR form: row i holds the (term, value) pairs
// indices[indptr[i] .. indptr[i + 1]) and values[...]. Terms are any uint32
// (a vocabulary id or a hashed feature); within a row they may come in any
// order and repeat (repeats are summed).
struct SparseRows {
    span<const uint64_t> indptr;  // rows() + 1 offsets, starting at 0, non-decreasing, ending at indices.size()
    span<const uint32_t> indices;
    span<const float> values;     // one per index

    size_t rows() const noexcept { return indptr.empty() ? 0 : indptr.size() - 1; }
};

// An index of sparse vectors (issue #59), scored by inner product: a
// result's distance is -(q . d), so the largest dot product comes first, as
// with Metric::InnerProduct. For keyword-style retrieval with BM25 or learned
// sparse weights (SPLADE) computed by the caller. add_batch and query_batch
// check the rows and make each one canonical (see canonical_row), handle ids,
// metadata and filters, and call the index's *_impl; InvertedIndex is exact.
class SparseIndex : public IndexBase {
public:
    // Adds rows.rows() vectors, with ids and metadata as in DenseIndex::add_batch.
    // Zero values are dropped. Throws std::invalid_argument, adding nothing,
    // on malformed rows (see SparseRows), bad ids or bad metadata. Returns the
    // vectors' ids, in order.
    vector<int64_t> add_batch(const SparseRows& rows, const int64_t* ids = nullptr, const Metadata& metadata = {});

    // Up to k vectors with the largest dot product with each query row,
    // largest first, run in parallel. Throws std::invalid_argument on
    // malformed rows.
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k) const;

    // Filtered search, as in DenseIndex: only vectors whose id is in `allowed`,
    // whose metadata matches `where`, or both, can be returned.
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k, span<const int64_t> allowed) const;
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k, const Expr& where) const;
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k, const Expr& where,
                                         span<const int64_t> allowed) const;

protected:
    SparseIndex() = default;

    // A row's (term, value) pairs.
    using Terms = vector<std::pair<uint32_t, float>>;

    // Row i of rows, sorted by term, repeats summed (in their original order)
    // and zeros dropped, into out.
    static void canonical_row(const SparseRows& rows, size_t i, Terms& out);

private:
    // What each sparse index implements. add_batch_impl gets well-formed rows
    // and must throw on bad ids before changing anything. query_impl gets one
    // canonical query and k >= 1; a non-null filter holds the live slots that
    // may be returned (count > 0), and without one every live slot may.
    virtual void add_batch_impl(const SparseRows& rows, const int64_t* ids) = 0;
    virtual vector<Neighbor> query_impl(const Terms& terms, size_t k, const detail::SlotFilter* filter) const = 0;

    // query_batch() for every overload; filter is nullptr for an unfiltered query.
    vector<vector<Neighbor>> search(const SparseRows& queries, size_t k, const detail::SlotFilter* filter) const;
};

} // namespace vecengine
