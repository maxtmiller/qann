#pragma once

#include "vecengine/index.hpp"
#include "vecengine/id_map.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <unordered_map>
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

// Exact inverted index over sparse vectors (issue #59), for keyword-style
// retrieval with BM25 or learned sparse weights (SPLADE) computed by the
// caller. Scores by inner product: a result's distance is -(q . d), so the
// largest dot product comes first, as with Metric::InnerProduct. A query
// reads only its own terms' posting lists, and returns only vectors that
// share at least one nonzero term with it.
class SparseIndex : public IndexBase {
public:
    // custom_ids makes add_batch() take user ids.
    explicit SparseIndex(bool custom_ids = false) : id_map_(custom_ids) {}

    // Adds rows.rows() vectors, with ids and metadata as in Index::add_batch.
    // Zero values are dropped. Throws std::invalid_argument, adding nothing,
    // on malformed rows (see SparseRows), bad ids or bad metadata. Returns the
    // vectors' ids, in order.
    vector<int64_t> add_batch(const SparseRows& rows, const int64_t* ids = nullptr, const Metadata& metadata = {});

    // Up to k vectors with the largest dot product with each query row,
    // largest first, run in parallel. Throws std::invalid_argument on
    // malformed rows.
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k) const;

    // Filtered search, as in Index: only vectors whose id is in `allowed`,
    // whose metadata matches `where`, or both, can be returned.
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k, span<const int64_t> allowed) const;
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k, const Expr& where) const;
    vector<vector<Neighbor>> query_batch(const SparseRows& queries, size_t k, const Expr& where,
                                         span<const int64_t> allowed) const;

    // IndexBase overrides; see IndexBase for the contract of each.
    size_t remove(span<const int64_t> ids) override { return id_map_.remove(ids); }
    size_t compact() override;
    void save(std::ostream& out) const override;
    size_t slots() const noexcept override { return id_map_.slots(); }
    bool ids_are_slots() const noexcept override { return id_map_.ids_are_slots(); }
    bool custom_ids() const noexcept override { return id_map_.custom_ids(); }
    size_t size() const noexcept override { return id_map_.live(); }

    // Distinct terms with a posting list, and stored (term, vector) pairs,
    // including those of deleted vectors not yet compacted away.
    size_t terms() const noexcept { return lists_.size(); }
    size_t postings() const noexcept;

    // Rebuilds a SparseIndex from the body save() wrote, after load_index()
    // has read the header. Throws std::runtime_error on corrupt data.
    static std::unique_ptr<SparseIndex> load_body(std::istream& in, uint32_t version);

private:
    struct Posting {
        uint32_t slot;
        float weight;
    };

    const detail::IdMap& id_map() const noexcept override { return id_map_; }

    // query_batch() for every overload; filter is nullptr for an unfiltered query.
    vector<vector<Neighbor>> search(const SparseRows& queries, size_t k, const detail::SlotFilter* filter) const;

    // The posting lists of each term; a list holds its vectors in increasing slot order.
    std::unordered_map<uint32_t, vector<Posting>> lists_;
    detail::IdMap id_map_;
};

} // namespace vecengine
