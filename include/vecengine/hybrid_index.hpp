#pragma once

#include "vecengine/index.hpp"
#include "vecengine/id_map.hpp"
#include "vecengine/fusion.hpp"
#include "vecengine/sparse_index.hpp"

#include <array>
#include <cstddef>
#include <iosfwd>
#include <memory>
#include <span>
#include <vector>

namespace vecengine {

using std::size_t;
using std::span;
using std::vector;

// Hybrid search (issue #60): one index holding a dense and a sparse
// representation of every document, searched together and fused.
//
// Holds both by reference, as a RefineIndex holds its base: the caller keeps
// them alive, trains the dense one if it needs training, and can keep tuning
// either (nprobe, ef_search, pruning). A HybridIndex returned by load_index()
// owns them instead. User ids and metadata live here; the two indexes get
// plain slots and are only changed through this index, so slot i is the same
// document in both.
class HybridIndex : public IndexBase {
public:
    // dense and sparse must be empty, use no custom ids and not be wrapped
    // already (by a RefineIndex or another HybridIndex). Marks both wrapped.
    // custom_ids makes add_batch() take user ids.
    HybridIndex(DenseIndex& dense, SparseIndex& sparse, bool custom_ids = false);

    // Clears the wrapped marks, so caller-owned indexes can be used alone again.
    ~HybridIndex() override;

    // Adds n documents: row i of dense_vecs (n * dim() floats) and row i of
    // sparse_rows are the same document. ids and metadata as in
    // DenseIndex::add_batch. Checks both representations, ids and metadata
    // first, and throws std::invalid_argument without adding anything on a
    // row count mismatch, malformed sparse rows, a zero dense row under
    // Metric::Cosine, bad ids or bad metadata. Returns the documents' ids.
    vector<int64_t> add_batch(span<const float> dense_vecs, const SparseRows& sparse_rows,
                              const int64_t* ids = nullptr, const Metadata& metadata = {});

    // Up to k fused results per query, best first: row q of dense_queries (n *
    // dim() floats) and of sparse_queries is one query. Each side is searched
    // for k * depth_factor() results with the same filter, then the two lists
    // are fused with method() and weights(); scores are FusedHit scores,
    // higher is better. Filtered overloads restrict results as in DenseIndex.
    // Throws std::invalid_argument if the two have different row counts or the
    // sparse rows are malformed.
    vector<vector<FusedHit>> query_batch(span<const float> dense_queries, const SparseRows& sparse_queries,
                                         size_t k) const;
    vector<vector<FusedHit>> query_batch(span<const float> dense_queries, const SparseRows& sparse_queries,
                                         size_t k, span<const int64_t> allowed) const;
    vector<vector<FusedHit>> query_batch(span<const float> dense_queries, const SparseRows& sparse_queries,
                                         size_t k, const Expr& where) const;
    vector<vector<FusedHit>> query_batch(span<const float> dense_queries, const SparseRows& sparse_queries,
                                         size_t k, const Expr& where, span<const int64_t> allowed) const;

    // IndexBase overrides; see IndexBase for the contract of each. remove()
    // deletes the same slots in both indexes, and compact() compacts both with
    // the same remap.
    size_t remove(span<const int64_t> ids) override;
    size_t compact() override;

    // Writes this index's header and settings with both indexes nested inside.
    void save(std::ostream& out) const override;

    // Rebuilds a HybridIndex and the two indexes nested in it from the body
    // save() wrote. Throws std::runtime_error on corrupt data.
    static std::unique_ptr<HybridIndex> load_body(std::istream& in, uint32_t version);

    size_t slots() const noexcept override { return id_map_.slots(); }
    bool ids_are_slots() const noexcept override { return id_map_.ids_are_slots(); }
    bool custom_ids() const noexcept override { return id_map_.custom_ids(); }
    size_t size() const noexcept override { return id_map_.live(); }
    size_t dim() const noexcept { return dense_.dim(); }

    // The wrapped indexes, e.g. to tune nprobe or ef_search on a loaded index.
    DenseIndex& dense() noexcept { return dense_; }
    const DenseIndex& dense() const noexcept { return dense_; }
    SparseIndex& sparse() noexcept { return sparse_; }
    const SparseIndex& sparse() const noexcept { return sparse_; }

    // Fusion settings, used by every query until changed, and saved with the
    // index. weights() is {dense, sparse}: each finite and >= 0, not both 0
    // (default {1, 1}). rrf_k() is RRF's constant, finite and > 0 (default 60).
    // depth_factor() >= 1 (default 5): each side fetches k * depth_factor
    // results; with Linear it also sets which hits normalize to 0. The setters
    // throw std::invalid_argument on bad values.
    FusionMethod method() const noexcept { return method_; }
    void set_method(FusionMethod method) noexcept { method_ = method; }
    std::array<float, 2> weights() const noexcept { return weights_; }
    void set_weights(float dense, float sparse);
    float rrf_k() const noexcept { return rrf_k_; }
    void set_rrf_k(float k);
    size_t depth_factor() const noexcept { return depth_factor_; }
    void set_depth_factor(size_t depth_factor);

private:
    const detail::IdMap& id_map() const noexcept override { return id_map_; }

    // query_batch() for every overload; filter is nullptr for an unfiltered query.
    vector<vector<FusedHit>> search(span<const float> dense_queries, const SparseRows& sparse_queries, size_t k,
                                    const detail::SlotFilter* filter) const;

    // Used by load_body(): takes ownership of already filled indexes and skips
    // the public constructor's empty checks.
    HybridIndex(std::unique_ptr<DenseIndex> dense, std::unique_ptr<SparseIndex> sparse);

    std::unique_ptr<DenseIndex> owned_dense_;    // set on file load; nullptr when wrapping caller-owned indexes
    std::unique_ptr<SparseIndex> owned_sparse_;
    DenseIndex& dense_;
    SparseIndex& sparse_;
    FusionMethod method_ = FusionMethod::RRF;
    std::array<float, 2> weights_{1.0f, 1.0f};
    float rrf_k_ = 60.0f;
    size_t depth_factor_ = 5;
    detail::IdMap id_map_;
};

} // namespace vecengine
