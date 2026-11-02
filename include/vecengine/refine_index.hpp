#pragma once

#include "vecengine/index.hpp"
#include "vecengine/id_map.hpp"

#include <cstddef>
#include <vector>
#include <span>
#include <iosfwd>
#include <memory>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

// Re-ranks an approximate index's results with exact distances.
// query() asks `base` for k * k_factor candidates, re-scores each against
// the raw vector stored here, and returns the exact top k.
//
// Only useful when `base` ranks with approximate distances (e.g. IVFIndex
// with PQ). Holds `base` by reference: the caller keeps it alive, trains it
// before the first add(), and can keep tuning it (nprobe, pq_distance). A
// RefineIndex returned by load_index() owns its base instead.
//
// User ids live here; the base gets plain slots, so base slot i and this
// index's slot i are always the same vector.
class RefineIndex : public DenseIndex {
public:
    // base must have no slots yet, ids equal to its slots (no custom ids, never compacted), and not already be wrapped by
    // another RefineIndex, so its slots line up with this index's raw storage.
    // k_factor >= 1. custom_ids makes add_batch() take user ids. Marks base as
    // wrapped (see IndexBase::wrapped()).
    explicit RefineIndex(DenseIndex& base, size_t k_factor = 10, bool custom_ids = false);

    // Clears the base's wrapped mark, so a caller-owned base can be used on
    // its own again.
    ~RefineIndex() override;

    // DenseIndex overrides; see DenseIndex for the contract of each. Adds go to the base
    // too (without ids), and remove() deletes the same slots in the base. The
    // metric is the base's. query_batch fetches every query's candidates with
    // one base query_batch call (so IVFIndex's batched coarse search applies),
    // then re-ranks in parallel. A filtered query passes its filter (base
    // slots are this index's slots) to the base.
    size_t remove(span<const int64_t> ids) override;

    // Compacts this index's ids and raw vectors and the base's storage with
    // the same remap, keeping them aligned.
    size_t compact() override;

    // Writes this index's header and fields with the base nested inside.
    void save(std::ostream& out) const override;

    // Rebuilds a RefineIndex, and the Flat or IVF base nested in it, from the
    // body save() wrote. Throws std::runtime_error on corrupt data.
    static std::unique_ptr<RefineIndex> load_body(std::istream& in, uint32_t version);

    size_t size() const noexcept override { return id_map_.live(); }
    size_t dim() const noexcept override { return base_.dim(); }
    bool stores_vectors() const noexcept override { return true; }

    size_t slots() const noexcept override { return id_map_.slots(); }
    bool custom_ids() const noexcept override { return id_map_.custom_ids(); }
    bool ids_are_slots() const noexcept override { return id_map_.ids_are_slots(); }

    // The wrapped approximate index, e.g. to tune nprobe on a loaded index.
    DenseIndex& base() noexcept { return base_; }
    const DenseIndex& base() const noexcept { return base_; }

    // Candidates fetched from the base per result (k * k_factor).
    // set_k_factor throws std::invalid_argument when k_factor is 0.
    size_t k_factor() const noexcept { return k_factor_; }
    void set_k_factor(size_t k_factor);

private:
    // DenseIndex implementations: vectors already prepared for the metric, and
    // passed to the base's *_impl so they are prepared only once.
    void add_batch_impl(span<const float> vecs, size_t n, const int64_t* ids) override;
    vector<Neighbor> query_impl(span<const float> vec, size_t k, const detail::SlotFilter* filter) const override;
    const float* stored_vector(uint32_t slot) const override { return data_.data() + size_t{slot} * dim(); }
    vector<vector<Neighbor>> query_batch_impl(span<const float> queries, size_t num_queries, size_t k,
                                              const detail::SlotFilter* filter) const override;
    const detail::IdMap& id_map() const noexcept override { return id_map_; }

    // Candidates to fetch from the base for k results: k * k_factor, capped
    // at the vectors that may be returned.
    size_t candidates(size_t k, const detail::SlotFilter* filter) const;

    // compact() for a RefineIndex that is the dense side of a HybridIndex:
    // ids become the new slots.
    size_t compact_as_base() override;

    // compact() and compact_as_base(): keep_ids as in IdMap::compact.
    size_t compact_impl(bool keep_ids);

    // Exact top k of the base's candidates for vec (candidate ids are base slots).
    vector<Neighbor> rerank(span<const float> vec, const vector<Neighbor>& candidates, size_t k) const;

    // Whether a filtered query skips the base and scores the allowed vectors
    // directly (see direct_scan).
    bool use_direct(const detail::SlotFilter* filter) const;

    // Exact top k among the filter's slots, from the raw vectors stored here.
    vector<Neighbor> direct_scan(span<const float> vec, size_t k, const detail::SlotFilter& filter) const;

    // Used by load_body(): takes ownership of an already filled base and skips
    // the public constructor's empty-base check.
    RefineIndex(std::unique_ptr<DenseIndex> base, size_t k_factor)
        : DenseIndex(base->metric()), owned_base_(std::move(base)), base_(*owned_base_), k_factor_(k_factor) {
        base_.wrapped_ = true;
    }

    std::unique_ptr<DenseIndex> owned_base_; // set on file load; nullptr when wrapping a caller-owned base.
    DenseIndex& base_; // approximate first-stage index
    size_t k_factor_; // candidates fetched per result: k * k_factor
    vector<float> data_; // raw vectors, row-major: id_map_.slots() * dim
    detail::IdMap id_map_;
};

} // namespace vecengine
