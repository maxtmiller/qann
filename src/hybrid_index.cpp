#include "vecengine/hybrid_index.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/inverted_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/sparse_approx_index.hpp"
#include "parallel.hpp"
#include "serialize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace vecengine {

HybridIndex::HybridIndex(DenseIndex& dense, SparseIndex& sparse, bool custom_ids)
    : dense_(dense), sparse_(sparse), id_map_(custom_ids) {
    for (const IndexBase* index : {static_cast<const IndexBase*>(&dense), static_cast<const IndexBase*>(&sparse)}) {
        if (index->slots() != 0) throw std::invalid_argument("the dense and sparse indexes must be empty");
        if (!index->ids_are_slots())
            throw std::invalid_argument("the dense and sparse indexes must not use custom ids; give the HybridIndex custom_ids instead");
        if (index->wrapped()) throw std::invalid_argument("the dense and sparse indexes must not already be wrapped by another index");
    }
    dense.wrapped_ = true;
    sparse.wrapped_ = true;
}

HybridIndex::HybridIndex(std::unique_ptr<DenseIndex> dense, std::unique_ptr<SparseIndex> sparse)
    : owned_dense_(std::move(dense)), owned_sparse_(std::move(sparse)), dense_(*owned_dense_), sparse_(*owned_sparse_) {
    dense_.wrapped_ = true;
    sparse_.wrapped_ = true;
}

HybridIndex::~HybridIndex() {
    dense_.wrapped_ = false;
    sparse_.wrapped_ = false;
}

void HybridIndex::set_weights(float dense, float sparse) {
    if (!std::isfinite(dense) || !std::isfinite(sparse) || dense < 0.0f || sparse < 0.0f)
        throw std::invalid_argument("weights must be finite and >= 0");
    if (dense == 0.0f && sparse == 0.0f) throw std::invalid_argument("weights must not both be 0");
    weights_ = {dense, sparse};
}

void HybridIndex::set_rrf_k(float k) {
    if (!std::isfinite(k) || k <= 0.0f) throw std::invalid_argument("rrf_k must be finite and > 0");
    rrf_k_ = k;
}

void HybridIndex::set_depth_factor(size_t depth_factor) {
    if (depth_factor == 0) throw std::invalid_argument("depth_factor must be >= 1");
    depth_factor_ = depth_factor;
}

vector<int64_t> HybridIndex::add_batch(span<const float> dense_vecs, const SparseRows& sparse_rows, const int64_t* ids,
                                       const Metadata& metadata) {
    // check everything before changing either index, so a throw adds nothing anywhere
    SparseIndex::check_rows(sparse_rows);
    const size_t n = sparse_rows.rows();
    if (dense_vecs.size() != n * dim())
        throw std::invalid_argument("the dense rows (" + std::to_string(dense_vecs.size() / std::max<size_t>(dim(), 1)) +
                                    ") and sparse rows (" + std::to_string(n) + ") must be the same documents");
    id_map_.check(n, ids);
    metadata_.check(n, metadata);
    vector<float> unit;
    if (dense_.metric() == Metric::Cosine) unit = dense_.normalize_rows(dense_vecs, n);

    // the dense add can still refuse (an untrained IVFIndex), but before it changes anything
    const size_t first = slots();
    dense_.add_batch_impl(unit.empty() ? dense_vecs : span<const float>(unit), n, nullptr);
    sparse_.add_batch_impl(sparse_rows, nullptr);
    id_map_.add(n, ids);
    metadata_.append(first, n, metadata);

    vector<int64_t> added(n);
    for (size_t i = 0; i < n; ++i) added[i] = id_map_.label(static_cast<uint32_t>(first + i));
    return added;
}

vector<vector<FusedHit>> HybridIndex::query_batch(span<const float> dense_queries, const SparseRows& sparse_queries,
                                                  size_t k) const {
    return search(dense_queries, sparse_queries, k, nullptr);
}

vector<vector<FusedHit>> HybridIndex::query_batch(span<const float> dense_queries, const SparseRows& sparse_queries,
                                                  size_t k, span<const int64_t> allowed) const {
    const detail::SlotFilter filter = id_map_.filter(allowed);
    return search(dense_queries, sparse_queries, k, &filter);
}

vector<vector<FusedHit>> HybridIndex::query_batch(span<const float> dense_queries, const SparseRows& sparse_queries,
                                                  size_t k, const Expr& where) const {
    const detail::SlotFilter filter = where_filter(where, nullptr);
    return search(dense_queries, sparse_queries, k, &filter);
}

vector<vector<FusedHit>> HybridIndex::query_batch(span<const float> dense_queries, const SparseRows& sparse_queries,
                                                  size_t k, const Expr& where, span<const int64_t> allowed) const {
    const detail::SlotFilter filter = where_filter(where, &allowed);
    return search(dense_queries, sparse_queries, k, &filter);
}

vector<vector<FusedHit>> HybridIndex::search(span<const float> dense_queries, const SparseRows& sparse_queries, size_t k,
                                             const detail::SlotFilter* filter) const {
    SparseIndex::check_rows(sparse_queries);
    const size_t n = sparse_queries.rows();
    if (dense_queries.size() != n * dim())
        throw std::invalid_argument("the dense and sparse queries must have the same number of rows");
    vector<vector<FusedHit>> results(n);
    const size_t reachable = filter ? filter->count : id_map_.live();
    if (k == 0 || reachable == 0 || n == 0) return results;

    // both sides use the same slots, so the filter applies to each as it is
    const size_t depth = std::min(k > std::numeric_limits<size_t>::max() / depth_factor_
                                      ? std::numeric_limits<size_t>::max() : k * depth_factor_,
                                  reachable);
    const auto dense_hits = dense_.query_prepared(dense_queries, n, depth, filter, true);
    const auto sparse_hits = sparse_.search(sparse_queries, depth, filter);

    detail::parallel_for(n, [&](size_t q) {
        vector<int64_t> ids[2];
        vector<float> dists[2];
        for (int side = 0; side < 2; ++side) {
            const vector<Neighbor>& hits = side == 0 ? dense_hits[q] : sparse_hits[q];
            for (const Neighbor& h : hits) {
                ids[side].push_back(static_cast<int64_t>(h.index));   // a slot: both indexes' ids are their slots
                dists[side].push_back(h.distance);
            }
        }
        const span<const int64_t> id_lists[2] = {ids[0], ids[1]};
        const span<const float> dist_lists[2] = {dists[0], dists[1]};
        results[q] = fuse(method_, id_lists, dist_lists, weights_, rrf_k_, k);
        for (FusedHit& hit : results[q]) hit.id = id_map_.label(static_cast<uint32_t>(hit.id));
    });
    return results;
}

size_t HybridIndex::remove(span<const int64_t> ids) {
    vector<uint32_t> slots;
    const size_t removed = id_map_.remove(ids, &slots);
    const vector<int64_t> sub_ids(slots.begin(), slots.end());
    dense_.remove(sub_ids);
    sparse_.remove(sub_ids);
    return removed;
}

size_t HybridIndex::compact() {
    const vector<uint32_t> remap = id_map_.compact(true);
    if (remap.empty()) {
        metadata_.drop_unused_strings();
        return 0;
    }
    // both drop the same deleted slots in order, so their new slots match the remap
    dense_.compact_as_base();
    sparse_.compact_as_base();
    metadata_.compact(remap);
    return remap.size() - id_map_.slots();
}

void HybridIndex::save(std::ostream& out) const {
    using namespace detail;
    write_header(out, IndexKind::Hybrid);
    write_pod(out, static_cast<uint8_t>(method_));
    write_pod(out, weights_[0]);
    write_pod(out, weights_[1]);
    write_pod(out, rrf_k_);
    write_pod(out, static_cast<uint64_t>(depth_factor_));
    write_pod(out, static_cast<uint64_t>(id_map_.slots()));
    dense_.save(out);
    sparse_.save(out);
    id_map_.save(out);
    metadata_.save(out);
}

std::unique_ptr<HybridIndex> HybridIndex::load_body(std::istream& in, uint32_t version) {
    using namespace detail;
    const auto method = read_pod<uint8_t>(in);
    const auto w_dense = read_pod<float>(in), w_sparse = read_pod<float>(in), rrf_k = read_pod<float>(in);
    const auto depth_factor = read_pod<uint64_t>(in);
    const auto count = read_pod<uint64_t>(in);
    if (method > static_cast<uint8_t>(FusionMethod::Linear) || depth_factor == 0 || !std::isfinite(rrf_k) || rrf_k <= 0.0f ||
        !std::isfinite(w_dense) || !std::isfinite(w_sparse) || w_dense < 0.0f || w_sparse < 0.0f ||
        (w_dense == 0.0f && w_sparse == 0.0f))
        throw std::runtime_error("corrupt file: invalid HybridIndex settings");

    Header header = read_header(in);
    if (header.version != version) throw std::runtime_error("corrupt file: HybridIndex parts have a different format version");
    std::unique_ptr<DenseIndex> dense;
    switch (header.kind) {
        case IndexKind::Flat: dense = FlatIndex::load_body(in, version); break;
        case IndexKind::IVF: dense = IVFIndex::load_body(in, version); break;
        case IndexKind::HNSW: dense = HNSWIndex::load_body(in, version); break;
        case IndexKind::Refine: dense = RefineIndex::load_body(in, version); break;
        default: throw std::runtime_error("corrupt file: a HybridIndex's dense index must be Flat, IVF, HNSW or Refine");
    }
    header = read_header(in);
    if (header.version != version) throw std::runtime_error("corrupt file: HybridIndex parts have a different format version");
    std::unique_ptr<SparseIndex> sparse;
    switch (header.kind) {
        case IndexKind::Inverted: sparse = InvertedIndex::load_body(in, version); break;
        case IndexKind::SparseApprox: sparse = SparseApproxIndex::load_body(in, version); break;
        default: throw std::runtime_error("corrupt file: a HybridIndex's sparse index must be Inverted or SparseApprox");
    }
    for (const IndexBase* part : {static_cast<const IndexBase*>(dense.get()), static_cast<const IndexBase*>(sparse.get())})
        if (!part->ids_are_slots() || part->slots() != count)
            throw std::runtime_error("corrupt file: a HybridIndex's parts must hold its slots, with ids equal to slots");

    auto index = std::unique_ptr<HybridIndex>(new HybridIndex(std::move(dense), std::move(sparse)));
    index->method_ = static_cast<FusionMethod>(method);
    index->weights_ = {w_dense, w_sparse};
    index->rrf_k_ = rrf_k;
    index->depth_factor_ = depth_factor;
    index->id_map_ = IdMap::load(in, count, version);
    index->metadata_ = MetadataStore::load(in, count);
    return index;
}

} // namespace vecengine
