#include "vecengine/refine_index.hpp"
#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "parallel.hpp"
#include "profile.hpp"
#include "serialize.hpp"

#include <cstddef>
#include <vector>
#include <span>
#include <cassert>
#include <stdexcept>
#include <utility>
#include <algorithm>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;
using std::pair;

RefineIndex::RefineIndex(Index& base, size_t k_factor, bool custom_ids) : Index(base.metric()), base_(base), k_factor_(k_factor), id_map_(custom_ids) {
    if (base.slots() != 0) throw std::invalid_argument("base index must be empty");
    if (k_factor == 0) throw std::invalid_argument("k_factor must be >= 1");
    if (!base.ids_are_slots()) throw std::invalid_argument("base index must not use custom ids or have been compacted");
    if (base.wrapped()) throw std::invalid_argument("base index is already wrapped by another RefineIndex");
    base.wrapped_ = true;
}

RefineIndex::~RefineIndex() {
    base_.wrapped_ = false;
}

void RefineIndex::set_k_factor(size_t k_factor) {
    if (k_factor == 0) throw std::invalid_argument("k_factor must be >= 1");
    k_factor_ = k_factor;
}

void RefineIndex::add_batch_impl(span<const float> vecs, size_t n, const int64_t* ids) {
    id_map_.check(n, ids);
    base_.add_batch_impl(vecs, n, nullptr);
    id_map_.add(n, ids);
    data_.insert(data_.end(), vecs.begin(), vecs.end());
}

size_t RefineIndex::candidates(size_t k, const detail::SlotFilter* filter) const {
    return std::min(k * k_factor_, filter ? filter->count : id_map_.live());
}

vector<Neighbor> RefineIndex::query_impl(span<const float> vec, size_t k, const detail::SlotFilter* filter) const {
    assert(vec.size() == dim());
    if (id_map_.live() == 0) return {};

    return rerank(vec, base_.query_impl(vec, candidates(k, filter), filter), k);
}

vector<vector<Neighbor>> RefineIndex::query_batch_impl(span<const float> queries, size_t num_queries, size_t k,
                                                       const detail::SlotFilter* filter) const {
    assert(queries.size() == num_queries * dim());
    if (id_map_.live() == 0) return vector<vector<Neighbor>>(num_queries);

    auto fetched = base_.query_batch_impl(queries, num_queries, candidates(k, filter), filter);
    vector<vector<Neighbor>> results(num_queries);
    detail::parallel_for(num_queries, [&](size_t i) {
        results[i] = rerank({queries.data() + i * dim(), dim()}, fetched[i], k);
    });
    return results;
}

vector<Neighbor> RefineIndex::rerank(span<const float> vec, const vector<Neighbor>& candidatesRaw, size_t k) const {
    VECENGINE_PROF_START(t_rerank);
    vector<pair<float, size_t>> candidates;
    candidates.reserve(candidatesRaw.size());
    const size_t d = dim();

    for (size_t i = 0; i < candidatesRaw.size(); ++i) {
        constexpr size_t kAhead = 8;   // how many candidates ahead to prefetch
        if (i + kAhead < candidatesRaw.size()) {
            const char* p = reinterpret_cast<const char*>(data_.data() + candidatesRaw[i + kAhead].index * dim());
            for (size_t b = 0; b < dim() * sizeof(float); b += 64) __builtin_prefetch(p + b);
        }
        float dst = exact_score(vec.data(), data_.data() + (candidatesRaw[i].index * d), d);
        candidates.emplace_back(dst, candidatesRaw[i].index);
    }

    const size_t keep = std::min(k, candidates.size());
    std::nth_element(candidates.begin(), candidates.begin() + keep, candidates.end());
    std::sort(candidates.begin(), candidates.begin() + keep);

    vector<Neighbor> results(keep);
    for (size_t i = 0; i < keep; ++i) {
        auto [distance, index] = candidates[i];
        results[i] = {static_cast<size_t>(id_map_.label(static_cast<uint32_t>(index))), distance};
    }

    VECENGINE_PROF_STOP(t_rerank, kProfRerank);
    return results;
}

size_t RefineIndex::remove(span<const int64_t> ids) {
    vector<uint32_t> slots;
    const size_t removed = id_map_.remove(ids, &slots);
    const vector<int64_t> base_ids(slots.begin(), slots.end());
    base_.remove(base_ids);
    return removed;
}

size_t RefineIndex::compact() {
    auto remap = id_map_.compact(true);
    if (remap.empty()) return 0;

    base_.compact_as_base();
    metadata_.compact(remap);
    size_t write_slot = 0;
    for (size_t i = 0; i < remap.size(); ++i) {
        uint32_t new_slot = remap[i];
        if (remap[i] == detail::IdMap::kRemoved) continue;

        if (new_slot != i) {
            float* src = data_.data() + i * dim();
            float* dst = data_.data() + new_slot * dim();
            std::copy_n(src, dim(), dst);
        }
        ++write_slot;
    }
    data_.resize(write_slot * dim());
    data_.shrink_to_fit();

    return remap.size() - id_map_.slots();
}

size_t RefineIndex::compact_as_base() {
    assert(base_.slots() == id_map_.slots());
    throw std::logic_error("a RefineIndex cannot be the base of another RefineIndex");
}

void RefineIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::Refine);
    write_pod(out, static_cast<uint64_t>(k_factor_));
    write_pod(out, static_cast<uint64_t>(id_map_.slots()));
    base_.save(out);
    write_vec(out, data_);
    id_map_.save(out);
    metadata_.save(out);
}

std::unique_ptr<RefineIndex> RefineIndex::load_body(std::istream& in, uint32_t version) {
    using namespace detail;

    const auto k_factor = read_pod<uint64_t>(in);
    const auto count = read_pod<uint64_t>(in);
    if (k_factor == 0) throw std::runtime_error("corrupt file: invalid RefineIndex shape");

    std::unique_ptr<Index> base;

    const Header header = read_header(in);
    if (header.version != version)
        throw std::runtime_error("corrupt file: RefineIndex base has a different format version");
    switch (header.kind) {
        case IndexKind::Flat: base = FlatIndex::load_body(in, header.version); break;
        case IndexKind::IVF:  base = IVFIndex::load_body(in, header.version); break;
        case IndexKind::HNSW: base = HNSWIndex::load_body(in, header.version); break;
        default: throw std::runtime_error("corrupt file: RefineIndex base must be Flat, IVF or HNSW");
    }
    if (!base->ids_are_slots())
        throw std::runtime_error("corrupt file: RefineIndex base ids must equal its slots");

    const uint64_t dim = base->dim();
    if (count != base->slots() || count > UINT64_MAX / dim) throw std::runtime_error("corrupt file: invalid RefineIndex shape");

    auto data = read_vec<float>(in, count * dim);
    if (data.size() != count * dim) throw std::runtime_error("corrupt file: unexpected number of vectors");

    auto index = std::unique_ptr<RefineIndex>(new RefineIndex(std::move(base), k_factor));

    index->data_ = std::move(data);

    if (version >= 2) index->id_map_ = detail::IdMap::load(in, count, version);
    else index->id_map_.add(count, nullptr);
    if (version >= 9) index->metadata_ = detail::MetadataStore::load(in, count);

    return index;
}

} // namespace vecengine
