#include "vecengine/refine_index.hpp"
#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "parallel.hpp"
#include "profile.hpp"
#include "serialize.hpp"

#include <cstddef>
#include <vector>
#include <span>
#include <cassert>
#include <stdexcept>
#include <queue>
#include <utility>
#include <algorithm>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;
using std::pair;

RefineIndex::RefineIndex(Index& base, size_t k_factor, bool custom_ids) : base_(base), k_factor_(k_factor), id_map_(custom_ids) {
    if (base.slots() != 0) throw std::invalid_argument("base index must be empty");
    if (k_factor == 0) throw std::invalid_argument("k_factor must be >= 1");
    if (base.custom_ids()) throw std::invalid_argument("base index must not use custom ids");
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

void RefineIndex::add(span<const float> vec) {
    assert(vec.size() == dim());

    id_map_.check(1, nullptr);
    base_.add(vec);
    id_map_.add(1, nullptr);
    data_.insert(data_.end(), vec.begin(), vec.end());
}

void RefineIndex::add_batch(span<const float> vecs, size_t n, const int64_t* ids) {
    id_map_.check(n, ids);
    base_.add_batch(vecs, n);
    id_map_.add(n, ids);
    data_.insert(data_.end(), vecs.begin(), vecs.end());
}

vector<Neighbor> RefineIndex::query(span<const float> vec, size_t k) const {
    assert(vec.size() == dim());
    if (id_map_.live() == 0) return {};

    return rerank(vec, base_.query(vec, std::min(k * k_factor_, id_map_.live())), k);
}

vector<vector<Neighbor>> RefineIndex::query_batch(span<const float> queries, size_t num_queries, size_t k) const {
    assert(queries.size() == num_queries * dim());
    if (id_map_.live() == 0) return vector<vector<Neighbor>>(num_queries);

    auto candidates = base_.query_batch(queries, num_queries, std::min(k * k_factor_, id_map_.live()));
    vector<vector<Neighbor>> results(num_queries);
    detail::parallel_for(num_queries, [&](size_t i) {
        results[i] = rerank({queries.data() + i * dim(), dim()}, candidates[i], k);
    });
    return results;
}

vector<Neighbor> RefineIndex::rerank(span<const float> vec, const vector<Neighbor>& candidatesRaw, size_t k) const {
    VECENGINE_PROF_START(t_rerank);
    vector<pair<float, size_t>> candidates;
    candidates.reserve(candidatesRaw.size());

    for (size_t i = 0; i < candidatesRaw.size(); ++i) {
        float dst = l2_distance(vec.data(), data_.data() + (candidatesRaw[i].index * dim()), dim());
        candidates.emplace_back(dst, candidatesRaw[i].index);
    }

    std::priority_queue<pair<float, size_t>, vector<pair<float, size_t>>, std::less<pair<float, size_t>>> maxHeap(candidates.begin(), candidates.end());
    while (maxHeap.size() > k) maxHeap.pop();

    const size_t n = maxHeap.size();
    vector<Neighbor> results(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        auto [distance, index] = maxHeap.top();
        results[i] = {static_cast<size_t>(id_map_.label(static_cast<uint32_t>(index))), distance};
        maxHeap.pop();
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

void RefineIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::Refine);
    write_pod(out, static_cast<uint64_t>(k_factor_));
    write_pod(out, static_cast<uint64_t>(id_map_.slots()));
    base_.save(out);
    write_vec(out, data_);
    id_map_.save(out);
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
        default: throw std::runtime_error("corrupt file: RefineIndex base must be Flat or IVF");
    }
    if (base->custom_ids())
        throw std::runtime_error("corrupt file: RefineIndex base must not use custom ids");

    const uint64_t dim = base->dim();
    if (count != base->slots() || count > UINT64_MAX / dim) throw std::runtime_error("corrupt file: invalid RefineIndex shape");

    auto data = read_vec<float>(in, count * dim);
    if (data.size() != count * dim) throw std::runtime_error("corrupt file: unexpected number of vectors");

    auto index = std::unique_ptr<RefineIndex>(new RefineIndex(std::move(base), k_factor));

    index->data_ = std::move(data);

    if (version >= 2) index->id_map_ = detail::IdMap::load(in, count);
    else index->id_map_.add(count, nullptr);

    return index;
}

} // namespace vecengine
