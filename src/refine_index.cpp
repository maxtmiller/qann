#include "vecengine/refine_index.hpp"
#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"
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

RefineIndex::RefineIndex(Index& base, size_t k_factor) : base_(base), k_factor_(k_factor) {
    if (base.size() != 0) throw std::invalid_argument("base index must be empty");
    if (k_factor == 0) throw std::invalid_argument("k_factor must be >= 1");
}

void RefineIndex::set_k_factor(size_t k_factor) {
    if (k_factor == 0) throw std::invalid_argument("k_factor must be >= 1");
    k_factor_ = k_factor;
}

void RefineIndex::add(span<const float> vec) {
    assert(vec.size() == dim());

    base_.add(vec);
    data_.insert(data_.end(), vec.begin(), vec.end());
    ++count_;

}

void RefineIndex::add_batch(span<const float> vecs, size_t n) {
    base_.add_batch(vecs, n);
    data_.insert(data_.end(), vecs.begin(), vecs.end());
    count_ += n;
}

vector<Neighbor> RefineIndex::query(span<const float> vec, size_t k) const {
    assert(vec.size() == dim());

    vector<Neighbor> candidatesRaw = base_.query(vec, std::min(k * k_factor_, count_));
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
        results[i] = {index, distance};
        maxHeap.pop();
    }

    VECENGINE_PROF_STOP(t_rerank, kProfRerank);
    return results;
}

void RefineIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::Refine);
    write_pod(out, static_cast<uint64_t>(k_factor_));
    write_pod(out, static_cast<uint64_t>(count_));
    base_.save(out);
    write_vec(out, data_);
}

std::unique_ptr<RefineIndex> RefineIndex::load_body(std::istream& in) {
    using namespace detail;

    const auto k_factor = read_pod<uint64_t>(in);
    const auto count = read_pod<uint64_t>(in);
    if (k_factor == 0) throw std::runtime_error("corrupt file: invalid RefineIndex shape");

    std::unique_ptr<Index> base;
    switch (read_header(in)) {
        case IndexKind::Flat: base = FlatIndex::load_body(in); break;
        case IndexKind::IVF:  base = IVFIndex::load_body(in); break;
        default: throw std::runtime_error("corrupt file: RefineIndex base must be Flat or IVF");
    }

    const uint64_t dim = base->dim();
    if (count != base->size() || count > UINT64_MAX / dim) throw std::runtime_error("corrupt file: invalid RefineIndex shape");

    auto data = read_vec<float>(in, count * dim);
    if (data.size() != count * dim) throw std::runtime_error("corrupt file: unexpected number of vectors");

    auto index = std::unique_ptr<RefineIndex>(new RefineIndex(std::move(base), k_factor));

    index->count_ = count;
    index->data_ = std::move(data);

    return index;
}

} // namespace vecengine
