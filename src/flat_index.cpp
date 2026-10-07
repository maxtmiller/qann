#include "vecengine/flat_index.hpp"
#include "vecengine/distance.hpp"
#include "serialize.hpp"
#include <cstddef>
#include <vector>
#include <span>
#include <stdexcept>
#include <cassert>
#include <queue>


namespace vecengine {

using std::size_t;
using std::vector;
using std::span;
using std::pair;

FlatIndex::FlatIndex(size_t dim, size_t capacity): dim_(dim), count_(0)  {
    if (dim == 0) throw std::invalid_argument("dim must be >= 1");

    data_.reserve(dim_ * capacity);
}

void FlatIndex::add(span<const float> vec) {
    assert(vec.size() == dim_);
    data_.insert(data_.end(), vec.begin(), vec.end());
    ++count_;
}

vector<Neighbor> FlatIndex::query(span<const float> vec, size_t k) const {
    std::priority_queue<pair<float, size_t>, vector<pair<float, size_t>>, std::less<pair<float, size_t>>> maxHeap;
    for (size_t i = 0; i < count_; ++i) {
        const float* start = data_.data() + i * dim_;
        float dist = l2_distance(start, vec.data(), dim_);

        if (maxHeap.size() < k) {
            maxHeap.emplace(dist, i);
        } else if (dist < maxHeap.top().first) {
            maxHeap.pop();
            maxHeap.emplace(dist, i);
        }
    }

    const size_t n = maxHeap.size();
    vector<Neighbor> results(n);
    for (int i = static_cast<int>(n) - 1; i >= 0; --i) {
        auto [dist, idx] = maxHeap.top();
        results[i] = {idx,dist};
        maxHeap.pop();
    }

    return results;
}

void FlatIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::Flat);
    write_pod(out, static_cast<uint64_t>(dim_));
    write_pod(out, static_cast<uint64_t>(count_));
    write_vec(out, data_);
}

std::unique_ptr<FlatIndex> FlatIndex::load_body(std::istream& in) {
    using namespace detail;

    const auto dim = read_pod<uint64_t>(in);
    const auto count = read_pod<uint64_t>(in);

    if (dim == 0 || count > UINT64_MAX / dim) throw std::runtime_error("corrupt file: invalid FlatIndex shape");

    auto index = std::make_unique<FlatIndex>(dim, 0);

    index->data_ = read_vec<float>(in, count * dim);
    if (index->data_.size() != count * dim) throw std::runtime_error("corrupt file: FlatIndex data length mismatch");
    index->count_ = count;

    return index;
}

}
