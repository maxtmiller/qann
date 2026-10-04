#pragma once

#include "vecengine/index.hpp"

#include <cstddef>
#include <vector>
#include <span>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

class FlatIndex : public Index {
public:
    explicit FlatIndex(size_t dim, size_t capacity = 1024);

    void add(span<const float> vec) override;
    vector<Neighbor> query(span<const float> vec, size_t k) const override;
    vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k) const override;

    size_t size() const noexcept override { return count_; }
    size_t dim() const noexcept override { return dim_; }

private:
    size_t dim_;
    size_t count_{0};
    vector<float> data_; // row-major: count_ * dim_
};

} // namespace vecengine
