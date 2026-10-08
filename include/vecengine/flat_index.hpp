#pragma once

#include "vecengine/index.hpp"
#include "vecengine/id_map.hpp"

#include <cstddef>
#include <vector>
#include <span>
#include <memory>
#include <iosfwd>
#include <cstdint>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

// Exact brute-force index: query() computes the distance to every live
// vector. 100% recall, cost linear in the number of vectors.
class FlatIndex : public Index {
public:
    // dim floats per vector; capacity is an initial reserve (vectors, not a
    // limit). custom_ids makes add_batch() take user ids.
    explicit FlatIndex(size_t dim, size_t capacity = 1024, bool custom_ids = false);

    // Index overrides; see Index for the contract of each.
    void add(span<const float> vec) override;
    void add_batch(span<const float> vecs, size_t n, const int64_t* ids = nullptr) override;
    vector<Neighbor> query(span<const float> vec, size_t k) const override;
    size_t remove(span<const int64_t> ids) override;

    void save(std::ostream& out) const override;

    // Rebuilds a FlatIndex from the body save() wrote, after load_index() has
    // read the header. Throws std::runtime_error on corrupt data.
    static std::unique_ptr<FlatIndex> load_body(std::istream& in, uint32_t version);

    size_t slots() const noexcept override { return id_map_.slots(); }
    bool custom_ids() const noexcept override { return id_map_.custom_ids(); }

    size_t size() const noexcept override { return id_map_.live(); }
    size_t dim() const noexcept override { return dim_; }

private:
    size_t dim_;
    vector<float> data_; // row-major, one row per slot: slots() * dim_
    detail::IdMap id_map_;
};

} // namespace vecengine
