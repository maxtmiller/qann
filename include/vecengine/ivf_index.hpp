#pragma once

#include "vecengine/index.hpp"
#include "vecengine/pq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <memory>
#include <unordered_map>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

struct InvertedList {
    vector<uint32_t> ids;
    vector<float> vecs; // populated when PQ is disabled
    vector<uint8_t> codes; // populated when PQ is enabled: num_subspaces bytes per vector, encodes (vec - its cluster's coarse centroid)
};

class IVFIndex : public Index {
public:
    explicit IVFIndex(size_t dim, size_t nlist, size_t nprobe = 10);

    // Enables PQ compression of residuals (vec - assigned coarse centroid).
    // Must be called before train()/add(). Once enabled, add() stores codes
    // in InvertedList::codes instead of raw floats in InvertedList::vecs.
    void enable_pq(size_t num_subspaces, size_t centroids_per_subspace = 256);

    size_t nprobe() const noexcept { return nprobe_; }
    void set_nprobe(size_t nprobe);

    PQDistance pq_distance() const noexcept { return pq_distance_; }
    void set_pq_distance(PQDistance mode) noexcept { pq_distance_ = mode; }

    void add(span<const float> vec) override;
    vector<Neighbor> query(span<const float> vec, size_t k) const override;
    vector<vector<Neighbor>> query_batch(span<const float> queries, size_t num_queries, size_t k) const override;

    void train(span<const float> vectors, size_t num_vectors, size_t max_iters = 25);

    size_t size() const noexcept override { return n_total_; }
    size_t dim() const noexcept override { return dim_; }

private:
    bool pq_enabled() const noexcept { return pq_ != nullptr; }

    size_t dim_; // dimension of each vector
    size_t nlist_; // num of coarse centroids
    size_t nprobe_; // num of coarse centroids to pick from during quering
    uint32_t n_total_ = 0; // num of vectors
    bool trained_ = false; // was function train run

    vector<float> coarse_centroids_; // coarse centroid vectors
    std::unordered_map<uint16_t,InvertedList> data_; // map of coarse centroid ID to inverted list
    std::unique_ptr<PQCodebook> pq_; // pointer to PQ LUT if using PQ
    PQDistance pq_distance_ = PQDistance::ADC; // type of quantization used

};

} // namespace vecengine
