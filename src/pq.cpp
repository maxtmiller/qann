#include "kmeans.hpp"
#include "vecengine/pq.hpp"
#include "vecengine/distance.hpp"

#include <cassert>
#include <vector>
#include <span>
#include <stdexcept>
#include <algorithm>


namespace vecengine {

using std::span;
using std::vector;
using std::size_t;

PQCodebook::PQCodebook(size_t dim, size_t num_subspaces, size_t centroids_per_subspace)
    : dim_(dim), num_subspaces_(num_subspaces), centroids_per_subspace_(centroids_per_subspace) {
    if (dim % num_subspaces != 0)
        throw std::invalid_argument("dim must be divisible by num_subspaces");
    if (centroids_per_subspace == 0 || centroids_per_subspace > 256)
        throw std::invalid_argument("centroids_per_subspace must be in [1, 256]");

    sub_dim_ = dim_ / num_subspaces_;
    centroids_.resize(num_subspaces_ * centroids_per_subspace_ * sub_dim_);
}

void PQCodebook::train(span<const float> vectors, size_t num_vectors, size_t max_iters) {
    assert(vectors.size() == num_vectors * dim_);
    if (num_vectors < centroids_per_subspace_) throw std::invalid_argument("training set must have at least centroids_per_subspace_ vectors");

    // run kmeans on each subspace
    for (size_t i = 0; i < num_subspaces_; ++i) {
        auto res = vecengine::detail::kmeans(vectors.data() + i*sub_dim_, num_vectors, dim_, sub_dim_, centroids_per_subspace_, max_iters);
        float* dst = centroids_.data() + i * centroids_per_subspace_ * sub_dim_;
        std::copy_n(res.centroids.data(), res.centroids.size(), dst);
    }

    // build SDC table
    sdc_table_.assign(num_subspaces_ * centroids_per_subspace_ * centroids_per_subspace_, 0.0f);
    for (size_t i = 0; i < num_subspaces_; ++i) {
        const float* vec = centroids_.data() + i * centroids_per_subspace_ * sub_dim_;
        float* sdc_subspace = sdc_table_.data() + (i * centroids_per_subspace_ * centroids_per_subspace_);

        for (size_t j = 0; j < centroids_per_subspace_; ++j) {
            for (size_t k = j + 1; k < centroids_per_subspace_; ++k) {
                float dst = l2_distance(vec + (j * sub_dim_), vec + (k * sub_dim_), sub_dim_);
                sdc_subspace[(j * centroids_per_subspace_) + k] = dst;
                sdc_subspace[(k * centroids_per_subspace_) + j] = dst;
            }
        }
    }
}

vector<uint8_t> PQCodebook::encode(span<const float> vec) const {
    assert(vec.size() == dim_);

    // converts a full dimensional vector into array of cluster IDs
    vector<uint8_t> res(num_subspaces_, 0);
    for (size_t i = 0; i < num_subspaces_; ++i) {
        const float* vec_pos = vec.data() + (i * sub_dim_);
        const float* centroid_pos = centroids_.data() + (i * centroids_per_subspace_ * sub_dim_);
        size_t closest_index = vecengine::detail::nearest_centroid(vec_pos, centroid_pos, centroids_per_subspace_, sub_dim_);
        res[i] = static_cast<uint8_t>(closest_index);
    }

    return res;
}

vector<float> PQCodebook::compute_adc_table(span<const float> query) const {
    assert(query.size() == dim_);

    // builds adc table
    vector<float> res(num_subspaces_ * centroids_per_subspace_);
    for (size_t i = 0; i < num_subspaces_; ++i) {
        const float* query_pos = query.data() + (i * sub_dim_);
        for (size_t j = 0; j < centroids_per_subspace_; ++j) {
            const float* centroid_pos = centroids_.data() + (i * centroids_per_subspace_ + j) * sub_dim_;
            float dst = l2_distance(query_pos, centroid_pos, sub_dim_);
            res[i * centroids_per_subspace_ + j] = dst;
        }
    }

    return res;
}

float PQCodebook::distance_adc(span<const float> table, span<const uint8_t> code) const {
    assert(table.size() == num_subspaces_ * centroids_per_subspace_);
    assert(code.size() == num_subspaces_);

    // Same shape as distance_sdc, indexing table[s * K + code[s]].
    
    float sum = 0.0f;
    for (size_t i = 0; i < num_subspaces_; ++i) {
        sum += table[(i * centroids_per_subspace_) + code[i]];
    }

    return sum;
}

float PQCodebook::distance_sdc(span<const uint8_t> query_code, span<const uint8_t> code) const {
    assert(query_code.size() == num_subspaces_);
    assert(code.size() == num_subspaces_);

    // calculates sum of distances between two vectors encoded by closest centroid
    float sum = 0.0f;
    for (size_t i = 0; i < num_subspaces_; ++i) {
        sum += sdc_table_[(i * centroids_per_subspace_ * centroids_per_subspace_) + (query_code[i] * centroids_per_subspace_) + code[i]];
    }

    return sum;
}

} // namespace vecengine
