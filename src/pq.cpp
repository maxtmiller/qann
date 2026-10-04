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
    centroids_t_.resize(num_subspaces_ * centroids_per_subspace_ * sub_dim_);
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

    for (size_t i = 0; i < num_subspaces_; ++i) {
        for (size_t j = 0; j < sub_dim_; ++j) {
            for (size_t k = 0; k < centroids_per_subspace_; ++k) {
                centroids_t_[(i * sub_dim_ + j) * centroids_per_subspace_ + k] = centroids_[(i * centroids_per_subspace_ + k) * sub_dim_ + j];
            }
        }
    }

    // build SDC table
    sdc_table_.assign(num_subspaces_ * centroids_per_subspace_ * centroids_per_subspace_, 0.0f);
    for (size_t i = 0; i < num_subspaces_; ++i) {
        float* sdc_subspace = sdc_table_.data() + (i * centroids_per_subspace_ * centroids_per_subspace_);

        for (size_t j = 0; j < sub_dim_; ++j) {
            const float* cent_t_pos = centroids_t_.data() + (i  * sub_dim_ + j) * centroids_per_subspace_;

            for (size_t k = 0; k < centroids_per_subspace_; ++k) {
                float val1 = cent_t_pos[k];
                float* sdc_row = sdc_subspace + (k * centroids_per_subspace_);

                for (size_t l = 0; l < centroids_per_subspace_; ++l) {
                    float diff = val1 - cent_t_pos[l];
                    sdc_row[l] += diff * diff;
                }
            }
        }
    }
}

vector<uint8_t> PQCodebook::encode(span<const float> vec) const {
    assert(vec.size() == dim_);

    // converts a full dimensional vector into array of cluster IDs
    vector<float> table(num_subspaces_ * centroids_per_subspace_);
    compute_adc_table(vec, table);

    // finds closest centroid to vec in each subspace using adc table
    vector<uint8_t> res(num_subspaces_, 0);
    for (size_t i = 0; i < num_subspaces_; ++i) {
        const float* cur_dist = table.data() + (i * centroids_per_subspace_);
        float min_dist = cur_dist[0];
        for (size_t j = 0; j < centroids_per_subspace_; ++j) min_dist = std::min(min_dist, cur_dist[j]);

        size_t closest_index = 0;
        while (cur_dist[closest_index] != min_dist) ++closest_index;
        res[i] = static_cast<uint8_t>(closest_index);
    }

    return res;
}

void PQCodebook::compute_adc_table(span<const float> query, span<float> out) const {
    assert(query.size() == dim_);
    assert(out.size() == num_subspaces_ * centroids_per_subspace_);

    // builds adc table
    for (size_t i = 0; i < num_subspaces_; ++i) {
        float* out_pos = out.data() + (i * centroids_per_subspace_);
        std::fill_n(out_pos, centroids_per_subspace_, 0.0f);
        
        for (size_t j = 0; j < sub_dim_; ++j) {
            float q_d = query[i * sub_dim_ + j];
            const float* cent_d = centroids_t_.data() + (i * sub_dim_ + j) * centroids_per_subspace_;

            for (size_t k = 0; k < centroids_per_subspace_; ++k) {
                float diff = q_d - cent_d[k];
                out_pos[k] += diff * diff;
            }
        }
    }
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
