#include "kmeans.hpp"
#include "parallel.hpp"
#include "serialize.hpp"
#include "vecengine/pq.hpp"
#include "vecengine/distance.hpp"

#include <cassert>
#include <vector>
#include <span>
#include <stdexcept>
#include <algorithm>
#include <bit>
#include <cstring>
#include <numeric>
#include <random>

namespace vecengine {

using std::span;
using std::vector;
using std::size_t;

PQCodebook::PQCodebook(size_t dim, size_t num_subspaces, size_t centroids_per_subspace)
    : dim_(dim), num_subspaces_(num_subspaces), centroids_per_subspace_(centroids_per_subspace) {

    if (dim == 0 || num_subspaces == 0) throw std::invalid_argument("dim and num_subspaces must be >= 1");

    if (dim % num_subspaces != 0)
        throw std::invalid_argument("dim must be divisible by num_subspaces");
    if (centroids_per_subspace == 0 || centroids_per_subspace > 256)
        throw std::invalid_argument("centroids_per_subspace must be in [1, 256]");

    sub_dim_ = dim_ / num_subspaces_;
    centroids_.resize(num_subspaces_ * centroids_per_subspace_ * sub_dim_);
    centroids_t_.resize(num_subspaces_ * centroids_per_subspace_ * sub_dim_);
}

void PQCodebook::train(span<const float> vectors, size_t num_vectors, size_t max_iters, std::optional<uint32_t> seed) {
    assert(vectors.size() == num_vectors * dim_);
    if (num_vectors < centroids_per_subspace_) throw std::invalid_argument("training set must have at least centroids_per_subspace_ vectors");

    // Train on at most 256 points per centroid, as FAISS does: on landmark-dino-768
    // (PQ384x4) the full 131k sample made training ~5x slower than FAISS (#170)
    // without a better codebook.
    const size_t cap = 256 * centroids_per_subspace_;
    vector<float> subset;
    if (num_vectors > cap) {
        std::mt19937 rng(seed ? *seed : std::random_device{}());
        vector<size_t> rows(num_vectors);
        std::iota(rows.begin(), rows.end(), 0);
        subset.resize(cap * dim_);
        for (size_t i = 0; i < cap; ++i) {
            std::uniform_int_distribution<size_t> pick(i, num_vectors - 1);
            std::swap(rows[i], rows[pick(rng)]);
            std::copy_n(vectors.data() + rows[i] * dim_, dim_, subset.data() + i * dim_);
        }
        vectors = subset;
        num_vectors = cap;
    }

    // run kmeans on each subspace
    detail::parallel_for(num_subspaces_, [&](size_t i) {
        auto sub_seed = seed ? std::optional<uint32_t>(*seed + static_cast<uint32_t>(i)) : std::nullopt;
        auto res = vecengine::detail::kmeans(vectors.data() + i*sub_dim_, num_vectors, dim_, sub_dim_, centroids_per_subspace_, max_iters, sub_seed);
        float* dst = centroids_.data() + i * centroids_per_subspace_ * sub_dim_;
        std::copy_n(res.centroids.data(), res.centroids.size(), dst);
    });

    build_tables();
}

vector<uint8_t> PQCodebook::encode(span<const float> vec) const {
    vector<uint8_t> res(num_subspaces_);
    encode(vec, res.data());
    return res;
}

namespace {

// PQCodebook::encode with the centroid count K fixed at compile time (0 = runtime k),
// so the distance, min and position loops vectorize. With the branchless position
// search for K <= 32, PQ384x4 encoding went from 10.0 to 2.4 us per vector on an M3
// (#170).
template <size_t K>
void encode_subspaces(const float* centroids_t, const float* vec, size_t m, size_t sub_dim, size_t k, uint8_t* out) {
    const size_t kk = K ? K : k;
    float dist[256];
    int32_t bits[256];
    for (size_t i = 0; i < m; ++i) {
        const float* cent = centroids_t + i * sub_dim * kk;
        const float q0 = vec[i * sub_dim];
        for (size_t c = 0; c < kk; ++c) {
            const float diff = q0 - cent[c];
            dist[c] = diff * diff;
        }
        for (size_t j = 1; j < sub_dim; ++j) {
            const float q = vec[i * sub_dim + j];
            const float* cent_d = cent + j * kk;
            for (size_t c = 0; c < kk; ++c) {
                const float diff = q - cent_d[c];
                dist[c] += diff * diff;
            }
        }

        // First index of the minimum. Distances are sums of squares (>= +0), whose
        // bit patterns order like the floats as int32, and an integer min vectorizes
        // where a float min can't (NaN rules); signed, because baseline x86-64 (SSE2)
        // has no unsigned 32-bit min.
        std::memcpy(bits, dist, kk * sizeof(float));
        int32_t min_bits = bits[0];
        for (size_t c = 1; c < kk; ++c) min_bits = std::min(min_bits, bits[c]);
        size_t best = 0;
        if constexpr (K > 0 && K <= 32) {
            uint32_t hit = 0;
            for (size_t c = 0; c < K; ++c) hit |= uint32_t(bits[c] == min_bits) << c;
            best = static_cast<size_t>(std::countr_zero(hit));
        } else {
            while (bits[best] != min_bits) ++best;
        }
        out[i] = static_cast<uint8_t>(best);
    }
}

} // namespace

void PQCodebook::encode(span<const float> vec, uint8_t* out) const {
    assert(vec.size() == dim_);
    const float* c = centroids_t_.data();
    if (centroids_per_subspace_ == 16) encode_subspaces<16>(c, vec.data(), num_subspaces_, sub_dim_, 16, out);
    else if (centroids_per_subspace_ == 256) encode_subspaces<256>(c, vec.data(), num_subspaces_, sub_dim_, 256, out);
    else encode_subspaces<0>(c, vec.data(), num_subspaces_, sub_dim_, centroids_per_subspace_, out);
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

void PQCodebook::compute_inner_table(span<const float> query, span<float> out) const {
    assert(query.size() == dim_);
    assert(out.size() == num_subspaces_ * centroids_per_subspace_);

    for (size_t i = 0; i < num_subspaces_; ++i) {
        float* out_pos = out.data() + (i * centroids_per_subspace_);
        std::fill_n(out_pos, centroids_per_subspace_, 0.0f);
        
        for (size_t j = 0; j < sub_dim_; ++j) {
            float q_d = query[i * sub_dim_ + j];
            const float* cent_d = centroids_t_.data() + (i * sub_dim_ + j) * centroids_per_subspace_;

            for (size_t k = 0; k < centroids_per_subspace_; ++k) {
                float diff = -2 * q_d * cent_d[k];
                out_pos[k] += diff;
            }
        }
    }
}

void PQCodebook::compute_list_term(span<const float> coarse_centroid, span<float> out) const {
    assert(coarse_centroid.size() == dim_);
    assert(out.size() == num_subspaces_ * centroids_per_subspace_);

    for (size_t i = 0; i < num_subspaces_; ++i) {
        float* out_pos = out.data() + (i * centroids_per_subspace_);
        std::fill_n(out_pos, centroids_per_subspace_, 0.0f);
        
        for (size_t j = 0; j < sub_dim_; ++j) {
            float c_d = coarse_centroid[i * sub_dim_ + j];
            const float* cent_d = centroids_t_.data() + (i * sub_dim_ + j) * centroids_per_subspace_;

            for (size_t k = 0; k < centroids_per_subspace_; ++k) {
                out_pos[k] += cent_d[k] * (cent_d[k] + 2 * c_d);
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

void PQCodebook::save(std::ostream& out) const {
    using namespace detail;

    write_pod(out, static_cast<uint64_t>(dim_));
    write_pod(out, static_cast<uint64_t>(num_subspaces_));
    write_pod(out, static_cast<uint64_t>(centroids_per_subspace_));
    write_vec(out, centroids_);
}

PQCodebook PQCodebook::load(std::istream& in) {
    using namespace detail;

    const auto dim = read_pod<uint64_t>(in);
    const auto num_subspaces = read_pod<uint64_t>(in);
    const auto centroids_per_subspace = read_pod<uint64_t>(in);

    PQCodebook cb(dim, num_subspaces, centroids_per_subspace);

    const uint64_t expected_size = num_subspaces * centroids_per_subspace * cb.sub_dim_;

    cb.centroids_ = read_vec<float>(in, expected_size);
    if (cb.centroids_.size() != expected_size) throw std::runtime_error("corrupt file: unexpected number of centroids");

    cb.build_tables();
    return cb;
}


void PQCodebook::build_tables() {
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

} // namespace vecengine
