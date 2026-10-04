#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

// ADC keeps the query full-precision (more accurate); SDC encodes it too.
enum class PQDistance { ADC, SDC };

// Product Quantization codebook. Splits a `dim`-dimensional vector into
// `num_subspaces` contiguous sub-vectors and quantizes each sub-vector
// independently to one of `centroids_per_subspace` learned centroids.
// A vector is then represented as `num_subspaces` single-byte codes
// instead of `dim` floats.
class PQCodebook {
public:
    // dim must be divisible by num_subspaces.
    // centroids_per_subspace must fit in uint8_t (<= 256); 256 is standard.
    PQCodebook(size_t dim, size_t num_subspaces, size_t centroids_per_subspace = 256);

    // Runs k-means independently within each subspace to learn centroids_,
    // then (always) builds sdc_table_ from the resulting centroids.
    void train(span<const float> vectors, size_t num_vectors, size_t max_iters = 25);

    // Encodes one full vector into num_subspaces_ codes (one byte each).
    vector<uint8_t> encode(span<const float> vec) const;

    // ADC step 1: distances from each query slice to every centroid in its
    // subspace. Returns num_subspaces_ * centroids_per_subspace_ floats,
    // table[s * K + c]. Build once per query (per probed list in IVF) and
    // reuse it for every code scanned.
    vector<float> compute_adc_table(span<const float> query) const;

    // ADC step 2: query stays full-precision; only `code` is quantized.
    // Sums table[s * K + code[s]] over subspaces - lookups only, no float math.
    float distance_adc(span<const float> table, span<const uint8_t> code) const;

    // SDC: both sides are quantized. Looks up precomputed centroid-pair
    // distances in sdc_table_ built once during train() - no float math.
    float distance_sdc(span<const uint8_t> query_code, span<const uint8_t> code) const;

    size_t dim() const noexcept { return dim_; }
    size_t num_subspaces() const noexcept { return num_subspaces_; }
    size_t centroids_per_subspace() const noexcept { return centroids_per_subspace_; }

private:
    size_t dim_; // dimension of raw vectors
    size_t num_subspaces_; // num of subspaces each vector is split into
    size_t sub_dim_; // dim of vectors in each subspace --> dim_ / num_subspaces_
    size_t centroids_per_subspace_; // num of centroids per subspace

    vector<float> centroids_; // m * k * sub_dim_, row-major per subspace
    vector<float> sdc_table_; // m * k * k, built once in train()
};

} // namespace vecengine
