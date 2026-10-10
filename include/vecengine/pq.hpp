#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>
#include <span>
#include <cassert>
#include <iosfwd>

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
    // dim and num_subspaces must be >= 1 and dim divisible by num_subspaces.
    // centroids_per_subspace must fit in uint8_t (<= 256); 256 is standard.
    PQCodebook(size_t dim, size_t num_subspaces, size_t centroids_per_subspace = 256);

    // Runs k-means independently within each subspace to learn centroids_,
    // then (always) builds sdc_table_ from the resulting centroids.
    // A seed makes training reproducible; subspace i uses seed + i.
    void train(span<const float> vectors, size_t num_vectors, size_t max_iters = 25,
               std::optional<uint32_t> seed = std::nullopt);

    // Encodes one full vector into num_subspaces_ codes (one byte each).
    vector<uint8_t> encode(span<const float> vec) const;

    // ADC step 1: distances from each query slice to every centroid in its
    // subspace. Returns num_subspaces_ * centroids_per_subspace_ floats,
    // table[s * K + c]. Build once per query (per probed list in IVF) and
    // reuse it for every code scanned.
    void compute_adc_table(span<const float> query, span<float> out) const;

    // Precomputed-table term C: out[s * K + j] = -2 <query_s, y_sj>, where
    // y_sj is centroid j of subspace s. Build once per query and share it
    // across every probed list. Same layout as compute_adc_table.
    void compute_inner_table(span<const float> query, span<float> out) const;

    // Precomputed-table term B for one coarse centroid c:
    // out[s * K + j] = ||y_sj||^2 + 2 <c_s, y_sj>. Built once per list at
    // train/load time. Per subspace, ||q_s - c_s||^2 + B + C equals the ADC
    // table entry of the residual (q - c).
    void compute_list_term(span<const float> coarse_centroid, span<float> out) const;

    // ADC step 2: query stays full-precision; only `code` is quantized.
    // Sums table[s * K + code[s]] over subspaces - lookups only, no float math.
    float distance_adc(span<const float> table, span<const uint8_t> code) const {
        assert(table.size() == num_subspaces_ * centroids_per_subspace_);
        assert(code.size() == num_subspaces_);

        float sum = 0.0f;
        for (size_t i = 0; i < num_subspaces_; ++i) {
            sum += table[(i * centroids_per_subspace_) + code[i]];
        }
        return sum;
    }

    // ADC step 3: computes distances for n codes at once, storing them in out.
    // table is the same as in distance_adc; codes is n * num_subspaces_ bytes.
    void distances_adc(span<const float> table, const uint8_t* codes, size_t n, float* out) const {
        assert(table.size() == num_subspaces_ * centroids_per_subspace_);

        // calculates distance for 4 vectors at a time,
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            const uint8_t* c = codes + i * num_subspaces_;
            float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            for (size_t j = 0; j < num_subspaces_; ++j) {
                const float* row = table.data() + j * centroids_per_subspace_;
                d0 += row[c[j]];
                d1 += row[c[num_subspaces_ + j]];
                d2 += row[c[2 * num_subspaces_ + j]];
                d3 += row[c[3 * num_subspaces_ + j]];
            }
            out[i] = d0; out[i + 1] = d1; out[i + 2] = d2; out[i + 3] = d3;
        }
        for (; i < n; ++i) {
            const uint8_t* c = codes + i * num_subspaces_;
            float d = 0;
            for (size_t s = 0; s < num_subspaces_; ++s) d += table[s * centroids_per_subspace_ + c[s]];
            out[i] = d;
        }
    }

    // SDC: both sides are quantized. Looks up precomputed centroid-pair
    // distances in sdc_table_ built once during train() - no float math.
    float distance_sdc(span<const uint8_t> query_code, span<const uint8_t> code) const;

    // Writes the codebook's shape and centroids (no header; it is stored
    // inside an IVFIndex body).
    void save(std::ostream& out) const;

    // Reads what save() wrote and rebuilds the derived tables. Throws
    // std::runtime_error on corrupt data.
    static PQCodebook load(std::istream& in);

    // Shape of the codebook.
    size_t dim() const noexcept { return dim_; }
    size_t num_subspaces() const noexcept { return num_subspaces_; }
    size_t centroids_per_subspace() const noexcept { return centroids_per_subspace_; }

private:
    size_t dim_; // dimension of raw vectors
    size_t num_subspaces_; // num of subspaces each vector is split into
    size_t sub_dim_; // dim of vectors in each subspace --> dim_ / num_subspaces_
    size_t centroids_per_subspace_; // num of centroids per subspace

    vector<float> centroids_; // m * k * sub_dim_, row-major per subspace
    vector<float> centroids_t_; // m * sub_dim_ * k, contiguous over centroids for the ADC/SDC table loops
    vector<float> sdc_table_; // m * k * k, built once in train()

    // Builds centroids_t_ and sdc_table_ from centroids_; called by train() and load().
    void build_tables();
};

} // namespace vecengine
