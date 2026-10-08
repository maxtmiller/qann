#pragma once

#include "vecengine/index.hpp"
#include "vecengine/pq.hpp"
#include "vecengine/id_map.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <unordered_map>
#include <iosfwd>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

// The vectors assigned to one coarse cluster.
struct InvertedList {
    vector<uint32_t> ids; // slot of each vector; id_map_ maps slots to user ids
    vector<float> vecs; // populated when PQ is disabled
    vector<uint8_t> codes; // populated when PQ is enabled: num_subspaces bytes per vector, encodes (vec - its cluster's coarse centroid)
};

// Inverted-file index: k-means splits vectors into nlist clusters, and a
// query scans only the nprobe clusters nearest to it. Optionally stores PQ
// codes of each vector's residual instead of the raw floats.
class IVFIndex : public Index {
public:
    // nlist must be in [100, 65535] and nprobe in [1, nlist]; throws
    // std::invalid_argument otherwise. custom_ids makes add_batch() take user
    // ids.
    explicit IVFIndex(size_t dim, size_t nlist, size_t nprobe = 10, bool custom_ids = false);

    // Enables PQ compression of residuals (vec - assigned coarse centroid).
    // Must be called before train()/add(). Once enabled, add() stores codes
    // in InvertedList::codes instead of raw floats in InvertedList::vecs.
    void enable_pq(size_t num_subspaces, size_t centroids_per_subspace = 256);

    // Clusters scanned per query. set_nprobe throws std::invalid_argument
    // unless 1 <= nprobe <= nlist; it can change at any time.
    size_t nprobe() const noexcept { return nprobe_; }
    void set_nprobe(size_t nprobe);

    // How PQ codes are scored: ADC (default, more accurate) or SDC. Has no
    // effect without PQ.
    PQDistance pq_distance() const noexcept { return pq_distance_; }
    void set_pq_distance(PQDistance mode) noexcept { pq_distance_ = mode; }

    // Index overrides; see Index for the contract of each. add/add_batch
    // throw std::logic_error before train().
    void add(span<const float> vec) override;
    void add_batch(span<const float> vecs, size_t n, const int64_t* ids = nullptr) override;
    vector<Neighbor> query(span<const float> vec, size_t k) const override;
    size_t remove(span<const int64_t> ids) override;

    // Learns the nlist coarse centroids (and the PQ codebook, if enabled) from
    // num_vectors sample vectors (at least nlist). Call once, before any add.
    // A seed makes training reproducible: coarse k-means uses seed, the PQ
    // codebook (if enabled) uses seed + 1 onward.
    void train(span<const float> vectors, size_t num_vectors, size_t max_iters = 25,
               std::optional<uint32_t> seed = std::nullopt);

    void save(std::ostream& out) const override;

    // Rebuilds an IVFIndex from the body save() wrote, after load_index() has
    // read the header. Throws std::runtime_error on corrupt data.
    static std::unique_ptr<IVFIndex> load_body(std::istream& in, uint32_t version);

    size_t slots() const noexcept override { return id_map_.slots(); }
    bool custom_ids() const noexcept override { return id_map_.custom_ids(); }

    size_t size() const noexcept override { return id_map_.live(); }
    size_t dim() const noexcept override { return dim_; }

private:
    // Whether enable_pq() was called.
    bool pq_enabled() const noexcept { return pq_ != nullptr; }

    size_t dim_; // dimension of each vector
    size_t nlist_; // num of coarse centroids
    size_t nprobe_; // num of coarse centroids to pick from during quering
    bool trained_ = false; // was function train run

    vector<float> coarse_centroids_; // coarse centroid vectors
    std::unordered_map<uint16_t,InvertedList> data_; // map of coarse centroid ID to inverted list
    std::unique_ptr<PQCodebook> pq_; // pointer to PQ LUT if using PQ
    PQDistance pq_distance_ = PQDistance::ADC; // type of quantization used
    detail::IdMap id_map_;
};

} // namespace vecengine
