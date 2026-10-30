#pragma once

#include "vecengine/index.hpp"
#include "vecengine/pq.hpp"
#include "vecengine/id_map.hpp"
#include "vecengine/sq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <memory>
#include <optional>
#include <unordered_map>
#include <iosfwd>
#include <stdexcept>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;

// The vectors assigned to one coarse cluster.
struct InvertedList {
    vector<uint32_t> ids; // slot of each vector; id_map_ maps slots to user ids
    vector<float> vecs; // populated without PQ or SQ
    vector<uint8_t> codes; // with PQ: num_subspaces bytes per vector, encoding (vec - its cluster's coarse centroid);
                           // with SQ: each vector's SQ codes (detail::SQ::code_bytes() per vector)
};

// Inverted-file index: k-means splits vectors into nlist clusters, and a
// query scans only the nprobe clusters nearest to it. Optionally stores PQ
// codes of each vector's residual, or SQ codes of each vector, instead of the
// raw floats.
class IVFIndex : public Index {
public:
    // nlist must be in [100, 65535] and nprobe in [1, nlist]; throws
    // std::invalid_argument otherwise. custom_ids makes add_batch() take user
    // ids.
    explicit IVFIndex(size_t dim, size_t nlist, size_t nprobe = 10, bool custom_ids = false, Metric metric = Metric::L2);

    // Enables PQ compression of residuals (vec - assigned coarse centroid).
    // Must be called before train()/add(). Once enabled, add() stores codes
    // in InvertedList::codes instead of raw floats in InvertedList::vecs.
    // With rotate, train() also learns a PCA-balanced rotation (see
    // src/rotation.hpp) and every vector is rotated by it before the coarse
    // and PQ steps (issue #41). Throws std::logic_error after enable_sq().
    void enable_pq(size_t num_subspaces, size_t centroids_per_subspace = 256, bool fast_scan = false,
                   bool rotate = false);

    // Whether enable_pq(..., rotate = true) was used.
    bool pq_rotate() const noexcept { return rotate_; }

    // Stores each vector as scalar codes of `bits` (4 or 8; std::invalid_argument
    // otherwise) per dimension instead of raw floats: 8x or 4x less list
    // memory, approximate distances. train() then also learns each
    // dimension's range from its sample. Must be called before train()/add(),
    // and not together with enable_pq(); throws std::logic_error otherwise.
    void enable_sq(int bits = 8);

    // Bits per dimension of the SQ codes: 4 or 8, or 0 without enable_sq().
    int sq_bits() const noexcept { return sq_bits_; }

    // Clusters scanned per query. set_nprobe throws std::invalid_argument
    // unless 1 <= nprobe <= nlist; it can change at any time.
    size_t nprobe() const noexcept { return nprobe_; }
    size_t nlist() const noexcept { return nlist_; }
    void set_nprobe(size_t nprobe);

    // How PQ codes are scored: ADC (default, more accurate) or SDC. Has no
    // effect without PQ.
    PQDistance pq_distance() const noexcept { return pq_distance_; }
    void set_pq_distance(PQDistance mode) { 
        if (fast_scan_ && mode == PQDistance::SDC) 
            throw std::invalid_argument("cannot use SDC with fast-scan");
        if (metric_ == Metric::InnerProduct && mode == PQDistance::SDC)
            throw std::invalid_argument("cannot use SDC with Metric::InnerProduct");
        pq_distance_ = mode; 
    }

    // Whether ADC queries use precomputed per-list tables instead of building
    // a full ADC table for every probed list. Enabled by default; the getter
    // is true only once the tables exist (PQ on, trained, and nlist * m * K
    // floats within kPrecomputedMaxBytes). Disabling frees them. Results
    // match the per-list path up to float rounding. Never built for
    // Metric::InnerProduct, whose one table per query serves every list.
    bool precomputed_tables() const noexcept { return !precomputed_.empty(); }
    void set_precomputed_tables(bool enabled);

    // Whether a filtered query probes more lists as its filter narrows, about
    // 0.75 * nprobe * s^(-1/3) for a fraction s allowed, so its recall stays
    // near an unfiltered query's. On by default; off, it probes nprobe lists
    // and then only until it has k allowed vectors (faster, lower recall).
    // Not saved: a loaded index has it on.
    bool filter_probe_scaling() const noexcept { return filter_probe_scaling_; }
    void set_filter_probe_scaling(bool enabled) noexcept { filter_probe_scaling_ = enabled; }

    // Index overrides; see Index for the contract of each. Adds throw
    // std::logic_error before train(). query_batch picks every query's nprobe
    // lists with one matrix multiply per block of queries (with BLAS), then
    // scans the lists in parallel; results match query() except that a
    // near-tie between coarse centroids can pick a different list.
    size_t remove(span<const int64_t> ids) override;
    size_t compact() override;

    // Learns the nlist coarse centroids (and the PQ codebook or SQ ranges, if enabled) from
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
    bool ids_are_slots() const noexcept override { return id_map_.ids_are_slots(); }
    bool fast_scan() const noexcept { return fast_scan_; }

    size_t size() const noexcept override { return id_map_.live(); }
    size_t dim() const noexcept override { return dim_; }

private:
    // Index implementations: vectors already prepared for the metric.
    void add_batch_impl(span<const float> vecs, size_t n, const int64_t* ids) override;
    vector<Neighbor> query_impl(span<const float> vec, size_t k, const detail::SlotFilter* filter) const override;
    vector<vector<Neighbor>> query_batch_impl(span<const float> queries, size_t num_queries, size_t k,
                                              const detail::SlotFilter* filter) const override;
    const detail::IdMap& id_map() const noexcept override { return id_map_; }

    size_t compact_as_base() override;

    // Shared by compact() and compact_as_base(): compacts id_map_, then the
    // storage, and returns the slots reclaimed.
    size_t compact_impl(bool keep_ids);

    // Moves every live vector from its old slot to remap[old] (see
    // IdMap::compact) and drops deleted ones, keeping their order.
    void compact_storage(const vector<uint32_t>& remap);

    // Whether enable_pq() was called.
    bool pq_enabled() const noexcept { return pq_ != nullptr; }

    // A copy of the n rows of vecs rotated by rotation_ (requires a trained
    // rotation). Used by train, add_batch_impl and the query paths.
    vector<float> rotated(span<const float> vecs, size_t n) const;

    // Bytes of one vector's SQ codes (with SQ enabled).
    size_t sq_code_bytes() const noexcept { return sq_bits_ == 8 ? dim_ : (dim_ + 1) / 2; }

    // Fills precomputed_ with PQCodebook::compute_list_term for every coarse
    // centroid, or leaves it empty if disabled, untrained, without PQ or over
    // the size cap. Called by train(), load_body() and set_precomputed_tables().
    void build_precomputed();

    // Fills centroid_sqnorms_ from coarse_centroids_. Called by train() and load_body().
    void update_centroid_sqnorms();

    // One probed list: its exact coarse distance ||q - c||^2 (term A of the
    // precomputed tables), or -(q . c) for InnerProduct, and its cluster id.
    // InnerProduct still clusters and assigns vectors by L2 k-means and only
    // probes by -(q . c): measured on lastfm factors and VIBE's yi and
    // llama-128-ip, that beat clustering the L2-transformed vectors (as HNSW
    // links them) or probing by L2 in that space.
    struct Probe {
        float dist;
        uint16_t list;

        // Nearer first; ties by list id, so the order is deterministic.
        bool operator<(const Probe& o) const noexcept { return dist < o.dist || (dist == o.dist && list < o.list); }
    };

    // The nprobe lists nearest to vec, nearest first.
    vector<Probe> coarse(span<const float> vec) const;

    // Every list, for a filtered query: the nprobe nearest to vec first,
    // nearest first, then the rest in any order (scan() sorts them if it
    // needs them).
    vector<Probe> coarse_all(span<const float> vec) const;

    // Coarse step for n queries (query i at queries + i * dim_): writes each
    // query's nprobe probes to out + i * nprobe_, in any order.
    void coarse_batch(const float* queries, size_t n, Probe* out) const;

    // Fine step: the top k over the probed lists. Without a filter, probes
    // holds the nprobe lists to scan. With one, it holds coarse_all()'s lists,
    // and lists past the first nprobe are scanned, nearest first, only until
    // min(k, filter->count) allowed vectors have been found.
    vector<Neighbor> scan(span<const float> vec, span<Probe> probes, size_t k, const detail::SlotFilter* filter) const;

    // Whether a filtered query scores the allowed vectors directly instead of
    // probing lists (see direct_scan).
    bool use_direct(size_t k, const detail::SlotFilter* filter) const;

    // Top k among the filter's slots, found through slot_list_ / slot_pos_.
    vector<Neighbor> direct_scan(span<const float> vec, size_t k, const detail::SlotFilter& filter) const;

    // Rebuilds slot_list_ and slot_pos_ from the lists (after compact or load);
    // leaves them empty with PQ.
    void rebuild_slot_lookup();

    static constexpr size_t kPrecomputedMaxBytes = size_t{256} << 20;
    // With Accelerate, a single query's coarse step uses one sgemv from this dimension up;
    // below it the call's fixed cost outweighs the gain (measured on #62).
    static constexpr size_t kBlasCoarseMinDim = 256;

    size_t dim_; // dimension of each vector
    size_t nlist_; // num of coarse centroids
    size_t nprobe_; // num of coarse centroids to pick from during quering
    bool trained_ = false; // was function train run

    vector<float> coarse_centroids_; // coarse centroid vectors
    vector<float> centroid_sqnorms_; // |c|^2 per coarse centroid, for the BLAS coarse steps
    std::unordered_map<uint16_t,InvertedList> data_; // map of coarse centroid ID to inverted list
    std::unique_ptr<PQCodebook> pq_; // pointer to PQ LUT if using PQ
    int sq_bits_ = 0; // 0 (no SQ), 4 or 8
    detail::SQ sq_; // SQ ranges, learned by train() when SQ is enabled
    PQDistance pq_distance_ = PQDistance::ADC; // type of quantization used
    bool use_precomputed_ = true; // set_precomputed_tables() setting
    bool filter_probe_scaling_ = true; // set_filter_probe_scaling() setting
    vector<float> precomputed_; // term B: nlist * m * K floats, list-major; empty when not in use
    detail::IdMap id_map_;
    bool fast_scan_ = false;
    bool rotate_ = false; // enable_pq(..., rotate) setting
    vector<float> rotation_; // dim_ x dim_, row-major (see src/rotation.hpp); empty until trained
    vector<float> rotation_panels_; // rotation_ in the layout detail::rotate reads
    // list holding each slot and its position there, for direct_scan; plain and
    // SQ only (empty with PQ), not saved but rebuilt on load and compact
    vector<uint16_t> slot_list_;
    vector<uint32_t> slot_pos_;
};

} // namespace vecengine
