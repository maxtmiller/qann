#pragma once

#include "vecengine/index.hpp"

#include <cstddef>
#include <memory>
#include <filesystem>
#include <iosfwd>

namespace vecengine {

using std::size_t;

// Index types make_index() can build.
enum class IndexType {
    Flat,
    IVF,
    HNSW,
};

// Per-type construction options. Only the fields relevant to `type` are read.
struct IndexOptions {
    size_t capacity = 1024; // FlatIndex: initial reserve
    size_t nlist = 100;     // IVFIndex: number of coarse clusters
    size_t nprobe = 10;     // IVFIndex: clusters scanned per query
    size_t pq_subspaces = 0;   // IVFIndex: PQ subspaces, 0 = PQ off
    size_t pq_centroids = 256; // IVFIndex: PQ centroids per subspace
    bool pq_rotate = false;    // IVFIndex: learn a PCA-balanced rotation before PQ (needs pq_subspaces)
    int ivf_sq_bits = 0;       // IVFIndex: 4 or 8 stores SQ codes of that many bits instead of floats; 0 = floats
    size_t hnsw_m = 16;            // HNSWIndex: neighbors per node (2x on level 0)
    size_t ef_construction = 200;  // HNSWIndex: beam width while inserting
    size_t ef_search = 64;         // HNSWIndex: beam width per query
    int hnsw_sq_bits = 0;          // HNSWIndex: 4 or 8 stores codes of that many bits (call train() before adding); 0 = floats
    bool custom_ids = false;   // all: add_batch() takes user ids
    Metric metric = Metric::L2; // all: how vectors are compared
};

// Constructs the requested index type. Caller owns the result via unique_ptr
// and interacts with it entirely through the Index interface.
std::unique_ptr<Index> make_index(IndexType type, size_t dim, const IndexOptions& opts = {});

// Loads an index from a stream. The stream must have been written by save().
// Returns the concrete type; dynamic_cast to Index for a dense one.
std::unique_ptr<IndexBase> load_index(std::istream& in);

// Saves an index to a file. Writes to path + ".tmp" first and renames it into
// place, so a failed save never leaves a partial file at `path`.
void save_index(const IndexBase& index, const std::filesystem::path& path);

// Loads an index saved by save_index. Every problem with the file's contents
// is reported as std::runtime_error.
std::unique_ptr<IndexBase> load_index(const std::filesystem::path& path);

} // namespace vecengine
