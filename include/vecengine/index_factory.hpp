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
    // HNSW,  // add here once implemented
};

// Per-type construction options. Only the fields relevant to `type` are read.
struct IndexOptions {
    size_t capacity = 1024; // FlatIndex: initial reserve
    size_t nlist = 100;     // IVFIndex: number of coarse clusters
    size_t nprobe = 10;     // IVFIndex: clusters scanned per query
    size_t pq_subspaces = 0;   // IVFIndex: PQ subspaces, 0 = PQ off
    size_t pq_centroids = 256; // IVFIndex: PQ centroids per subspace
    bool custom_ids = false;   // both: add_batch() takes user ids
};

// Constructs the requested index type. Caller owns the result via unique_ptr
// and interacts with it entirely through the Index interface.
std::unique_ptr<Index> make_index(IndexType type, size_t dim, const IndexOptions& opts = {});

// Loads an index from a stream. The stream must have been written by Index::save().
std::unique_ptr<Index> load_index(std::istream& in);

// Saves an index to a file. Writes to path + ".tmp" first and renames it into
// place, so a failed save never leaves a partial file at `path`.
void save_index(const Index& index, const std::filesystem::path& path);

// Loads an index saved by save_index. Every problem with the file's contents
// is reported as std::runtime_error.
std::unique_ptr<Index> load_index(const std::filesystem::path& path);

} // namespace vecengine
