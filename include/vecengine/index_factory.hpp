#pragma once

#include "vecengine/index.hpp"

#include <cstddef>
#include <memory>

namespace vecengine {

using std::size_t;

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
};

// Constructs the requested index type. Caller owns the result via unique_ptr
// and interacts with it entirely through the Index interface.
std::unique_ptr<Index> make_index(IndexType type, size_t dim, const IndexOptions& opts = {});

} // namespace vecengine
