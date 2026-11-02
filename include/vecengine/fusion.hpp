#pragma once

// Reciprocal rank fusion (issue #60): merges ranked id lists from several
// retrievers, such as a dense and a sparse index over the same documents,
// into one ranking. Only rank positions count, so retrievers whose scores
// aren't comparable (BM25 vs cosine distance) can be combined.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vecengine {

using std::size_t;
using std::span;
using std::vector;

// One fused result: an id and its RRF score, higher is better.
struct FusedHit {
    int64_t id;
    float score;
};

// Fuses one query's ranked lists: score(id) = sum over the lists holding it of
// weights[i] / (k + rank), rank counting from 1 at the head of the list.
// Negative ids (padding) are skipped, an id repeated within a list counts only
// at its first position, and lists with weight 0 are ignored. Returns up to
// limit hits, best first, ties broken by the smaller id. weights is empty (all
// 1) or holds one finite weight >= 0 per list; k must be finite and > 0.
// Throws std::invalid_argument otherwise.
vector<FusedHit> rrf_fuse(span<const span<const int64_t>> lists, span<const float> weights, float k, size_t limit);

// rrf_fuse() for num_queries queries at once, in parallel: lists[i] holds
// num_queries rows of widths[i] ids (row-major), and row q of every list is
// fused into result q. Throws std::invalid_argument as rrf_fuse() does, or
// when a list's size isn't num_queries * widths[i].
vector<vector<FusedHit>> rrf_fuse_batch(span<const span<const int64_t>> lists, span<const size_t> widths,
                                        size_t num_queries, span<const float> weights, float k, size_t limit);

} // namespace vecengine
