#pragma once

// Result fusion (issue #60): merges ranked results from several retrievers,
// such as a dense and a sparse index over the same documents, into one
// ranking, by reciprocal rank fusion or by a weighted sum of normalized scores.

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vecengine {

using std::size_t;
using std::span;
using std::vector;

// How fuse() scores an id, summed over the lists holding it with weight w:
// - RRF (reciprocal rank fusion): w / (k + rank), rank counting from 1. Uses
//   only positions, so it needs no distances and no tuning beyond k.
// - Linear: w * (max - d) / (max - min), the list's distance d min-max
//   normalized per query over the hits it returned (1 for the nearest, 0 for
//   the farthest; 1 for all when they are equal). Needs distances; with
//   weights tuned on labeled queries it usually beats RRF.
enum class FusionMethod { RRF, Linear };

// One fused result: an id and its score, higher is better.
struct FusedHit {
    int64_t id;
    float score;
};

// Fuses one query's ranked lists. dists is empty for RRF, or holds one
// distance per id of each list (smaller is nearer) for Linear. Negative ids
// (padding) and, for Linear, non-finite distances are skipped; an id repeated
// within a list counts only at its first position; lists with weight 0 are
// ignored. Returns up to limit hits, best first, ties broken by the smaller
// id. weights is empty (all 1) or holds one finite weight >= 0 per list; k
// (RRF only) must be finite and > 0. Throws std::invalid_argument otherwise.
vector<FusedHit> fuse(FusionMethod method, span<const span<const int64_t>> ids, span<const span<const float>> dists,
                      span<const float> weights, float k, size_t limit);

// fuse() for num_queries queries at once, in parallel: ids[i] (and dists[i])
// hold num_queries rows of widths[i] entries, row-major, and row q of every
// list is fused into result q. Throws std::invalid_argument as fuse() does, or
// when a list's size isn't num_queries * widths[i].
vector<vector<FusedHit>> fuse_batch(FusionMethod method, span<const span<const int64_t>> ids,
                                    span<const span<const float>> dists, span<const size_t> widths,
                                    size_t num_queries, span<const float> weights, float k, size_t limit);

} // namespace vecengine
