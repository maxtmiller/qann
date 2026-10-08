#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace vecengine::detail {

using std::size_t;
using std::vector;

// Output of kmeans().
struct KMeansResult {
    vector<float> centroids; // k * d, row-major
    vector<uint32_t> assignments; // n entries, centroid id in [0, k) for each point
};

// Lloyd's k-means over n points of d floats each, where point i starts at
// data + i * stride. stride == d for packed vectors (IVF); stride > d lets
// PQ cluster one subspace slice of every row without copying it out.
// Requires n >= k. Without a seed, initial centroids come from std::random_device.
KMeansResult kmeans(const float* data, size_t n, size_t stride, size_t d, size_t k, size_t max_iters,
                    std::optional<uint32_t> seed = std::nullopt);

// Index in [0, k) of the centroid closest (squared L2) to the d-float point.
// centroids is k * d, row-major.
size_t nearest_centroid(const float* point, const float* centroids, size_t k, size_t d);

// Nearest centroid for each of n points (point i at data + i * stride).
// Same result as calling nearest_centroid per point, computed in blocks.
void assign_nearest(const float* data, size_t n, size_t stride, const float* centroids, size_t k, size_t d, uint32_t* out);

// Moves each empty cluster's centroid onto one of the points furthest from its
// own centroid, so no cluster stays empty. Updates centroids and cluster_sizes
// in place; called by kmeans() between iterations.
void reseed_empty_clusters(const float* data, size_t n, size_t stride, size_t dim, float* centroids, size_t k, const std::vector<uint32_t>& assignments, std::vector<size_t>& cluster_sizes);

} // namespace vecengine::detail
