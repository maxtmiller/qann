#include "kmeans.hpp"
#include "vecengine/distance.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <limits>
#include <random>
#include <algorithm>
#include <numeric>
#include <utility>

#ifdef VECENGINE_USE_ACCELERATE
#include <Accelerate/Accelerate.h>
#endif

namespace vecengine::detail {

using std::size_t;
using std::vector;


void assign_nearest(const float* data, size_t n, size_t stride, const float* centroids, size_t k, size_t d, uint32_t* out) {
    #ifdef VECENGINE_USE_ACCELERATE
        // Compute norm_sq[c] = ||c||^2 for each centroid
        std::vector<float> cent_norms(k, 0.0f);
        for (size_t i = 0; i < k; ++i) {
            float sum = 0.0f;
            const float* cent = centroids + i * d;
            for (size_t j = 0; j < d; ++j) {
                sum += cent[j] * cent[j];
            }
            cent_norms[i] = sum;
        }

        // Scratchpad buffer for block matrix multiplication
        constexpr size_t B = 1024; // Block size
        std::vector<float> dots(B * k);

        // Process in blocks of B points
        for (size_t i = 0; i < n; i += B) {
            size_t b = std::min(B, n - i);

            // Compute dots = data_block * centroids^T
            // Result matrix 'dots' has dimensions (b x k)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(b), static_cast<int>(k), static_cast<int>(d),
                1.0f,
                data + i * stride, static_cast<int>(stride),
                centroids, static_cast<int>(d),
                0.0f,
                dots.data(), static_cast<int>(k)
            );

            // Find min (||c||^2 - 2 * dot) for each point in the block
            for (size_t j = 0; j < b; ++j) {
                const float* row_dots = dots.data() + j * k;
                
                uint32_t best_c = 0;
                float best_score = cent_norms[0] - 2.0f * row_dots[0];

                for (size_t l = 1; l < k; ++l) {
                    float score = cent_norms[l] - 2.0f * row_dots[l];
                    bool closer = score < best_score;
                    best_score = closer ? score : best_score;
                    best_c = closer ? static_cast<uint32_t>(l) : best_c;
                }

                out[i + j] = best_c;
            }
        }
    #else
        for (size_t i = 0; i < n; ++i) {
            out[i] = nearest_centroid(data + i * stride, centroids, k, d);
        }
    #endif
}

size_t nearest_centroid(const float* point, const float* centroids, size_t k, size_t d) {
    
    std::pair<float, size_t> closest = { std::numeric_limits<float>::max(), 0 };
    for (size_t i = 0; i < k; ++i) {
        const float* centroid_ptr = centroids + (i * d);
        float dist = l2_distance(point, centroid_ptr, d);
        
        if (dist < closest.first) {
            closest = { dist, i };
        }
    }

    return closest.second;
}

KMeansResult kmeans(const float* data, size_t n, size_t stride, size_t d, size_t k, size_t max_iters) {
    if (n < k) throw std::invalid_argument("kmeans: need at least k points");

    std::random_device rd;
    std::mt19937 rng(rd());

    vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    // swap indicies based on random indexes
    for (size_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<size_t> dist(i, n - 1);
        size_t pick = dist(rng);
        std::swap(indices[i], indices[pick]);
    }

    // add the randomly chosen vectors to be initial centroids
    vector<float> test_centroids(k * d);
    for (size_t i = 0; i < k; ++i) {
        size_t chosen_row = indices[i];
        
        const float* src = data + (chosen_row * stride);
        float* dst = test_centroids.data() + (i * d);
        
        std::copy(src, src + d, dst);
    }

    // run iterations of k-means to choose meaningful centroids
    vector<uint32_t> assignments(n);
    for (size_t i = 0; i < max_iters; ++i) {

        // find closest centroid to each vector
        assign_nearest(data, n, stride, test_centroids.data(), k, d, assignments.data());
        
        // for each centroid sum all the dimensions of the vectors closest, and track total number of vectors per centroid
        vector<float> new_centroids(k * d, 0.0f);
        vector<size_t> count_vector(k, 0);
        for (size_t j = 0; j < n; ++j) {
            uint32_t cluster_id = assignments[j];

            const float* vector_ptr = data + (j * stride);
            float* sum_ptr = new_centroids.data() + (cluster_id * d);
            
            for (size_t l = 0; l < d; ++l) {
                sum_ptr[l] += vector_ptr[l];
            }
            ++count_vector[cluster_id];
        }

        // update centroids with mean of current closest vectors or furthest vector if empty
        for (uint32_t j = 0; j < k; ++j) {
            if (count_vector[j] > 0) {
                float* sum_ptr = new_centroids.data() + (j * d);
                float count = static_cast<float>(count_vector[j]);

                for (size_t l = 0; l < d; ++l) {
                    sum_ptr[l] /= count;
                }
            } else {
                size_t worst_vec_idx = 0;
                float max_loss = -1.0f;

                // assign empty centroid with vector furthest away from all other vectors
                for (size_t l = 0; l < n; ++l) {
                    uint32_t current_cluster = assignments[l];
                    const float* vec_ptr = data + (l * stride);
                    const float* centroid_ptr = test_centroids.data() + (current_cluster * d);
                    
                    float dist = l2_distance(vec_ptr, centroid_ptr, d);
                    if (dist > max_loss && dist > 1e-6f) {
                        max_loss = dist;
                        worst_vec_idx = l;
                    }
                }

                std::copy_n(data + worst_vec_idx * stride, d, new_centroids.data() + (j * d));
            }
        }

        test_centroids = std::move(new_centroids);
    }

    assign_nearest(data, n, stride, test_centroids.data(), k, d, assignments.data());

    return { std::move(test_centroids), std::move(assignments) };
}

} // namespace vecengine::detail
