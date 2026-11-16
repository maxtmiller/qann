#include "kmeans.hpp"
#include "parallel.hpp"
#include "vecengine/distance.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <limits>
#include <random>
#include <algorithm>
#include <numbers>
#include <functional>
#include <numeric>
#include <utility>
#include <queue>

#ifdef VECENGINE_USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

namespace vecengine::detail {

using std::size_t;
using std::vector;
using std::pair;


void assign_nearest(const float* data, size_t n, size_t stride, const float* centroids, size_t k, size_t d, uint32_t* out) {
    #ifdef VECENGINE_USE_BLAS
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
        constexpr size_t B = 1024;
        std::vector<float> dots(B * k);

        // Process in blocks of B points
        for (size_t i = 0; i < n; i += B) {
            size_t b = std::min(B, n - i);

            // Compute dots = data_block * centroids^T
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
            auto nearest = [&](size_t j) {
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
            };
            // Coarse assignment (IVF add, k-means over nlist) splits the scan
            // across threads; OpenBLAS's idle threads spin while it runs (#170).
            // Small ones stay serial: PQ training calls this per subspace from
            // inside its own parallel_for.
            constexpr size_t kRows = 128;
            if (b * k >= (size_t{1} << 18)) {
                detail::parallel_for((b + kRows - 1) / kRows, [&](size_t r) {
                    for (size_t j = r * kRows; j < std::min(b, (r + 1) * kRows); ++j) nearest(j);
                });
            } else {
                for (size_t j = 0; j < b; ++j) nearest(j);
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

void reseed_empty_clusters(const float* data, size_t n, size_t stride, size_t dim, float* centroids, size_t k, const std::vector<uint32_t>& assignments, std::vector<size_t>& cluster_sizes) {
    
    // track indicies of empty clusters
    std::vector<size_t> empty_clusters;
    for (size_t i = 0; i < k; ++i) {
        if (cluster_sizes[i] == 0) {
            empty_clusters.emplace_back(i);
        }
    }
    if (empty_clusters.empty()) return;

    // get empty_clusters.size() furthest vectors
    size_t empty_num = empty_clusters.size();
    std::priority_queue<pair<float, size_t>, std::vector<pair<float, size_t>>, std::greater<pair<float, size_t>>> minHeapVectors;
    for (size_t i = 0; i < n; ++i) {
        float dist = l2_distance(data + i * stride, centroids + assignments[i] * dim, dim);
        if (dist < 1e-6f) continue;

        if (minHeapVectors.size() < empty_num) {
            minHeapVectors.emplace(dist, i);
        } else if (dist > minHeapVectors.top().first) {
            minHeapVectors.pop();
            minHeapVectors.emplace(dist, i);
        }
    }

    // update empty centroids with furthest vectors; fallback to picking a vector from largest cluster
    for (size_t i = 0; i < empty_num; ++i) {
        if (!minHeapVectors.empty()) {
            auto [distance, index] = minHeapVectors.top();
            minHeapVectors.pop();
            std::copy_n(data + index * stride, dim, centroids + empty_clusters[i] * dim);
        } else {
            // find largest cluster
            size_t largest_c = std::distance(
                cluster_sizes.begin(),
                std::max_element(cluster_sizes.begin(), cluster_sizes.end())
            );

            // get pointer to largest cluster centroid and empty cluster centroid
            const float eps = 1.0f / 1024.0f;
            float* src = centroids + largest_c * dim;
            float* dst = centroids + empty_clusters[i] * dim;

            // push both clusters away from each other
            for (size_t j = 0; j < dim; ++j) {
                dst[j] = src[j] * (1.0f + eps);
                src[j] *= (1.0f - eps);
            }

            // update cluster sizes so next empty cluster chooses largest cluster after split
            cluster_sizes[empty_clusters[i]] = cluster_sizes[largest_c] / 2;
            cluster_sizes[largest_c] -= cluster_sizes[empty_clusters[i]];
        }
    }
}

KMeansResult kmeans(const float* data, size_t n, size_t stride, size_t d, size_t k, size_t max_iters, std::optional<uint32_t> seed,
                    bool spherical) {
    if (n < k) throw std::invalid_argument("kmeans: need at least k points");

    std::mt19937 rng(seed ? *seed : std::random_device{}());

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
                if (spherical) {
                    double sq = 0;
                    for (size_t l = 0; l < d; ++l) sq += double(sum_ptr[l]) * sum_ptr[l];
                    if (sq > 0) {
                        const float scale = static_cast<float>(1.0 / std::sqrt(sq));
                        for (size_t l = 0; l < d; ++l) sum_ptr[l] *= scale;
                    }
                }
            }
        }

        // reseed empty clusters with furthest vectors
        reseed_empty_clusters(data, n, stride, d, new_centroids.data(), k, assignments, count_vector);

        test_centroids = std::move(new_centroids);
    }

    assign_nearest(data, n, stride, test_centroids.data(), k, d, assignments.data());

    return { std::move(test_centroids), std::move(assignments) };
}

namespace {

// Unit-length copy of each row of c (k x d); zero rows stay zero.
void unit_rows(const float* c, size_t k, size_t d, float* out) {
    for (size_t j = 0; j < k; ++j) {
        double sq = 0;
        for (size_t l = 0; l < d; ++l) sq += double(c[j * d + l]) * c[j * d + l];
        const float scale = sq > 0 ? static_cast<float>(1.0 / std::sqrt(sq)) : 0.0f;
        for (size_t l = 0; l < d; ++l) out[j * d + l] = c[j * d + l] * scale;
    }
}

size_t round_power_of_two(double v) {
    if (v <= 1) return 1;
    const size_t low = size_t{1} << static_cast<int>(std::floor(std::log2(v)));
    const size_t high = 2 * low;
    return v / double(low) < double(high) / v ? low : high;
}

} // namespace

// e-means (Bessa et al. 2026, arXiv 2611.09501; reference code in faiss/contrib/e_means):
// mini-batch k-means with EMA centroids. Constants are those of faiss's NumpyEmeans:
// the throughput (d > 16) or quality batch law, the step law, a cosine-annealed step
// down to 0.1 alpha0, initial mass 2^-16, and every 5 updates clusters under 0.1x the
// mean mass are refilled by splitting donors drawn in proportion to their mass.
KMeansResult emeans(const float* data, size_t n, size_t stride, size_t d, size_t k, size_t passes,
                    std::optional<uint32_t> seed, bool spherical) {
    if (n < k) throw std::invalid_argument("emeans: need at least k points");
    if (passes == 0) throw std::invalid_argument("emeans: need at least one pass");
    std::mt19937 rng(seed ? *seed : std::random_device{}());

    const double T = double(passes), nd = double(n), kd = double(k);
    const bool quality = d <= 16;
    size_t batch = round_power_of_two((quality ? 0.00439175 : 0.035134) * std::pow(kd, 0.134749) *
                                      std::pow(nd, 0.876473) * std::pow(T, 0.813344));
    batch = quality ? std::min(batch, n / 2) : std::min(batch, round_power_of_two(nd / 2));
    batch = std::max<size_t>(1, std::min(batch, n));
    const double alpha0 = std::clamp(0.8244 * std::pow(kd, -0.4617) * std::pow(double(batch), 0.9003) *
                                         std::pow(nd, -0.4498) * std::pow(T, -0.1057),
                                     1e-4, 1.0);
    const size_t steps_per_pass = (n + batch - 1) / batch, total_steps = passes * steps_per_pass;
    const float tau = d <= 2 ? 0.0f : 0.1f;

    // initial centroids: k distinct random rows
    vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    for (size_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<size_t> pick(i, n - 1);
        std::swap(order[i], order[pick(rng)]);
    }
    vector<float> pos(k * d);
    for (size_t j = 0; j < k; ++j) std::copy_n(data + order[j] * stride, d, pos.data() + j * d);
    if (spherical) unit_rows(pos.data(), k, d, pos.data());
    vector<float> mass(k, 1.0f / 65536.0f);

    vector<float> readout(spherical ? k * d : 0), rows(batch * d), sums(k * d), pass_sums(k * d, 0.0f);
    vector<double> pass_counts(k, 0.0);
    vector<uint32_t> counts(k), labels(batch);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    size_t step = 0;
    for (size_t pass = 0; pass < passes; ++pass) {
        std::shuffle(order.begin(), order.end(), rng);
        const bool last = pass + 1 == passes;
        for (size_t start = 0; start < n; start += batch, ++step) {
            const size_t b = std::min(batch, n - start);
            for (size_t r = 0; r < b; ++r) std::copy_n(data + order[start + r] * stride, d, rows.data() + r * d);
            const float* cent = pos.data();
            if (spherical) {
                unit_rows(pos.data(), k, d, readout.data());
                cent = readout.data();
            }
            assign_nearest(rows.data(), b, d, cent, k, d, labels.data());

            std::fill(counts.begin(), counts.end(), 0u);
            for (size_t r = 0; r < b; ++r) ++counts[labels[r]];
            std::fill(sums.begin(), sums.end(), 0.0f);
            // each thread sums a slice of the dimensions over every row
            constexpr size_t kDims = 64;
            parallel_for((d + kDims - 1) / kDims, [&](size_t c) {
                const size_t lo = c * kDims, hi = std::min(d, lo + kDims);
                for (size_t r = 0; r < b; ++r) {
                    float* dst = sums.data() + labels[r] * d;
                    const float* src = rows.data() + r * d;
                    for (size_t l = lo; l < hi; ++l) dst[l] += src[l];
                }
            });
            if (last) {
                for (size_t j = 0; j < k; ++j) pass_counts[j] += counts[j];
                for (size_t i = 0; i < k * d; ++i) pass_sums[i] += sums[i];
            }

            const double frac = double(std::min(step, total_steps)) / double(std::max<size_t>(1, total_steps));
            const double alpha = alpha0 * (0.1 + 0.9 * 0.5 * (1.0 + std::cos(std::numbers::pi * frac)));
            const bool replace = alpha == 1.0;
            if (!replace)
                for (float& m : mass) m = static_cast<float>(m * (1.0 - alpha));
            for (size_t j = 0; j < k; ++j) {
                if (counts[j] == 0) continue;
                const float nj = static_cast<float>(counts[j]);
                const float new_mass = (replace ? 0.0f : mass[j]) + static_cast<float>(alpha) * nj;
                const float w = static_cast<float>(alpha) * nj / new_mass;
                float* c = pos.data() + j * d;
                const float* s = sums.data() + j * d;
                for (size_t l = 0; l < d; ++l) {
                    const float mean = s[l] / nj;
                    c[l] = w == 1.0f ? mean : c[l] + w * (mean - c[l]);
                }
                mass[j] = new_mass;
            }

            // revival: split heavy donors into clusters that lost their mass
            if (tau > 0 && (step + 1) % 5 == 0) {
                float total = 0;
                for (float m : mass) total += m;
                const float threshold = total * tau / static_cast<float>(k);
                vector<size_t> unused;
                vector<std::pair<double, size_t>> donors;
                for (size_t j = 0; j < k; ++j) {
                    if (mass[j] < threshold) unused.push_back(j);
                    else if (mass[j] > 0) donors.emplace_back(0.0, j);
                }
                const size_t nsplit = std::min(unused.size(), donors.size());
                if (nsplit > 0) {
                    // Gumbel top-k: sampling without replacement, proportional to mass
                    for (auto& [key, j] : donors)
                        key = std::log(double(mass[j])) - std::log(-std::log(std::max(unif(rng), 1e-300)));
                    std::partial_sort(donors.begin(), donors.begin() + nsplit, donors.end(), std::greater<>());
                    for (size_t t = 0; t < nsplit; ++t) {
                        const size_t src = donors[t].second, dst = unused[t];
                        float* a = pos.data() + src * d;
                        float* c = pos.data() + dst * d;
                        bool same = true;
                        for (size_t l = 0; l < d; ++l) {
                            const float off = (l % 2 == 0 ? 1e-4f : -1e-4f);
                            const float v = a[l];
                            a[l] = v * (1.0f - off);
                            c[l] = v * (1.0f + off);
                            same = same && a[l] == c[l];
                        }
                        if (same) c[0] = std::nextafter(c[0], std::numeric_limits<float>::infinity());
                        mass[src] = mass[dst] = mass[src] / 2.0f;
                    }
                }
            }
        }
    }

    // closing: means of the last pass's assignments; unassigned clusters keep their position
    for (size_t j = 0; j < k; ++j) {
        if (pass_counts[j] <= 0) continue;
        const float cnt = static_cast<float>(pass_counts[j]);
        for (size_t l = 0; l < d; ++l) pos[j * d + l] = pass_sums[j * d + l] / cnt;
    }
    if (spherical) unit_rows(pos.data(), k, d, pos.data());

    vector<uint32_t> assignments(n);
    assign_nearest(data, n, stride, pos.data(), k, d, assignments.data());
    return { std::move(pos), std::move(assignments) };
}

} // namespace vecengine::detail
