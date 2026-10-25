#include "rotation.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#ifdef VECENGINE_USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

namespace vecengine::detail {

void symmetric_eigen(vector<double> a, size_t d, vector<double>& values, vector<double>& vectors) {
    vectors.assign(d * d, 0.0);
    for (size_t i = 0; i < d; ++i) vectors[i * d + i] = 1.0;

    double total = 0.0;
    for (double x : a) total += x * x;

    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (size_t p = 0; p < d; ++p)
            for (size_t q = p + 1; q < d; ++q) off += a[p * d + q] * a[p * d + q];
        if (off <= 1e-24 * total) break;

        for (size_t p = 0; p < d; ++p) {
            for (size_t q = p + 1; q < d; ++q) {
                const double apq = a[p * d + q];
                if (apq == 0.0) continue;

                // the rotation in the (p, q) plane that zeroes a[p][q]: t = tan(angle),
                // the smaller root of t^2 + 2 theta t - 1 = 0 (Numerical Recipes 11.1)
                const double theta = (a[q * d + q] - a[p * d + p]) / (2.0 * apq);
                const double t = std::abs(theta) > 1e150 ? 1.0 / (2.0 * theta)
                                                         : (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;

                // a = J^T a J, and vectors = vectors J, with J the identity except
                // J[p][p] = J[q][q] = c, J[p][q] = s, J[q][p] = -s
                for (size_t k = 0; k < d; ++k) {
                    const double akp = a[k * d + p], akq = a[k * d + q];
                    a[k * d + p] = c * akp - s * akq;
                    a[k * d + q] = s * akp + c * akq;
                }
                for (size_t k = 0; k < d; ++k) {
                    const double apk = a[p * d + k], aqk = a[q * d + k];
                    a[p * d + k] = c * apk - s * aqk;
                    a[q * d + k] = s * apk + c * aqk;
                }
                for (size_t k = 0; k < d; ++k) {
                    const double vkp = vectors[k * d + p], vkq = vectors[k * d + q];
                    vectors[k * d + p] = c * vkp - s * vkq;
                    vectors[k * d + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    values.resize(d);
    for (size_t i = 0; i < d; ++i) values[i] = a[i * d + i];
}

vector<float> balanced_pca_rotation(span<const float> vecs, size_t n, size_t d, size_t m) {
    if (m == 0 || d % m != 0) throw std::invalid_argument("dim must be divisible by num_subspaces");
    if (n == 0) throw std::invalid_argument("training set must not be empty");

    vector<double> mean(d, 0.0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) mean[j] += vecs[i * d + j];
    for (double& x : mean) x /= static_cast<double>(n);

    // covariance in double, accumulated over blocks of centered rows (upper triangle)
    vector<double> cov(d * d, 0.0);
    constexpr size_t kBlock = 4096;
    vector<double> block(std::min(n, kBlock) * d);
    for (size_t start = 0; start < n; start += kBlock) {
        const size_t rows = std::min(kBlock, n - start);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < d; ++j) block[i * d + j] = vecs[(start + i) * d + j] - mean[j];
#ifdef VECENGINE_USE_BLAS
        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, static_cast<int>(d), static_cast<int>(rows), 1.0,
                    block.data(), static_cast<int>(d), 1.0, cov.data(), static_cast<int>(d));
#else
        for (size_t i = 0; i < rows; ++i) {
            const double* x = block.data() + i * d;
            for (size_t p = 0; p < d; ++p)
                for (size_t q = p; q < d; ++q) cov[p * d + q] += x[p] * x[q];
        }
#endif
    }
    const double scale = n > 1 ? 1.0 / static_cast<double>(n - 1) : 1.0;
    for (size_t p = 0; p < d; ++p) {
        for (size_t q = p; q < d; ++q) {
            cov[p * d + q] *= scale;
            cov[q * d + p] = cov[p * d + q];
        }
    }

    vector<double> values, vectors;
    symmetric_eigen(std::move(cov), d, values, vectors);

    // eigenvalue allocation: largest first, each into the subspace with room
    // whose product of eigenvalues (sum of logs) is smallest so far
    vector<size_t> order(d);
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return values[a] > values[b]; });

    const size_t width = d / m;
    vector<size_t> filled(m, 0);
    vector<double> log_sum(m, 0.0);
    vector<float> r(d * d);
    for (size_t e : order) {
        size_t best = m;
        for (size_t b = 0; b < m; ++b)
            if (filled[b] < width && (best == m || log_sum[b] < log_sum[best])) best = b;
        log_sum[best] += std::log(std::max(values[e], 1e-12));

        const size_t col = best * width + filled[best]++;
        for (size_t row = 0; row < d; ++row) r[row * d + col] = static_cast<float>(vectors[row * d + e]);
    }
    return r;
}

bool is_orthogonal(span<const float> r, size_t d) {
    for (size_t a = 0; a < d; ++a) {
        for (size_t b = a; b < d; ++b) {
            double dot = 0.0;
            for (size_t k = 0; k < d; ++k) dot += double(r[k * d + a]) * r[k * d + b];
            if (std::abs(dot - (a == b ? 1.0 : 0.0)) > 1e-3) return false;
        }
    }
    return true;
}

// Column panel widths: kPanel, then 8 for the remainder, then 1. A panel of
// width w holds R[k][j..j+w) for k = 0..d-1, contiguous, so rotating streams
// it in order with w running sums in registers.
constexpr size_t kPanel = 32;

template <typename F>
static void for_each_panel(size_t d, F&& f) {
    size_t j = 0;
    for (; j + kPanel <= d; j += kPanel) f(std::integral_constant<size_t, kPanel>{}, j);
    for (; j + 8 <= d; j += 8) f(std::integral_constant<size_t, 8>{}, j);
    for (; j < d; ++j) f(std::integral_constant<size_t, 1>{}, j);
}

vector<float> pack_rotation(span<const float> r, size_t d) {
    vector<float> panels(d * d);
    float* p = panels.data();
    for_each_panel(d, [&](auto w, size_t j) {
        for (size_t k = 0; k < d; ++k, p += w) std::copy_n(r.data() + k * d + j, w, p);
    });
    return panels;
}

void rotate(span<const float> panels, const float* in, size_t n, size_t d, float* out) {
    // each output is fma(x[k], R[k][j], sum) for k = 0..d-1 in order, spelled
    // out with std::fma so no compiler can fuse, split or reorder it: a vector
    // rotates to the same floats on every platform (saved indexes hold rotated codes)
    constexpr size_t kRows = 64;
    parallel_for((n + kRows - 1) / kRows, [&](size_t block) {
        const size_t end = std::min(n, (block + 1) * kRows);
        for (size_t i = block * kRows; i < end; ++i) {
            const float* x = in + i * d;
            const float* p = panels.data();
            for_each_panel(d, [&](auto w, size_t j) {
                float acc[decltype(w)::value] = {};
                for (size_t k = 0; k < d; ++k, p += w) {
                    const float xk = x[k];
                    for (size_t c = 0; c < w; ++c) acc[c] = std::fma(xk, p[c], acc[c]);
                }
                std::copy_n(acc, w, out + i * d + j);
            });
        }
    });
}

} // namespace vecengine::detail
