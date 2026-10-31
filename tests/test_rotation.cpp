#include <catch2/catch_test_macros.hpp>

#include "load_dense.hpp"

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"
#include "rotation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vecengine;
using namespace vecengine::detail;

namespace {

std::vector<float> rot_gaussian(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = g(rng);
    return v;
}

// A random orthogonal d x d matrix: the eigenvectors of a random symmetric one.
std::vector<float> random_orthogonal(size_t d, unsigned seed) {
    const auto b = rot_gaussian(d * d, 1, seed);
    std::vector<double> a(d * d);
    for (size_t i = 0; i < d; ++i)
        for (size_t j = 0; j < d; ++j) a[i * d + j] = b[i * d + j] + b[j * d + i];
    std::vector<double> values, vectors;
    symmetric_eigen(a, d, values, vectors);
    return {vectors.begin(), vectors.end()};
}

// Data that plain PQ encodes wastefully: dimension j + d/2 is a near-copy of
// dimension j, so every value is quantized twice, in two different subspaces.
// The rotation folds each pair into one high-variance and one near-constant
// direction. (A random rotation of a steep Gaussian spectrum is not such a
// case: there the balanced rotation is no better than plain PQ.)
std::vector<float> correlated(size_t n, size_t d, unsigned seed) {
    const size_t h = d / 2;
    const auto z = rot_gaussian(n, h, seed);
    const auto noise = rot_gaussian(n, h, seed + 1);
    std::vector<float> x(n * d);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < h; ++j) {
            x[i * d + j] = z[i * h + j];
            x[i * d + h + j] = z[i * h + j] + 0.05f * noise[i * h + j];
        }
    return x;
}

double rot_recall(const std::vector<std::vector<Neighbor>>& got, const std::vector<std::vector<Neighbor>>& truth) {
    double sum = 0;
    for (size_t q = 0; q < truth.size(); ++q) {
        std::set<size_t> ids;
        for (const auto& nb : truth[q]) ids.insert(nb.index);
        size_t hit = 0;
        for (const auto& nb : got[q]) hit += ids.count(nb.index);
        sum += static_cast<double>(hit) / truth[q].size();
    }
    return sum / truth.size();
}

void rot_require_identical(const std::vector<std::vector<Neighbor>>& a, const std::vector<std::vector<Neighbor>>& b) {
    REQUIRE(a.size() == b.size());
    for (size_t q = 0; q < a.size(); ++q) {
        REQUIRE(a[q].size() == b[q].size());
        for (size_t i = 0; i < a[q].size(); ++i) {
            REQUIRE(a[q][i].index == b[q][i].index);
            REQUIRE(a[q][i].distance == b[q][i].distance);
        }
    }
}

// Max |(M^T M - I)_ab| over a d x d matrix.
double orthogonality_error(const std::vector<double>& m, size_t d) {
    double worst = 0;
    for (size_t a = 0; a < d; ++a)
        for (size_t b = 0; b < d; ++b) {
            double dot = 0;
            for (size_t k = 0; k < d; ++k) dot += m[k * d + a] * m[k * d + b];
            worst = std::max(worst, std::abs(dot - (a == b ? 1.0 : 0.0)));
        }
    return worst;
}

// A trained IVF-PQ index, rotated or not, not yet filled.
std::unique_ptr<IVFIndex> rot_ivf(bool rotate, const std::vector<float>& data, size_t n, size_t d, size_t m,
                                  Metric metric = Metric::L2) {
    auto ivf = std::make_unique<IVFIndex>(d, 100, 100, false, metric);
    ivf->enable_pq(m, 256, false, rotate);
    ivf->train(data, n, 25, 1);
    return ivf;
}

} // namespace

TEST_CASE("symmetric_eigen decomposes symmetric matrices", "[rotation]") {
    SECTION("random symmetric matrix: Q diag(values) Q^T rebuilds it and Q is orthogonal") {
        const size_t d = 24;
        const auto b = rot_gaussian(d * d, 1, 1);
        std::vector<double> a(d * d);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j) a[i * d + j] = b[i * d + j] + b[j * d + i];

        std::vector<double> values, q;
        symmetric_eigen(a, d, values, q);
        REQUIRE(values.size() == d);
        REQUIRE(orthogonality_error(q, d) < 1e-10);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j) {
                double rebuilt = 0;
                for (size_t k = 0; k < d; ++k) rebuilt += q[i * d + k] * values[k] * q[j * d + k];
                REQUIRE(std::abs(rebuilt - a[i * d + j]) < 1e-9);
            }
    }

    SECTION("a diagonal matrix is its own decomposition") {
        const std::vector<double> a = {3, 0, 0, 0, -1, 0, 0, 0, 2};
        std::vector<double> values, q;
        symmetric_eigen(a, 3, values, q);
        REQUIRE(values == std::vector<double>{3, -1, 2});
        REQUIRE(q == std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
    }

    SECTION("2x2: [[2, 1], [1, 2]] has eigenvalues 1 and 3 along the diagonals") {
        std::vector<double> values, q;
        symmetric_eigen({2, 1, 1, 2}, 2, values, q);
        std::vector<double> sorted = values;
        std::sort(sorted.begin(), sorted.end());
        REQUIRE(std::abs(sorted[0] - 1) < 1e-12);
        REQUIRE(std::abs(sorted[1] - 3) < 1e-12);
        for (size_t k = 0; k < 2; ++k) REQUIRE(std::abs(std::abs(q[k]) - std::sqrt(0.5)) < 1e-12);
    }
}

TEST_CASE("balanced_pca_rotation is orthogonal, decorrelates and balances subspaces", "[rotation]") {
    const size_t n = 20000, d = 32, m = 8, w = d / m;
    const auto x = correlated(n, d, 2);
    const auto r = balanced_pca_rotation(x, n, d, m);
    REQUIRE(r.size() == d * d);
    REQUIRE(is_orthogonal(r, d));

    std::vector<float> y(n * d);
    rotate(pack_rotation(r, d), x.data(), n, d, y.data());
    std::vector<double> mean(d, 0), cov(d * d, 0);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) mean[j] += y[i * d + j] / n;
    for (size_t i = 0; i < n; ++i)
        for (size_t a = 0; a < d; ++a)
            for (size_t b = 0; b < d; ++b) cov[a * d + b] += (y[i * d + a] - mean[a]) * (y[i * d + b] - mean[b]) / (n - 1);

    // decorrelated: the rotated covariance is diagonal up to float rounding
    double off = 0, total = 0;
    for (size_t a = 0; a < d; ++a)
        for (size_t b = 0; b < d; ++b) {
            total += cov[a * d + b] * cov[a * d + b];
            if (a != b) off += cov[a * d + b] * cov[a * d + b];
        }
    REQUIRE(off / total < 1e-8);

    // balanced: each subspace's product of variances is closer to the
    // others' than with the axes in sorted order (largest variances together)
    std::vector<double> var(d);
    for (size_t a = 0; a < d; ++a) var[a] = cov[a * d + a];
    std::vector<double> sorted = var;
    std::sort(sorted.rbegin(), sorted.rend());
    auto spread = [&](const std::vector<double>& v) {
        double lo = INFINITY, hi = -INFINITY;
        for (size_t s = 0; s < m; ++s) {
            double logs = 0;
            for (size_t j = 0; j < w; ++j) logs += std::log(v[s * w + j]);
            lo = std::min(lo, logs);
            hi = std::max(hi, logs);
        }
        return hi - lo;
    };
    REQUIRE(spread(var) < 0.5 * spread(sorted));

    REQUIRE_THROWS_AS(balanced_pca_rotation(x, n, d, 5), std::invalid_argument);
}

TEST_CASE("rotate multiplies rows by R and preserves distances", "[rotation]") {
    const size_t n = 300, d = 20;
    const auto x = rot_gaussian(n, d, 3);
    const auto r = random_orthogonal(d, 4);
    std::vector<float> y(n * d);
    rotate(pack_rotation(r, d), x.data(), n, d, y.data());

    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < d; ++j) {
            double naive = 0;
            for (size_t k = 0; k < d; ++k) naive += double(x[i * d + k]) * r[k * d + j];
            REQUIRE(std::abs(y[i * d + j] - naive) < 1e-4);
        }
    for (size_t i = 1; i < n; ++i) {
        const float before = l2_distance(x.data(), x.data() + i * d, d);
        const float after = l2_distance(y.data(), y.data() + i * d, d);
        REQUIRE(std::abs(before - after) < 1e-3f * std::max(1.0f, before));
    }
}

TEST_CASE("Rotated IVF-PQ beats plain IVF-PQ on correlated data", "[rotation][ivf]") {
    const size_t n = 20000, d = 32, m = 8, nq = 200, k = 10;
    const auto data = correlated(n, d, 5);
    const auto queries = correlated(nq, d, 6);
    FlatIndex flat(d);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    auto plain = rot_ivf(false, data, n, d, m);
    plain->add_batch(data, n);
    auto rotated = rot_ivf(true, data, n, d, m);
    rotated->add_batch(data, n);
    REQUIRE_FALSE(plain->pq_rotate());
    REQUIRE(rotated->pq_rotate());

    const double before = rot_recall(plain->query_batch(queries, nq, k), truth);
    const double after = rot_recall(rotated->query_batch(queries, nq, k), truth);
    CAPTURE(before, after);
    REQUIRE(after >= before + 0.1);
}

TEST_CASE("Rotated IVF-PQ keeps distances, batch results and file round trips", "[rotation][ivf]") {
    const size_t n = 5000, d = 16, m = 8, nq = 50, k = 10;
    const auto data = correlated(n, d, 7);
    const auto queries = correlated(nq, d, 8);

    SECTION("re-ranking returns exact distances, so the rotation keeps the geometry") {
        auto ivf = rot_ivf(true, data, n, d, m);
        RefineIndex refine(*ivf, 50);
        refine.add_batch(data, n);
        FlatIndex flat(d);
        flat.add_batch(data, n);
        const auto got = refine.query_batch(queries, nq, k);
        REQUIRE(rot_recall(got, flat.query_batch(queries, nq, k)) >= 0.99);
        for (size_t q = 0; q < nq; ++q)
            for (const auto& nb : got[q]) {
                const float exact = l2_distance(queries.data() + q * d, data.data() + nb.index * d, d);
                REQUIRE(std::abs(nb.distance - exact) <= 1e-4f * std::max(1.0f, exact));
            }
    }

    SECTION("query matches query_batch, filtered and unfiltered (no double rotation)") {
        auto ivf = rot_ivf(true, data, n, d, m);
        ivf->add_batch(data, n);
        std::vector<int64_t> allowed;
        for (int64_t id = 0; id < static_cast<int64_t>(n); id += 3) allowed.push_back(id);
        const auto batch = ivf->query_batch(queries, nq, k);
        const auto batch_filtered = ivf->query_batch(queries, nq, k, allowed);
        for (size_t q = 0; q < nq; ++q) {
            const std::span<const float> one(queries.data() + q * d, d);
            // a near-tie between coarse centroids may differ in the batched path; with
            // nprobe = nlist every list is scanned, so results match exactly
            rot_require_identical({ivf->query(one, k)}, {batch[q]});
            rot_require_identical({ivf->query(one, k, allowed)}, {batch_filtered[q]});
        }
    }

    SECTION("cosine: the rotation is learned on unit vectors") {
        FlatIndex flat(d, 1024, false, Metric::Cosine);
        flat.add_batch(data, n);
        auto ivf = rot_ivf(true, data, n, d, m, Metric::Cosine);
        ivf->add_batch(data, n);
        const auto got = ivf->query_batch(queries, nq, k);
        REQUIRE(rot_recall(got, flat.query_batch(queries, nq, k)) >= 0.85);
        for (const auto& row : got)
            for (const auto& nb : row) REQUIRE(nb.distance <= 2.05f);
    }

    SECTION("save/load and compact give identical results") {
        auto ivf = rot_ivf(true, data, n, d, m);
        ivf->add_batch(data, n);
        std::vector<int64_t> dead;
        for (int64_t id = 0; id < static_cast<int64_t>(n); id += 4) dead.push_back(id);
        ivf->remove(dead);
        const auto before = ivf->query_batch(queries, nq, k);

        std::stringstream s;
        ivf->save(s);
        auto loaded = load_dense(s);
        REQUIRE(dynamic_cast<IVFIndex&>(*loaded).pq_rotate());
        rot_require_identical(loaded->query_batch(queries, nq, k), before);

        REQUIRE(ivf->compact() == dead.size());
        rot_require_identical(ivf->query_batch(queries, nq, k), before);
    }

    SECTION("an untrained rotated index round-trips and trains after loading") {
        IVFIndex ivf(d, 100, 100);
        ivf.enable_pq(m, 256, false, true);
        std::stringstream s;
        ivf.save(s);
        auto loaded = load_dense(s);
        auto& back = dynamic_cast<IVFIndex&>(*loaded);
        REQUIRE(back.pq_rotate());
        back.train(data, n, 25, 1);
        back.add_batch(data, n);
        REQUIRE(back.size() == n);
    }

    SECTION("make_index passes IndexOptions::pq_rotate") {
        IndexOptions opts;
        opts.nlist = 100;
        opts.pq_subspaces = m;
        opts.pq_rotate = true;
        auto index = make_index(IndexType::IVF, d, opts);
        REQUIRE(dynamic_cast<IVFIndex&>(*index).pq_rotate());
    }

    SECTION("rotation must be requested before training") {
        IVFIndex ivf(d, 100);
        ivf.train(data, n, 25, 1);
        REQUIRE_THROWS_AS(ivf.enable_pq(m, 256, false, true), std::logic_error);
    }
}

TEST_CASE("Rotated IVF-PQ: load rejects bad rotation data", "[rotation][ivf][serialize]") {
    const size_t n = 2000, d = 16, m = 8;
    const auto data = correlated(n, d, 9);
    auto ivf = rot_ivf(true, data, n, d, m);
    ivf->add_batch(data, n);
    std::stringstream s;
    ivf->save(s);
    const std::string bytes = s.str();

    // the fast-scan byte (0), the rotate flag (1), then the matrix length d * d
    const uint64_t len = d * d;
    std::string marker = std::string("\x00\x01", 2) + std::string(reinterpret_cast<const char*>(&len), sizeof(len));
    const size_t at = bytes.find(marker);
    REQUIRE(at != std::string::npos);
    REQUIRE(bytes.find(marker, at + 1) == std::string::npos);
    const size_t flag_at = at + 1, matrix_at = at + 2 + sizeof(len);

    auto rejects = [](const std::string& b) {
        std::stringstream t(b);
        REQUIRE_THROWS_AS(load_dense(t), std::runtime_error);
    };

    std::string bad_flag = bytes;
    bad_flag[flag_at] = 2;
    rejects(bad_flag);

    // rotation off but a matrix present
    std::string flag_off = bytes;
    flag_off[flag_at] = 0;
    rejects(flag_off);

    // trained but no matrix
    std::string no_matrix = bytes;
    const uint64_t zero = 0;
    std::memcpy(no_matrix.data() + flag_at + 1, &zero, sizeof(zero));
    no_matrix.erase(matrix_at, d * d * sizeof(float));
    rejects(no_matrix);

    // not orthogonal: one entry doubled
    std::string skewed = bytes;
    float entry;
    std::memcpy(&entry, skewed.data() + matrix_at, sizeof(entry));
    entry = entry * 2.0f + 0.5f;
    std::memcpy(skewed.data() + matrix_at, &entry, sizeof(entry));
    rejects(skewed);

    rejects(bytes.substr(0, matrix_at + 10));
}
