#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/pq.hpp"

#include <cmath>
#include <vector>
#include <random>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

static constexpr float kEps = 1e-4f;

// Generates `n` random D-dim vectors, flattened row-major.
static std::vector<float> random_vectors(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (float& v : data) v = dist(rng);
    return data;
}

// ---------------------------------------------------------------------------
// L2 distance
// ---------------------------------------------------------------------------

TEST_CASE("l2_distance_scalar: identical vectors", "[l2][scalar]") {
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
    REQUIRE_THAT(vecengine::l2_distance_scalar(v.data(), v.data(), v.size()),
                 WithinAbs(0.0f, kEps));
}

TEST_CASE("l2_distance_scalar: orthogonal unit vectors", "[l2][scalar]") {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    // squared L2 = (1-0)^2 + (0-1)^2 = 2
    REQUIRE_THAT(vecengine::l2_distance_scalar(a.data(), b.data(), a.size()),
                 WithinAbs(2.0f, kEps));
}

TEST_CASE("l2_distance_scalar: known values", "[l2][scalar]") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    // (3^2 + 3^2 + 3^2) = 27
    REQUIRE_THAT(vecengine::l2_distance_scalar(a.data(), b.data(), a.size()),
                 WithinAbs(27.0f, kEps));
}

TEST_CASE("FlatIndex: query returns self at rank 0", "[index]") {
    vecengine::FlatIndex idx(3);
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    idx.add(a);
    idx.add(b);
    auto res = idx.query(a,2);
    REQUIRE(res[0].index == 0);
    REQUIRE(res[1].index == 1);
}

TEST_CASE("FlatIndex: query returns more than k", "[index]") {
    vecengine::FlatIndex idx(3);
    std::vector<float> a = {0.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 0.0f};
    idx.add(a);
    idx.add(b);
    auto res = idx.query(a, 5);
    REQUIRE(res.size() == 2);
}

TEST_CASE("FlatIndex: correct distance ordering", "[index]") {
    vecengine::FlatIndex idx(3);
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {2.0f, 0.0f, 0.0f}; // dist from a = 1
    std::vector<float> c = {0.0f, 10.0f, 0.0f}; // dist from a = 101
    idx.add(a);
    idx.add(b);
    idx.add(c);
    auto res = idx.query(a, 3);
    REQUIRE(res[0].index == 0); // closest is itself
    REQUIRE(res[1].index == 1); // second is b
    REQUIRE(res[2].index == 2); // farthest is c
}

TEST_CASE("FlatIndex: distance values are correct", "[index]") {
    vecengine::FlatIndex idx(3);
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {2.0f, 0.0f, 0.0f}; // squared L2 from a = 1
    idx.add(a);
    idx.add(b);
    auto res = idx.query(a, 2);
    REQUIRE_THAT(res[0].distance, WithinAbs(0.0f, kEps)); // self
    REQUIRE_THAT(res[1].distance, WithinAbs(1.0f, kEps)); // (2-1)^2 = 1
}

TEST_CASE("FlatIndex: query_batch matches per-query results", "[index][batch]") {
    vecengine::FlatIndex idx(3);
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {2.0f, 0.0f, 0.0f}; // dist from a = 1
    std::vector<float> c = {0.0f, 10.0f, 0.0f}; // dist from a = 101
    idx.add(a);
    idx.add(b);
    idx.add(c);

    // Two queries packed into one flattened row-major buffer: a, then b.
    std::vector<float> queries = {1.0f, 0.0f, 0.0f,
                                   2.0f, 0.0f, 0.0f};
    auto batch_res = idx.query_batch(queries, 2, 3);
    REQUIRE(batch_res.size() == 2);

    auto expected_a = idx.query(a, 3);
    auto expected_b = idx.query(b, 3);

    REQUIRE(batch_res[0].size() == expected_a.size());
    for (size_t i = 0; i < expected_a.size(); ++i) {
        REQUIRE(batch_res[0][i].index == expected_a[i].index);
        REQUIRE_THAT(batch_res[0][i].distance, WithinAbs(expected_a[i].distance, kEps));
    }

    REQUIRE(batch_res[1].size() == expected_b.size());
    for (size_t i = 0; i < expected_b.size(); ++i) {
        REQUIRE(batch_res[1][i].index == expected_b[i].index);
        REQUIRE_THAT(batch_res[1][i].distance, WithinAbs(expected_b[i].distance, kEps));
    }
}

TEST_CASE("FlatIndex: query_batch preserves query order", "[index][batch]") {
    vecengine::FlatIndex idx(3);
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};
    idx.add(a);
    idx.add(b);

    // Query b first, then a - results should follow query order, not add order.
    std::vector<float> queries = {0.0f, 1.0f, 0.0f,
                                   1.0f, 0.0f, 0.0f};
    auto batch_res = idx.query_batch(queries, 2, 1);
    REQUIRE(batch_res.size() == 2);
    REQUIRE(batch_res[0][0].index == 1); // b is closest to first query (b itself)
    REQUIRE(batch_res[1][0].index == 0); // a is closest to second query (a itself)
}


// ---------------------------------------------------------------------------
// IVFIndex
// ---------------------------------------------------------------------------

TEST_CASE("IVFIndex: constructor rejects out-of-range nlist", "[ivf]") {
    REQUIRE_THROWS_AS(vecengine::IVFIndex(8, 99, 10), std::invalid_argument);
    REQUIRE_THROWS_AS(vecengine::IVFIndex(8, 65536, 10), std::invalid_argument);
    REQUIRE_NOTHROW(vecengine::IVFIndex(8, 100, 10));
}

TEST_CASE("IVFIndex: train rejects fewer training vectors than nlist", "[ivf]") {
    vecengine::IVFIndex idx(8, 100, 10);
    auto data = random_vectors(50, 8, /*seed=*/1);
    REQUIRE_THROWS_AS(idx.train(data, 50, 5), std::invalid_argument);
}

TEST_CASE("IVFIndex: query returns self at rank 0", "[ivf]") {
    const size_t dim = 16, n = 2000, nlist = 100, nprobe = 10;
    auto data = random_vectors(n, dim, /*seed=*/42);

    vecengine::IVFIndex idx(dim, nlist, nprobe);
    idx.train(data, n, 25);
    for (size_t i = 0; i < n; ++i)
        idx.add(std::span<const float>(data.data() + i * dim, dim));

    REQUIRE(idx.size() == n);

    std::span<const float> query(data.data() + 42 * dim, dim);
    auto res = idx.query(query, 5);
    REQUIRE(res.size() == 5);
    REQUIRE(res[0].index == 42);
    REQUIRE_THAT(res[0].distance, WithinAbs(0.0f, kEps));
}

TEST_CASE("IVFIndex: correct distance ordering", "[ivf]") {
    const size_t dim = 16, n = 2000, nlist = 100, nprobe = 20;
    auto data = random_vectors(n, dim, /*seed=*/7);

    vecengine::IVFIndex idx(dim, nlist, nprobe);
    idx.train(data, n, 25);
    for (size_t i = 0; i < n; ++i)
        idx.add(std::span<const float>(data.data() + i * dim, dim));

    std::span<const float> query(data.data() + 100 * dim, dim);
    auto res = idx.query(query, 10);

    for (size_t i = 1; i < res.size(); ++i)
        REQUIRE(res[i].distance >= res[i - 1].distance);
}

TEST_CASE("IVFIndex: query_batch matches per-query results", "[ivf][batch]") {
    const size_t dim = 16, n = 1000, nlist = 100, nprobe = 15;
    auto data = random_vectors(n, dim, /*seed=*/99);

    vecengine::IVFIndex idx(dim, nlist, nprobe);
    idx.train(data, n, 25);
    for (size_t i = 0; i < n; ++i)
        idx.add(std::span<const float>(data.data() + i * dim, dim));

    // Two queries: vectors 10 and 200 from the training set.
    std::vector<float> queries(2 * dim);
    std::copy_n(data.data() + 10 * dim, dim, queries.data());
    std::copy_n(data.data() + 200 * dim, dim, queries.data() + dim);

    auto batch_res = idx.query_batch(queries, 2, 5);
    REQUIRE(batch_res.size() == 2);

    auto expected_0 = idx.query(std::span<const float>(queries.data(), dim), 5);
    auto expected_1 = idx.query(std::span<const float>(queries.data() + dim, dim), 5);

    REQUIRE(batch_res[0].size() == expected_0.size());
    for (size_t i = 0; i < expected_0.size(); ++i) {
        REQUIRE(batch_res[0][i].index == expected_0[i].index);
        REQUIRE_THAT(batch_res[0][i].distance, WithinAbs(expected_0[i].distance, kEps));
    }

    REQUIRE(batch_res[1].size() == expected_1.size());
    for (size_t i = 0; i < expected_1.size(); ++i) {
        REQUIRE(batch_res[1][i].index == expected_1[i].index);
        REQUIRE_THAT(batch_res[1][i].distance, WithinAbs(expected_1[i].distance, kEps));
    }
}

// ---------------------------------------------------------------------------
// PQCodebook
// ---------------------------------------------------------------------------

TEST_CASE("PQCodebook: constructor rejects invalid config", "[pq]") {
    REQUIRE_THROWS_AS(vecengine::PQCodebook(10, 4, 16), std::invalid_argument);  // 10 % 4 != 0
    REQUIRE_THROWS_AS(vecengine::PQCodebook(8, 4, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(vecengine::PQCodebook(8, 4, 257), std::invalid_argument);
    REQUIRE_NOTHROW(vecengine::PQCodebook(8, 4, 256));
}

TEST_CASE("PQCodebook: train rejects fewer vectors than centroids", "[pq]") {
    vecengine::PQCodebook pq(8, 4, 16);
    auto data = random_vectors(10, 8, /*seed=*/1);
    REQUIRE_THROWS_AS(pq.train(data, 10, 5), std::invalid_argument);
}

TEST_CASE("PQCodebook: encode returns one in-range code per subspace", "[pq]") {
    const size_t dim = 16, m = 4, K = 16, n = 500;
    auto data = random_vectors(n, dim, /*seed=*/3);
    vecengine::PQCodebook pq(dim, m, K);
    pq.train(data, n, 10);

    for (size_t i = 0; i < 20; ++i) {
        auto code = pq.encode(std::span<const float>(data.data() + i * dim, dim));
        REQUIRE(code.size() == m);
        for (uint8_t c : code) REQUIRE(c < K);
    }
}

// With n == K every training point becomes its own centroid, so encoding is
// lossless and ADC must equal the exact distance.
TEST_CASE("PQCodebook: ADC is exact when vectors are centroids", "[pq][adc]") {
    const size_t dim = 8, m = 4, K = 16;
    auto data = random_vectors(K, dim, /*seed=*/5);
    vecengine::PQCodebook pq(dim, m, K);
    pq.train(data, K, 10);

    auto query = random_vectors(1, dim, /*seed=*/6);
    auto table = pq.compute_adc_table(query);
    REQUIRE(table.size() == m * K);

    for (size_t i = 0; i < K; ++i) {
        const float* x = data.data() + i * dim;
        auto code = pq.encode(std::span<const float>(x, dim));
        float exact = vecengine::l2_distance(query.data(), x, dim);
        REQUIRE_THAT(pq.distance_adc(table, code), WithinAbs(exact, kEps));
    }
}

TEST_CASE("PQCodebook: SDC is zero on self and symmetric", "[pq][sdc]") {
    const size_t dim = 16, m = 4, K = 16, n = 500;
    auto data = random_vectors(n, dim, /*seed=*/8);
    vecengine::PQCodebook pq(dim, m, K);
    pq.train(data, n, 10);

    auto a = pq.encode(std::span<const float>(data.data(), dim));
    auto b = pq.encode(std::span<const float>(data.data() + dim, dim));
    REQUIRE_THAT(pq.distance_sdc(a, a), WithinAbs(0.0f, kEps));
    REQUIRE_THAT(pq.distance_sdc(a, b), WithinAbs(pq.distance_sdc(b, a), kEps));
}

TEST_CASE("PQCodebook: ADC approximates true distance better than SDC", "[pq][adc][sdc]") {
    const size_t dim = 32, m = 8, K = 64, n = 4000;
    auto data = random_vectors(n, dim, /*seed=*/11);
    vecengine::PQCodebook pq(dim, m, K);
    pq.train(data, n, 15);

    auto query = random_vectors(1, dim, /*seed=*/12);
    auto table = pq.compute_adc_table(query);
    auto query_code = pq.encode(query);

    double adc_err = 0.0, sdc_err = 0.0;
    for (size_t i = 0; i < 200; ++i) {
        const float* x = data.data() + i * dim;
        auto code = pq.encode(std::span<const float>(x, dim));
        float exact = vecengine::l2_distance(query.data(), x, dim);
        adc_err += std::abs(pq.distance_adc(table, code) - exact);
        sdc_err += std::abs(pq.distance_sdc(query_code, code) - exact);
    }
    REQUIRE(adc_err < sdc_err);
}

// Recall@10 against exact FlatIndex results. Measured on this data: plain IVF
// ~0.80, ADC ~0.63, SDC ~0.48; thresholds leave headroom for k-means randomness.
TEST_CASE("IVFIndex+PQ: recall vs FlatIndex, ADC beats SDC", "[ivf][pq]") {
    const size_t dim = 32, n = 5000, k = 10, num_queries = 100;
    auto data = random_vectors(n, dim, /*seed=*/42);

    vecengine::FlatIndex flat(dim);
    vecengine::IVFIndex idx(dim, 100, 20);
    idx.enable_pq(8, 256);
    idx.train(data, n, 15);
    for (size_t i = 0; i < n; ++i) {
        std::span<const float> v(data.data() + i * dim, dim);
        flat.add(v);
        idx.add(v);
    }

    auto recall = [&] {
        size_t hits = 0;
        for (size_t q = 0; q < num_queries; ++q) {
            std::span<const float> query(data.data() + q * 37 * dim, dim);
            auto truth = flat.query(query, k);
            for (const auto& r : idx.query(query, k))
                for (const auto& t : truth)
                    if (r.index == t.index) { ++hits; break; }
        }
        return static_cast<double>(hits) / (num_queries * k);
    };

    REQUIRE(idx.pq_distance() == vecengine::PQDistance::ADC);
    double adc = recall();
    idx.set_pq_distance(vecengine::PQDistance::SDC);
    double sdc = recall();

    REQUIRE(adc > 0.5);
    REQUIRE(sdc > 0.3);
    REQUIRE(adc > sdc);
}

#if defined(__AVX2__)
TEST_CASE("l2_distance_avx2 matches scalar", "[l2][avx2]") {
    std::vector<float> a(64), b(64);
    for (int i = 0; i < 64; ++i) { a[i] = static_cast<float>(i); b[i] = static_cast<float>(i * 2 + 1); }

    float ref  = vecengine::l2_distance_scalar(a.data(), b.data(), a.size());
    float fast = vecengine::l2_distance_avx2  (a.data(), b.data(), a.size());
    REQUIRE_THAT(fast, WithinRel(ref, 1e-4f));
}

TEST_CASE("l2_distance_avx2: non-multiple-of-8 length", "[l2][avx2]") {
    std::vector<float> a(13, 1.0f), b(13, 3.0f);
    float ref  = vecengine::l2_distance_scalar(a.data(), b.data(), a.size());
    float fast = vecengine::l2_distance_avx2  (a.data(), b.data(), a.size());
    REQUIRE_THAT(fast, WithinRel(ref, 1e-4f));
}
#endif

#if defined(__ARM_NEON)
TEST_CASE("l2_distance_neon matches scalar", "[l2][neon]") {
    std::vector<float> a(64), b(64);
    for (int i = 0; i < 64; ++i) { a[i] = static_cast<float>(i); b[i] = static_cast<float>(i * 2 + 1); }

    float ref  = vecengine::l2_distance_scalar(a.data(), b.data(), a.size());
    float fast = vecengine::l2_distance_neon  (a.data(), b.data(), a.size());
    REQUIRE_THAT(fast, WithinRel(ref, 1e-4f));
}

TEST_CASE("l2_distance_neon: non-multiple-of-4 length", "[l2][neon]") {
    std::vector<float> a(13, 1.0f), b(13, 3.0f);
    float ref  = vecengine::l2_distance_scalar(a.data(), b.data(), a.size());
    float fast = vecengine::l2_distance_neon  (a.data(), b.data(), a.size());
    REQUIRE_THAT(fast, WithinRel(ref, 1e-4f));
}

TEST_CASE("cosine_distance_neon matches scalar", "[cosine][neon]") {
    std::vector<float> a(64), b(64);
    for (int i = 0; i < 64; ++i) { a[i] = std::sin(i * 0.1f); b[i] = std::cos(i * 0.1f); }

    float ref  = vecengine::cosine_distance_scalar(a.data(), b.data(), a.size());
    float fast = vecengine::cosine_distance_neon  (a.data(), b.data(), a.size());
    REQUIRE_THAT(fast, WithinRel(ref, 1e-4f));
}
#endif

// ---------------------------------------------------------------------------
// Cosine distance
// ---------------------------------------------------------------------------

TEST_CASE("cosine_distance_scalar: identical vectors", "[cosine][scalar]") {
    std::vector<float> v = {1.0f, 2.0f, 3.0f};
    REQUIRE_THAT(vecengine::cosine_distance_scalar(v.data(), v.data(), v.size()),
                 WithinAbs(0.0f, kEps));
}

TEST_CASE("cosine_distance_scalar: orthogonal vectors", "[cosine][scalar]") {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f};
    REQUIRE_THAT(vecengine::cosine_distance_scalar(a.data(), b.data(), a.size()),
                 WithinAbs(1.0f, kEps));
}

TEST_CASE("cosine_distance_scalar: opposite vectors", "[cosine][scalar]") {
    std::vector<float> a = {1.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f};
    REQUIRE_THAT(vecengine::cosine_distance_scalar(a.data(), b.data(), a.size()),
                 WithinAbs(2.0f, kEps));
}

TEST_CASE("cosine_distance_scalar: zero vector returns 1", "[cosine][scalar]") {
    std::vector<float> a = {0.0f, 0.0f, 0.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    REQUIRE_THAT(vecengine::cosine_distance_scalar(a.data(), b.data(), a.size()),
                 WithinAbs(1.0f, kEps));
}

#if defined(__AVX2__)
TEST_CASE("cosine_distance_avx2 matches scalar", "[cosine][avx2]") {
    std::vector<float> a(64), b(64);
    for (int i = 0; i < 64; ++i) { a[i] = std::sin(i * 0.1f); b[i] = std::cos(i * 0.1f); }

    float ref  = vecengine::cosine_distance_scalar(a.data(), b.data(), a.size());
    float fast = vecengine::cosine_distance_avx2  (a.data(), b.data(), a.size());
    REQUIRE_THAT(fast, WithinRel(ref, 1e-4f));
}
#endif
