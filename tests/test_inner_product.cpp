#include <catch2/catch_test_macros.hpp>

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace vecengine;

// Gaussian rows scaled by lengths spread over 0.1x to 10x, so the inner product
// ranking differs from the cosine one.
static std::vector<float> ip_random(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::uniform_real_distribution<float> log_len(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (size_t i = 0; i < n; ++i) {
        const float len = std::pow(10.0f, log_len(rng));
        for (size_t j = 0; j < dim; ++j) v[i * dim + j] = d(rng) * len;
    }
    return v;
}

static double dot_ref(const float* a, const float* b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += double(a[i]) * b[i];
    return s;
}

// The k largest inner products with q, as (-dot, slot), best first.
static std::vector<std::pair<double, size_t>> ip_truth(const std::vector<float>& data, size_t n, size_t dim, const float* q, size_t k) {
    std::vector<std::pair<double, size_t>> all(n);
    for (size_t i = 0; i < n; ++i) all[i] = {-dot_ref(q, data.data() + i * dim, dim), i};
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    all.resize(k);
    return all;
}

static void require_same(const std::vector<std::vector<Neighbor>>& a, const std::vector<std::vector<Neighbor>>& b) {
    REQUIRE(a.size() == b.size());
    for (size_t q = 0; q < a.size(); ++q) {
        REQUIRE(a[q].size() == b[q].size());
        for (size_t i = 0; i < a[q].size(); ++i) {
            REQUIRE(a[q][i].index == b[q][i].index);
            REQUIRE(a[q][i].distance == b[q][i].distance);
        }
    }
}

static std::unique_ptr<Index> roundtrip(const Index& index) {
    std::stringstream s;
    index.save(s);
    return load_index(s);
}

TEST_CASE("inner_product kernels match a double-precision reference", "[ip][distance]") {
    std::mt19937 rng(1);
    std::normal_distribution<float> d(0.0f, 1.0f);
    for (size_t n = 1; n <= 37; ++n) {   // every SIMD tail length
        std::vector<float> a(n), b(n);
        for (size_t i = 0; i < n; ++i) a[i] = d(rng), b[i] = d(rng);
        const double ref = dot_ref(a.data(), b.data(), n);
        REQUIRE(std::fabs(inner_product_scalar(a.data(), b.data(), n) - ref) < 1e-4);
        REQUIRE(std::fabs(inner_product_avx2(a.data(), b.data(), n) - ref) < 1e-4);
        REQUIRE(std::fabs(inner_product_neon(a.data(), b.data(), n) - ref) < 1e-4);
        REQUIRE(std::fabs(inner_product(std::span<const float>(a), std::span<const float>(b)) - ref) < 1e-4);
    }
}

TEST_CASE("inner product FlatIndex returns the exact ranking and -dot distances", "[ip]") {
    const size_t n = 2000, dim = 24, nq = 30, k = 10;
    auto data = ip_random(n, dim, 2);
    auto queries = ip_random(nq, dim, 3);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    REQUIRE(flat.metric() == Metric::InnerProduct);
    flat.add_batch(data, n);

    FlatIndex cosine(dim, 1024, false, Metric::Cosine);
    cosine.add_batch(data, n);
    size_t differs = 0;

    for (size_t q = 0; q < nq; ++q) {
        const float* qv = queries.data() + q * dim;
        const auto truth = ip_truth(data, n, dim, qv, k);
        const auto got = flat.query({qv, dim}, k);
        REQUIRE(got.size() == k);
        for (size_t i = 0; i < k; ++i) {
            REQUIRE(got[i].index == truth[i].second);
            REQUIRE(std::fabs(got[i].distance - truth[i].first) < 1e-3 * std::max(1.0, std::fabs(truth[i].first)));
        }
        differs += cosine.query({qv, dim}, 1)[0].index != got[0].index;
    }
    REQUIRE(differs > 0);   // vectors are not normalized: the ranking is not cosine's
}

TEST_CASE("inner product: query and query_batch agree exactly", "[ip][batch]") {
    const size_t n = 1000, dim = 16, nq = 30, k = 10;
    auto data = ip_random(n, dim, 4);
    auto queries = ip_random(nq, dim, 5);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, n);
    std::vector<std::vector<Neighbor>> single;
    for (size_t q = 0; q < nq; ++q) single.push_back(flat.query({queries.data() + q * dim, dim}, k));
    require_same(single, flat.query_batch(queries, nq, k));
}

TEST_CASE("inner product accepts zero vectors", "[ip]") {
    const size_t dim = 8;
    const std::vector<float> zero(dim, 0.0f);
    auto data = ip_random(5, dim, 6);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, 5);
    flat.add(zero);
    const auto got = flat.query(zero, 6);
    REQUIRE(got.size() == 6);
    for (const auto& nb : got) REQUIRE(nb.distance == 0.0f);
}

TEST_CASE("RefineIndex re-ranks by exact inner product", "[ip][refine]") {
    const size_t n = 1500, dim = 16, nq = 20, k = 10;
    auto data = ip_random(n, dim, 7);
    auto queries = ip_random(nq, dim, 8);
    FlatIndex base(dim, 1024, false, Metric::InnerProduct);
    RefineIndex refine(base, 3);
    REQUIRE(refine.metric() == Metric::InnerProduct);
    refine.add_batch(data, n);

    for (size_t q = 0; q < nq; ++q) {
        const float* qv = queries.data() + q * dim;
        const auto truth = ip_truth(data, n, dim, qv, k);
        const auto got = refine.query({qv, dim}, k);
        REQUIRE(got.size() == k);
        for (size_t i = 0; i < k; ++i) REQUIRE(got[i].index == truth[i].second);
    }

    SECTION("filtered search keeps only allowed ids") {
        std::vector<int64_t> allowed;
        for (int64_t id = 0; id < static_cast<int64_t>(n); id += 3) allowed.push_back(id);
        for (size_t q = 0; q < nq; ++q) {
            const auto got = refine.query({queries.data() + q * dim, dim}, k, allowed);
            REQUIRE(got.size() == k);
            for (const auto& nb : got) REQUIRE(nb.index % 3 == 0);
        }
    }
}

TEST_CASE("the inner product metric survives save/load", "[ip][serialize]") {
    const size_t n = 800, dim = 16, nq = 20, k = 10;
    auto data = ip_random(n, dim, 9);
    auto queries = ip_random(nq, dim, 10);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, n);
    FlatIndex base(dim, 1024, false, Metric::InnerProduct);
    RefineIndex refine(base, 4);
    refine.add_batch(data, n);

    for (const Index* index : {static_cast<const Index*>(&flat), static_cast<const Index*>(&refine)}) {
        auto loaded = roundtrip(*index);
        REQUIRE(loaded->metric() == Metric::InnerProduct);
        require_same(loaded->query_batch(queries, nq, k), index->query_batch(queries, nq, k));
    }
}

TEST_CASE("indexes without inner product support reject it", "[ip]") {
    REQUIRE_THROWS_AS(IVFIndex(8, 100, 10, false, Metric::InnerProduct), std::invalid_argument);
    REQUIRE_THROWS_AS(HNSWIndex(8, 16, 100, false, 1, Metric::InnerProduct), std::invalid_argument);

    IndexOptions opts;
    opts.metric = Metric::InnerProduct;
    REQUIRE(make_index(IndexType::Flat, 8, opts)->metric() == Metric::InnerProduct);
}
