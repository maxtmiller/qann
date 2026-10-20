#include <catch2/catch_test_macros.hpp>

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/metadata.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/threads.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <set>
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

// Rows around 20 Gaussian cluster centers, scaled by lengths over 0.1x to 10x,
// and queries near the same centers: directions that follow the data, as in
// real embeddings (on Gaussian rows whose directions ignore their lengths, the
// graph HNSW builds for inner product does poorly).
static void ip_clustered(size_t n, size_t nq, size_t dim, unsigned seed, std::vector<float>& data, std::vector<float>& queries) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::uniform_real_distribution<float> log_len(-1.0f, 1.0f);
    std::uniform_int_distribution<size_t> pick(0, 19);
    std::vector<float> centers(20 * dim);
    for (auto& x : centers) x = d(rng);
    auto fill = [&](std::vector<float>& out, size_t rows, bool scale) {
        out.resize(rows * dim);
        for (size_t i = 0; i < rows; ++i) {
            const float* c = centers.data() + pick(rng) * dim;
            const float len = scale ? std::pow(10.0f, log_len(rng)) : 1.0f;
            for (size_t j = 0; j < dim; ++j) out[i * dim + j] = (c[j] + 0.5f * d(rng)) * len;
        }
    };
    fill(data, n, true);
    fill(queries, nq, false);
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

// Mean fraction of each truth row's ids found in the matching result row.
static double recall(const std::vector<std::vector<Neighbor>>& got, const std::vector<std::vector<Neighbor>>& truth) {
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

// Inserts on one thread while alive, so a seeded HNSW graph is the same every time.
struct OneThread {
    OneThread() { set_num_threads(1); }
    ~OneThread() { set_num_threads(0); }
};

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

TEST_CASE("make_index passes the inner product metric", "[ip]") {
    IndexOptions opts;
    opts.metric = Metric::InnerProduct;
    for (IndexType type : {IndexType::Flat, IndexType::IVF, IndexType::HNSW})
        REQUIRE(make_index(type, 8, opts)->metric() == Metric::InnerProduct);
    opts.pq_subspaces = 4;
    REQUIRE(make_index(IndexType::IVF, 8, opts)->metric() == Metric::InnerProduct);
}

TEST_CASE("IVF-PQ inner product: SDC is rejected and no precomputed tables are built", "[ip][ivf][pq]") {
    const size_t n = 1000, dim = 16;
    std::vector<float> data, queries;
    ip_clustered(n, 1, dim, 23, data, queries);
    IVFIndex ivf(dim, 100, 10, false, Metric::InnerProduct);
    ivf.enable_pq(4);
    REQUIRE_THROWS_AS(ivf.set_pq_distance(PQDistance::SDC), std::invalid_argument);
    ivf.train(data, n, 10, 1);
    REQUIRE_FALSE(ivf.precomputed_tables());
}

TEST_CASE("IVF-PQ inner product is exact when every residual is a centroid", "[ip][ivf][pq]") {
    // 256 vectors and 256 centroids per subspace: k-means gives each residual
    // slice its own centroid, so the table sums equal -(q . x)
    const size_t n = 256, dim = 16, nq = 20, k = 10;
    std::vector<float> data, queries;
    ip_clustered(n, nq, dim, 24, data, queries);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    IVFIndex ivf(dim, 100, 100, false, Metric::InnerProduct);
    ivf.enable_pq(4, 256);
    ivf.train(data, n, 25, 1);
    ivf.add_batch(data, n);
    const auto got = ivf.query_batch(queries, nq, k);
    for (size_t q = 0; q < nq; ++q)
        for (size_t i = 0; i < k; ++i) {
            REQUIRE(got[q][i].index == truth[q][i].index);
            REQUIRE(std::fabs(got[q][i].distance - truth[q][i].distance) < 1e-3f * std::max(1.0f, std::fabs(truth[q][i].distance)));
        }
}

TEST_CASE("IVF-PQ and fast-scan find inner product neighbors under a RefineIndex", "[ip][ivf][pq]") {
    const size_t n = 5000, dim = 32, nq = 100, k = 10;
    std::vector<float> data, queries;
    ip_clustered(n, nq, dim, 25, data, queries);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    struct Config { size_t m, K; bool fast, rotate; };
    for (const Config c : {Config{8, 256, false, false}, Config{16, 16, true, false}, Config{16, 16, true, true}}) {
        CAPTURE(c.m, c.fast, c.rotate);
        IVFIndex ivf(dim, 100, 20, false, Metric::InnerProduct);
        ivf.enable_pq(c.m, c.K, c.fast, c.rotate);
        ivf.train(data, n, 10, 1);
        RefineIndex refine(ivf, 10);
        refine.add_batch(data, n);

        const auto got = refine.query_batch(queries, nq, k);
        REQUIRE(recall(got, truth) >= 0.95);
        // without re-ranking, still far better than chance
        REQUIRE(recall(ivf.query_batch(queries, nq, k), truth) >= 0.5);

        std::vector<std::vector<Neighbor>> single;
        for (size_t q = 0; q < nq; ++q) single.push_back(refine.query({queries.data() + q * dim, dim}, k));
        require_same(single, got);

        auto loaded = roundtrip(refine);
        require_same(loaded->query_batch(queries, nq, k), got);
    }
}

TEST_CASE("IVF probing every list returns the exact inner product ranking", "[ip][ivf]") {
    const size_t n = 3000, dim = 16, nq = 40, k = 10;
    std::vector<float> data, queries;
    ip_clustered(n, nq, dim, 21, data, queries);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, n);

    IVFIndex ivf(dim, 100, 100, false, Metric::InnerProduct);
    ivf.train(data, n, 10, 1);
    ivf.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);
    const auto got = ivf.query_batch(queries, nq, k);
    for (size_t q = 0; q < nq; ++q) {
        REQUIRE(got[q].size() == k);
        for (size_t i = 0; i < k; ++i) {
            REQUIRE(got[q][i].index == truth[q][i].index);
            REQUIRE(std::fabs(got[q][i].distance - truth[q][i].distance) < 1e-4f * std::max(1.0f, std::fabs(truth[q][i].distance)));
        }
    }
}

TEST_CASE("IVF finds inner product neighbors with floats and SQ", "[ip][ivf]") {
    const size_t n = 5000, dim = 32, nq = 100, k = 10;
    std::vector<float> data, queries;
    ip_clustered(n, nq, dim, 22, data, queries);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    for (int bits : {0, 8, 4}) {
        CAPTURE(bits);
        IVFIndex ivf(dim, 100, 20, false, Metric::InnerProduct);
        if (bits) ivf.enable_sq(bits);
        ivf.train(data, n, 10, 1);
        Index* index = &ivf;
        std::unique_ptr<RefineIndex> refine;
        if (bits == 4) {   // 4 bits needs re-ranking
            refine = std::make_unique<RefineIndex>(ivf, 4);
            index = refine.get();
        }
        index->add_batch(data, n);

        const auto got = index->query_batch(queries, nq, k);
        REQUIRE(recall(got, truth) >= 0.95);
        std::vector<std::vector<Neighbor>> single;
        for (size_t q = 0; q < nq; ++q) single.push_back(index->query({queries.data() + q * dim, dim}, k));
        require_same(single, got);

        std::vector<int64_t> allowed;
        for (int64_t id = 0; id < static_cast<int64_t>(n); id += 5) allowed.push_back(id);
        for (size_t q = 0; q < nq; ++q) {
            const auto filtered = index->query({queries.data() + q * dim, dim}, k, allowed);
            REQUIRE(filtered.size() == k);   // probes past nprobe until it has k allowed vectors
            for (const auto& nb : filtered) REQUIRE(nb.index % 5 == 0);
        }

        auto loaded = roundtrip(*index);
        REQUIRE(loaded->metric() == Metric::InnerProduct);
        require_same(loaded->query_batch(queries, nq, k), got);
    }
}

TEST_CASE("HNSW finds inner product neighbors at exact -dot distances", "[ip][hnsw]") {
    const size_t n = 5000, dim = 32, nq = 100, k = 10;
    std::vector<float> data, queries;
    ip_clustered(n, nq, dim, 11, data, queries);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    HNSWIndex hnsw(dim, 16, 100, false, 1, Metric::InnerProduct);
    hnsw.set_ef_search(200);
    hnsw.add_batch(data, n);
    const auto got = hnsw.query_batch(queries, nq, k);
    REQUIRE(recall(got, truth) >= 0.97);
    for (size_t q = 0; q < nq; ++q) {
        for (size_t i = 0; i < got[q].size(); ++i) {
            const double dot = dot_ref(queries.data() + q * dim, data.data() + got[q][i].index * dim, dim);
            REQUIRE(std::fabs(got[q][i].distance + dot) < 1e-3 * std::max(1.0, std::fabs(dot)));
            if (i > 0) REQUIRE(got[q][i - 1].distance <= got[q][i].distance);
        }
    }

    std::vector<std::vector<Neighbor>> single;
    for (size_t q = 0; q < nq; ++q) single.push_back(hnsw.query({queries.data() + q * dim, dim}, k));
    require_same(single, got);

    SECTION("8-bit SQ, re-ranked by a RefineIndex") {
        HNSWIndex sq(dim, 16, 100, false, 1, Metric::InnerProduct);
        sq.set_ef_search(200);
        sq.enable_sq(8);
        sq.train(data, n);
        RefineIndex refine(sq, 2);
        refine.add_batch(data, n);
        REQUIRE(recall(refine.query_batch(queries, nq, k), truth) >= 0.97);
    }
    SECTION("filtered search") {
        std::vector<int64_t> allowed;
        for (int64_t id = 0; id < static_cast<int64_t>(n); id += 2) allowed.push_back(id);
        for (size_t q = 0; q < nq; ++q)
            for (const auto& nb : hnsw.query({queries.data() + q * dim, dim}, k, allowed)) REQUIRE(nb.index % 2 == 0);
    }
}

TEST_CASE("HNSW inner product builds like L2 HNSW on the transformed vectors", "[ip][hnsw]") {
    // x -> [x, sqrt(M^2 - |x|^2)], q -> [q, 0]: the graph should be as good as
    // an L2 graph over those vectors, at every ef
    const size_t n = 5000, dim = 32, nq = 100, k = 10;
    std::vector<float> data, queries;
    ip_clustered(n, nq, dim, 12, data, queries);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    double max_sq = 0;
    for (size_t i = 0; i < n; ++i) max_sq = std::max(max_sq, dot_ref(data.data() + i * dim, data.data() + i * dim, dim));
    std::vector<float> aug(n * (dim + 1)), aug_q(nq * (dim + 1), 0.0f);
    for (size_t i = 0; i < n; ++i) {
        std::copy_n(data.data() + i * dim, dim, aug.data() + i * (dim + 1));
        aug[i * (dim + 1) + dim] = static_cast<float>(std::sqrt(std::max(0.0, max_sq - dot_ref(data.data() + i * dim, data.data() + i * dim, dim))));
    }
    for (size_t q = 0; q < nq; ++q) std::copy_n(queries.data() + q * dim, dim, aug_q.data() + q * (dim + 1));

    HNSWIndex ip(dim, 16, 100, false, 1, Metric::InnerProduct);
    ip.add_batch(data, n);
    HNSWIndex l2(dim + 1, 16, 100, false, 1);
    l2.add_batch(aug, n);
    for (size_t ef : {20, 50, 100}) {
        CAPTURE(ef);
        ip.set_ef_search(ef);
        l2.set_ef_search(ef);
        const double r_ip = recall(ip.query_batch(queries, nq, k), truth);
        const double r_l2 = recall(l2.query_batch(aug_q, nq, k), truth);
        REQUIRE(r_ip >= r_l2 - 0.03);
    }
}

TEST_CASE("HNSW inner product keeps recall when later adds have larger norms", "[ip][hnsw]") {
    // the first batch fixes M; every vector of the second is 10x longer
    const size_t n = 2500, dim = 32, nq = 100, k = 10;
    std::vector<float> data, queries;
    ip_clustered(2 * n, nq, dim, 13, data, queries);
    for (size_t i = 0; i < n * dim; ++i) data[i] *= 0.1f;
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, 2 * n);
    const auto truth = flat.query_batch(queries, nq, k);

    HNSWIndex once(dim, 16, 100, false, 1, Metric::InnerProduct);
    once.set_ef_search(100);
    once.add_batch(data, 2 * n);
    HNSWIndex twice(dim, 16, 100, false, 1, Metric::InnerProduct);
    twice.set_ef_search(100);
    twice.add_batch({data.data(), n * dim}, n);
    twice.add_batch({data.data() + n * dim, n * dim}, n);
    const double r_once = recall(once.query_batch(queries, nq, k), truth);
    const double r_twice = recall(twice.query_batch(queries, nq, k), truth);
    REQUIRE(r_twice >= r_once - 0.05);
    REQUIRE(r_twice >= 0.7);
}

TEST_CASE("HNSW inner product: save/load and compact keep building the same way", "[ip][hnsw][serialize]") {
    OneThread one_thread;
    const size_t n = 1500, dim = 16, nq = 30, k = 10;
    auto data = ip_random(n + 300, dim, 15);
    auto queries = ip_random(nq, dim, 16);

    for (int bits : {0, 8, 4}) {
        CAPTURE(bits);
        HNSWIndex hnsw(dim, 12, 100, false, 9, Metric::InnerProduct);
        if (bits) {
            hnsw.enable_sq(bits);
            hnsw.train(data, n + 300);
        }
        hnsw.add_batch({data.data(), n * dim}, n);
        std::vector<int64_t> dead{3, 30, 300};
        hnsw.remove(dead);

        // M comes back from the stored vectors, so later adds link the same way
        auto loaded = roundtrip(hnsw);
        REQUIRE(loaded->metric() == Metric::InnerProduct);
        require_same(loaded->query_batch(queries, nq, k), hnsw.query_batch(queries, nq, k));
        hnsw.add_batch({data.data() + n * dim, 100 * dim}, 100);
        loaded->add_batch({data.data() + n * dim, 100 * dim}, 100);
        require_same(loaded->query_batch(queries, nq, k), hnsw.query_batch(queries, nq, k));

        // compact recomputes M from the remaining vectors, as a load would
        REQUIRE(hnsw.compact() == dead.size());
        auto compacted = roundtrip(hnsw);
        hnsw.add_batch({data.data() + (n + 100) * dim, 200 * dim}, 200);
        compacted->add_batch({data.data() + (n + 100) * dim, 200 * dim}, 200);
        require_same(compacted->query_batch(queries, nq, k), hnsw.query_batch(queries, nq, k));
    }
}

TEST_CASE("inner product with custom ids and where= filters on every index", "[ip][metadata]") {
    const size_t n = 3000, dim = 16, nq = 30, k = 10;
    std::vector<float> data, queries;
    ip_clustered(n, nq, dim, 31, data, queries);
    std::vector<int64_t> ids(n), group(n);
    for (size_t i = 0; i < n; ++i) ids[i] = 1000 + 7 * static_cast<int64_t>(i), group[i] = static_cast<int64_t>(i % 4);
    const Metadata md{{"group", group}};
    const Expr where = Expr::compare("group", Expr::Op::Eq, int64_t{1});

    // exact answer: the k largest dot products among rows with group 1, by custom id
    std::vector<std::vector<int64_t>> truth(nq);
    for (size_t q = 0; q < nq; ++q) {
        std::vector<std::pair<double, int64_t>> all;
        for (size_t i = 1; i < n; i += 4) all.push_back({-dot_ref(queries.data() + q * dim, data.data() + i * dim, dim), ids[i]});
        std::partial_sort(all.begin(), all.begin() + k, all.end());
        for (size_t i = 0; i < k; ++i) truth[q].push_back(all[i].second);
    }
    auto check = [&](const Index& index, double min_recall) {
        double hit = 0;
        for (size_t q = 0; q < nq; ++q) {
            const auto got = index.query({queries.data() + q * dim, dim}, k, where);
            REQUIRE(got.size() == k);
            for (const auto& nb : got) {
                REQUIRE((static_cast<int64_t>(nb.index) - 1000) % 7 == 0);
                REQUIRE(((static_cast<int64_t>(nb.index) - 1000) / 7) % 4 == 1);
                hit += std::count(truth[q].begin(), truth[q].end(), static_cast<int64_t>(nb.index));
            }
        }
        REQUIRE(hit / (nq * k) >= min_recall);
    };

    FlatIndex flat(dim, 1024, true, Metric::InnerProduct);
    flat.add_batch(data, n, ids.data(), md);
    check(flat, 1.0);

    HNSWIndex hnsw(dim, 16, 100, true, 1, Metric::InnerProduct);
    hnsw.set_ef_search(100);
    hnsw.add_batch(data, n, ids.data(), md);
    check(hnsw, 0.95);

    IVFIndex ivf(dim, 100, 100, true, Metric::InnerProduct);   // every list: exact
    ivf.train(data, n, 10, 1);
    ivf.add_batch(data, n, ids.data(), md);
    check(ivf, 1.0);

    IVFIndex fs(dim, 100, 20, false, Metric::InnerProduct);
    fs.enable_pq(8, 16, true);
    fs.train(data, n, 10, 1);
    RefineIndex refine(fs, 10, true);
    refine.add_batch(data, n, ids.data(), md);
    check(refine, 0.95);
}

TEST_CASE("IVF inner product: remove and compact keep results and ids", "[ip][ivf][compact]") {
    const size_t n = 3000, dim = 16, nq = 30, k = 10;
    std::vector<float> data, queries;
    ip_clustered(n, nq, dim, 32, data, queries);
    std::vector<int64_t> dead;
    for (int64_t id = 0; id < static_cast<int64_t>(n); id += 3) dead.push_back(id);

    for (int mode : {0, 1, 2}) {   // floats, SQ8, fast-scan PQ
        CAPTURE(mode);
        IVFIndex ivf(dim, 100, 20, false, Metric::InnerProduct);
        if (mode == 1) ivf.enable_sq(8);
        if (mode == 2) ivf.enable_pq(8, 16, true);
        ivf.train(data, n, 10, 1);
        ivf.add_batch(data, n);
        REQUIRE(ivf.remove(dead) == dead.size());

        const auto before = ivf.query_batch(queries, nq, k);
        for (const auto& row : before)
            for (const auto& nb : row) REQUIRE(nb.index % 3 != 0);
        REQUIRE(ivf.compact() == dead.size());
        require_same(ivf.query_batch(queries, nq, k), before);
        require_same(roundtrip(ivf)->query_batch(queries, nq, k), before);
    }
}

TEST_CASE("HNSW inner product with 4-bit SQ, re-ranked", "[ip][hnsw][sq]") {
    const size_t n = 5000, dim = 32, nq = 100, k = 10;
    std::vector<float> data, queries;
    ip_clustered(n, nq, dim, 33, data, queries);
    FlatIndex flat(dim, 1024, false, Metric::InnerProduct);
    flat.add_batch(data, n);

    HNSWIndex sq(dim, 16, 100, false, 1, Metric::InnerProduct);
    sq.set_ef_search(200);
    sq.enable_sq(4);
    sq.train(data, n);
    RefineIndex refine(sq, 8);
    refine.add_batch(data, n);
    // 16 levels per dimension spread over 100x norms: 0.89 to 1.00 over seeds
    // 33-38 (8 bits and floats: 0.98 to 1.00)
    REQUIRE(recall(refine.query_batch(queries, nq, k), flat.query_batch(queries, nq, k)) >= 0.85);
    require_same(roundtrip(refine)->query_batch(queries, nq, k), refine.query_batch(queries, nq, k));
}
