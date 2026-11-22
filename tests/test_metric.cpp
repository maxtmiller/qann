#include <catch2/catch_test_macros.hpp>

#include "load_dense.hpp"

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"
#include "serialize.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vecengine;
using namespace vecengine::detail;

static std::vector<float> metric_random(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = d(rng);
    return v;
}

// Exposes the protected helper for direct tests.
struct NormalizeProbe : FlatIndex {
    using FlatIndex::FlatIndex;
    using DenseIndex::normalize_rows;
};

// Mean fraction of each truth row's ids found in the matching result row.
static double metric_recall(const std::vector<std::vector<Neighbor>>& got, const std::vector<std::vector<Neighbor>>& truth) {
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

static void require_identical(const std::vector<std::vector<Neighbor>>& a, const std::vector<std::vector<Neighbor>>& b) {
    REQUIRE(a.size() == b.size());
    for (size_t q = 0; q < a.size(); ++q) {
        REQUIRE(a[q].size() == b[q].size());
        for (size_t i = 0; i < a[q].size(); ++i) {
            REQUIRE(a[q][i].index == b[q][i].index);
            REQUIRE(a[q][i].distance == b[q][i].distance);
        }
    }
}

static std::unique_ptr<DenseIndex> metric_roundtrip(const DenseIndex& index) {
    std::stringstream s;
    index.save(s);
    return load_dense(s);
}

TEST_CASE("normalize_rows scales each row to unit length", "[metric]") {
    const size_t dim = 7;
    NormalizeProbe probe(dim);
    auto data = metric_random(50, dim, 1);
    for (size_t i = 0; i < 50; ++i)   // rows of very different lengths
        for (size_t d = 0; d < dim; ++d) data[i * dim + d] *= static_cast<float>(std::pow(10.0, static_cast<double>(i % 7) - 3));

    const auto unit = probe.normalize_rows(data, 50);
    REQUIRE(unit.size() == data.size());
    for (size_t i = 0; i < 50; ++i) {
        const float* in = data.data() + i * dim;
        const float* out = unit.data() + i * dim;
        double norm = 0, len = 0;
        for (size_t d = 0; d < dim; ++d) {
            norm += double(out[d]) * out[d];
            len += double(in[d]) * in[d];
        }
        REQUIRE(std::fabs(norm - 1.0) < 1e-5);
        for (size_t d = 0; d < dim; ++d)   // same direction: out = in / |in|
            REQUIRE(std::fabs(out[d] - in[d] / std::sqrt(len)) < 1e-5);
    }

    REQUIRE(probe.normalize_rows(std::span<const float>(), 0).empty());

    std::vector<float> zero_row = metric_random(3, dim, 2);
    std::fill(zero_row.begin() + dim, zero_row.begin() + 2 * dim, 0.0f);
    REQUIRE_THROWS_AS(probe.normalize_rows(zero_row, 3), std::invalid_argument);
}

TEST_CASE("cosine FlatIndex returns the exact cosine ranking", "[metric]") {
    const size_t n = 2000, dim = 24, nq = 30, k = 10;
    auto data = metric_random(n, dim, 3);
    auto queries = metric_random(nq, dim, 4);
    FlatIndex flat(dim, 1024, false, Metric::Cosine);
    REQUIRE(flat.metric() == Metric::Cosine);
    flat.add_batch(data, n);

    for (size_t q = 0; q < nq; ++q) {
        const float* qv = queries.data() + q * dim;
        std::vector<std::pair<float, size_t>> all(n);
        for (size_t i = 0; i < n; ++i) all[i] = {cosine_distance(qv, data.data() + i * dim, dim), i};
        std::partial_sort(all.begin(), all.begin() + k, all.end());

        const auto got = flat.query({qv, dim}, k);
        REQUIRE(got.size() == k);
        for (size_t i = 0; i < k; ++i) {
            REQUIRE(got[i].index == all[i].second);
            REQUIRE(std::fabs(got[i].distance - all[i].first) < 1e-5f);
            REQUIRE(got[i].distance >= -1e-6f);
            REQUIRE(got[i].distance <= 2.0f + 1e-6f);
        }
    }
}

TEST_CASE("cosine results ignore vector length", "[metric]") {
    const size_t n = 1000, dim = 16, nq = 20, k = 10;
    auto data = metric_random(n, dim, 5);
    auto queries = metric_random(nq, dim, 6);
    auto scaled = data;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> scale(0.01f, 100.0f);
    for (size_t i = 0; i < n; ++i) {
        const float s = scale(rng);
        for (size_t d = 0; d < dim; ++d) scaled[i * dim + d] *= s;
    }

    FlatIndex a(dim, 1024, false, Metric::Cosine), b(dim, 1024, false, Metric::Cosine);
    a.add_batch(data, n);
    b.add_batch(scaled, n);
    const auto ra = a.query_batch(queries, nq, k), rb = b.query_batch(queries, nq, k);
    for (size_t q = 0; q < nq; ++q)
        for (size_t i = 0; i < k; ++i) {
            REQUIRE(ra[q][i].index == rb[q][i].index);
            REQUIRE(std::fabs(ra[q][i].distance - rb[q][i].distance) < 1e-5f);
        }
}

TEST_CASE("cosine indexes reject zero vectors", "[metric]") {
    const size_t dim = 8;
    auto data = metric_random(10, dim, 8);
    std::fill(data.begin() + 3 * dim, data.begin() + 4 * dim, 0.0f);
    const std::vector<float> zero(dim, 0.0f);

    FlatIndex flat(dim, 1024, false, Metric::Cosine);
    REQUIRE_THROWS_AS(flat.add_batch(data, 10), std::invalid_argument);
    REQUIRE(flat.size() == 0);   // nothing added
    flat.add_batch({data.data(), 3 * dim}, 3);
    REQUIRE_THROWS_AS(flat.query(zero, 1), std::invalid_argument);

    FlatIndex l2(dim);   // L2 has no such restriction
    l2.add(zero);
    REQUIRE(l2.query(zero, 1)[0].distance == 0.0f);
}

TEST_CASE("approximate indexes find cosine neighbors", "[metric]") {
    const size_t n = 5000, dim = 32, nq = 100, k = 10;
    auto data = metric_random(n, dim, 9);
    auto queries = metric_random(nq, dim, 10);
    FlatIndex flat(dim, 1024, false, Metric::Cosine);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    // nprobe 50: random 32-d data is hard for IVF (L2 gets the same recall here)
    SECTION("IVF + PQ + RefineIndex") {
        IVFIndex ivf(dim, 100, 50, false, Metric::Cosine);
        ivf.enable_pq(8);
        ivf.train(data, n, 10, 1);
        RefineIndex refine(ivf, 10);
        REQUIRE(refine.metric() == Metric::Cosine);
        refine.add_batch(data, n);
        const auto got = refine.query_batch(queries, nq, k);
        REQUIRE(metric_recall(got, truth) >= 0.9);
        for (size_t q = 0; q < nq; ++q)   // re-ranked distances are exact cosine distances
            for (const auto& nb : got[q])
                REQUIRE(std::fabs(nb.distance - cosine_distance(queries.data() + q * dim, data.data() + nb.index * dim, dim)) < 1e-5f);
    }
    SECTION("IVF fast-scan + RefineIndex") {
        IVFIndex ivf(dim, 100, 50, false, Metric::Cosine);
        ivf.enable_pq(16, 16, true);
        ivf.train(data, n, 10, 1);
        RefineIndex refine(ivf, 10);
        refine.add_batch(data, n);
        REQUIRE(metric_recall(refine.query_batch(queries, nq, k), truth) >= 0.9);
    }
    SECTION("HNSW") {
        HNSWIndex hnsw(dim, 16, 100, false, 1, Metric::Cosine);
        hnsw.add_batch(data, n);
        const auto got = hnsw.query_batch(queries, nq, k);
        REQUIRE(metric_recall(got, truth) >= 0.95);
        std::vector<std::vector<Neighbor>> single;
        for (size_t q = 0; q < nq; ++q) single.push_back(hnsw.query({queries.data() + q * dim, dim}, k));
        require_identical(single, got);   // each query is normalized exactly once on both paths
    }
}

TEST_CASE("cosine Flat: query and query_batch agree exactly", "[metric]") {
    const size_t n = 1000, dim = 16, nq = 30, k = 10;
    auto data = metric_random(n, dim, 11);
    auto queries = metric_random(nq, dim, 12);
    FlatIndex flat(dim, 1024, false, Metric::Cosine);
    flat.add_batch(data, n);
    std::vector<std::vector<Neighbor>> single;
    for (size_t q = 0; q < nq; ++q) single.push_back(flat.query({queries.data() + q * dim, dim}, k));
    require_identical(single, flat.query_batch(queries, nq, k));
}

TEST_CASE("the metric survives save/load", "[metric][serialize]") {
    const size_t n = 1500, dim = 16, nq = 20, k = 10;
    auto data = metric_random(n, dim, 13);
    auto queries = metric_random(nq, dim, 14);

    FlatIndex flat(dim, 1024, false, Metric::Cosine);
    flat.add_batch(data, n);

    IVFIndex ivf(dim, 100, 20, false, Metric::Cosine);
    ivf.enable_pq(8, 16, true);
    ivf.train(data, n, 10, 1);
    RefineIndex refine(ivf, 5);
    refine.add_batch(data, n);

    HNSWIndex hnsw(dim, 8, 50, false, 1, Metric::Cosine);
    hnsw.add_batch(data, n);

    for (const DenseIndex* index : {static_cast<const DenseIndex*>(&flat), static_cast<const DenseIndex*>(&refine),
                               static_cast<const DenseIndex*>(&hnsw)}) {
        auto loaded = metric_roundtrip(*index);
        REQUIRE(loaded->metric() == Metric::Cosine);
        require_identical(loaded->query_batch(queries, nq, k), index->query_batch(queries, nq, k));
    }
    auto loaded_refine = metric_roundtrip(refine);
    REQUIRE(dynamic_cast<RefineIndex&>(*loaded_refine).base().metric() == Metric::Cosine);
}

TEST_CASE("cosine IVF-PQ ranks as well as L2 on the same unit vectors", "[metric]") {
    // Wide clusters: each one's plain mean is much shorter than its unit centroid. PQ
    // residuals taken against the unit centroid were larger, and the codes ranked worse
    // than L2's on the same unit vectors; against each list's mean they rank about as
    // well. One dataset differs by +-0.02 either way, so the gap is averaged over eight:
    // -0.031 with unit-centroid residuals, +0.004 with the means (M3).
    // The data comes from mt19937's raw output (Box-Muller), the same on every standard
    // library: std::normal_distribution isn't.
    const size_t n = 5000, dim = 32, nq = 100, k = 10, clusters = 20, datasets = 8;
    double gap = 0;
    for (unsigned seed = 1; seed <= datasets; ++seed) {
        std::mt19937 rng(seed);
        auto uniform = [&] { return (static_cast<double>(rng()) + 0.5) / 4294967296.0; };
        auto gauss = [&] {
            const double u = uniform(), v = uniform();
            return static_cast<float>(std::sqrt(-2.0 * std::log(u)) * std::cos(6.283185307179586 * v));
        };
        auto unit_rows = [&](std::vector<float> v) {
            for (size_t i = 0; i < v.size() / dim; ++i) {
                float norm = 0;
                for (size_t d = 0; d < dim; ++d) norm += v[i * dim + d] * v[i * dim + d];
                for (size_t d = 0; d < dim; ++d) v[i * dim + d] /= std::sqrt(norm);
            }
            return v;
        };
        std::vector<float> centers(clusters * dim);
        for (auto& x : centers) x = gauss();
        centers = unit_rows(centers);
        auto sample = [&](size_t count) {
            std::vector<float> v(count * dim);
            for (size_t i = 0; i < count; ++i) {
                const size_t c = rng() % clusters;
                for (size_t d = 0; d < dim; ++d) v[i * dim + d] = centers[c * dim + d] + 0.35f * gauss();
            }
            return v;
        };
        const auto data = sample(n), queries = sample(nq);

        FlatIndex flat(dim, 1024, false, Metric::Cosine);
        flat.add_batch(data, n);
        const auto truth = flat.query_batch(queries, nq, k);

        // every list probed, no re-ranking: the codes' ranking alone
        IVFIndex cosine(dim, 100, 100, false, Metric::Cosine);
        cosine.enable_pq(8);
        cosine.train(data, n, 10, 1);
        cosine.set_cosine_inner_product(false);
        cosine.add_batch(data, n);

        const auto unit_data = unit_rows(data), unit_queries = unit_rows(queries);
        IVFIndex l2(dim, 100, 100, false, Metric::L2);
        l2.enable_pq(8);
        l2.train(unit_data, n, 10, 1);
        l2.add_batch(unit_data, n);

        const double cos_recall = metric_recall(cosine.query_batch(queries, nq, k), truth);
        const double l2_recall = metric_recall(l2.query_batch(unit_queries, nq, k), truth);
        gap += cos_recall - l2_recall;

        if (seed == 1) {   // the residual centers are saved: a loaded index ranks exactly the same
            std::stringstream s;
            cosine.save(s);
            auto loaded = load_dense(s);
            require_identical(loaded->query_batch(queries, nq, k), cosine.query_batch(queries, nq, k));
        }
    }
    gap /= datasets;
    REQUIRE(gap >= -0.01);
}

TEST_CASE("a cosine IVF-PQ file with the wrong number of residual centers is rejected", "[metric][serialize]") {
    const size_t n = 500, dim = 8, nlist = 100;
    auto data = metric_random(n, dim, 16);
    IVFIndex ivf(dim, nlist, 10, false, Metric::Cosine);
    ivf.enable_pq(4);
    ivf.train(data, n, 10, 1);
    ivf.add_batch(data, n);
    std::stringstream s;
    ivf.save(s);
    std::string bytes = s.str();
    // the body ends with the residual centers: a u64 count, then nlist * dim floats
    const size_t count_at = bytes.size() - nlist * dim * sizeof(float) - sizeof(uint64_t);
    uint64_t count;
    std::memcpy(&count, bytes.data() + count_at, sizeof(count));
    REQUIRE(count == nlist * dim);
    count -= dim;   // one list short; the floats left over then follow the body
    std::memcpy(bytes.data() + count_at, &count, sizeof(count));
    std::stringstream t(bytes);
    REQUIRE_THROWS_AS(load_dense(t), std::runtime_error);
}

TEST_CASE("an unknown metric byte is rejected on load", "[metric][serialize]") {
    FlatIndex flat(4, 1024, false, Metric::Cosine);
    auto data = metric_random(10, 4, 15);
    flat.add_batch(data, 10);
    std::stringstream s;
    flat.save(s);
    std::string bytes = s.str();
    // the metric comes just before the metadata section, which is a column count of 0 here
    const size_t metric_at = bytes.size() - 1 - sizeof(uint64_t);
    REQUIRE(bytes[metric_at] == static_cast<char>(Metric::Cosine));
    bytes[metric_at] = 7;
    std::stringstream t(bytes);
    REQUIRE_THROWS_AS(load_dense(t), std::runtime_error);
}

TEST_CASE("make_index passes the metric", "[metric]") {
    IndexOptions opts;
    opts.metric = Metric::Cosine;
    for (IndexType type : {IndexType::Flat, IndexType::IVF, IndexType::HNSW})
        REQUIRE(make_dense(type, 8, opts)->metric() == Metric::Cosine);
    REQUIRE(make_dense(IndexType::Flat, 8)->metric() == Metric::L2);
}

TEST_CASE("files from format version 4 (0.1.0a10) still load, as L2", "[serialize][compat]") {
    // Saved by 0.1.0a10, before the metric byte: flat with deletions,
    // IVF fast-scan, and HNSW with deletions and ef_search 32.
    const std::filesystem::path dir = QANN_TEST_DATA_DIR;
    struct Expect { const char* name; size_t size; };
    for (const Expect e : {Expect{"v4_flat.qann", 1285}, Expect{"v4_ivf_fs.qann", 1500}, Expect{"v4_hnsw.qann", 1200}}) {
        CAPTURE(e.name);
        auto index = load_dense(dir / e.name);
        REQUIRE(index->size() == e.size);
        REQUIRE(index->dim() == 16);
        REQUIRE(index->metric() == Metric::L2);
        if (auto* h = dynamic_cast<HNSWIndex*>(index.get())) REQUIRE(h->ef_search() == 32);

        auto query = metric_random(1, 16, 16);
        const auto before = index->query(query, 10);
        REQUIRE(before.size() == 10);

        // Re-saving writes the current version and keeps results.
        std::stringstream s;
        index->save(s);
        REQUIRE(read_header(s).version == kFormatVersion);
        s.seekg(0);
        auto reloaded = load_dense(s);
        REQUIRE(reloaded->metric() == Metric::L2);
        const auto again = reloaded->query(query, 10);
        for (size_t i = 0; i < before.size(); ++i) {
            REQUIRE(again[i].index == before[i].index);
            REQUIRE(again[i].distance == before[i].distance);
        }
    }
}
