#include <catch2/catch_test_macros.hpp>

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/threads.hpp"
#include "serialize.hpp"

#include <cstdint>
#include <cstring>
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

static std::vector<float> hnsw_random(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = d(rng);
    return v;
}

// Mean fraction of each truth row's ids found in the matching result row.
static double hnsw_recall(const std::vector<std::vector<Neighbor>>& got, const std::vector<std::vector<Neighbor>>& truth) {
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

// Inserts on one thread while alive, so a seeded graph is identical every
// time; parallel inserts interleave differently from run to run.
struct OneThread {
    OneThread() { set_num_threads(1); }
    ~OneThread() { set_num_threads(0); }
};

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

TEST_CASE("HNSW: high recall against exact search", "[hnsw]") {
    const size_t n = 5000, dim = 32, nq = 200, k = 10;
    auto data = hnsw_random(n, dim, 1);
    auto queries = hnsw_random(nq, dim, 2);

    HNSWIndex hnsw(dim, 16, 200, false, 7);
    hnsw.add_batch(data, n);
    FlatIndex flat(dim);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    hnsw.set_ef_search(64);
    const double r64 = hnsw_recall(hnsw.query_batch(queries, nq, k), truth);
    REQUIRE(r64 >= 0.95);

    // a wider beam never loses recall here
    hnsw.set_ef_search(256);
    REQUIRE(hnsw_recall(hnsw.query_batch(queries, nq, k), truth) >= r64);
}

TEST_CASE("HNSW: results are sorted, distinct and at exact distances", "[hnsw]") {
    const size_t n = 2000, dim = 16, nq = 50, k = 20;
    auto data = hnsw_random(n, dim, 3);
    auto queries = hnsw_random(nq, dim, 4);
    HNSWIndex hnsw(dim, 8, 100, false, 1);
    hnsw.add_batch(data, n);

    for (size_t q = 0; q < nq; ++q) {
        const float* qv = queries.data() + q * dim;
        const auto got = hnsw.query({qv, dim}, k);
        REQUIRE(got.size() == k);
        std::set<size_t> seen;
        for (size_t i = 0; i < got.size(); ++i) {
            REQUIRE(seen.insert(got[i].index).second);
            REQUIRE(got[i].distance == l2_distance(qv, data.data() + got[i].index * dim, dim));
            if (i > 0) REQUIRE(got[i - 1].distance <= got[i].distance);
        }
    }
}

TEST_CASE("HNSW: the same seed builds the same graph", "[hnsw]") {
    OneThread one_thread;
    const size_t n = 2000, dim = 16, nq = 50, k = 10;
    auto data = hnsw_random(n, dim, 5);
    auto queries = hnsw_random(nq, dim, 6);
    HNSWIndex a(dim, 12, 100, false, 42), b(dim, 12, 100, false, 42);
    a.add_batch(data, n);
    b.add_batch(data, n);
    require_identical(a.query_batch(queries, nq, k), b.query_batch(queries, nq, k));
}

TEST_CASE("HNSW: query and query_batch agree", "[hnsw]") {
    const size_t n = 2000, dim = 16, nq = 50, k = 10;
    auto data = hnsw_random(n, dim, 7);
    auto queries = hnsw_random(nq, dim, 8);
    HNSWIndex hnsw(dim, 16, 100, false, 3);
    hnsw.add_batch(data, n);

    const auto batch = hnsw.query_batch(queries, nq, k);
    std::vector<std::vector<Neighbor>> single;
    for (size_t q = 0; q < nq; ++q) single.push_back(hnsw.query({queries.data() + q * dim, dim}, k));
    require_identical(single, batch);
}

TEST_CASE("HNSW: add one at a time matches add_batch", "[hnsw]") {
    OneThread one_thread;
    const size_t n = 500, dim = 8, nq = 20, k = 5;
    auto data = hnsw_random(n, dim, 9);
    auto queries = hnsw_random(nq, dim, 10);
    HNSWIndex a(dim, 8, 50, false, 5), b(dim, 8, 50, false, 5);
    a.add_batch(data, n);
    for (size_t i = 0; i < n; ++i) b.add({data.data() + i * dim, dim});
    require_identical(a.query_batch(queries, nq, k), b.query_batch(queries, nq, k));
}

TEST_CASE("HNSW: small indexes", "[hnsw]") {
    const size_t dim = 4;
    auto data = hnsw_random(5, dim, 11);
    const std::span<const float> q(data.data(), dim);

    HNSWIndex hnsw(dim, 4, 10, false, 1);
    REQUIRE(hnsw.query(q, 3).empty());   // empty index

    hnsw.add(q);
    auto one = hnsw.query(q, 3);
    REQUIRE(one.size() == 1);
    REQUIRE(one[0].index == 0);
    REQUIRE(one[0].distance == 0.0f);

    hnsw.add_batch({data.data() + dim, 4 * dim}, 4);
    auto all = hnsw.query(q, 10);   // k > size: every vector, once
    REQUIRE(all.size() == 5);
    std::set<size_t> ids;
    for (const auto& nb : all) ids.insert(nb.index);
    REQUIRE(ids.size() == 5);
    REQUIRE(hnsw.query(q, 0).empty());
}

TEST_CASE("HNSW: removed ids are never returned", "[hnsw]") {
    const size_t n = 2000, dim = 16, nq = 50, k = 10;
    auto data = hnsw_random(n, dim, 12);
    auto queries = hnsw_random(nq, dim, 13);
    HNSWIndex hnsw(dim, 16, 100, false, 2);
    hnsw.add_batch(data, n);

    std::vector<int64_t> dead;
    for (int64_t id = 0; id < static_cast<int64_t>(n); id += 4) dead.push_back(id);
    REQUIRE(hnsw.remove(dead) == dead.size());
    REQUIRE(hnsw.size() == n - dead.size());

    for (const auto& row : hnsw.query_batch(queries, nq, k)) {
        REQUIRE(row.size() == k);
        for (const auto& nb : row) REQUIRE(nb.index % 4 != 0);
    }
}

TEST_CASE("HNSW: custom ids are reported", "[hnsw]") {
    const size_t n = 300, dim = 8;
    auto data = hnsw_random(n, dim, 14);
    std::vector<int64_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = 1000 + 7 * static_cast<int64_t>(i);

    HNSWIndex hnsw(dim, 8, 50, true, 1);
    REQUIRE_THROWS_AS(hnsw.add({data.data(), dim}), std::invalid_argument);   // no id to give it
    hnsw.add_batch(data, n, ids.data());
    for (size_t i = 0; i < n; i += 37) REQUIRE(hnsw.query({data.data() + i * dim, dim}, 1)[0].index == 1000 + 7 * i);
}

TEST_CASE("HNSW: settings are validated", "[hnsw]") {
    REQUIRE_THROWS_AS(HNSWIndex(0), std::invalid_argument);
    REQUIRE_THROWS_AS(HNSWIndex(8, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(HNSWIndex(8, 16, 0), std::invalid_argument);
    HNSWIndex hnsw(8);
    REQUIRE_THROWS_AS(hnsw.set_ef_search(0), std::invalid_argument);
    REQUIRE(hnsw.M() == 16);
    REQUIRE(hnsw.ef_construction() == 200);
    REQUIRE(hnsw.ef_search() == 64);
}

static std::unique_ptr<Index> hnsw_roundtrip(const Index& index) {
    std::stringstream s;
    index.save(s);
    return load_index(s);
}

TEST_CASE("HNSW: save/load returns identical results and keeps adding the same way", "[hnsw][serialize]") {
    OneThread one_thread;
    const size_t n = 1500, dim = 16, nq = 50, k = 10;
    auto data = hnsw_random(n + 200, dim, 20);
    auto queries = hnsw_random(nq, dim, 21);

    HNSWIndex hnsw(dim, 12, 100, false, 9);
    hnsw.set_ef_search(40);
    hnsw.add_batch({data.data(), n * dim}, n);
    std::vector<int64_t> dead{3, 30, 300};
    hnsw.remove(dead);

    auto loaded_base = hnsw_roundtrip(hnsw);
    auto* loaded = dynamic_cast<HNSWIndex*>(loaded_base.get());
    REQUIRE(loaded != nullptr);
    REQUIRE(loaded->size() == n - dead.size());
    REQUIRE(loaded->M() == 12);
    REQUIRE(loaded->ef_construction() == 100);
    REQUIRE(loaded->ef_search() == 40);
    require_identical(loaded->query_batch(queries, nq, k), hnsw.query_batch(queries, nq, k));

    // the generator state is saved, so new vectors get the same levels and links
    hnsw.add_batch({data.data() + n * dim, 200 * dim}, 200);
    loaded->add_batch({data.data() + n * dim, 200 * dim}, 200);
    require_identical(loaded->query_batch(queries, nq, k), hnsw.query_batch(queries, nq, k));
}

TEST_CASE("HNSW: save/load keeps custom ids and an empty index", "[hnsw][serialize]") {
    const size_t n = 200, dim = 8;
    auto data = hnsw_random(n, dim, 22);
    std::vector<int64_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = 50 + 3 * static_cast<int64_t>(i);

    HNSWIndex custom(dim, 8, 50, true, 1);
    custom.add_batch(data, n, ids.data());
    auto loaded = hnsw_roundtrip(custom);
    REQUIRE(loaded->custom_ids());
    REQUIRE(loaded->query({data.data() + 10 * dim, dim}, 1)[0].index == 80);

    HNSWIndex empty(dim);
    auto loaded_empty = hnsw_roundtrip(empty);
    REQUIRE(loaded_empty->size() == 0);
    REQUIRE(loaded_empty->query({data.data(), dim}, 5).empty());
}

TEST_CASE("HNSW: load rejects truncated and corrupt data", "[hnsw][serialize]") {
    const size_t n = 300, dim = 8;
    auto data = hnsw_random(n, dim, 23);
    HNSWIndex hnsw(dim, 8, 50, false, 1);
    hnsw.add_batch(data, n);
    std::stringstream s;
    hnsw.save(s);
    const std::string bytes = s.str();

    SECTION("truncated") {
        for (size_t cut : {size_t{9}, size_t{40}, bytes.size() / 2, bytes.size() - 1}) {
            std::stringstream t(bytes.substr(0, cut));
            REQUIRE_THROWS_AS(load_index(t), std::runtime_error);
        }
    }
    SECTION("neighbor out of range") {
        // the first level-0 neighbor of slot 0 sits after the header, the shape
        // fields, the generator state, the vectors and the levels
        std::stringstream probe(bytes);
        (void)read_header(probe);
        for (int i = 0; i < 5; ++i) (void)read_pod<uint64_t>(probe);
        (void)read_pod<int32_t>(probe);
        (void)read_pod<uint32_t>(probe);
        (void)read_vec<char>(probe, 1 << 16);
        (void)read_vec<float>(probe, n * dim);
        (void)read_vec<uint8_t>(probe, n);
        const auto level0_at = static_cast<size_t>(probe.tellg()) + sizeof(uint64_t);   // past its length
        std::string bad = bytes;
        const uint32_t out_of_range = static_cast<uint32_t>(n);
        std::memcpy(bad.data() + level0_at + sizeof(uint32_t), &out_of_range, sizeof(uint32_t));
        std::stringstream t(bad);
        REQUIRE_THROWS_AS(load_index(t), std::runtime_error);
    }
}

TEST_CASE("HNSW: make_index builds one with the given options", "[hnsw]") {
    IndexOptions opts;
    opts.hnsw_m = 8;
    opts.ef_construction = 40;
    opts.ef_search = 20;
    auto index = make_index(IndexType::HNSW, 12, opts);
    auto* hnsw = dynamic_cast<HNSWIndex*>(index.get());
    REQUIRE(hnsw != nullptr);
    REQUIRE(hnsw->dim() == 12);
    REQUIRE(hnsw->M() == 8);
    REQUIRE(hnsw->ef_construction() == 40);
    REQUIRE(hnsw->ef_search() == 20);
}

TEST_CASE("HNSW: compact drops deleted vectors and keeps ids", "[hnsw][compact]") {
    const size_t n = 3000, dim = 16, nq = 100, k = 10;
    auto data = hnsw_random(n, dim, 24);
    auto queries = hnsw_random(nq, dim, 25);
    HNSWIndex hnsw(dim, 16, 100, false, 4);
    hnsw.add_batch(data, n);

    std::vector<int64_t> dead;
    for (int64_t id = 0; id < static_cast<int64_t>(n); id += 3) dead.push_back(id);
    hnsw.remove(dead);
    REQUIRE(hnsw.compact() == dead.size());
    REQUIRE(hnsw.slots() == n - dead.size());
    REQUIRE(hnsw.size() == n - dead.size());
    REQUIRE(hnsw.compact() == 0);   // nothing left to reclaim

    // the rebuilt graph still finds the live neighbors, by their old ids
    FlatIndex flat(dim);
    flat.add_batch(data, n);
    flat.remove(dead);
    const auto truth = flat.query_batch(queries, nq, k);
    const auto got = hnsw.query_batch(queries, nq, k);
    for (const auto& row : got)
        for (const auto& nb : row) REQUIRE(nb.index % 3 != 0);
    REQUIRE(hnsw_recall(got, truth) >= 0.95);

    // new vectors continue after the highest id ever assigned
    hnsw.add({queries.data(), dim});
    REQUIRE(hnsw.query({queries.data(), dim}, 1)[0].index == n);

    // and the compacted index survives a save/load
    auto loaded = hnsw_roundtrip(hnsw);
    require_identical(loaded->query_batch(queries, nq, k), hnsw.query_batch(queries, nq, k));
}

TEST_CASE("HNSW: queries still return k live results after most vectors are deleted", "[hnsw]") {
    const size_t n = 3000, dim = 16, nq = 100, k = 10;
    auto data = hnsw_random(n, dim, 26);
    auto queries = hnsw_random(nq, dim, 27);
    HNSWIndex hnsw(dim, 16, 100, false, 6);
    hnsw.add_batch(data, n);
    hnsw.set_ef_search(16);   // a narrow beam, so deleted nodes would crowd it out

    std::vector<int64_t> dead;
    for (int64_t id = 0; id < static_cast<int64_t>(n); ++id)
        if (id % 10 != 0) dead.push_back(id);   // keep every 10th
    hnsw.remove(dead);

    FlatIndex flat(dim);
    flat.add_batch(data, n);
    flat.remove(dead);
    const auto truth = flat.query_batch(queries, nq, k);
    const auto got = hnsw.query_batch(queries, nq, k);
    for (const auto& row : got) {
        REQUIRE(row.size() == k);
        for (const auto& nb : row) REQUIRE(nb.index % 10 == 0);
    }
    REQUIRE(hnsw_recall(got, truth) >= 0.9);
}

TEST_CASE("HNSW: a parallel build matches a serial one in quality", "[hnsw]") {
    const size_t n = 20000, dim = 32, nq = 200, k = 10;
    auto data = hnsw_random(n, dim, 30);
    auto queries = hnsw_random(nq, dim, 31);
    FlatIndex flat(dim);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    double serial_recall;
    {
        OneThread one_thread;
        HNSWIndex serial(dim, 16, 100, false, 1);
        serial.add_batch(data, n);
        serial_recall = hnsw_recall(serial.query_batch(queries, nq, k), truth);
    }

    set_num_threads(8);
    struct Reset { ~Reset() { set_num_threads(0); } } reset;
    for (int build = 0; build < 3; ++build) {   // several builds, to shake out races
        HNSWIndex parallel(dim, 16, 100, false, 1);
        parallel.add_batch({data.data(), 5000 * dim}, 5000);   // two batches: the second joins a built graph
        parallel.add_batch({data.data() + 5000 * dim, (n - 5000) * dim}, n - 5000);
        REQUIRE(parallel.size() == n);
        REQUIRE(hnsw_recall(parallel.query_batch(queries, nq, k), truth) >= serial_recall - 0.02);
    }
}

TEST_CASE("HNSW: a RefineIndex over HNSW saves and loads", "[hnsw][serialize][refine]") {
    OneThread one_thread;
    const size_t n = 1000, dim = 16, nq = 30, k = 10;
    auto data = hnsw_random(n, dim, 40);
    auto queries = hnsw_random(nq, dim, 41);
    HNSWIndex base(dim, 8, 50, false, 1);
    RefineIndex refine(base, 2);
    refine.add_batch(data, n);

    auto loaded = hnsw_roundtrip(refine);
    REQUIRE(dynamic_cast<RefineIndex*>(loaded.get()) != nullptr);
    require_identical(loaded->query_batch(queries, nq, k), refine.query_batch(queries, nq, k));
}
