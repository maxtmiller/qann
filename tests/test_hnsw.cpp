#include <catch2/catch_test_macros.hpp>

#include "load_dense.hpp"

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/sq.hpp"
#include "vecengine/threads.hpp"
#include "serialize.hpp"

#include <algorithm>
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

TEST_CASE("HNSW: exact duplicates stay connected", "[hnsw]") {
    // 2,000 distinct vectors, each stored 4 times in shuffled order. Ties make
    // ids ambiguous, so recall counts results at most the true k-th distance.
    const size_t distinct = 2000, copies = 4, n = distinct * copies, dim = 16, nq = 200, k = 10;
    const auto base = hnsw_random(distinct, dim, 40);
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i % distinct;
    std::shuffle(order.begin(), order.end(), std::mt19937(41));
    std::vector<float> data(n * dim);
    for (size_t i = 0; i < n; ++i) std::copy_n(base.data() + order[i] * dim, dim, data.data() + i * dim);
    const auto queries = hnsw_random(nq, dim, 42);

    OneThread one_thread;
    HNSWIndex hnsw(dim, 8, 100, false, 1);
    hnsw.add_batch(data, n);
    hnsw.set_ef_search(32);
    FlatIndex flat(dim);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);
    const auto got = hnsw.query_batch(queries, nq, k);

    double hits = 0;
    for (size_t q = 0; q < nq; ++q)
        for (const auto& nb : got[q]) hits += nb.distance <= truth[q][k - 1].distance;
    REQUIRE(hits / (nq * k) >= 0.75);   // 0.84 here; pruning ties as well gave 0.52
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

static std::unique_ptr<DenseIndex> hnsw_roundtrip(const DenseIndex& index) {
    std::stringstream s;
    index.save(s);
    return load_dense(s);
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
            REQUIRE_THROWS_AS(load_dense(t), std::runtime_error);
        }
    }
    SECTION("neighbor out of range") {
        // the first level-0 neighbor of slot 0 sits after the header, the shape
        // fields, the generator state, the storage byte, the vectors and the levels
        std::stringstream probe(bytes);
        (void)read_header(probe);
        for (int i = 0; i < 5; ++i) (void)read_pod<uint64_t>(probe);
        (void)read_pod<int32_t>(probe);
        (void)read_pod<uint32_t>(probe);
        (void)read_pod<uint64_t>(probe);
        (void)read_pod<uint8_t>(probe);
        (void)read_vec<float>(probe, n * dim);
        (void)read_vec<uint8_t>(probe, n);
        const auto level0_at = static_cast<size_t>(probe.tellg()) + sizeof(uint64_t);   // past its length
        std::string bad = bytes;
        const uint32_t out_of_range = static_cast<uint32_t>(n);
        std::memcpy(bad.data() + level0_at + sizeof(uint32_t), &out_of_range, sizeof(uint32_t));
        std::stringstream t(bad);
        REQUIRE_THROWS_AS(load_dense(t), std::runtime_error);
    }
}

TEST_CASE("HNSW: make_index builds one with the given options", "[hnsw]") {
    IndexOptions opts;
    opts.hnsw_m = 8;
    opts.ef_construction = 40;
    opts.ef_search = 20;
    auto index = make_dense(IndexType::HNSW, 12, opts);
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

// An SQ HNSWIndex with codes of `bits`, trained on data and holding all of it.
static std::unique_ptr<HNSWIndex> sq_hnsw(int bits, const std::vector<float>& data, size_t n, size_t dim, uint32_t seed,
                                          Metric metric = Metric::L2) {
    auto hnsw = std::make_unique<HNSWIndex>(dim, 16, 100, false, seed, metric);
    hnsw->enable_sq(bits);
    hnsw->train(data, n);
    hnsw->add_batch(data, n);
    return hnsw;
}

TEST_CASE("SQ: codes round to the nearest level and the kernel matches the decoded vector", "[sq]") {
    for (int bits : {8, 4}) {
        // with and without SIMD tails, and odd sizes that leave a 4-bit code's high nibble unused
        for (size_t dim : {size_t{3}, size_t{16}, size_t{33}, size_t{48}, size_t{80}, size_t{100}, size_t{128}}) {
            CAPTURE(bits, dim);
            auto data = hnsw_random(200, dim, 50);
            SQ q;
            q.bits = bits;
            q.train(data, 200, dim);
            REQUIRE(q.code_bytes() == (bits == 8 ? dim : (dim + 1) / 2));
            std::vector<uint8_t> code(q.code_bytes());
            std::vector<float> decoded(dim);
            auto query = hnsw_random(1, dim, 51);
            for (size_t i = 0; i < 200; ++i) {
                const float* v = data.data() + i * dim;
                q.encode(v, code.data());
                q.decode(code.data(), decoded.data());
                for (size_t d = 0; d < dim; ++d) REQUIRE(std::abs(decoded[d] - v[d]) <= q.step[d] * 0.5f + 1e-5f);
                const float exact = l2_distance_scalar(query.data(), decoded.data(), dim);
                REQUIRE(std::abs(q.distance(query.data(), code.data()) - exact) <= 1e-4f * (1.0f + exact));
                const float dot = inner_product_scalar(query.data(), decoded.data(), dim);
                REQUIRE(std::abs(q.inner_product(query.data(), code.data()) - dot) <= 1e-4f * (1.0f + std::abs(dot)));
            }
        }
    }

    // out-of-range values clamp to the end codes; a constant dimension decodes to its value
    for (int bits : {8, 4}) {
        CAPTURE(bits);
        SQ q;
        q.bits = bits;
        const std::vector<float> train{0.0f, 5.0f, 1.0f, 1.0f, 5.0f, 1.0f};   // dims 1 and 2 constant
        q.train(train, 2, 3);
        uint8_t code[3] = {};
        float back[3];
        const float high[3] = {3.0f, 9.0f, 2.0f};
        q.encode(high, code);
        q.decode(code, back);
        REQUIRE(back[0] == 1.0f);
        REQUIRE(back[1] == 5.0f);
        REQUIRE(back[2] == 1.0f);
        if (bits == 4) {
            REQUIRE(code[0] == 0x0F);   // dimension 0 at the top code, dimension 1 at 0
            REQUIRE(code[1] == 0x00);   // dimension 2 alone in byte 1
        }
        const float low[3] = {-3.0f, 9.0f, 2.0f};
        q.encode(low, code);
        q.decode(code, back);
        REQUIRE(back[0] == 0.0f);
    }
}

TEST_CASE("HNSW SQ: recall close to the float graph, exact again under a RefineIndex", "[hnsw][sq]") {
    const size_t n = 5000, dim = 32, nq = 200, k = 10;
    auto data = hnsw_random(n, dim, 52);
    auto queries = hnsw_random(nq, dim, 53);
    FlatIndex flat(dim);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    struct Case { int bits; double plain; size_t k_factor; };
    for (const Case c : {Case{8, 0.9, 4}, Case{4, 0.5, 8}}) {
        CAPTURE(c.bits);
        auto owned = sq_hnsw(c.bits, data, n, dim, 7);
        HNSWIndex& hnsw = *owned;
        REQUIRE(hnsw.sq_bits() == c.bits);
        REQUIRE(hnsw.trained());
        hnsw.set_ef_search(64);
        REQUIRE(hnsw_recall(hnsw.query_batch(queries, nq, k), truth) >= c.plain);

        HNSWIndex base(dim, 16, 100, false, 7);
        base.enable_sq(c.bits);
        base.train(data, n);
        base.set_ef_search(64);
        RefineIndex refine(base, c.k_factor);
        refine.add_batch(data, n);
        const auto got = refine.query_batch(queries, nq, k);
        REQUIRE(hnsw_recall(got, truth) >= 0.95);
        for (size_t q = 0; q < nq; ++q)
            for (const auto& nb : got[q])
                REQUIRE(nb.distance == l2_distance(queries.data() + q * dim, data.data() + nb.index * dim, dim));
    }
}

TEST_CASE("HNSW SQ: enable_sq and train are checked", "[hnsw][sq]") {
    auto data = hnsw_random(100, 8, 54);
    HNSWIndex plain(8);
    REQUIRE(plain.sq_bits() == 0);
    REQUIRE_THROWS_AS(plain.train(data, 100), std::logic_error);
    REQUIRE_FALSE(plain.trained());
    plain.add_batch(data, 100);
    REQUIRE_THROWS_AS(plain.enable_sq(8), std::logic_error);

    HNSWIndex hnsw(8);
    for (int bad : {0, 1, 2, 6, 16}) REQUIRE_THROWS_AS(hnsw.enable_sq(bad), std::invalid_argument);
    REQUIRE(hnsw.sq_bits() == 0);
    hnsw.enable_sq(4);
    REQUIRE_THROWS_AS(hnsw.add_batch(data, 100), std::logic_error);
    REQUIRE(hnsw.size() == 0);
    REQUIRE_THROWS_AS(hnsw.train(data, 0), std::invalid_argument);
    hnsw.train(data, 100);
    REQUIRE_THROWS_AS(hnsw.train(data, 100), std::logic_error);
    REQUIRE_THROWS_AS(hnsw.enable_sq(8), std::logic_error);
    hnsw.add_batch(data, 100);
    REQUIRE(hnsw.size() == 100);

    IndexOptions opts;
    opts.hnsw_sq_bits = 4;
    auto made = make_dense(IndexType::HNSW, 8, opts);
    REQUIRE(dynamic_cast<HNSWIndex&>(*made).sq_bits() == 4);
    opts.hnsw_sq_bits = 5;
    REQUIRE_THROWS_AS(make_dense(IndexType::HNSW, 8, opts), std::invalid_argument);
}

TEST_CASE("HNSW SQ: save/load keeps codes, ranges and an untrained index", "[hnsw][sq][serialize]") {
    OneThread one_thread;
    const size_t n = 1500, dim = 17, nq = 50, k = 10;   // odd dim: 4-bit rows end in half a byte
    auto data = hnsw_random(n + 200, dim, 55);
    auto queries = hnsw_random(nq, dim, 56);
    for (int bits : {8, 4}) {
        CAPTURE(bits);
        HNSWIndex hnsw(dim, 12, 100, false, 9);
        hnsw.enable_sq(bits);
        hnsw.train(data, n + 200);
        hnsw.add_batch({data.data(), n * dim}, n);
        std::vector<int64_t> dead{3, 30, 300};
        hnsw.remove(dead);

        auto loaded_base = hnsw_roundtrip(hnsw);
        auto& loaded = dynamic_cast<HNSWIndex&>(*loaded_base);
        REQUIRE(loaded.sq_bits() == bits);
        REQUIRE(loaded.trained());
        require_identical(loaded.query_batch(queries, nq, k), hnsw.query_batch(queries, nq, k));
        hnsw.add_batch({data.data() + n * dim, 200 * dim}, 200);
        loaded.add_batch({data.data() + n * dim, 200 * dim}, 200);
        require_identical(loaded.query_batch(queries, nq, k), hnsw.query_batch(queries, nq, k));

        HNSWIndex untrained(dim);
        untrained.enable_sq(bits);
        auto loaded_untrained = hnsw_roundtrip(untrained);
        auto& u = dynamic_cast<HNSWIndex&>(*loaded_untrained);
        REQUIRE(u.sq_bits() == bits);
        REQUIRE_FALSE(u.trained());
        u.train(data, n);
        u.add_batch({data.data(), n * dim}, n);
        REQUIRE(u.size() == n);

        // a refined SQ graph nests the same way
        HNSWIndex base(dim, 8, 50, false, 1);
        base.enable_sq(bits);
        base.train(data, n);
        RefineIndex refine(base, 2);
        refine.add_batch({data.data(), n * dim}, n);
        auto loaded_refine = hnsw_roundtrip(refine);
        require_identical(loaded_refine->query_batch(queries, nq, k), refine.query_batch(queries, nq, k));
    }
}

TEST_CASE("HNSW SQ: load rejects a bad storage byte and a trained-less index with vectors", "[hnsw][sq][serialize]") {
    const size_t n = 100, dim = 8;
    auto data = hnsw_random(n, dim, 57);
    auto owned = sq_hnsw(8, data, n, dim, 1);
    HNSWIndex& hnsw = *owned;
    std::stringstream s;
    hnsw.save(s);
    const std::string bytes = s.str();

    // the storage byte (bits per dimension) follows the header, the shape fields and the generator state
    const size_t storage_at = 9 + 5 * 8 + 4 + 4 + 8;
    REQUIRE(bytes[storage_at] == 8);
    for (char bad_bits : {1, 2, 5, 16}) {
        std::string bad = bytes;
        bad[storage_at] = bad_bits;
        std::stringstream t(bad);
        REQUIRE_THROWS_AS(load_dense(t), std::runtime_error);
    }

    // 8-bit codes read as 4-bit: twice the code bytes the shape allows
    std::string narrowed = bytes;
    narrowed[storage_at] = 4;
    std::stringstream t(narrowed);
    REQUIRE_THROWS_AS(load_dense(t), std::runtime_error);

    // ranges cut to length 0 while slots hold codes
    std::string no_ranges = bytes;
    const uint64_t zero = 0;
    std::memcpy(no_ranges.data() + storage_at + 1, &zero, sizeof(zero));
    no_ranges.erase(storage_at + 1 + 8, dim * sizeof(float));
    std::stringstream u(no_ranges);
    REQUIRE_THROWS_AS(load_dense(u), std::runtime_error);
}

TEST_CASE("HNSW SQ: compact rebuilds from the codes and keeps ids", "[hnsw][sq][compact]") {
    const size_t n = 3000, dim = 16, nq = 100, k = 10;
    auto data = hnsw_random(n, dim, 58);
    auto queries = hnsw_random(nq, dim, 59);
    FlatIndex flat(dim);
    flat.add_batch(data, n);
    std::vector<int64_t> dead;
    for (int64_t id = 0; id < static_cast<int64_t>(n); id += 3) dead.push_back(id);
    flat.remove(dead);
    const auto truth = flat.query_batch(queries, nq, k);

    for (int bits : {8, 4}) {
        CAPTURE(bits);
        auto owned = sq_hnsw(bits, data, n, dim, 4);
        HNSWIndex& hnsw = *owned;
        hnsw.remove(dead);
        REQUIRE(hnsw.compact() == dead.size());
        REQUIRE(hnsw.size() == n - dead.size());

        const auto got = hnsw.query_batch(queries, nq, k);
        for (const auto& row : got)
            for (const auto& nb : row) REQUIRE(nb.index % 3 != 0);
        REQUIRE(hnsw_recall(got, truth) >= (bits == 8 ? 0.9 : 0.5));
    }
}

TEST_CASE("HNSW SQ: cosine trains on unit vectors", "[hnsw][sq][metric]") {
    const size_t n = 2000, dim = 16, nq = 50, k = 10;
    auto data = hnsw_random(n, dim, 60);
    for (size_t i = 0; i < n * dim; ++i) data[i] *= 1.0f + static_cast<float>(i % 7);   // lengths don't matter
    auto queries = hnsw_random(nq, dim, 61);
    FlatIndex flat(dim, 1024, false, Metric::Cosine);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);
    for (int bits : {8, 4}) {
        CAPTURE(bits);
        auto owned = sq_hnsw(bits, data, n, dim, 2, Metric::Cosine);
        REQUIRE(hnsw_recall(owned->query_batch(queries, nq, k), truth) >= (bits == 8 ? 0.9 : 0.5));
    }
}
