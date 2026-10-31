#include <catch2/catch_test_macros.hpp>

#include "load_dense.hpp"

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"

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

static std::vector<float> sq_random(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = d(rng);
    return v;
}

// Mean fraction of each truth row's ids found in the matching result row.
static double sq_recall(const std::vector<std::vector<Neighbor>>& got, const std::vector<std::vector<Neighbor>>& truth) {
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

static void sq_require_identical(const std::vector<std::vector<Neighbor>>& a, const std::vector<std::vector<Neighbor>>& b) {
    REQUIRE(a.size() == b.size());
    for (size_t q = 0; q < a.size(); ++q) {
        REQUIRE(a[q].size() == b[q].size());
        for (size_t i = 0; i < a[q].size(); ++i) {
            REQUIRE(a[q][i].index == b[q][i].index);
            REQUIRE(a[q][i].distance == b[q][i].distance);
        }
    }
}

static std::unique_ptr<Index> sq_roundtrip(const Index& index) {
    std::stringstream s;
    index.save(s);
    return load_dense(s);
}

// A trained IVFIndex with SQ codes of `bits` (0 = floats), not yet filled.
static std::unique_ptr<IVFIndex> sq_ivf(int bits, const std::vector<float>& data, size_t n, size_t dim,
                                        Metric metric = Metric::L2) {
    auto ivf = std::make_unique<IVFIndex>(dim, 100, 20, false, metric);
    if (bits) ivf->enable_sq(bits);
    ivf->train(data, n, 25, 1);
    return ivf;
}

TEST_CASE("IVF SQ: 8 bits keeps float recall, 4 bits does after re-ranking", "[ivf][sq]") {
    const size_t n = 20000, dim = 32, nq = 200, k = 10;
    auto data = sq_random(n, dim, 1);
    auto queries = sq_random(nq, dim, 2);
    FlatIndex flat(dim);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);

    auto floats_owned = sq_ivf(0, data, n, dim);
    IVFIndex& floats = *floats_owned;
    floats.add_batch(data, n);
    const double float_recall = sq_recall(floats.query_batch(queries, nq, k), truth);

    auto sq8_owned = sq_ivf(8, data, n, dim);
    IVFIndex& sq8 = *sq8_owned;
    REQUIRE(sq8.sq_bits() == 8);
    sq8.add_batch(data, n);
    REQUIRE(sq_recall(sq8.query_batch(queries, nq, k), truth) >= float_recall - 0.03);

    // query and query_batch agree (query_batch picks probes with its own batched coarse step)
    for (size_t q = 0; q < 20; ++q) {
        const auto one = sq8.query({queries.data() + q * dim, dim}, k);
        REQUIRE(one.size() == k);
        for (size_t i = 1; i < one.size(); ++i) REQUIRE(one[i - 1].distance <= one[i].distance);
    }

    auto sq4_owned = sq_ivf(4, data, n, dim);
    IVFIndex& sq4 = *sq4_owned;
    RefineIndex refine(sq4, 8);
    refine.add_batch(data, n);
    const auto got = refine.query_batch(queries, nq, k);
    REQUIRE(sq_recall(got, truth) >= float_recall - 0.02);
    for (size_t q = 0; q < nq; ++q)
        for (const auto& nb : got[q])
            REQUIRE(nb.distance == l2_distance(queries.data() + q * dim, data.data() + nb.index * dim, dim));
}

TEST_CASE("IVF SQ: enable_sq is checked", "[ivf][sq]") {
    auto data = sq_random(500, 8, 3);
    IVFIndex ivf(8, 100);
    REQUIRE(ivf.sq_bits() == 0);
    for (int bad : {0, 1, 2, 6, 16}) REQUIRE_THROWS_AS(ivf.enable_sq(bad), std::invalid_argument);
    ivf.enable_sq(4);
    REQUIRE_THROWS_AS(ivf.enable_pq(4), std::logic_error);   // not with PQ, either order
    IVFIndex pq(8, 100);
    pq.enable_pq(4);
    REQUIRE_THROWS_AS(pq.enable_sq(8), std::logic_error);

    REQUIRE_THROWS_AS(ivf.add_batch(data, 500), std::logic_error);   // not trained
    ivf.train(data, 500, 10, 1);
    REQUIRE_THROWS_AS(ivf.enable_sq(8), std::logic_error);   // after train
    ivf.add_batch(data, 500);
    REQUIRE(ivf.size() == 500);

    IndexOptions opts;
    opts.ivf_sq_bits = 8;
    auto made = make_index(IndexType::IVF, 8, opts);
    REQUIRE(dynamic_cast<IVFIndex&>(*made).sq_bits() == 8);
    opts.pq_subspaces = 4;
    REQUIRE_THROWS_AS(make_index(IndexType::IVF, 8, opts), std::logic_error);
}

TEST_CASE("IVF SQ: save/load keeps codes, ranges and an untrained index", "[ivf][sq][serialize]") {
    const size_t n = 3000, dim = 17, nq = 50, k = 10;   // odd dim: 4-bit rows end in half a byte
    auto data = sq_random(n, dim, 4);
    auto queries = sq_random(nq, dim, 5);
    for (int bits : {8, 4}) {
        CAPTURE(bits);
        auto ivf_owned = sq_ivf(bits, data, n, dim);
    IVFIndex& ivf = *ivf_owned;
        ivf.add_batch(data, n);
        std::vector<int64_t> dead{3, 30, 300};
        ivf.remove(dead);

        auto loaded_base = sq_roundtrip(ivf);
        auto& loaded = dynamic_cast<IVFIndex&>(*loaded_base);
        REQUIRE(loaded.sq_bits() == bits);
        REQUIRE(loaded.size() == n - dead.size());
        sq_require_identical(loaded.query_batch(queries, nq, k), ivf.query_batch(queries, nq, k));

        IVFIndex untrained(dim, 100);
        untrained.enable_sq(bits);
        auto loaded_untrained = sq_roundtrip(untrained);
        auto& u = dynamic_cast<IVFIndex&>(*loaded_untrained);
        REQUIRE(u.sq_bits() == bits);
        u.train(data, n, 10, 1);
        u.add_batch(data, n);
        REQUIRE(u.size() == n);
    }
}

TEST_CASE("IVF SQ: load rejects bad SQ settings", "[ivf][sq][serialize]") {
    const size_t n = 1000, dim = 8;
    auto data = sq_random(n, dim, 6);
    auto ivf_owned = sq_ivf(8, data, n, dim);
    IVFIndex& ivf = *ivf_owned;
    ivf.add_batch(data, n);
    std::stringstream s;
    ivf.save(s);
    const std::string bytes = s.str();

    // the SQ byte follows the header, the shape fields, the coarse centroids and the PQ flag
    const size_t sq_at = 9 + 4 * 8 + 1 + 1 + 8 + 100 * dim * 4 + 1;
    REQUIRE(bytes[sq_at] == 8);
    for (char bad : {1, 2, 5, 16}) {
        std::string b = bytes;
        b[sq_at] = bad;
        std::stringstream t(b);
        REQUIRE_THROWS_AS(load_dense(t), std::runtime_error);
    }

    // 8-bit codes read as 4-bit: list lengths no longer match
    std::string narrowed = bytes;
    narrowed[sq_at] = 4;
    std::stringstream t(narrowed);
    REQUIRE_THROWS_AS(load_dense(t), std::runtime_error);

    // ranges cut to length 0 on a trained index
    std::string no_ranges = bytes;
    const uint64_t zero = 0;
    std::memcpy(no_ranges.data() + sq_at + 1, &zero, sizeof(zero));
    no_ranges.erase(sq_at + 1 + 8, dim * sizeof(float));
    std::stringstream u(no_ranges);
    REQUIRE_THROWS_AS(load_dense(u), std::runtime_error);

    SECTION("truncated") {
        for (size_t cut : {sq_at, sq_at + 5, bytes.size() / 2, bytes.size() - 1}) {
            std::stringstream c(bytes.substr(0, cut));
            REQUIRE_THROWS_AS(load_dense(c), std::runtime_error);
        }
    }
}

TEST_CASE("IVF SQ: compact moves codes and keeps ids", "[ivf][sq][compact]") {
    const size_t n = 5000, dim = 16, nq = 100, k = 10;
    auto data = sq_random(n, dim, 7);
    auto queries = sq_random(nq, dim, 8);
    std::vector<int64_t> dead;
    for (int64_t id = 0; id < static_cast<int64_t>(n); id += 3) dead.push_back(id);

    for (int bits : {8, 4}) {
        CAPTURE(bits);
        auto ivf_owned = sq_ivf(bits, data, n, dim);
    IVFIndex& ivf = *ivf_owned;
        ivf.add_batch(data, n);
        ivf.remove(dead);
        const auto before = ivf.query_batch(queries, nq, k);
        REQUIRE(ivf.compact() == dead.size());
        REQUIRE(ivf.size() == n - dead.size());
        sq_require_identical(ivf.query_batch(queries, nq, k), before);   // same codes, same results
        for (const auto& row : before)
            for (const auto& nb : row) REQUIRE(nb.index % 3 != 0);
    }
}

TEST_CASE("IVF SQ: cosine learns ranges on unit vectors", "[ivf][sq][metric]") {
    const size_t n = 5000, dim = 16, nq = 50, k = 10;
    auto data = sq_random(n, dim, 9);
    for (size_t i = 0; i < n * dim; ++i) data[i] *= 1.0f + static_cast<float>(i % 7);   // lengths don't matter
    auto queries = sq_random(nq, dim, 10);
    FlatIndex flat(dim, 1024, false, Metric::Cosine);
    flat.add_batch(data, n);
    auto ivf_owned = sq_ivf(8, data, n, dim, Metric::Cosine);
    IVFIndex& ivf = *ivf_owned;
    ivf.add_batch(data, n);
    const auto got = ivf.query_batch(queries, nq, k);
    REQUIRE(sq_recall(got, flat.query_batch(queries, nq, k)) >= 0.85);
    for (const auto& row : got)
        for (const auto& nb : row) REQUIRE(nb.distance <= 2.05f);   // cosine distances, roughly in [0, 2]
}
