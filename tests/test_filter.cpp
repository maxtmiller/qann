#include <catch2/catch_test_macros.hpp>

#include "vecengine/distance.hpp"
#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <span>
#include <string>
#include <vector>

using namespace vecengine;

static std::vector<float> filter_random(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = d(rng);
    return v;
}

// Every `step`-th id below n, starting at `offset`.
static std::vector<int64_t> every(size_t n, size_t step, size_t offset = 0) {
    std::vector<int64_t> ids;
    for (size_t i = offset; i < n; i += step) ids.push_back(static_cast<int64_t>(i));
    return ids;
}

// Exact top k of the rows of data whose id (row number) is in allowed.
static std::vector<std::vector<Neighbor>> filtered_truth(const std::vector<float>& data, size_t dim,
                                                         const std::vector<float>& queries, size_t nq, size_t k,
                                                         const std::vector<int64_t>& allowed) {
    std::vector<std::vector<Neighbor>> truth(nq);
    for (size_t q = 0; q < nq; ++q) {
        std::vector<Neighbor> all;
        for (int64_t id : allowed)
            all.push_back({static_cast<size_t>(id), l2_distance(queries.data() + q * dim, data.data() + id * dim, dim)});
        std::sort(all.begin(), all.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.distance < b.distance || (a.distance == b.distance && a.index < b.index);
        });
        all.resize(std::min(k, all.size()));
        truth[q] = all;
    }
    return truth;
}

static double filter_recall(const std::vector<std::vector<Neighbor>>& got, const std::vector<std::vector<Neighbor>>& truth) {
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

static void require_only_allowed(const std::vector<std::vector<Neighbor>>& got, const std::vector<int64_t>& allowed) {
    const std::set<int64_t> ok(allowed.begin(), allowed.end());
    for (const auto& row : got)
        for (const auto& nb : row) REQUIRE(ok.count(static_cast<int64_t>(nb.index)) == 1);
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

namespace {

// One index under test plus whatever it wraps.
struct Built {
    std::string name;
    std::unique_ptr<Index> base;   // the RefineIndex's base, or nullptr
    std::unique_ptr<Index> index;  // what queries run on
};

} // namespace

// Every index type, filled with data (row i gets id i).
static std::vector<Built> build_all(const std::vector<float>& data, size_t n, size_t dim) {
    std::vector<Built> out;

    auto flat = std::make_unique<FlatIndex>(dim);
    flat->add_batch(data, n);
    out.push_back({"flat", nullptr, std::move(flat)});

    auto ivf = std::make_unique<IVFIndex>(dim, 100, 10);
    ivf->train(data, n, 25, 1);
    ivf->add_batch(data, n);
    out.push_back({"ivf", nullptr, std::move(ivf)});

    auto ivf_sq = std::make_unique<IVFIndex>(dim, 100, 10);
    ivf_sq->enable_sq(8);
    ivf_sq->train(data, n, 25, 1);
    ivf_sq->add_batch(data, n);
    out.push_back({"ivf-sq8", nullptr, std::move(ivf_sq)});

    auto ivf_fs = std::make_unique<IVFIndex>(dim, 100, 10);
    ivf_fs->enable_pq(dim / 2, 16, true);
    ivf_fs->train(data, n, 25, 1);
    auto refine_fs = std::make_unique<RefineIndex>(*ivf_fs, 10);
    refine_fs->add_batch(data, n);
    out.push_back({"ivf-fastscan+R", std::move(ivf_fs), std::move(refine_fs)});

    auto hnsw = std::make_unique<HNSWIndex>(dim, 16, 100, false, 1);
    hnsw->add_batch(data, n);
    out.push_back({"hnsw", nullptr, std::move(hnsw)});

    auto hnsw_sq = std::make_unique<HNSWIndex>(dim, 16, 100, false, 1);
    hnsw_sq->enable_sq(8);
    hnsw_sq->train(data, n);
    auto refine_hnsw = std::make_unique<RefineIndex>(*hnsw_sq, 4);
    refine_hnsw->add_batch(data, n);
    out.push_back({"hnsw-sq8+R", std::move(hnsw_sq), std::move(refine_hnsw)});

    return out;
}

TEST_CASE("Filtered search returns only allowed ids, with good recall, on every index", "[filter]") {
    const size_t n = 10000, dim = 16, nq = 50, k = 10;
    const auto data = filter_random(n, dim, 1);
    const auto queries = filter_random(nq, dim, 2);
    auto indexes = build_all(data, n, dim);

    // with 10% allowed, the filtered top k lie about as deep as the unfiltered
    // top 10 * k, so that is the recall a filtered query should match
    const auto truth_deep = filtered_truth(data, dim, queries, nq, 10 * k, every(n, 1));
    for (const auto& [name, base, index] : indexes) {
        INFO(name);
        const double min_recall = filter_recall(index->query_batch(queries, nq, 10 * k), truth_deep) - 0.05;
        INFO("unfiltered recall at 10k " << min_recall + 0.05);

        // 10% selectivity
        const auto tenth = every(n, 10, 3);
        const auto got = index->query_batch(queries, nq, k, tenth);
        require_only_allowed(got, tenth);
        for (const auto& row : got) REQUIRE(row.size() == k);
        REQUIRE(filter_recall(got, filtered_truth(data, dim, queries, nq, k, tenth)) >= min_recall);

        // 50%: HNSW searches its graph rather than scoring the allowed vectors directly
        const auto half = every(n, 2, 1);
        const auto got_half = index->query_batch(queries, nq, k, half);
        require_only_allowed(got_half, half);
        REQUIRE(filter_recall(got_half, filtered_truth(data, dim, queries, nq, k, half)) >= min_recall);

        // a handful of ids: every one comes back
        const std::vector<int64_t> few = {5, 4321, 77, 9999, 1234};
        for (const auto& row : index->query_batch(queries, nq, k, few)) {
            std::set<int64_t> ids;
            for (const auto& nb : row) ids.insert(static_cast<int64_t>(nb.index));
            REQUIRE(ids == std::set<int64_t>(few.begin(), few.end()));
        }

        // an empty or all-unknown set returns nothing
        const std::vector<int64_t> none;
        const std::vector<int64_t> unknown = {-1, static_cast<int64_t>(n), int64_t{1} << 40};
        REQUIRE(index->query(std::span<const float>(queries.data(), dim), k, none).empty());
        for (const auto& row : index->query_batch(queries, nq, k, unknown)) REQUIRE(row.empty());

        // every id allowed gives the unfiltered results, and query matches query_batch
        const auto all = every(n, 1);
        const auto got_all = index->query_batch(queries, nq, k, all);
        for (size_t q = 0; q < nq; ++q) {
            const std::span<const float> one(queries.data() + q * dim, dim);
            require_identical({index->query(one, k)}, {got_all[q]});
            require_identical({index->query(one, k, tenth)}, {got[q]});
        }

        // deleted ids are never returned, even when allowed
        const auto fifth = every(n, 5);
        index->remove(every(n, 10));  // half of the allowed ids
        std::vector<int64_t> live;
        for (int64_t id : fifth) if (id % 10 != 0) live.push_back(id);
        const auto got_live = index->query_batch(queries, nq, k, fifth);
        require_only_allowed(got_live, live);
        REQUIRE(filter_recall(got_live, filtered_truth(data, dim, queries, nq, k, live)) >= min_recall);
    }
}

TEST_CASE("Filtered search with custom ids and after compaction", "[filter]") {
    const size_t n = 3000, dim = 8, nq = 20, k = 5;
    const auto data = filter_random(n, dim, 3);
    const auto queries = filter_random(nq, dim, 4);

    SECTION("custom ids") {
        // id of row i is 1000 + 3 * i
        std::vector<int64_t> ids(n);
        for (size_t i = 0; i < n; ++i) ids[i] = 1000 + 3 * static_cast<int64_t>(i);

        FlatIndex flat(dim, n, true);
        flat.add_batch(data, n, ids.data());
        HNSWIndex hnsw(dim, 16, 100, true, 1);
        hnsw.add_batch(data, n, ids.data());

        // rows 0, 10, 20, ... by custom id, plus ids that were never added
        std::vector<int64_t> allowed, rows;
        for (size_t i = 0; i < n; i += 10) {
            allowed.push_back(ids[i]);
            rows.push_back(static_cast<int64_t>(i));
        }
        allowed.push_back(0);
        allowed.push_back(1001);

        const auto truth = filtered_truth(data, dim, queries, nq, k, rows);
        for (Index* index : {static_cast<Index*>(&flat), static_cast<Index*>(&hnsw)}) {
            const auto got = index->query_batch(queries, nq, k, allowed);
            for (size_t q = 0; q < nq; ++q) {
                REQUIRE(got[q].size() == k);
                for (size_t i = 0; i < k; ++i) REQUIRE(got[q][i].index == ids[truth[q][i].index]);
            }
        }
    }

    SECTION("auto ids kept through compaction") {
        FlatIndex flat(dim);
        flat.add_batch(data, n);
        flat.remove(every(n, 2));  // drop the even ids, so odd id i now sits in slot i / 2
        REQUIRE(flat.compact() == n / 2);

        const auto allowed = every(n, 3);  // multiples of 3; the even ones are gone
        std::vector<int64_t> live;
        for (int64_t id : allowed) if (id % 2 == 1) live.push_back(id);

        const auto got = flat.query_batch(queries, nq, k, allowed);
        const auto truth = filtered_truth(data, dim, queries, nq, k, live);
        for (size_t q = 0; q < nq; ++q) {
            REQUIRE(got[q].size() == k);
            for (size_t i = 0; i < k; ++i) REQUIRE(got[q][i].index == truth[q][i].index);
        }
    }
}

TEST_CASE("Filtered IVF probes past nprobe until it has k allowed vectors", "[filter][ivf]") {
    const size_t n = 10000, dim = 16, nq = 30, k = 10;
    const auto data = filter_random(n, dim, 5);
    const auto queries = filter_random(nq, dim, 6);

    IVFIndex ivf(dim, 100, 1);
    ivf.train(data, n, 25, 1);
    ivf.add_batch(data, n);

    // 2%: about two allowed vectors per list, so one probe finds at most a few
    // (above nlist allowed, so the lists are probed rather than scored directly)
    const auto allowed = every(n, 50, 11);
    const auto got = ivf.query_batch(queries, nq, k, allowed);
    require_only_allowed(got, allowed);
    for (const auto& row : got) REQUIRE(row.size() == k);

    // lists are probed nearest first, so recall is at least what one probe gets without a filter
    const double unfiltered = filter_recall(ivf.query_batch(queries, nq, k), filtered_truth(data, dim, queries, nq, k, every(n, 1)));
    INFO("unfiltered recall " << unfiltered);
    REQUIRE(filter_recall(got, filtered_truth(data, dim, queries, nq, k, allowed)) >= unfiltered);
}

TEST_CASE("RefineIndex over IVF scores a narrow filter's vectors directly, exactly", "[filter][ivf]") {
    const size_t n = 10000, dim = 16, nq = 30, k = 10;
    const auto data = filter_random(n, dim, 7);
    const auto queries = filter_random(nq, dim, 8);

    // nlist 100, nprobe 1: the base path would miss most allowed vectors
    IVFIndex pq(dim, 100, 1);
    pq.enable_pq(8, 16, true);
    pq.train(data, n, 25, 1);
    std::vector<int64_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = static_cast<int64_t>(1000 + 3 * i);
    RefineIndex refine(pq, 4, true);
    refine.add_batch(data, n, ids.data());

    // 100 allowed (rows 5, 105, ...), under the 2 * nlist cutoff; two of them deleted
    std::vector<int64_t> rows = every(n, 100, 5), allowed;
    for (int64_t r : rows) allowed.push_back(ids[r]);
    refine.remove(std::vector<int64_t>{ids[rows[0]], ids[rows[1]]});
    const std::vector<int64_t> live_rows(rows.begin() + 2, rows.end());

    auto truth = filtered_truth(data, dim, queries, nq, k, live_rows);
    for (auto& row : truth)
        for (auto& nb : row) nb.index = static_cast<size_t>(ids[nb.index]);
    const auto batch = refine.query_batch(queries, nq, k, allowed);
    REQUIRE(batch.size() == nq);
    for (size_t q = 0; q < nq; ++q) {
        const auto one = refine.query(std::span<const float>(queries.data() + q * dim, dim), k, allowed);
        REQUIRE(batch[q].size() == k);
        REQUIRE(one.size() == k);
        for (size_t i = 0; i < k; ++i) {
            REQUIRE(batch[q][i].index == truth[q][i].index);
            REQUIRE(batch[q][i].distance == truth[q][i].distance);
            REQUIRE(one[i].index == truth[q][i].index);
            REQUIRE(one[i].distance == truth[q][i].distance);
        }
    }

    // fewer allowed than k: every one comes back
    const std::vector<int64_t> three = {ids[7], ids[8], ids[9]};
    const auto few = refine.query(std::span<const float>(queries.data(), dim), k, three);
    REQUIRE(few.size() == 3);

    // 500 allowed, above the cutoff: the base path, still only allowed vectors
    std::vector<int64_t> wide;
    for (int64_t r : every(n, 20, 3)) wide.push_back(ids[r]);
    const auto got = refine.query_batch(queries, nq, k, wide);
    require_only_allowed(got, wide);
    for (const auto& row : got) REQUIRE(row.size() == k);
}

TEST_CASE("Bare IVF scores a narrow filter's vectors directly: exact for plain, through compact and load", "[filter][ivf]") {
    const size_t n = 10000, dim = 16, nq = 30, k = 10;
    const auto data = filter_random(n, dim, 9);
    const auto queries = filter_random(nq, dim, 10);

    // nlist 100, nprobe 1: probing would miss most allowed vectors
    IVFIndex ivf(dim, 100, 1, true);
    ivf.train(data, n, 25, 1);
    std::vector<int64_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = static_cast<int64_t>(500 + 7 * i);
    ivf.add_batch(data, n, ids.data());

    // 80 allowed (under nlist), two of them deleted, then compacted
    const std::vector<int64_t> rows = every(n, 125, 4);
    std::vector<int64_t> allowed;
    for (int64_t r : rows) allowed.push_back(ids[r]);
    ivf.remove(std::vector<int64_t>{allowed[0], allowed[1], ids[1], ids[2]});
    const std::vector<int64_t> live_rows(rows.begin() + 2, rows.end());
    auto truth = filtered_truth(data, dim, queries, nq, k, live_rows);
    for (auto& row : truth)
        for (auto& nb : row) nb.index = static_cast<size_t>(ids[nb.index]);

    auto require_exact = [&](const Index& index) {
        const auto batch = index.query_batch(queries, nq, k, allowed);
        for (size_t q = 0; q < nq; ++q) {
            const auto one = index.query(std::span<const float>(queries.data() + q * dim, dim), k, allowed);
            REQUIRE(batch[q].size() == k);
            for (size_t i = 0; i < k; ++i) {
                REQUIRE(batch[q][i].index == truth[q][i].index);
                REQUIRE(batch[q][i].distance == truth[q][i].distance);
                REQUIRE(one[i].index == truth[q][i].index);
            }
        }
    };
    require_exact(ivf);
    ivf.compact();
    require_exact(ivf);
    std::stringstream buf;
    ivf.save(buf);
    require_exact(*load_index(buf));

    // vectors added after compaction are found through the lookup too
    const auto extra = filter_random(1, dim, 11);
    const int64_t extra_id = 99;
    ivf.add_batch(extra, 1, &extra_id);
    std::vector<int64_t> with_extra = allowed;
    with_extra.push_back(extra_id);
    const auto self = ivf.query(std::span<const float>(extra.data(), dim), 1, with_extra);
    REQUIRE(self[0].index == 99);
    REQUIRE(self[0].distance == 0.0f);
}

TEST_CASE("Bare IVF SQ8 under a narrow filter scores every allowed vector", "[filter][ivf]") {
    const size_t n = 10000, dim = 16, nq = 30, k = 10;
    const auto data = filter_random(n, dim, 12);
    const auto queries = filter_random(nq, dim, 13);
    IVFIndex sq(dim, 100, 1);
    sq.enable_sq(8);
    sq.train(data, n, 25, 1);
    sq.add_batch(data, n);
    const auto allowed = every(n, 125, 4);    // 80, under nlist
    const auto got = sq.query_batch(queries, nq, k, allowed);
    require_only_allowed(got, allowed);
    // every allowed vector is scored, so only SQ's rounding can cost recall
    REQUIRE(filter_recall(got, filtered_truth(data, dim, queries, nq, k, allowed)) >= 0.95);

    // just above nlist: the lists are probed as before
    const auto wide = every(n, 99, 4);    // 102
    const auto probed = sq.query_batch(queries, nq, k, wide);
    require_only_allowed(probed, wide);
    for (const auto& row : probed) REQUIRE(row.size() == k);
}

TEST_CASE("Filtered cosine search reports cosine distances", "[filter][metric]") {
    const size_t n = 2000, dim = 8, k = 5;
    const auto data = filter_random(n, dim, 7);
    const auto query = filter_random(1, dim, 8);

    FlatIndex flat(dim, n, false, Metric::Cosine);
    flat.add_batch(data, n);
    const auto allowed = every(n, 4);
    const auto got = flat.query(query, k, allowed);
    REQUIRE(got.size() == k);
    for (const auto& nb : got) {
        REQUIRE(nb.index % 4 == 0);
        const float expected = cosine_distance(query.data(), data.data() + nb.index * dim, dim);
        REQUIRE(std::abs(nb.distance - expected) < 1e-5f);
    }
}
