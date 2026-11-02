#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "vecengine/fusion.hpp"
#include "vecengine/threads.hpp"

#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

using namespace vecengine;
using Catch::Matchers::WithinRel;

static vector<FusedHit> fuse(const vector<vector<int64_t>>& lists, vector<float> weights = {}, float k = 60.0f,
                             size_t limit = 100) {
    vector<span<const int64_t>> spans(lists.begin(), lists.end());
    return vecengine::fuse(FusionMethod::RRF, spans, {}, weights, k, limit);
}

static vector<FusedHit> fuse_linear(const vector<vector<int64_t>>& lists, const vector<vector<float>>& dists,
                                    vector<float> weights = {}, size_t limit = 100) {
    vector<span<const int64_t>> id_spans(lists.begin(), lists.end());
    vector<span<const float>> dist_spans(dists.begin(), dists.end());
    return vecengine::fuse(FusionMethod::Linear, id_spans, dist_spans, weights, 60.0f, limit);
}

static vector<int64_t> ids_of(const vector<FusedHit>& hits) {
    vector<int64_t> out;
    for (const auto& h : hits) out.push_back(h.id);
    return out;
}

TEST_CASE("RRF scores ids by their ranks", "[fusion]") {
    // 7: rank 2 and 1; 3: rank 1 only; 9: rank 3 and 3; 4: rank 2 only
    const auto hits = fuse({{3, 7, 9}, {7, 4, 9}});
    REQUIRE(ids_of(hits) == vector<int64_t>{7, 9, 3, 4});
    REQUIRE_THAT(hits[0].score, WithinRel(1.0f / 62 + 1.0f / 61, 1e-6f));
    REQUIRE_THAT(hits[1].score, WithinRel(2.0f / 63, 1e-6f));
    REQUIRE_THAT(hits[2].score, WithinRel(1.0f / 61, 1e-6f));
    REQUIRE_THAT(hits[3].score, WithinRel(1.0f / 62, 1e-6f));
}

TEST_CASE("RRF: agreement beats a single top rank", "[fusion]") {
    // 5 is 5th in both lists, 1 is 1st in only one: 2/65 > 1/61
    const auto hits = fuse({{1, 10, 11, 12, 5}, {20, 21, 22, 23, 5}});
    REQUIRE(hits[0].id == 5);
}

TEST_CASE("RRF: k, weights, limit and ties", "[fusion]") {
    SECTION("k changes the scores") {
        const auto hits = fuse({{1, 2}}, {}, 1.0f);
        REQUIRE_THAT(hits[0].score, WithinRel(0.5f, 1e-6f));
        REQUIRE_THAT(hits[1].score, WithinRel(1.0f / 3, 1e-6f));
    }
    SECTION("weights scale each list") {
        const auto hits = fuse({{1}, {2}}, {1.0f, 3.0f});
        REQUIRE(ids_of(hits) == vector<int64_t>{2, 1});
        REQUIRE_THAT(hits[0].score, WithinRel(3.0f / 61, 1e-6f));
    }
    SECTION("weight 0 ignores a list entirely") {
        REQUIRE(ids_of(fuse({{1, 2}, {3, 4}}, {1.0f, 0.0f})) == vector<int64_t>{1, 2});
    }
    SECTION("limit keeps the best") {
        REQUIRE(ids_of(fuse({{4, 3, 2, 1}}, {}, 60.0f, 2)) == vector<int64_t>{4, 3});
    }
    SECTION("equal scores: smaller id first") {
        REQUIRE(ids_of(fuse({{9, 2}, {2, 9}})) == vector<int64_t>{2, 9});
        REQUIRE(ids_of(fuse({{8}, {3}})) == vector<int64_t>{3, 8});
    }
}

TEST_CASE("RRF: padding, repeats and empty lists", "[fusion]") {
    SECTION("negative ids are skipped") {
        REQUIRE(ids_of(fuse({{5, -1, -1}, {-1}})) == vector<int64_t>{5});
    }
    SECTION("a repeat within a list counts once, at its first rank") {
        const auto hits = fuse({{6, 6, 7}});
        REQUIRE(ids_of(hits) == vector<int64_t>{6, 7});
        REQUIRE_THAT(hits[0].score, WithinRel(1.0f / 61, 1e-6f));
        REQUIRE_THAT(hits[1].score, WithinRel(1.0f / 63, 1e-6f));
    }
    SECTION("empty lists give nothing") {
        REQUIRE(fuse({{}, {}}).empty());
    }
}

TEST_CASE("fuse rejects bad arguments", "[fusion]") {
    REQUIRE_THROWS_AS(fuse({}), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}, {2}}, {1.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}}, {-1.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}}, {std::numeric_limits<float>::infinity()}), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}}, {}, 0.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}}, {}, std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);

    const vector<int64_t> a{1, 2, 3, 4}, b{1, 2, 3};
    const vector<span<const int64_t>> lists{a, b};
    const vector<size_t> widths{2, 2};
    REQUIRE_THROWS_AS(fuse_batch(FusionMethod::RRF, lists, {}, widths, 2, {}, 60.0f, 2), std::invalid_argument);   // b is 3 ids, not 2 x 2
}

TEST_CASE("fuse_batch with RRF matches single queries", "[fusion]") {
    const size_t rows = 300, wa = 20, wb = 7;
    std::mt19937 rng(5);
    vector<int64_t> a(rows * wa), b(rows * wb);
    for (auto& x : a) x = static_cast<int64_t>(rng() % 50) - 2;   // repeats and some padding
    for (auto& x : b) x = static_cast<int64_t>(rng() % 50) - 2;
    const vector<span<const int64_t>> lists{a, b};
    const vector<size_t> widths{wa, wb};
    const vector<float> weights{1.0f, 2.0f};

    set_num_threads(4);
    const auto batch = fuse_batch(FusionMethod::RRF, lists, {}, widths, rows, weights, 60.0f, 10);
    set_num_threads(0);
    REQUIRE(batch.size() == rows);
    for (size_t q = 0; q < rows; ++q) {
        const vector<span<const int64_t>> row{span<const int64_t>(a).subspan(q * wa, wa),
                                              span<const int64_t>(b).subspan(q * wb, wb)};
        const auto one = vecengine::fuse(FusionMethod::RRF, row, {}, weights, 60.0f, 10);
        REQUIRE(ids_of(batch[q]) == ids_of(one));
        for (size_t i = 0; i < one.size(); ++i) REQUIRE(batch[q][i].score == one[i].score);
    }
}

TEST_CASE("linear fusion: min-max normalized distances, weighted", "[fusion]") {
    // list a: 1 nearest (d 0 -> 1), 2 at d 5 (0.5), 3 farthest (d 10 -> 0); list b: 3 at -4 (1), 1 at -2 (0)
    const vector<vector<int64_t>> ids{{1, 2, 3}, {3, 1}};
    const vector<vector<float>> dists{{0.0f, 5.0f, 10.0f}, {-4.0f, -2.0f}};
    SECTION("equal weights") {
        const auto hits = fuse_linear(ids, dists);
        REQUIRE(ids_of(hits) == vector<int64_t>{1, 3, 2});   // 1 + 0, 0 + 1 (tie: smaller id), 0.5
        REQUIRE_THAT(hits[0].score, WithinRel(1.0f, 1e-6f));
        REQUIRE_THAT(hits[2].score, WithinRel(0.5f, 1e-6f));
    }
    SECTION("weights decide") {
        const auto hits = fuse_linear(ids, dists, {0.25f, 1.0f});
        REQUIRE(ids_of(hits) == vector<int64_t>{3, 1, 2});   // 0 + 1, 0.25 + 0, 0.125
        REQUIRE_THAT(hits[0].score, WithinRel(1.0f, 1e-6f));
    }
}

TEST_CASE("linear fusion: equal distances, padding, repeats, errors", "[fusion]") {
    SECTION("a list whose hits are all equally near scores them all 1") {
        const auto hits = fuse_linear({{4, 5}}, {{2.0f, 2.0f}});
        REQUIRE_THAT(hits[0].score, WithinRel(1.0f, 1e-6f));
        REQUIRE_THAT(hits[1].score, WithinRel(1.0f, 1e-6f));
    }
    SECTION("padding and non-finite distances are skipped, also for min and max") {
        const float inf = std::numeric_limits<float>::infinity();
        const auto hits = fuse_linear({{1, 2, -1}}, {{1.0f, 3.0f, inf}});
        REQUIRE(ids_of(hits) == vector<int64_t>{1, 2});
        REQUIRE_THAT(hits[0].score, WithinRel(1.0f, 1e-6f));
        REQUIRE(hits[1].score == 0.0f);
        REQUIRE(ids_of(fuse_linear({{7, 8}}, {{std::numeric_limits<float>::quiet_NaN(), 1.0f}})) == vector<int64_t>{8});
    }
    SECTION("a repeat within a list counts once, at its first position") {
        const auto hits = fuse_linear({{6, 7, 6}}, {{0.0f, 1.0f, 2.0f}});
        REQUIRE(ids_of(hits) == vector<int64_t>{6, 7});
        REQUIRE_THAT(hits[0].score, WithinRel(1.0f, 1e-6f));
    }
    SECTION("distances are required, one per id") {
        const vector<int64_t> a{1, 2};
        const vector<float> d{0.0f};
        const vector<span<const int64_t>> ids{a};
        const vector<span<const float>> one{d};
        REQUIRE_THROWS_AS(vecengine::fuse(FusionMethod::Linear, ids, {}, {}, 60.0f, 5), std::invalid_argument);
        REQUIRE_THROWS_AS(vecengine::fuse(FusionMethod::Linear, ids, one, {}, 60.0f, 5), std::invalid_argument);
    }
    SECTION("k is not used, so any k is accepted") {
        REQUIRE(fuse_linear({{1}}, {{0.0f}}).size() == 1);
        const vector<int64_t> a{1};
        const vector<float> d{0.0f};
        const vector<span<const int64_t>> ids{a};
        const vector<span<const float>> ds{d};
        REQUIRE(vecengine::fuse(FusionMethod::Linear, ids, ds, {}, 0.0f, 5).size() == 1);
    }
}

TEST_CASE("fuse_batch with linear fusion matches single queries", "[fusion]") {
    const size_t rows = 200, wa = 12, wb = 5;
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> u(-5.0f, 5.0f);
    vector<int64_t> a(rows * wa), b(rows * wb);
    vector<float> da(rows * wa), db(rows * wb);
    for (auto& x : a) x = static_cast<int64_t>(rng() % 40) - 2;
    for (auto& x : b) x = static_cast<int64_t>(rng() % 40) - 2;
    for (auto& x : da) x = u(rng);
    for (auto& x : db) x = u(rng);
    const vector<span<const int64_t>> lists{a, b};
    const vector<span<const float>> dists{da, db};
    const vector<size_t> widths{wa, wb};
    const vector<float> weights{0.7f, 0.3f};

    set_num_threads(4);
    const auto batch = fuse_batch(FusionMethod::Linear, lists, dists, widths, rows, weights, 60.0f, 8);
    set_num_threads(0);
    for (size_t q = 0; q < rows; ++q) {
        const vector<span<const int64_t>> id_row{span<const int64_t>(a).subspan(q * wa, wa),
                                                 span<const int64_t>(b).subspan(q * wb, wb)};
        const vector<span<const float>> dist_row{span<const float>(da).subspan(q * wa, wa),
                                                 span<const float>(db).subspan(q * wb, wb)};
        const auto one = vecengine::fuse(FusionMethod::Linear, id_row, dist_row, weights, 60.0f, 8);
        REQUIRE(ids_of(batch[q]) == ids_of(one));
        for (size_t i = 0; i < one.size(); ++i) REQUIRE(batch[q][i].score == one[i].score);
    }
}
