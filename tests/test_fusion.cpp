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
    return rrf_fuse(spans, weights, k, limit);
}

static vector<int64_t> ids_of(const vector<FusedHit>& hits) {
    vector<int64_t> out;
    for (const auto& h : hits) out.push_back(h.id);
    return out;
}

TEST_CASE("rrf_fuse scores ids by their ranks", "[fusion]") {
    // 7: rank 2 and 1; 3: rank 1 only; 9: rank 3 and 3; 4: rank 2 only
    const auto hits = fuse({{3, 7, 9}, {7, 4, 9}});
    REQUIRE(ids_of(hits) == vector<int64_t>{7, 9, 3, 4});
    REQUIRE_THAT(hits[0].score, WithinRel(1.0f / 62 + 1.0f / 61, 1e-6f));
    REQUIRE_THAT(hits[1].score, WithinRel(2.0f / 63, 1e-6f));
    REQUIRE_THAT(hits[2].score, WithinRel(1.0f / 61, 1e-6f));
    REQUIRE_THAT(hits[3].score, WithinRel(1.0f / 62, 1e-6f));
}

TEST_CASE("rrf_fuse: agreement beats a single top rank", "[fusion]") {
    // 5 is 5th in both lists, 1 is 1st in only one: 2/65 > 1/61
    const auto hits = fuse({{1, 10, 11, 12, 5}, {20, 21, 22, 23, 5}});
    REQUIRE(hits[0].id == 5);
}

TEST_CASE("rrf_fuse: k, weights, limit and ties", "[fusion]") {
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

TEST_CASE("rrf_fuse: padding, repeats and empty lists", "[fusion]") {
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

TEST_CASE("rrf_fuse rejects bad arguments", "[fusion]") {
    REQUIRE_THROWS_AS(fuse({}), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}, {2}}, {1.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}}, {-1.0f}), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}}, {std::numeric_limits<float>::infinity()}), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}}, {}, 0.0f), std::invalid_argument);
    REQUIRE_THROWS_AS(fuse({{1}}, {}, std::numeric_limits<float>::quiet_NaN()), std::invalid_argument);

    const vector<int64_t> a{1, 2, 3, 4}, b{1, 2, 3};
    const vector<span<const int64_t>> lists{a, b};
    const vector<size_t> widths{2, 2};
    REQUIRE_THROWS_AS(rrf_fuse_batch(lists, widths, 2, {}, 60.0f, 2), std::invalid_argument);   // b is 3 ids, not 2 x 2
}

TEST_CASE("rrf_fuse_batch fuses row by row, like rrf_fuse", "[fusion]") {
    const size_t rows = 300, wa = 20, wb = 7;
    std::mt19937 rng(5);
    vector<int64_t> a(rows * wa), b(rows * wb);
    for (auto& x : a) x = static_cast<int64_t>(rng() % 50) - 2;   // repeats and some padding
    for (auto& x : b) x = static_cast<int64_t>(rng() % 50) - 2;
    const vector<span<const int64_t>> lists{a, b};
    const vector<size_t> widths{wa, wb};
    const vector<float> weights{1.0f, 2.0f};

    set_num_threads(4);
    const auto batch = rrf_fuse_batch(lists, widths, rows, weights, 60.0f, 10);
    set_num_threads(0);
    REQUIRE(batch.size() == rows);
    for (size_t q = 0; q < rows; ++q) {
        const vector<span<const int64_t>> row{span<const int64_t>(a).subspan(q * wa, wa),
                                              span<const int64_t>(b).subspan(q * wb, wb)};
        const auto one = rrf_fuse(row, weights, 60.0f, 10);
        REQUIRE(ids_of(batch[q]) == ids_of(one));
        for (size_t i = 0; i < one.size(); ++i) REQUIRE(batch[q][i].score == one[i].score);
    }
}
