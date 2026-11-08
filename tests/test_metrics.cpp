#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "vecengine/metrics.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace vecengine;
using Catch::Approx;

TEST_CASE("rank_score computes recall, nDCG and MRR", "[metrics]") {
    Relevance labels{{9, 1.0f}, {3, 2.0f}, {5, 0.0f}};
    check_relevance(labels);
    REQUIRE(labels.front().first == 3);
    const std::vector<int64_t> ranked{7, 3, 5, 9};

    REQUIRE(rank_score(RankMetric::Recall, ranked, labels, 4) == Approx(1.0));
    REQUIRE(rank_score(RankMetric::Recall, ranked, labels, 2) == Approx(0.5));   // grade 0 doesn't count as relevant
    REQUIRE(rank_score(RankMetric::MRR, ranked, labels, 4) == Approx(0.5));
    REQUIRE(rank_score(RankMetric::MRR, ranked, labels, 1) == 0.0);

    const double dcg = 2.0 / std::log2(3.0) + 1.0 / std::log2(5.0);
    const double ideal = 2.0 / std::log2(2.0) + 1.0 / std::log2(3.0);
    REQUIRE(rank_score(RankMetric::NDCG, ranked, labels, 4) == Approx(dcg / ideal));
    const std::vector<int64_t> best{3, 9};
    REQUIRE(rank_score(RankMetric::NDCG, best, labels, 10) == Approx(1.0));
    // the ideal is cut at k too
    REQUIRE(rank_score(RankMetric::NDCG, std::vector<int64_t>{3}, labels, 1) == Approx(1.0));
}

TEST_CASE("rank_score skips padding and repeated ids", "[metrics]") {
    Relevance labels{{1, 1.0f}, {2, 1.0f}};
    check_relevance(labels);
    const std::vector<int64_t> ranked{-1, 1, 1, 2};
    REQUIRE(rank_score(RankMetric::Recall, ranked, labels, 4) == Approx(1.0));
    REQUIRE(rank_score(RankMetric::MRR, ranked, labels, 4) == Approx(0.5));
    const double dcg = 1.0 / std::log2(3.0) + 1.0 / std::log2(5.0);
    REQUIRE(rank_score(RankMetric::NDCG, ranked, labels, 4) == Approx(dcg / (1.0 + 1.0 / std::log2(3.0))));
}

TEST_CASE("bad relevance labels and k are refused", "[metrics]") {
    Relevance repeated{{1, 1.0f}, {1, 2.0f}};
    REQUIRE_THROWS_AS(check_relevance(repeated), std::invalid_argument);
    Relevance none{{1, 0.0f}};
    REQUIRE_THROWS_AS(check_relevance(none), std::invalid_argument);
    Relevance empty;
    REQUIRE_THROWS_AS(check_relevance(empty), std::invalid_argument);
    Relevance nan{{1, NAN}};
    REQUIRE_THROWS_AS(check_relevance(nan), std::invalid_argument);
    Relevance ok{{1, 1.0f}};
    REQUIRE_THROWS_AS(rank_score(RankMetric::Recall, std::vector<int64_t>{1}, ok, 0), std::invalid_argument);
}
