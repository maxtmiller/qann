#include <catch2/catch_test_macros.hpp>

#include "vecengine/index_factory.hpp"
#include "vecengine/sparse_index.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace vecengine;

namespace {

using Row = std::vector<std::pair<uint32_t, float>>;

// Owns the arrays a SparseRows points into.
struct Csr {
    std::vector<uint64_t> indptr{0};
    std::vector<uint32_t> indices;
    std::vector<float> values;

    explicit Csr(const std::vector<Row>& rows) {
        for (const Row& row : rows) {
            for (auto [t, v] : row) {
                indices.push_back(t);
                values.push_back(v);
            }
            indptr.push_back(indices.size());
        }
    }
    SparseRows view() const { return {indptr, indices, values}; }
};

// n random rows over `vocab` terms, about `nnz` distinct terms each, drawn so
// low term ids are common (like stopwords). Integer weights when exact, so
// every sum is exact whatever the order or fusing of the arithmetic.
std::vector<Row> random_rows(size_t n, uint32_t vocab, size_t nnz, bool exact, std::mt19937& rng) {
    std::vector<Row> rows(n);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (Row& row : rows) {
        std::map<uint32_t, float> terms;
        for (size_t j = 0; j < nnz; ++j) {
            const auto t = static_cast<uint32_t>(std::pow(u(rng), 3.0f) * vocab);
            terms[t] = exact ? std::floor(u(rng) * 5.0f) + 1.0f : u(rng) * 3.0f;
        }
        row.assign(terms.begin(), terms.end());
    }
    return rows;
}

// Brute-force scores by dot product for every doc sharing a term with q and
// accepted by keep(), as (distance, id), nearest first, ties by id.
std::vector<std::pair<float, size_t>> brute_force(const std::vector<Row>& docs, const std::vector<int64_t>& ids,
                                                  const Row& q, const std::function<bool(size_t)>& keep) {
    std::vector<std::pair<float, size_t>> scored;
    for (size_t d = 0; d < docs.size(); ++d) {
        if (!keep(d)) continue;
        float score = 0;
        bool shared = false;
        for (auto [t, w] : q) {  // query terms in increasing order, as the index sums them
            for (auto [dt, dw] : docs[d])
                if (dt == t) {
                    score += w * dw;
                    shared = true;
                }
        }
        if (shared) scored.emplace_back(-score, static_cast<size_t>(ids[d]));
    }
    std::sort(scored.begin(), scored.end());
    return scored;
}

// got is a correct top k of want (brute_force's list): the k best distances,
// each with an id that truly has it. Which of several docs tied at the k-th
// distance is kept isn't specified, as in TopK.
void require_top_k(const std::vector<Neighbor>& got, const std::vector<std::pair<float, size_t>>& want, size_t k,
                   float tolerance) {
    REQUIRE(got.size() == std::min(k, want.size()));
    std::map<size_t, float> score_of;
    for (auto [dist, id] : want) score_of[id] = dist;
    for (size_t i = 0; i < got.size(); ++i) {
        REQUIRE(std::abs(got[i].distance - want[i].first) <= tolerance);
        REQUIRE(score_of.count(got[i].index) == 1);
        REQUIRE(std::abs(got[i].distance - score_of[got[i].index]) <= tolerance);
        if (i > 0) REQUIRE(got[i].index != got[i - 1].index);
    }
}

std::vector<int64_t> iota_ids(size_t n) {
    std::vector<int64_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = static_cast<int64_t>(i);
    return ids;
}

} // namespace

TEST_CASE("SparseIndex matches brute-force dot products (fuzz)", "[sparse]") {
    std::mt19937 rng(7);
    for (bool exact : {true, false}) {
        const auto docs = random_rows(2000, 500, 12, exact, rng);
        const auto queries = random_rows(100, 500, 6, exact, rng);
        SparseIndex index;
        const Csr csr(docs);
        REQUIRE(index.add_batch(csr.view()) == iota_ids(docs.size()));
        REQUIRE(index.size() == 2000);

        for (size_t k : {1, 10, 5000}) {
            const auto results = index.query_batch(Csr(queries).view(), k);
            REQUIRE(results.size() == queries.size());
            for (size_t i = 0; i < queries.size(); ++i)
                require_top_k(results[i], brute_force(docs, iota_ids(docs.size()), queries[i], [](size_t) { return true; }), k,
                              exact ? 0.0f : 1e-4f);
        }
    }
}

TEST_CASE("SparseIndex: input order, repeats and zeros don't change scores", "[sparse]") {
    SparseIndex index;
    // doc 0 as given out of order with a repeat and a zero; doc 1 canonical
    const Csr docs({{{9, 1.0f}, {3, 2.0f}, {9, 2.0f}, {4, 0.0f}}, {{3, 2.0f}, {9, 3.0f}}});
    index.add_batch(docs.view());
    REQUIRE(index.terms() == 2);    // term 4 had only a zero
    REQUIRE(index.postings() == 4);

    const Csr q({{{9, 1.0f}, {3, 1.0f}, {9, 1.0f}}});  // repeats sum: 3 -> 1, 9 -> 2
    const auto r = index.query_batch(q.view(), 10)[0];
    REQUIRE(r.size() == 2);
    REQUIRE(r[0].distance == -8.0f);
    REQUIRE(r[1].distance == -8.0f);
    REQUIRE(r[0].index == 0);    // ties by id

    // no shared term, an empty row, or a term never added: no results
    REQUIRE(index.query_batch(Csr({{{4, 1.0f}}}).view(), 10)[0].empty());
    REQUIRE(index.query_batch(Csr({{}}).view(), 10)[0].empty());
    REQUIRE(index.query_batch(Csr({{{9, 1.0f}}}).view(), 0)[0].empty());
    REQUIRE(index.query_batch(Csr({}).view(), 10).empty());
}

TEST_CASE("SparseIndex: custom ids, remove, compact, filters and metadata", "[sparse]") {
    std::mt19937 rng(11);
    const auto docs = random_rows(300, 50, 8, true, rng);
    const auto queries = random_rows(20, 50, 4, true, rng);
    std::vector<int64_t> ids(docs.size());
    std::vector<int64_t> parity(docs.size());
    for (size_t i = 0; i < docs.size(); ++i) {
        ids[i] = 1000 + 3 * static_cast<int64_t>(i);
        parity[i] = static_cast<int64_t>(i % 2);
    }
    SparseIndex index(true);
    REQUIRE(index.add_batch(Csr(docs).view(), ids.data(), {{"odd", parity}}) == ids);
    REQUIRE_THROWS_AS(index.add_batch(Csr({{{1, 1.0f}}}).view()), std::invalid_argument);  // needs ids

    std::vector<bool> live(docs.size(), true);
    std::vector<int64_t> gone;
    for (size_t i = 0; i < docs.size(); i += 7) {
        gone.push_back(ids[i]);
        live[i] = false;
    }
    REQUIRE(index.remove(gone) == gone.size());
    std::vector<int64_t> allowed;
    for (size_t i = 0; i < docs.size(); i += 3) allowed.push_back(ids[i]);
    const Expr odd = Expr::compare("odd", Expr::Op::Eq, int64_t{1});

    auto check = [&](const SparseIndex& idx) {
        const Csr q(queries);
        const auto plain = idx.query_batch(q.view(), 10);
        const auto by_ids = idx.query_batch(q.view(), 10, allowed);
        const auto by_where = idx.query_batch(q.view(), 10, odd);
        const auto both = idx.query_batch(q.view(), 10, odd, allowed);
        for (size_t i = 0; i < queries.size(); ++i) {
            require_top_k(plain[i], brute_force(docs, ids, queries[i], [&](size_t d) { return live[d]; }), 10, 0);
            require_top_k(by_ids[i], brute_force(docs, ids, queries[i], [&](size_t d) { return live[d] && d % 3 == 0; }), 10, 0);
            require_top_k(by_where[i], brute_force(docs, ids, queries[i], [&](size_t d) { return live[d] && d % 2 == 1; }), 10, 0);
            require_top_k(both[i], brute_force(docs, ids, queries[i],
                                              [&](size_t d) { return live[d] && d % 2 == 1 && d % 3 == 0; }), 10, 0);
        }
    };
    check(index);
    REQUIRE(index.query_batch(Csr(queries).view(), 10, std::vector<int64_t>{}).at(0).empty());

    const size_t before = index.postings();
    REQUIRE(index.compact() == gone.size());
    REQUIRE(index.postings() < before);
    REQUIRE(index.size() == docs.size() - gone.size());
    check(index);
    REQUIRE(std::get<std::vector<int64_t>>(index.get_metadata(std::vector<int64_t>{ids[1]})[0].values) ==
            std::vector<int64_t>{1});

    // save and load, then keep adding: ids continue to be checked
    std::stringstream s;
    index.save(s);
    std::unique_ptr<IndexBase> loaded = load_index(s);
    auto* sparse = dynamic_cast<SparseIndex*>(loaded.get());
    REQUIRE(sparse != nullptr);
    REQUIRE(sparse->custom_ids());
    REQUIRE(sparse->postings() == index.postings());
    check(*sparse);
    REQUIRE_THROWS_AS(sparse->add_batch(Csr({{{1, 1.0f}}}).view(), &ids[1]), std::invalid_argument);  // id taken

    // the same index always writes the same bytes
    std::stringstream again;
    sparse->save(again);
    REQUIRE(again.str() == s.str());
}

TEST_CASE("SparseIndex rejects malformed rows and corrupt files", "[sparse]") {
    SparseIndex index;
    const std::vector<uint32_t> indices = {1, 2};
    const std::vector<float> values = {1.0f, 1.0f};
    const std::vector<uint64_t> bad_start = {1, 2}, decreasing = {0, 2, 1}, too_long = {0, 3};
    REQUIRE_THROWS_AS(index.add_batch({bad_start, indices, values}), std::invalid_argument);
    REQUIRE_THROWS_AS(index.add_batch({decreasing, indices, values}), std::invalid_argument);
    REQUIRE_THROWS_AS(index.add_batch({too_long, indices, values}), std::invalid_argument);
    REQUIRE_THROWS_AS(index.add_batch({{}, indices, values}), std::invalid_argument);
    REQUIRE_THROWS_AS(index.add_batch(Csr({{{1, 1.0f}}}).view(), nullptr, {{"x", std::vector<int64_t>{1, 2}}}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(index.query_batch({decreasing, indices, values}, 1), std::invalid_argument);
    REQUIRE(index.size() == 0);
    REQUIRE(index.terms() == 0);

    index.add_batch(Csr({{{1, 1.0f}}, {{1, 2.0f}}}).view());
    std::stringstream s;
    index.save(s);
    const std::string bytes = s.str();
    // header (9 bytes), slots, list count, term, list length, then (slot, weight) pairs
    const size_t first_slot = 9 + 8 + 8 + 4 + 8;
    auto corrupt = [&](size_t at, uint32_t value) {
        std::string b = bytes;
        std::memcpy(b.data() + at, &value, sizeof value);
        std::stringstream t(b);
        REQUIRE_THROWS_AS(load_index(t), std::runtime_error);
    };
    corrupt(first_slot, 1);      // slots 1, 1: out of order
    corrupt(first_slot, 7);      // slot beyond the index
    corrupt(first_slot + 8, 0);  // slots 0, 0
    std::stringstream truncated(bytes.substr(0, bytes.size() - 3));
    REQUIRE_THROWS_AS(load_index(truncated), std::runtime_error);
}
