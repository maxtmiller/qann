#include <catch2/catch_test_macros.hpp>

#include "vecengine/index_factory.hpp"
#include "vecengine/inverted_index.hpp"
#include "vecengine/sparse_approx_index.hpp"
#include "serialize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <random>
#include <set>
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
        for (auto [t, w] : q) {
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

TEST_CASE("InvertedIndex matches brute-force dot products (fuzz)", "[sparse]") {
    std::mt19937 rng(7);
    for (bool exact : {true, false}) {
        const auto docs = random_rows(2000, 500, 12, exact, rng);
        const auto queries = random_rows(100, 500, 6, exact, rng);
        InvertedIndex index;
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

TEST_CASE("InvertedIndex: MaxScore pruning matches full scoring, with negative weights and filters (fuzz)", "[sparse]") {
    std::mt19937 rng(23);
    std::uniform_int_distribution<int> coin(0, 3);
    for (int trial = 0; trial < 4; ++trial) {
        const bool negative = trial % 2 == 1;  // some weights below zero, in docs and queries
        auto docs = random_rows(3000, 2000, 30, trial < 2, rng);
        auto queries = random_rows(60, 2000, 8, trial < 2, rng);
        // like IDF, rarer (higher) terms weigh more, so lists' bounds differ and pruning has room
        for (Row& row : docs)
            for (auto& [t, v] : row) v *= trial < 2 ? (t < 40 ? 1.0f : 6.0f) : 6.0f / (1.0f + float(t) / 40.0f);
        if (negative) {
            for (auto* rows : {&docs, &queries})
                for (Row& row : *rows)
                    for (auto& [t, v] : row)
                        if (coin(rng) == 0) v = -v;
        }
        InvertedIndex index;
        index.add_batch(Csr(docs).view());
        std::vector<int64_t> gone;
        for (int64_t i = 0; i < 3000; i += 13) gone.push_back(i);
        index.remove(gone);
        std::vector<int64_t> allowed;
        for (int64_t i = 0; i < 3000; i += 2) allowed.push_back(i);

        const float tolerance = trial < 2 ? 0.0f : 1e-4f;
        for (size_t k : {1, 5, 100}) {
            CAPTURE(trial, k);
            const Csr q(queries);
            const auto plain = index.query_batch(q.view(), k);
            const auto filtered = index.query_batch(q.view(), k, allowed);
            index.set_pruning(false);
            const auto full = index.query_batch(q.view(), k);
            index.set_pruning(true);
            for (size_t i = 0; i < queries.size(); ++i) {
                require_top_k(plain[i], brute_force(docs, iota_ids(3000), queries[i], [](size_t d) { return d % 13 != 0; }),
                              k, tolerance);
                require_top_k(filtered[i],
                              brute_force(docs, iota_ids(3000), queries[i], [](size_t d) { return d % 13 != 0 && d % 2 == 0; }),
                              k, tolerance);
                REQUIRE(plain[i].size() == full[i].size());
                for (size_t j = 0; j < full[i].size(); ++j) REQUIRE(std::abs(plain[i][j].distance - full[i][j].distance) <= tolerance);
            }
        }
    }
}

TEST_CASE("InvertedIndex: input order, repeats and zeros don't change scores", "[sparse]") {
    InvertedIndex index;
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

TEST_CASE("InvertedIndex: custom ids, remove, compact, filters and metadata", "[sparse]") {
    std::mt19937 rng(11);
    const auto docs = random_rows(300, 50, 8, true, rng);
    const auto queries = random_rows(20, 50, 4, true, rng);
    std::vector<int64_t> ids(docs.size());
    std::vector<int64_t> parity(docs.size());
    for (size_t i = 0; i < docs.size(); ++i) {
        ids[i] = 1000 + 3 * static_cast<int64_t>(i);
        parity[i] = static_cast<int64_t>(i % 2);
    }
    InvertedIndex index(true);
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

    auto check = [&](const InvertedIndex& idx) {
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
    auto* sparse = dynamic_cast<InvertedIndex*>(loaded.get());
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

TEST_CASE("InvertedIndex rejects malformed rows and corrupt files", "[sparse]") {
    InvertedIndex index;
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

TEST_CASE("InvertedIndex file saved on macOS (format 9) loads", "[sparse][compat]") {
    // 1000 docs, custom ids 100, 103, ..., every ninth deleted (not compacted),
    // int / string / float metadata columns; results are checked in test_sparse.py
    const std::filesystem::path dir = QANN_TEST_DATA_DIR;
    std::unique_ptr<IndexBase> loaded = load_index(dir / "v9_sparse.qann");
    auto* index = dynamic_cast<InvertedIndex*>(loaded.get());
    REQUIRE(index != nullptr);
    REQUIRE(index->custom_ids());
    REQUIRE(index->slots() == 1000);
    REQUIRE(index->size() == 888);
    REQUIRE(index->terms() == 297);
    REQUIRE(index->postings() == 11200);
    REQUIRE(index->metadata_schema().size() == 3);

    const Csr q({{{0, 1.0f}, {7919, 1.0f}}});  // the two commonest terms
    const auto before = index->query_batch(q.view(), 10)[0];
    REQUIRE(before.size() == 10);
    REQUIRE(index->compact() == 112);
    REQUIRE(index->query_batch(q.view(), 10)[0].size() == 10);
    for (size_t i = 0; i < before.size(); ++i) REQUIRE(index->query_batch(q.view(), 10)[0][i].distance == before[i].distance);

    std::stringstream s;
    index->save(s);
    REQUIRE(detail::read_header(s).version == detail::kFormatVersion);
    s.seekg(0);
    std::unique_ptr<IndexBase> again = load_index(s);
    REQUIRE(again->size() == 888);
    REQUIRE(again->slots() == 888);
}

TEST_CASE("make_index builds an InvertedIndex, usable through SparseIndex", "[sparse]") {
    IndexOptions opts;
    opts.custom_ids = true;
    opts.pruning = false;
    std::unique_ptr<IndexBase> made = make_index(IndexType::Inverted, 0, opts);
    auto* inverted = dynamic_cast<InvertedIndex*>(made.get());
    REQUIRE(inverted != nullptr);
    REQUIRE(inverted->custom_ids());
    REQUIRE_FALSE(inverted->pruning());

    SparseIndex& sparse = *inverted;
    const std::vector<int64_t> ids = {5, 9};
    sparse.add_batch(Csr({{{1, 1.0f}}, {{1, 2.0f}, {2, 1.0f}}}).view(), ids.data());
    const auto r = sparse.query_batch(Csr({{{1, 1.0f}}}).view(), 5)[0];
    REQUIRE(r.size() == 2);
    REQUIRE((r[0].index == 9 && r[0].distance == -2.0f));
}

namespace {

// Each id's exact distance, from an InvertedIndex query returning every match.
std::map<size_t, float> exact_distances(const InvertedIndex& exact, const Row& q) {
    std::map<size_t, float> out;
    const auto results = exact.query_batch(Csr({q}).view(), exact.size());
    for (const Neighbor& n : results[0]) out[n.index] = n.distance;
    return out;
}

} // namespace

TEST_CASE("SparseApproxIndex: exact at full settings, exact distances when approximate", "[sparse][approx]") {
    std::mt19937 rng(31);
    const auto docs = random_rows(3000, 400, 20, false, rng);
    const auto queries = random_rows(40, 400, 12, false, rng);
    InvertedIndex exact;
    SparseApproxIndex approx(false, 0, 0, 1000);  // every term, every entry, every candidate
    exact.add_batch(Csr(docs).view());
    approx.add_batch(Csr(docs).view());
    REQUIRE(approx.postings() == exact.postings());
    const Csr q(queries);
    const auto truth = exact.query_batch(q.view(), 10);
    const auto full = approx.query_batch(q.view(), 10);
    for (size_t i = 0; i < queries.size(); ++i) {
        REQUIRE(full[i].size() == truth[i].size());
        for (size_t j = 0; j < full[i].size(); ++j) REQUIRE(full[i][j].distance == truth[i][j].distance);  // bit for bit
    }

    // a real approximation: good recall, and each distance is the exact one, bit for bit
    approx.set_query_terms(4);
    approx.set_list_depth(100);
    approx.set_k_factor(5);
    const auto found = approx.query_batch(q.view(), 10);
    double hits = 0;
    for (size_t i = 0; i < queries.size(); ++i) {
        const auto dist = exact_distances(exact, queries[i]);
        std::set<size_t> top;
        for (const Neighbor& n : truth[i]) top.insert(n.index);
        for (const Neighbor& n : found[i]) {
            REQUIRE(n.distance == dist.at(n.index));
            hits += top.count(n.index);
        }
    }
    REQUIRE(hits / (10.0 * queries.size()) > 0.5);

    // filters, removals and compaction (the forward index is rebuilt)
    std::vector<int64_t> allowed;
    for (int64_t i = 0; i < 3000; i += 3) allowed.push_back(i);
    for (const auto& row : approx.query_batch(q.view(), 10, allowed))
        for (const Neighbor& n : row) REQUIRE(n.index % 3 == 0);
    std::vector<int64_t> gone;
    for (const auto& row : found)
        if (!row.empty()) gone.push_back(static_cast<int64_t>(row[0].index));
    std::sort(gone.begin(), gone.end());
    gone.erase(std::unique(gone.begin(), gone.end()), gone.end());
    approx.remove(gone);
    exact.remove(gone);
    for (int pass = 0; pass < 2; ++pass) {
        const auto rows = approx.query_batch(q.view(), 10);
        for (size_t i = 0; i < queries.size(); ++i) {
            const auto dist = exact_distances(exact, queries[i]);
            for (const Neighbor& n : rows[i]) {
                REQUIRE_FALSE(std::binary_search(gone.begin(), gone.end(), static_cast<int64_t>(n.index)));
                REQUIRE(n.distance == dist.at(n.index));
            }
        }
        if (pass == 0) {
            REQUIRE(approx.compact() == gone.size());
            exact.compact();
        }
    }

    // settings are saved, and the forward index is rebuilt on load
    std::stringstream s;
    approx.save(s);
    std::unique_ptr<IndexBase> loaded = load_index(s);
    auto* back = dynamic_cast<SparseApproxIndex*>(loaded.get());
    REQUIRE(back != nullptr);
    REQUIRE((back->query_terms() == 4 && back->list_depth() == 100 && back->k_factor() == 5));
    const auto before = approx.query_batch(q.view(), 10), after = back->query_batch(q.view(), 10);
    for (size_t i = 0; i < queries.size(); ++i) {
        REQUIRE(after[i].size() == before[i].size());
        for (size_t j = 0; j < after[i].size(); ++j)
            REQUIRE((after[i][j].index == before[i][j].index && after[i][j].distance == before[i][j].distance));
    }
    std::stringstream again;
    back->save(again);
    REQUIRE(again.str() == s.str());

    REQUIRE_THROWS_AS(approx.set_k_factor(0), std::invalid_argument);
    REQUIRE_THROWS_AS(SparseApproxIndex(false, 10, 10, 0), std::invalid_argument);
}

TEST_CASE("SparseApproxIndex: lists stay ordered through many small adds", "[sparse][approx]") {
    // term 0 in every vector, so its list grows long and is merged again and
    // again as batches arrive; the others are random
    std::mt19937 rng(37);
    std::uniform_real_distribution<float> u(0.1f, 2.0f);
    std::uniform_int_distribution<uint32_t> term(1, 300);
    const size_t n = 40000;
    std::vector<Row> docs(n);
    for (Row& row : docs) {
        std::map<uint32_t, float> terms{{0, u(rng)}};
        for (int j = 0; j < 4; ++j) terms[term(rng)] = u(rng);
        row.assign(terms.begin(), terms.end());
    }
    SparseApproxIndex index(false, 0, 0, 100000);
    for (size_t first = 0; first < n;) {
        const size_t batch = 1 + rng() % 700;
        index.add_batch(Csr(std::vector<Row>(docs.begin() + first, docs.begin() + std::min(n, first + batch))).view());
        first += batch;
    }
    const std::vector<Row> queries = {{{0, 1.0f}}, {{0, 0.5f}, {7, 1.0f}}, {{3, 1.0f}, {150, 2.0f}}};
    const auto full = index.query_batch(Csr(queries).view(), 10);
    for (size_t i = 0; i < queries.size(); ++i)
        require_top_k(full[i], brute_force(docs, iota_ids(n), queries[i], [](size_t) { return true; }), 10, 1e-4f);

    // the heaviest entries of term 0 come first: depth 10 finds the true top 10 for {0: 1}
    index.set_list_depth(10);
    index.set_k_factor(1);
    const auto top = index.query_batch(Csr({{{0, 1.0f}}}).view(), 10)[0];
    REQUIRE(top.size() == 10);
    for (size_t j = 0; j < 10; ++j) REQUIRE(top[j].distance == full[0][j].distance);

    // the loader checks every list's order
    std::stringstream s;
    index.save(s);
    REQUIRE_NOTHROW(load_index(s));
}

TEST_CASE("SparseApproxIndex rejects corrupt files, and make_index builds one", "[sparse][approx]") {
    SparseApproxIndex index;
    index.add_batch(Csr({{{1, 1.0f}}, {{1, 2.0f}}}).view());  // two postings stay unsorted (slot order)
    std::stringstream s;
    index.save(s);
    const std::string bytes = s.str();
    // header (9 bytes), slots, query_terms, list_depth, k_factor, list count,
    // term, sorted count, list length, then (slot, weight) pairs
    const size_t k_factor = 9 + 8 + 16, sorted = k_factor + 8 + 8 + 4, first_slot = sorted + 8 + 8;
    auto corrupt = [&](std::initializer_list<std::pair<size_t, uint64_t>> writes) {
        std::string b = bytes;
        for (auto [at, value] : writes) std::memcpy(b.data() + at, &value, sizeof value);
        std::stringstream t(b);
        REQUIRE_THROWS_AS(load_index(t), std::runtime_error);
    };
    corrupt({{k_factor, 0}});
    corrupt({{sorted, 3}});  // more sorted than there are
    corrupt({{sorted, 2}});  // weights 1, 2 claimed heaviest first
    {
        std::string b = bytes;  // slots 1, 1
        const uint32_t one = 1;
        std::memcpy(b.data() + first_slot, &one, sizeof one);
        std::stringstream t(b);
        REQUIRE_THROWS_AS(load_index(t), std::runtime_error);
    }

    IndexOptions opts;
    opts.custom_ids = true;
    opts.query_terms = 3;
    opts.list_depth = 50;
    opts.k_factor = 7;
    std::unique_ptr<IndexBase> made = make_index(IndexType::SparseApprox, 0, opts);
    auto* approx = dynamic_cast<SparseApproxIndex*>(made.get());
    REQUIRE(approx != nullptr);
    REQUIRE((approx->custom_ids() && approx->query_terms() == 3 && approx->list_depth() == 50 && approx->k_factor() == 7));
}

TEST_CASE("SparseApproxIndex file saved on macOS (format 9) loads", "[sparse][approx][compat]") {
    // 1500 docs in two adds, custom ids 50, 52, ..., every eleventh deleted
    // (not compacted), a group column; results are checked in test_sparse.py
    const std::filesystem::path dir = QANN_TEST_DATA_DIR;
    std::unique_ptr<IndexBase> loaded = load_index(dir / "v9_sparse_approx.qann");
    auto* index = dynamic_cast<SparseApproxIndex*>(loaded.get());
    REQUIRE(index != nullptr);
    REQUIRE((index->custom_ids() && index->slots() == 1500 && index->size() == 1363));
    REQUIRE((index->terms() == 300 && index->postings() == 16999));
    REQUIRE((index->query_terms() == 3 && index->list_depth() == 40 && index->k_factor() == 4));
    const Csr q({{{0, 1.0f}, {7919, 1.0f}}});
    const auto before = index->query_batch(q.view(), 10)[0];
    REQUIRE(before.size() == 10);
    REQUIRE(index->compact() == 137);
    const auto after = index->query_batch(q.view(), 10)[0];
    REQUIRE(after.size() == 10);
    std::stringstream s;
    index->save(s);
    s.seekg(0);
    REQUIRE(load_index(s)->size() == 1363);
}
