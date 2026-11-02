#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/hybrid_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/inverted_index.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/sparse_approx_index.hpp"
#include "serialize.hpp"

#include <cstdint>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vecengine;

namespace {

// Owns the arrays a SparseRows points into: n random rows over vocab terms.
struct Csr {
    std::vector<uint64_t> indptr{0};
    std::vector<uint32_t> indices;
    std::vector<float> values;

    Csr(size_t n, uint32_t vocab, size_t nnz, std::mt19937& rng) {
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (size_t i = 0; i < n; ++i) {
            std::map<uint32_t, float> terms;
            for (size_t j = 0; j < nnz; ++j) terms[static_cast<uint32_t>(u(rng) * vocab)] = 0.1f + u(rng);
            for (auto [t, v] : terms) {
                indices.push_back(t);
                values.push_back(v);
            }
            indptr.push_back(indices.size());
        }
    }
    SparseRows view() const { return {indptr, indices, values}; }
};

std::vector<float> random_dense(size_t n, size_t dim, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (float& x : v) x = u(rng);
    return v;
}

constexpr size_t kDim = 16, kN = 600, kQ = 40;
constexpr uint32_t kVocab = 300;

// Documents, queries and ids shared by the tests: ids increase with the row,
// so standalone indexes over the same rows break ties the same way.
struct Data {
    std::mt19937 rng{7};
    std::vector<float> dense = random_dense(kN, kDim, rng);
    Csr sparse{kN, kVocab, 12, rng};
    std::vector<float> dense_q = random_dense(kQ, kDim, rng);
    Csr sparse_q{kQ, kVocab, 8, rng};
    std::vector<int64_t> ids = [] {
        std::vector<int64_t> v(kN);
        for (size_t i = 0; i < kN; ++i) v[i] = 1000 + 3 * static_cast<int64_t>(i);
        return v;
    }();
    Metadata group() const {
        std::vector<int64_t> g(kN);
        for (size_t i = 0; i < kN; ++i) g[i] = static_cast<int64_t>(i % 3);
        return {MetadataColumn("group", g)};
    }
};

// What a HybridIndex should return: the same documents in standalone indexes
// with the hybrid's ids, each searched for k * depth_factor, then fused.
std::vector<std::vector<FusedHit>> reference(const Data& d, const HybridIndex& h, size_t k, const int64_t* allowed_ids = nullptr,
                                             size_t allowed_n = 0) {
    FlatIndex dense(kDim, 16, true);
    InvertedIndex sparse(true);
    dense.add_batch(d.dense, kN, d.ids.data());
    sparse.add_batch(d.sparse.view(), d.ids.data());
    const size_t depth = std::min(k * h.depth_factor(), allowed_ids ? allowed_n : kN);
    const span<const int64_t> allowed(allowed_ids, allowed_n);
    const auto dh = allowed_ids ? dense.query_batch(d.dense_q, kQ, depth, allowed) : dense.query_batch(d.dense_q, kQ, depth);
    const auto sh = allowed_ids ? sparse.query_batch(d.sparse_q.view(), depth, allowed) : sparse.query_batch(d.sparse_q.view(), depth);
    std::vector<std::vector<FusedHit>> out(kQ);
    for (size_t q = 0; q < kQ; ++q) {
        std::vector<int64_t> ids[2];
        std::vector<float> dists[2];
        for (const auto& n : dh[q]) ids[0].push_back(static_cast<int64_t>(n.index)), dists[0].push_back(n.distance);
        for (const auto& n : sh[q]) ids[1].push_back(static_cast<int64_t>(n.index)), dists[1].push_back(n.distance);
        const span<const int64_t> il[2] = {ids[0], ids[1]};
        const span<const float> dl[2] = {dists[0], dists[1]};
        out[q] = fuse(h.method(), il, dl, h.weights(), h.rrf_k(), k);
    }
    return out;
}

void require_same(const std::vector<std::vector<FusedHit>>& a, const std::vector<std::vector<FusedHit>>& b) {
    REQUIRE(a.size() == b.size());
    for (size_t q = 0; q < a.size(); ++q) {
        REQUIRE(a[q].size() == b[q].size());
        for (size_t i = 0; i < a[q].size(); ++i) {
            REQUIRE(a[q][i].id == b[q][i].id);
            REQUIRE(a[q][i].score == b[q][i].score);
        }
    }
}

std::unique_ptr<HybridIndex> roundtrip(const HybridIndex& h) {
    std::stringstream s;
    h.save(s);
    auto loaded = load_index(s);
    REQUIRE(dynamic_cast<HybridIndex*>(loaded.get()) != nullptr);
    return std::unique_ptr<HybridIndex>(static_cast<HybridIndex*>(loaded.release()));
}

} // namespace

TEST_CASE("HybridIndex returns the fusion of its two sides", "[hybrid]") {
    const Data d;
    FlatIndex dense(kDim);
    InvertedIndex sparse;
    HybridIndex h(dense, sparse, true);
    REQUIRE(h.add_batch(d.dense, d.sparse.view(), d.ids.data()) == d.ids);
    REQUIRE(h.size() == kN);
    REQUIRE(dense.slots() == kN);
    REQUIRE(sparse.slots() == kN);

    const auto method = GENERATE(FusionMethod::RRF, FusionMethod::Linear);
    h.set_method(method);
    SECTION("defaults") { require_same(h.query_batch(d.dense_q, d.sparse_q.view(), 10), reference(d, h, 10)); }
    SECTION("weights, rrf_k and depth") {
        h.set_weights(0.3f, 1.5f);
        h.set_rrf_k(10.0f);
        h.set_depth_factor(2);
        require_same(h.query_batch(d.dense_q, d.sparse_q.view(), 7), reference(d, h, 7));
    }
    SECTION("one side weighted 0 gives that side's ranking alone") {
        h.set_weights(0.0f, 1.0f);
        const auto got = h.query_batch(d.dense_q, d.sparse_q.view(), 10);
        InvertedIndex alone(true);
        alone.add_batch(d.sparse.view(), d.ids.data());
        const auto want = alone.query_batch(d.sparse_q.view(), 10);
        for (size_t q = 0; q < kQ; ++q)
            for (size_t i = 0; i < got[q].size(); ++i) REQUIRE(got[q][i].id == static_cast<int64_t>(want[q][i].index));
    }
}

TEST_CASE("HybridIndex filters both sides the same way", "[hybrid][filter]") {
    const Data d;
    FlatIndex dense(kDim);
    InvertedIndex sparse;
    HybridIndex h(dense, sparse, true);
    h.add_batch(d.dense, d.sparse.view(), d.ids.data(), d.group());

    std::vector<int64_t> group0;
    for (size_t i = 0; i < kN; i += 3) group0.push_back(d.ids[i]);
    const auto want = reference(d, h, 10, group0.data(), group0.size());
    require_same(h.query_batch(d.dense_q, d.sparse_q.view(), 10, Expr::compare("group", Expr::Op::Eq, int64_t{0})), want);
    require_same(h.query_batch(d.dense_q, d.sparse_q.view(), 10, group0), want);
    // where and allowed together: group 0 and the first 20 of it
    const std::vector<int64_t> first(group0.begin(), group0.begin() + 20);
    for (const auto& row : h.query_batch(d.dense_q, d.sparse_q.view(), 10, Expr::compare("group", Expr::Op::Eq, int64_t{0}), first))
        for (const auto& hit : row) REQUIRE(std::find(first.begin(), first.end(), hit.id) != first.end());
    // a filter nothing passes
    for (const auto& row : h.query_batch(d.dense_q, d.sparse_q.view(), 10, Expr::compare("group", Expr::Op::Eq, int64_t{9})))
        REQUIRE(row.empty());
}

TEST_CASE("HybridIndex adds all or nothing", "[hybrid]") {
    Data d;
    FlatIndex dense(kDim, 16, false, Metric::Cosine);
    InvertedIndex sparse;
    HybridIndex h(dense, sparse, true);
    const SparseRows first10{span<const uint64_t>(d.sparse.indptr).first(11),
                             span<const uint32_t>(d.sparse.indices).first(d.sparse.indptr[10]),
                             span<const float>(d.sparse.values).first(d.sparse.indptr[10])};
    h.add_batch(span<const float>(d.dense).first(10 * kDim), first10, d.ids.data());
    auto unchanged = [&] {
        REQUIRE(h.slots() == 10);
        REQUIRE(dense.slots() == 10);
        REQUIRE(sparse.slots() == 10);
    };
    const std::vector<int64_t> new_ids{5000, 5001};
    const std::vector<uint64_t> indptr{0, 1, 2};
    const std::vector<uint32_t> terms{1, 2};
    const std::vector<float> values{1.0f, 1.0f};
    const SparseRows two{indptr, terms, values};
    std::vector<float> rows = random_dense(2, kDim, d.rng);

    SECTION("a zero dense row under Cosine") {
        std::fill(rows.begin() + kDim, rows.end(), 0.0f);
        REQUIRE_THROWS_AS(h.add_batch(rows, two, new_ids.data()), std::invalid_argument);
    }
    SECTION("malformed sparse rows") {
        const std::vector<uint64_t> bad{0, 2, 1};
        REQUIRE_THROWS_AS(h.add_batch(rows, SparseRows{bad, terms, values}, new_ids.data()), std::invalid_argument);
    }
    SECTION("different row counts") {
        REQUIRE_THROWS_AS(h.add_batch(span<const float>(rows).first(kDim), two, new_ids.data()), std::invalid_argument);
    }
    SECTION("an id already used") {
        const std::vector<int64_t> taken{5000, d.ids[0]};
        REQUIRE_THROWS_AS(h.add_batch(rows, two, taken.data()), std::invalid_argument);
    }
    SECTION("metadata of the wrong length") {
        const Metadata md{MetadataColumn("group", std::vector<int64_t>{1})};
        REQUIRE_THROWS_AS(h.add_batch(rows, two, new_ids.data(), md), std::invalid_argument);
    }
    unchanged();
    REQUIRE(h.add_batch(random_dense(2, kDim, d.rng), two, new_ids.data()) == new_ids);   // and the index still works
}

TEST_CASE("HybridIndex with an untrained dense side adds nothing", "[hybrid]") {
    IVFIndex dense(4, 100);
    InvertedIndex sparse;
    HybridIndex h(dense, sparse);
    const std::vector<float> rows(4, 1.0f);
    const std::vector<uint64_t> indptr{0, 1};
    const std::vector<uint32_t> terms{3};
    const std::vector<float> values{1.0f};
    REQUIRE_THROWS(h.add_batch(rows, SparseRows{indptr, terms, values}));
    REQUIRE(h.slots() == 0);
    REQUIRE(sparse.slots() == 0);
}

TEST_CASE("HybridIndex construction and settings", "[hybrid]") {
    SECTION("wraps only empty, auto-id, unwrapped indexes") {
        FlatIndex used(4);
        used.add(std::vector<float>(4, 1.0f));
        InvertedIndex sparse, custom(true);
        FlatIndex dense(4);
        REQUIRE_THROWS_AS(HybridIndex(used, sparse), std::invalid_argument);
        REQUIRE_THROWS_AS(HybridIndex(dense, custom), std::invalid_argument);
        {
            HybridIndex h(dense, sparse);
            REQUIRE(dense.wrapped());
            REQUIRE(sparse.wrapped());
            InvertedIndex other;
            REQUIRE_THROWS_AS(HybridIndex(dense, other), std::invalid_argument);
            REQUIRE_THROWS_AS(RefineIndex(dense), std::invalid_argument);
        }
        REQUIRE_FALSE(dense.wrapped());
        REQUIRE_FALSE(sparse.wrapped());
    }
    SECTION("settings are checked") {
        FlatIndex dense(4);
        InvertedIndex sparse;
        HybridIndex h(dense, sparse);
        REQUIRE_THROWS_AS(h.set_weights(-1.0f, 1.0f), std::invalid_argument);
        REQUIRE_THROWS_AS(h.set_weights(0.0f, 0.0f), std::invalid_argument);
        REQUIRE_THROWS_AS(h.set_weights(1.0f, std::numeric_limits<float>::infinity()), std::invalid_argument);
        REQUIRE_THROWS_AS(h.set_rrf_k(0.0f), std::invalid_argument);
        REQUIRE_THROWS_AS(h.set_depth_factor(0), std::invalid_argument);
        REQUIRE(h.weights() == std::array<float, 2>{1.0f, 1.0f});
        REQUIRE(h.depth_factor() == 5);
        // an empty index answers every query with nothing
        const std::vector<uint64_t> indptr{0, 0};
        REQUIRE(h.query_batch(std::vector<float>(4, 1.0f), SparseRows{indptr, {}, {}}, 5)[0].empty());
    }
}

TEST_CASE("HybridIndex remove, compact and save keep both sides aligned", "[hybrid][compact][serialize]") {
    const Data d;
    // dense sides of every kind a HybridIndex can load, including a RefineIndex over IVF-PQ
    const int kind = GENERATE(0, 1, 2);
    const bool approx_sparse = GENERATE(false, true);
    INFO("dense kind " << kind << ", approximate sparse " << approx_sparse);
    std::unique_ptr<DenseIndex> dense, ivf;
    if (kind == 0) {
        dense = std::make_unique<FlatIndex>(kDim);
    } else if (kind == 1) {
        dense = std::make_unique<HNSWIndex>(kDim, 8, 40, false, 1u);
    } else {
        auto base = std::make_unique<IVFIndex>(kDim, 100, 100);
        base->enable_pq(kDim / 2);
        base->train(d.dense, kN, 5, 1);
        ivf = std::move(base);
        dense = std::make_unique<RefineIndex>(*ivf, 8);
    }
    std::unique_ptr<SparseIndex> sparse;
    if (approx_sparse) sparse = std::make_unique<SparseApproxIndex>(false, 0, 0, 10);   // reads everything: exact
    else sparse = std::make_unique<InvertedIndex>();
    HybridIndex h(*dense, *sparse, true);
    h.add_batch(d.dense, d.sparse.view(), d.ids.data(), d.group());
    h.set_method(FusionMethod::Linear);
    h.set_weights(0.7f, 0.4f);

    std::vector<int64_t> gone;
    for (size_t i = 0; i < kN; i += 4) gone.push_back(d.ids[i]);
    REQUIRE(h.remove(gone) == gone.size());
    REQUIRE(h.remove(gone) == 0);
    const auto before = h.query_batch(d.dense_q, d.sparse_q.view(), 10);
    for (const auto& row : before)
        for (const auto& hit : row) REQUIRE(std::find(gone.begin(), gone.end(), hit.id) == gone.end());

    const auto loaded_deleted = roundtrip(h);   // saved with deletions not yet compacted
    require_same(loaded_deleted->query_batch(d.dense_q, d.sparse_q.view(), 10), before);

    REQUIRE(h.compact() == gone.size());
    REQUIRE(h.slots() == kN - gone.size());
    REQUIRE(dense->slots() == h.slots());
    REQUIRE(sparse->slots() == h.slots());
    REQUIRE(dense->ids_are_slots());
    REQUIRE(sparse->ids_are_slots());
    if (kind != 1) require_same(h.query_batch(d.dense_q, d.sparse_q.view(), 10), before);   // HNSW rebuilds its graph

    const auto after = h.query_batch(d.dense_q, d.sparse_q.view(), 10);
    const auto loaded = roundtrip(h);
    REQUIRE(loaded->method() == FusionMethod::Linear);
    REQUIRE(loaded->weights() == std::array<float, 2>{0.7f, 0.4f});
    REQUIRE(loaded->ids() == h.ids());
    require_same(loaded->query_batch(d.dense_q, d.sparse_q.view(), 10), after);
    require_same(loaded->query_batch(d.dense_q, d.sparse_q.view(), 10, Expr::compare("group", Expr::Op::Eq, int64_t{1})),
                 h.query_batch(d.dense_q, d.sparse_q.view(), 10, Expr::compare("group", Expr::Op::Eq, int64_t{1})));
}

TEST_CASE("HybridIndex load rejects corrupt files", "[hybrid][serialize]") {
    const Data d;
    FlatIndex dense(kDim);
    InvertedIndex sparse;
    HybridIndex h(dense, sparse);
    h.add_batch(d.dense, d.sparse.view());
    std::stringstream s;
    h.save(s);
    const std::string good = s.str();
    // header (magic, version, kind) then method, two weights, rrf_k, depth_factor, count
    size_t at = 0;
    {
        std::stringstream probe;
        detail::write_header(probe, detail::IndexKind::Hybrid);
        at = probe.str().size();
    }
    auto load = [](const std::string& bytes) {
        std::stringstream in(bytes);
        return load_index(in);
    };
    auto patched = [&](size_t offset, const void* value, size_t size) {
        std::string bytes = good;
        std::memcpy(bytes.data() + offset, value, size);
        return bytes;
    };
    REQUIRE(load(good) != nullptr);
    const uint8_t bad_method = 7;
    REQUIRE_THROWS_AS(load(patched(at, &bad_method, 1)), std::runtime_error);
    const float negative = -1.0f;
    REQUIRE_THROWS_AS(load(patched(at + 1, &negative, 4)), std::runtime_error);
    const uint64_t zero = 0, wrong_count = kN + 1;
    REQUIRE_THROWS_AS(load(patched(at + 13, &zero, 8)), std::runtime_error);          // depth_factor
    REQUIRE_THROWS_AS(load(patched(at + 21, &wrong_count, 8)), std::runtime_error);   // count vs the parts
    REQUIRE_THROWS_AS(load(good.substr(0, good.size() / 2)), std::runtime_error);    // truncated
    // the parts swapped: a sparse index where the dense one belongs
    const auto dense_kind_at = at + 29 + sizeof(detail::kMagic) + sizeof(detail::kFormatVersion);   // nested dense header's kind
    const auto inverted = static_cast<uint8_t>(detail::IndexKind::Inverted);
    REQUIRE_THROWS_AS(load(patched(dense_kind_at, &inverted, 1)), std::runtime_error);
}
