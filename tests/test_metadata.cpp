#include <catch2/catch_test_macros.hpp>

#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/metadata.hpp"
#include "vecengine/refine_index.hpp"
#include "serialize.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vecengine;
using namespace vecengine::detail;
using Op = Expr::Op;

namespace {

std::vector<float> md_random(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = g(rng);
    return v;
}

const std::vector<std::string> kColors = {"red", "green", "blue", "black"};

// Row i: price = i % 100 (int), weight = i * 0.5 (float), color = kColors[i % 4].
Metadata md_rows(size_t first, size_t n) {
    std::vector<int64_t> price(n);
    std::vector<double> weight(n);
    std::vector<std::string> color(n);
    for (size_t j = 0; j < n; ++j) {
        const size_t i = first + j;
        price[j] = static_cast<int64_t>(i % 100);
        weight[j] = static_cast<double>(i) * 0.5;
        color[j] = kColors[i % 4];
    }
    return {{"price", price}, {"weight", weight}, {"color", color}};
}

// The ids of rows [0, n) matching pred, as a sorted list.
std::vector<int64_t> ids_where(size_t n, const std::function<bool(size_t)>& pred) {
    std::vector<int64_t> out;
    for (size_t i = 0; i < n; ++i)
        if (pred(i)) out.push_back(static_cast<int64_t>(i));
    return out;
}

// The slots set in a bit vector.
std::vector<int64_t> bits_to_ids(const std::vector<uint64_t>& words, size_t slots) {
    std::vector<int64_t> out;
    for (size_t i = 0; i < slots; ++i)
        if ((words[i >> 6] >> (i & 63)) & 1) out.push_back(static_cast<int64_t>(i));
    return out;
}

void md_require_identical(const std::vector<std::vector<Neighbor>>& a, const std::vector<std::vector<Neighbor>>& b) {
    REQUIRE(a.size() == b.size());
    for (size_t q = 0; q < a.size(); ++q) {
        REQUIRE(a[q].size() == b[q].size());
        for (size_t i = 0; i < a[q].size(); ++i) {
            REQUIRE(a[q][i].index == b[q][i].index);
            REQUIRE(a[q][i].distance == b[q][i].distance);
        }
    }
}

Expr f(const char* field, Op op, Expr::Value v) { return Expr::compare(field, op, std::move(v)); }

} // namespace

TEST_CASE("MetadataStore: columns, nulls and validation", "[metadata]") {
    MetadataStore store;
    store.check(3, md_rows(0, 3));
    store.append(0, 3, md_rows(0, 3));
    REQUIRE(store.schema() == std::vector<std::pair<std::string, ColumnType>>{
                                  {"price", ColumnType::Int64}, {"weight", ColumnType::Float64}, {"color", ColumnType::String}});

    // a new column starts with earlier rows null; a left-out column gets null
    const Metadata only_tag = {{"tag", std::vector<std::string>{"a", "b"}}};
    store.check(2, only_tag);
    store.append(3, 2, only_tag);
    REQUIRE(bits_to_ids(store.evaluate(Expr::is_null("tag"), 5), 5) == std::vector<int64_t>{0, 1, 2});
    REQUIRE(bits_to_ids(store.evaluate(Expr::is_null("price"), 5), 5) == std::vector<int64_t>{3, 4});

    SECTION("bad batches are rejected before anything changes") {
        REQUIRE_THROWS_AS(store.check(2, {{"price", std::vector<int64_t>{1}}}), std::invalid_argument);      // length
        REQUIRE_THROWS_AS(store.check(1, {{"price", std::vector<double>{1.5}}}), std::invalid_argument);     // type
        REQUIRE_THROWS_AS(store.check(1, {{"", std::vector<int64_t>{1}}}), std::invalid_argument);           // name
        REQUIRE_THROWS_AS(store.check(1, {{"x", std::vector<int64_t>{1}}, {"x", std::vector<int64_t>{2}}}),
                          std::invalid_argument);                                                             // twice
    }

    SECTION("compact keeps rows aligned with their slots") {
        std::vector<uint32_t> remap = {0, IdMap::kRemoved, 1, IdMap::kRemoved, 2};
        store.compact(remap);
        REQUIRE(bits_to_ids(store.evaluate(Expr::is_null("tag"), 3), 3) == std::vector<int64_t>{0, 1});
        REQUIRE(bits_to_ids(store.evaluate(f("color", Op::Eq, std::string("blue")), 3), 3) == std::vector<int64_t>{1});
    }
}

TEST_CASE("MetadataStore: every operator matches a hand-computed mask", "[metadata]") {
    const size_t n = 1000;
    MetadataStore store;
    store.append(0, n, md_rows(0, n));
    auto got = [&](const Expr& e) { return bits_to_ids(store.evaluate(e, n), n); };
    auto price = [](size_t i) { return static_cast<int64_t>(i % 100); };
    auto weight = [](size_t i) { return static_cast<double>(i) * 0.5; };

    REQUIRE(got(f("price", Op::Eq, int64_t{7})) == ids_where(n, [&](size_t i) { return price(i) == 7; }));
    REQUIRE(got(f("price", Op::Ne, int64_t{7})) == ids_where(n, [&](size_t i) { return price(i) != 7; }));
    REQUIRE(got(f("price", Op::Lt, int64_t{10})) == ids_where(n, [&](size_t i) { return price(i) < 10; }));
    REQUIRE(got(f("price", Op::Le, int64_t{10})) == ids_where(n, [&](size_t i) { return price(i) <= 10; }));
    REQUIRE(got(f("price", Op::Gt, int64_t{90})) == ids_where(n, [&](size_t i) { return price(i) > 90; }));
    REQUIRE(got(f("price", Op::Ge, int64_t{90})) == ids_where(n, [&](size_t i) { return price(i) >= 90; }));

    // an int column against a float value compares as numbers
    REQUIRE(got(f("price", Op::Lt, 9.5)) == ids_where(n, [&](size_t i) { return price(i) <= 9; }));
    REQUIRE(got(f("price", Op::Eq, 7.0)) == got(f("price", Op::Eq, int64_t{7})));
    REQUIRE(got(f("weight", Op::Ge, int64_t{400})) == ids_where(n, [&](size_t i) { return weight(i) >= 400; }));

    REQUIRE(got(Expr::isin("price", {int64_t{1}, int64_t{2}, 3.0})) ==
            ids_where(n, [&](size_t i) { return price(i) >= 1 && price(i) <= 3; }));
    REQUIRE(got(Expr::isin("weight", {0.5, int64_t{2}})) == std::vector<int64_t>{1, 4});

    // strings: equality, set membership, lexicographic order; an unseen value matches nothing
    REQUIRE(got(f("color", Op::Eq, std::string("red"))) == ids_where(n, [](size_t i) { return i % 4 == 0; }));
    REQUIRE(got(f("color", Op::Ne, std::string("red"))) == ids_where(n, [](size_t i) { return i % 4 != 0; }));
    REQUIRE(got(Expr::isin("color", {std::string("green"), std::string("blue")})) ==
            ids_where(n, [](size_t i) { return i % 4 == 1 || i % 4 == 2; }));
    REQUIRE(got(f("color", Op::Lt, std::string("c"))) ==   // "black" and "blue"
            ids_where(n, [](size_t i) { return i % 4 == 2 || i % 4 == 3; }));
    REQUIRE(got(f("color", Op::Eq, std::string("purple"))).empty());

    // combinations
    const Expr cheap = f("price", Op::Lt, int64_t{20});
    const Expr red = f("color", Op::Eq, std::string("red"));
    REQUIRE(got(cheap & red) == ids_where(n, [&](size_t i) { return price(i) < 20 && i % 4 == 0; }));
    REQUIRE(got(cheap | red) == ids_where(n, [&](size_t i) { return price(i) < 20 || i % 4 == 0; }));
    REQUIRE(got(~cheap) == ids_where(n, [&](size_t i) { return price(i) >= 20; }));
    REQUIRE(got(~~cheap) == got(cheap));
    REQUIRE(got(Expr::is_null("price")).empty());

    // errors: unknown column, wrong kind of value, not a comparison
    REQUIRE_THROWS_AS(store.evaluate(f("size", Op::Eq, int64_t{1}), n), std::invalid_argument);
    REQUIRE_THROWS_AS(store.evaluate(f("price", Op::Eq, std::string("7")), n), std::invalid_argument);
    REQUIRE_THROWS_AS(store.evaluate(f("color", Op::Eq, int64_t{1}), n), std::invalid_argument);
    REQUIRE_THROWS_AS(Expr::compare("price", Op::And, int64_t{1}), std::invalid_argument);

    REQUIRE((cheap & ~red).to_string() == "((price < 20) & ~(color == 'red'))");
}

TEST_CASE("MetadataStore: nulls fail comparisons, and ~ includes them", "[metadata]") {
    MetadataStore store;
    store.append(0, 4, {{"price", std::vector<int64_t>{5, 50, 500, 5000}}});
    store.append(4, 2, {});  // two rows with no price
    auto got = [&](const Expr& e) { return bits_to_ids(store.evaluate(e, 6), 6); };
    REQUIRE(got(f("price", Op::Lt, int64_t{100})) == std::vector<int64_t>{0, 1});
    REQUIRE(got(f("price", Op::Ne, int64_t{5})) == std::vector<int64_t>{1, 2, 3});
    REQUIRE(got(~f("price", Op::Lt, int64_t{100})) == std::vector<int64_t>{2, 3, 4, 5});
    REQUIRE(got(~f("price", Op::Lt, int64_t{100}) & ~Expr::is_null("price")) == std::vector<int64_t>{2, 3});
}

TEST_CASE("Metadata filters search like the same ids passed as allowed, on every index", "[metadata]") {
    const size_t n = 4000, dim = 16, nq = 30, k = 10;
    const auto data = md_random(n, dim, 1);
    const auto queries = md_random(nq, dim, 2);
    const Metadata md = md_rows(0, n);

    auto flat = std::make_unique<FlatIndex>(dim);
    auto ivf = std::make_unique<IVFIndex>(dim, 100, 20);
    ivf->train(data, n, 25, 1);
    auto hnsw = std::make_unique<HNSWIndex>(dim, 16, 100, false, 1);
    auto pq = std::make_unique<IVFIndex>(dim, 100, 20);
    pq->enable_pq(8);
    pq->train(data, n, 25, 1);
    auto refine = std::make_unique<RefineIndex>(*pq, 8);
    std::vector<std::pair<std::string, Index*>> indexes = {
        {"flat", flat.get()}, {"ivf", ivf.get()}, {"hnsw", hnsw.get()}, {"refine", refine.get()}};

    // added in two batches, so rows line up across adds
    for (auto& [name, index] : indexes) {
        index->add_batch(std::span<const float>(data.data(), (n / 2) * dim), n / 2, nullptr, md_rows(0, n / 2));
        index->add_batch(std::span<const float>(data.data() + (n / 2) * dim, (n / 2) * dim), n / 2, nullptr,
                         md_rows(n / 2, n / 2));
    }

    const Expr where = (f("price", Op::Lt, int64_t{30}) & ~f("color", Op::Eq, std::string("red"))) |
                       f("weight", Op::Ge, 1900.0);
    const auto allowed = ids_where(n, [](size_t i) { return (i % 100 < 30 && i % 4 != 0) || i * 0.5 >= 1900.0; });
    const auto evens = ids_where(n, [](size_t i) { return i % 2 == 0; });
    std::vector<int64_t> both;
    for (int64_t id : allowed) if (id % 2 == 0) both.push_back(id);

    for (auto& [name, index] : indexes) {
        INFO(name);
        REQUIRE(index->metadata_schema().size() == 3);
        md_require_identical(index->query_batch(queries, nq, k, where), index->query_batch(queries, nq, k, allowed));
        md_require_identical(index->query_batch(queries, nq, k, where, evens), index->query_batch(queries, nq, k, both));
        for (size_t q = 0; q < nq; ++q) {
            const std::span<const float> one(queries.data() + q * dim, dim);
            md_require_identical({index->query(one, k, where)}, {index->query(one, k, allowed)});
            md_require_identical({index->query(one, k, where, evens)}, {index->query(one, k, both)});
        }
        REQUIRE_THROWS_AS(index->query(std::span<const float>(queries.data(), dim), k, f("size", Op::Eq, int64_t{1})),
                          std::invalid_argument);
    }

    SECTION("deleted vectors never match, and compaction keeps rows with their vectors") {
        for (auto& [name, index] : indexes) {
            INFO(name);
            const auto dead = ids_where(n, [](size_t i) { return i % 3 == 0; });
            index->remove(dead);
            std::vector<int64_t> live;
            for (int64_t id : allowed) if (id % 3 != 0) live.push_back(id);

            const auto before = index->query_batch(queries, nq, k, where);
            md_require_identical(before, index->query_batch(queries, nq, k, live));
            index->compact();
            if (name != "hnsw")  // HNSW rebuilds its graph on compact, so results can shift
                md_require_identical(index->query_batch(queries, nq, k, where), before);
            md_require_identical(index->query_batch(queries, nq, k, where), index->query_batch(queries, nq, k, live));
        }
    }
}

TEST_CASE("Metadata with custom ids, and a bad add changes nothing", "[metadata]") {
    const size_t n = 500, dim = 8;
    const auto data = md_random(n, dim, 3);
    std::vector<int64_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = 10000 + 7 * static_cast<int64_t>(i);

    FlatIndex flat(dim, n, true);
    flat.add_batch(data, n, ids.data(), md_rows(0, n));
    const auto got = flat.query(std::span<const float>(data.data(), dim), 5, f("color", Op::Eq, std::string("blue")));
    REQUIRE(got.size() == 5);
    for (const auto& nb : got) REQUIRE(((static_cast<int64_t>(nb.index) - 10000) / 7) % 4 == 2);

    // a duplicate id fails after the metadata check passed: neither side changes
    const std::vector<int64_t> dup = {ids[0]};
    REQUIRE_THROWS_AS(flat.add_batch(std::span<const float>(data.data(), dim), 1, dup.data(), md_rows(0, 1)),
                      std::invalid_argument);
    // a metadata type mismatch fails before the vectors are added
    const int64_t fresh = 1;
    REQUIRE_THROWS_AS(flat.add_batch(std::span<const float>(data.data(), dim), 1, &fresh,
                                     {{"price", std::vector<double>{1.5}}}),
                      std::invalid_argument);
    REQUIRE(flat.size() == n);
    flat.add_batch(std::span<const float>(data.data(), dim), 1, &fresh);  // no metadata: all null
    REQUIRE(flat.query(std::span<const float>(data.data(), dim), 1, Expr::is_null("price"))[0].index == 1);
}

TEST_CASE("Metadata is saved and loaded with every index", "[metadata][serialize]") {
    const size_t n = 600, dim = 8, nq = 10, k = 5;
    const auto data = md_random(n, dim, 4);
    const auto queries = md_random(nq, dim, 5);
    const Expr where = f("price", Op::Ge, int64_t{50}) & Expr::isin("color", {std::string("red"), std::string("blue")});

    FlatIndex flat(dim);
    IVFIndex pq(dim, 100, 20);
    pq.enable_pq(4);
    pq.train(data, n, 25, 1);
    RefineIndex refine(pq, 4);
    HNSWIndex hnsw(dim, 16, 100, false, 1);
    for (Index* index : {static_cast<Index*>(&flat), static_cast<Index*>(&refine), static_cast<Index*>(&hnsw)}) {
        index->add_batch(std::span<const float>(data.data(), 400 * dim), 400, nullptr, md_rows(0, 400));
        index->add_batch(std::span<const float>(data.data() + 400 * dim, 200 * dim), 200);  // nulls
        index->remove(std::vector<int64_t>{1, 2, 3});

        std::stringstream s;
        index->save(s);
        auto loaded = load_index(s);
        REQUIRE(loaded->metadata_schema() == index->metadata_schema());
        md_require_identical(loaded->query_batch(queries, nq, k, where), index->query_batch(queries, nq, k, where));
        md_require_identical(loaded->query_batch(queries, nq, k, Expr::is_null("price")),
                             index->query_batch(queries, nq, k, Expr::is_null("price")));
    }
}

TEST_CASE("Metadata: load rejects corrupt columns", "[metadata][serialize]") {
    const size_t n = 6;
    MetadataStore store;
    store.append(0, n, {{"c", std::vector<std::string>{"x", "y", "x", "z", "y", "x"}}});
    std::stringstream s;
    store.save(s);
    const std::string bytes = s.str();
    {
        std::stringstream t(bytes);
        REQUIRE(MetadataStore::load(t, n).schema().size() == 1);
    }

    auto rejects = [](const std::string& b, uint64_t slots) {
        std::stringstream t(b);
        REQUIRE_THROWS_AS(MetadataStore::load(t, slots), std::runtime_error);
    };
    // layout: count (8), name length (8) + "c", type (1), valid (8 + n), codes (8 + 4n), dict size (8), entries
    const size_t type_at = 8 + 8 + 1, valid_at = type_at + 1 + 8, codes_at = valid_at + n + 8;

    rejects(bytes, n + 1);  // rows don't match the index's slots
    std::string bad_type = bytes;
    bad_type[type_at] = 3;
    rejects(bad_type, n);
    std::string bad_null = bytes;
    bad_null[valid_at] = 2;
    rejects(bad_null, n);
    std::string bad_code = bytes;
    const uint32_t out_of_range = 3;
    std::memcpy(bad_code.data() + codes_at, &out_of_range, sizeof(out_of_range));
    rejects(bad_code, n);
    std::string no_name = bytes;
    const uint64_t zero = 0;
    std::memcpy(no_name.data() + 8, &zero, sizeof(zero));
    no_name.erase(16, 1);
    rejects(no_name, n);
    rejects(bytes.substr(0, bytes.size() - 2), n);

    // duplicate dictionary entries
    std::stringstream d;
    write_pod<uint64_t>(d, 1);
    write_vec(d, std::vector<char>{'c'});
    write_pod<uint8_t>(d, 2);
    write_vec(d, std::vector<uint8_t>{1, 1});
    write_vec(d, std::vector<uint32_t>{0, 1});
    write_pod<uint64_t>(d, 2);
    write_vec(d, std::vector<char>{'x'});
    write_vec(d, std::vector<char>{'x'});
    rejects(d.str(), 2);
}
