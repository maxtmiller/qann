#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/metadata.hpp"
#include "vecengine/refine_index.hpp"
#include "serialize.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
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

// Raises the parser's condition limit for one scope.
struct ConditionLimit {
    size_t saved = Expr::max_conditions();
    explicit ConditionLimit(size_t n) { Expr::set_max_conditions(n); }
    ~ConditionLimit() { Expr::set_max_conditions(saved); }
};

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

TEST_CASE("MetadataStore: compact frees strings only removed rows used", "[metadata]") {
    MetadataStore store;
    const Metadata md = {{"tag", std::vector<std::string>{"lost", "kept", "gone", "kept", "lost"}}};
    store.append(0, 5, md);
    store.append(5, 1, {});    // a null row
    store.compact({IdMap::kRemoved, 0, IdMap::kRemoved, 1, IdMap::kRemoved, 2});

    std::ostringstream out;
    store.save(out);
    REQUIRE(out.str().find("kept") != std::string::npos);
    REQUIRE(out.str().find("lost") == std::string::npos);
    REQUIRE(out.str().find("gone") == std::string::npos);

    const auto tag = [](const char* v) { return f("tag", Op::Eq, std::string(v)); };
    REQUIRE(bits_to_ids(store.evaluate(tag("kept"), 3), 3) == std::vector<int64_t>{0, 1});
    REQUIRE(bits_to_ids(store.evaluate(tag("lost"), 3), 3).empty());
    REQUIRE(bits_to_ids(store.evaluate(Expr::is_null("tag"), 3), 3) == std::vector<int64_t>{2});
    std::istringstream in(out.str());
    MetadataStore loaded = MetadataStore::load(in, 3);
    REQUIRE(bits_to_ids(loaded.evaluate(tag("kept"), 3), 3) == std::vector<int64_t>{0, 1});

    // a freed string can come back
    loaded.append(3, 2, {{"tag", std::vector<std::string>{"lost", "kept"}}});
    REQUIRE(bits_to_ids(loaded.evaluate(tag("lost"), 5), 5) == std::vector<int64_t>{3});
    REQUIRE(bits_to_ids(loaded.evaluate(tag("kept"), 5), 5) == std::vector<int64_t>{0, 1, 4});
}

TEST_CASE("MetadataStore: string filters stay exact through adds, updates, compaction and load (fuzz)", "[metadata]") {
    // A few common strings and many rare ones, so filters take both the row
    // pass and the per-string rows, and strings move between lists and bitmaps.
    std::mt19937 rng(21);
    auto draw = [&]() -> std::optional<std::string> {
        const unsigned r = rng() % 100;
        if (r < 5) return std::nullopt;
        if (r < 45) return std::string("s0");
        if (r < 60) return std::string("s1");
        if (r < 68) return std::string("s2");
        return "s" + std::to_string(3 + rng() % 400);
    };
    auto column = [](const std::vector<std::optional<std::string>>& v) {
        std::vector<std::string> values(v.size());
        std::vector<uint8_t> valid(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            valid[i] = v[i].has_value();
            if (v[i]) values[i] = *v[i];
        }
        return Metadata{{"tag", values, valid}};
    };

    MetadataStore store;
    std::vector<std::optional<std::string>> ref;
    auto check = [&](int step) {
        CAPTURE(step, ref.size());
        const std::vector<std::string> probes = {"s0", "s1", "s2", "s" + std::to_string(3 + rng() % 400),
                                                 "s" + std::to_string(3 + rng() % 400), "zz"};
        const std::string& a = probes[rng() % probes.size()];
        const std::string& b = probes[rng() % probes.size()];
        struct Case {
            Expr where;
            std::function<bool(const std::optional<std::string>&)> match;
        };
        const std::vector<Case> cases = {
            {f("tag", Op::Eq, a), [&](const auto& v) { return v && *v == a; }},
            {f("tag", Op::Ne, a), [&](const auto& v) { return v && *v != a; }},
            {f("tag", Op::Lt, b), [&](const auto& v) { return v && *v < b; }},
            {Expr::isin("tag", {a, b, std::string("zz")}), [&](const auto& v) { return v && (*v == a || *v == b); }},
            {Expr::is_null("tag"), [&](const auto& v) { return !v; }},
        };
        for (size_t c = 0; c < cases.size(); ++c) {
            CAPTURE(c, a, b);
            REQUIRE(bits_to_ids(store.evaluate(cases[c].where, ref.size()), ref.size()) ==
                    ids_where(ref.size(), [&](size_t i) { return cases[c].match(ref[i]); }));
        }
    };

    for (int step = 0; step < 120; ++step) {
        const unsigned op = ref.empty() ? 0 : rng() % 6;
        if (op <= 1) {  // add a batch, sometimes leaving the column out (all null)
            const size_t n = 1 + rng() % (rng() % 4 == 0 ? 4000 : 300);
            std::vector<std::optional<std::string>> batch(n);
            const bool left_out = !ref.empty() && rng() % 8 == 0;
            for (auto& v : batch) v = left_out ? std::nullopt : draw();
            store.append(ref.size(), n, left_out ? Metadata{} : column(batch));
            ref.insert(ref.end(), batch.begin(), batch.end());
        } else if (op == 2) {  // overwrite distinct rows, sometimes all to one string
            std::vector<uint32_t> slots(ref.size());
            std::iota(slots.begin(), slots.end(), 0u);
            std::shuffle(slots.begin(), slots.end(), rng);
            slots.resize(1 + rng() % std::min<size_t>(ref.size(), 2000));
            const auto same = draw();
            std::vector<std::optional<std::string>> batch(slots.size());
            for (auto& v : batch) v = rng() % 2 ? same : draw();
            store.update(slots, column(batch));
            for (size_t i = 0; i < slots.size(); ++i) ref[slots[i]] = batch[i];
        } else if (op == 3) {  // drop about a tenth of the rows
            std::vector<uint32_t> remap(ref.size());
            std::vector<std::optional<std::string>> kept;
            for (size_t i = 0; i < ref.size(); ++i) {
                if (rng() % 10 == 0) remap[i] = IdMap::kRemoved;
                else {
                    remap[i] = static_cast<uint32_t>(kept.size());
                    kept.push_back(ref[i]);
                }
            }
            store.compact(remap);
            ref = std::move(kept);
        } else if (op == 4) {
            store.drop_unused_strings();
        } else {
            std::stringstream file;
            store.save(file);
            store = MetadataStore::load(file, ref.size());
        }
        check(step);
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

TEST_CASE("Metadata filters on other index settings, and HNSW's graph path", "[metadata]") {
    const size_t n = 4000, dim = 16, nq = 20, k = 10;
    const auto data = md_random(n, dim, 6);
    const auto queries = md_random(nq, dim, 7);

    // narrow: HNSW scores the allowed vectors directly; wide (80%): it searches its graph
    const Expr narrow = f("price", Op::Lt, int64_t{30}) & ~f("color", Op::Eq, std::string("red"));
    const Expr wide = f("price", Op::Lt, int64_t{80});
    const auto narrow_ids = ids_where(n, [](size_t i) { return i % 100 < 30 && i % 4 != 0; });
    const auto wide_ids = ids_where(n, [](size_t i) { return i % 100 < 80; });

    std::vector<std::pair<std::string, std::function<std::pair<std::unique_ptr<Index>, std::unique_ptr<Index>>()>>> configs = {
        {"ivf-sq8", [&] {
             auto ivf = std::make_unique<IVFIndex>(dim, 100, 20);
             ivf->enable_sq(8);
             ivf->train(data, n, 25, 1);
             return std::make_pair(std::unique_ptr<Index>(), std::unique_ptr<Index>(std::move(ivf)));
         }},
        {"ivf-fastscan+R", [&] {
             auto ivf = std::make_unique<IVFIndex>(dim, 100, 20);
             ivf->enable_pq(8, 16, true);
             ivf->train(data, n, 25, 1);
             auto refine = std::make_unique<RefineIndex>(*ivf, 8);
             return std::make_pair(std::unique_ptr<Index>(std::move(ivf)), std::unique_ptr<Index>(std::move(refine)));
         }},
        {"ivf-pq-rotate+R", [&] {
             auto ivf = std::make_unique<IVFIndex>(dim, 100, 20);
             ivf->enable_pq(8, 256, false, true);
             ivf->train(data, n, 25, 1);
             auto refine = std::make_unique<RefineIndex>(*ivf, 8);
             return std::make_pair(std::unique_ptr<Index>(std::move(ivf)), std::unique_ptr<Index>(std::move(refine)));
         }},
        {"hnsw-sq8+R", [&] {
             auto hnsw = std::make_unique<HNSWIndex>(dim, 16, 100, false, 1);
             hnsw->enable_sq(8);
             hnsw->train(data, n);
             auto refine = std::make_unique<RefineIndex>(*hnsw, 4);
             return std::make_pair(std::unique_ptr<Index>(std::move(hnsw)), std::unique_ptr<Index>(std::move(refine)));
         }},
        {"hnsw", [&] {
             return std::make_pair(std::unique_ptr<Index>(),
                                   std::unique_ptr<Index>(std::make_unique<HNSWIndex>(dim, 16, 100, false, 1)));
         }},
        {"flat-cosine", [&] {
             return std::make_pair(std::unique_ptr<Index>(),
                                   std::unique_ptr<Index>(std::make_unique<FlatIndex>(dim, n, false, Metric::Cosine)));
         }},
        {"hnsw-cosine", [&] {
             return std::make_pair(std::unique_ptr<Index>(), std::unique_ptr<Index>(std::make_unique<HNSWIndex>(
                                                                 dim, 16, 100, false, 1, Metric::Cosine)));
         }},
    };

    for (auto& [name, make] : configs) {
        INFO(name);
        auto [base, index] = make();
        index->add_batch(data, n, nullptr, md_rows(0, n));
        for (const auto& [where, ids] : {std::make_pair(narrow, narrow_ids), std::make_pair(wide, wide_ids)}) {
            const auto got = index->query_batch(queries, nq, k, where);
            md_require_identical(got, index->query_batch(queries, nq, k, ids));
            const std::set<int64_t> ok(ids.begin(), ids.end());
            for (const auto& row : got) {
                REQUIRE(row.size() == k);
                for (const auto& nb : row) REQUIRE(ok.count(static_cast<int64_t>(nb.index)) == 1);
            }
        }
    }
}

TEST_CASE("Metadata: a re-added id gets its new row, and empty adds create no columns", "[metadata]") {
    const size_t dim = 4;
    const auto data = md_random(3, dim, 8);
    FlatIndex flat(dim, 16, true);
    const std::vector<int64_t> ids = {5, 6};
    flat.add_batch(std::span<const float>(data.data(), 2 * dim), 2, ids.data(),
                   {{"color", std::vector<std::string>{"red", "red"}}});
    flat.remove(std::vector<int64_t>{5});
    const int64_t again = 5;
    flat.add_batch(std::span<const float>(data.data() + 2 * dim, dim), 1, &again,
                   {{"color", std::vector<std::string>{"blue"}}});
    const std::span<const float> q(data.data(), dim);
    REQUIRE(flat.query(q, 5, f("color", Op::Eq, std::string("blue")))[0].index == 5);
    const auto red = flat.query(q, 5, f("color", Op::Eq, std::string("red")));
    REQUIRE(red.size() == 1);
    REQUIRE(red[0].index == 6);
    flat.compact();
    REQUIRE(flat.query(q, 5, f("color", Op::Eq, std::string("blue")))[0].index == 5);

    // no vectors, so no evidence of a type: the column isn't created
    FlatIndex empty(dim);
    empty.add_batch(std::span<const float>(), 0, nullptr, {{"a", std::vector<double>{}}});
    REQUIRE(empty.metadata_schema().empty());
    empty.add_batch(std::span<const float>(data.data(), dim), 1, nullptr, {{"a", std::vector<std::string>{"x"}}});
    REQUIRE(empty.metadata_schema() == std::vector<std::pair<std::string, ColumnType>>{{"a", ColumnType::String}});
}

TEST_CASE("update_metadata overwrites rows in place and validates first", "[metadata]") {
    const size_t dim = 4, n = 6;
    const auto data = md_random(n, dim, 11);
    FlatIndex flat(dim, n, true);
    const std::vector<int64_t> ids = {10, 11, 12, 13, 14, 15};
    flat.add_batch(data, n, ids.data(),
                   {{"price", std::vector<int64_t>{1, 2, 3, 4, 5, 6}},
                    {"color", std::vector<std::string>{"red", "red", "blue", "blue", "red", "blue"}}});
    const std::span<const float> q(data.data(), dim);
    const auto match = [&](const Expr& e) {
        std::vector<int64_t> out;
        for (const auto& nb : flat.query(q, n, e)) out.push_back(static_cast<int64_t>(nb.index));
        std::sort(out.begin(), out.end());
        return out;
    };
    const auto color = [](const char* c) { return f("color", Op::Eq, std::string(c)); };
    using Ids = std::vector<int64_t>;

    // only the given rows and columns change
    flat.update_metadata(Ids{12, 10}, {{"price", std::vector<int64_t>{30, 10}}});
    REQUIRE(match(f("price", Op::Eq, int64_t{30})) == Ids{12});
    REQUIRE(match(f("price", Op::Eq, int64_t{10})) == Ids{10});
    REQUIRE(match(color("red")) == Ids{10, 11, 14});

    // a null flag clears a value; a new string joins the dictionary
    flat.update_metadata(Ids{11, 13}, {{"color", std::vector<std::string>{"", "green"}, {0, 1}}});
    REQUIRE(match(Expr::is_null("color")) == Ids{11});
    REQUIRE(match(color("green")) == Ids{13});
    // an all-null update fits a column of any type
    flat.update_metadata(Ids{15}, {{"price", std::vector<std::string>{""}, {0}}});
    REQUIRE(match(Expr::is_null("price")) == Ids{15});

    SECTION("bad calls throw and change nothing") {
        const auto before = match(f("price", Op::Ge, int64_t{0}));
        const Metadata one = {{"price", std::vector<int64_t>{99}}};
        REQUIRE_THROWS_WITH(flat.update_metadata(Ids{99}, one), "id 99 is not in the index");
        flat.remove(Ids{14});
        REQUIRE_THROWS_WITH(flat.update_metadata(Ids{14}, one), "id 14 is not in the index");
        REQUIRE_THROWS_WITH(flat.update_metadata(Ids{10, 11, 10}, {{"price", std::vector<int64_t>{1, 2, 3}}}),
                            "id 10 appears twice");
        REQUIRE_THROWS_AS(flat.update_metadata(Ids{10, 11}, one), std::invalid_argument);                    // length
        REQUIRE_THROWS_AS(flat.update_metadata(Ids{10}, {{"price", std::vector<double>{1.5}}}), std::invalid_argument);
        REQUIRE_THROWS_WITH(flat.update_metadata(Ids{10}, {{"prcie", std::vector<int64_t>{1}}}),
                            "the index has no metadata column 'prcie'; add it with add_metadata_column");
        // a bad second column stops the first from being written too
        REQUIRE_THROWS_AS(flat.update_metadata(Ids{10}, {{"price", std::vector<int64_t>{77}}, {"color", std::vector<int64_t>{1}}}),
                          std::invalid_argument);
        REQUIRE(match(f("price", Op::Ge, int64_t{0})) == Ids{10, 11, 12, 13});
        REQUIRE(before == Ids{10, 11, 12, 13, 14});
        REQUIRE(match(f("price", Op::Eq, int64_t{77})).empty());
    }

    SECTION("add_metadata_column adds a typed all-null column to fill with updates") {
        flat.add_metadata_column("year", ColumnType::Int64);
        REQUIRE(match(Expr::is_null("year")) == Ids{10, 11, 12, 13, 14, 15});
        flat.update_metadata(Ids{13}, {{"year", std::vector<int64_t>{2024}}});
        REQUIRE(match(f("year", Op::Eq, int64_t{2024})) == Ids{13});
        REQUIRE_THROWS_WITH(flat.add_metadata_column("year", ColumnType::Float64), "metadata column 'year' already exists");
        REQUIRE_THROWS_AS(flat.add_metadata_column("", ColumnType::Int64), std::invalid_argument);
        REQUIRE_THROWS_AS(flat.update_metadata(Ids{13}, {{"year", std::vector<double>{1.5}}}), std::invalid_argument);
        // later adds fill it like any column
        const int64_t next = 16;
        flat.add_batch(std::span<const float>(data.data(), dim), 1, &next, {{"year", std::vector<int64_t>{2025}}});
        REQUIRE(match(f("year", Op::Eq, int64_t{2025})) == Ids{16});
    }

    SECTION("no ids change nothing, but an unknown column still throws") {
        flat.update_metadata(Ids{}, {{"price", std::vector<double>{}}});    // no values, so no type to clash
        REQUIRE(match(f("price", Op::Ge, int64_t{0})) == Ids{10, 11, 12, 13, 14});
        REQUIRE_THROWS_AS(flat.update_metadata(Ids{}, {{"year", std::vector<int64_t>{}}}), std::invalid_argument);
        REQUIRE(flat.metadata_schema().size() == 2);
    }

    SECTION("updates survive compact and save / load, and replaced strings are freed") {
        flat.update_metadata(Ids{12, 15}, {{"color", std::vector<std::string>{"teal", "teal"}}});
        flat.remove(Ids{11});
        flat.compact();
        std::stringstream buf;
        flat.save(buf);
        REQUIRE(buf.str().find("teal") != std::string::npos);
        auto loaded = load_index(buf);
        const auto got = loaded->query(q, n, color("teal"));
        REQUIRE(got.size() == 2);
        flat.update_metadata(Ids{12, 15}, {{"color", std::vector<std::string>{"red", "red"}}});
        flat.compact();
        std::stringstream again;
        flat.save(again);
        REQUIRE(again.str().find("teal") == std::string::npos);
    }
}

TEST_CASE("update_metadata on auto ids after compaction, and through a RefineIndex", "[metadata]") {
    const size_t dim = 4, n = 8;
    const auto data = md_random(n, dim, 12);
    using Ids = std::vector<int64_t>;

    FlatIndex flat(dim);
    flat.add_batch(data, n, nullptr, {{"price", std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7}}});
    flat.remove(Ids{0, 1});
    flat.compact();    // ids 2..7 now sit in slots 0..5
    flat.update_metadata(Ids{7, 2}, {{"price", std::vector<int64_t>{70, 20}}});
    const std::span<const float> q(data.data(), dim);
    REQUIRE(flat.query(q, n, f("price", Op::Eq, int64_t{70}))[0].index == 7);
    REQUIRE(flat.query(q, n, f("price", Op::Eq, int64_t{20}))[0].index == 2);
    REQUIRE_THROWS_AS(flat.update_metadata(Ids{0}, {{"price", std::vector<int64_t>{1}}}), std::invalid_argument);

    FlatIndex base(dim);
    RefineIndex refine(base, 2, true);
    const Ids rids = {100, 200, 300};
    refine.add_batch(std::span<const float>(data.data(), 3 * dim), 3, rids.data(),
                     {{"color", std::vector<std::string>{"a", "b", "c"}}});
    refine.update_metadata(Ids{200}, {{"color", std::vector<std::string>{"a"}}});
    REQUIRE(refine.query(q, 3, f("color", Op::Eq, std::string("a"))).size() == 2);
    REQUIRE(base.metadata_schema().empty());
    refine.add_metadata_column("rank", ColumnType::Float64);
    refine.update_metadata(Ids{300}, {{"rank", std::vector<double>{0.5}}});
    REQUIRE(refine.query(q, 3, f("rank", Op::Gt, 0.0))[0].index == 300);
    REQUIRE(base.metadata_schema().empty());

    // a column on an empty index: adds without it fill it with nulls
    FlatIndex fresh(dim);
    fresh.add_metadata_column("tag", ColumnType::String);
    fresh.add_batch(std::span<const float>(data.data(), dim), 1);
    REQUIRE(fresh.metadata_schema() == std::vector<std::pair<std::string, ColumnType>>{{"tag", ColumnType::String}});
    REQUIRE(fresh.query(q, 1, Expr::is_null("tag")).size() == 1);
}

TEST_CASE("add_metadata_column: a default, values by id, ints into floats, and drop", "[metadata]") {
    const size_t dim = 4, n = 5;
    const auto data = md_random(n, dim, 14);
    FlatIndex flat(dim, n, true);
    const std::vector<int64_t> ids = {1, 2, 3, 4, 5};
    flat.add_batch(data, n, ids.data(), {{"weight", std::vector<double>{0.5, 1.5, 2.5, 3.5, 4.5}}});
    const std::span<const float> q(data.data(), dim);
    using Ids = std::vector<int64_t>;
    const auto match = [&](const Expr& e) {
        Ids out;
        for (const auto& nb : flat.query(q, 16, e)) out.push_back(static_cast<int64_t>(nb.index));    // more than the index holds
        std::sort(out.begin(), out.end());
        return out;
    };

    // ints fit a float column, in adds and updates; floats don't fit an int column
    const int64_t six = 6;
    flat.add_batch(std::span<const float>(data.data(), dim), 1, &six, {{"weight", std::vector<int64_t>{7}}});
    flat.update_metadata(Ids{1}, {{"weight", std::vector<int64_t>{9}}});
    REQUIRE(match(f("weight", Op::Eq, 7.0)) == Ids{6});
    REQUIRE(match(f("weight", Op::Eq, int64_t{9})) == Ids{1});
    REQUIRE(flat.metadata_schema()[0].second == ColumnType::Float64);

    SECTION("a default fills every vector") {
        flat.add_metadata_column("in_stock", ColumnType::Int64, Expr::Value{int64_t{1}});
        REQUIRE(match(f("in_stock", Op::Eq, int64_t{1})) == Ids{1, 2, 3, 4, 5, 6});
        flat.add_metadata_column("score", ColumnType::Float64, Expr::Value{int64_t{2}});    // an int default for a float column
        REQUIRE(match(f("score", Op::Eq, 2.0)).size() == 6);
        flat.add_metadata_column("label", ColumnType::String, Expr::Value{std::string("new")});
        REQUIRE(match(f("label", Op::Eq, std::string("new"))).size() == 6);
        REQUIRE_THROWS_WITH(flat.add_metadata_column("bad", ColumnType::Int64, Expr::Value{1.5}),
                            "metadata column 'bad' holds int values; got float");
        REQUIRE_THROWS_AS(flat.add_metadata_column("bad", ColumnType::String, Expr::Value{int64_t{1}}), std::invalid_argument);
        REQUIRE(flat.metadata_schema().size() == 4);
        // a later add that leaves it out stores null
        const int64_t seven = 7;
        flat.add_batch(std::span<const float>(data.data(), dim), 1, &seven);
        REQUIRE(match(Expr::is_null("in_stock")) == Ids{7});
    }

    SECTION("values by id fill those vectors, null for the rest, in one step") {
        flat.add_metadata_column("tag", ColumnType::String, Ids{4, 2}, std::vector<std::string>{"d", ""}, {1, 0});
        REQUIRE(match(f("tag", Op::Eq, std::string("d"))) == Ids{4});
        REQUIRE(match(Expr::is_null("tag")) == Ids{1, 2, 3, 5, 6});
        flat.add_metadata_column("rank", ColumnType::Float64, Ids{3}, std::vector<int64_t>{8});
        REQUIRE(match(f("rank", Op::Eq, 8.0)) == Ids{3});
        // every error leaves the schema unchanged
        const auto schema = flat.metadata_schema();
        REQUIRE_THROWS_WITH(flat.add_metadata_column("x", ColumnType::Int64, Ids{99}, std::vector<int64_t>{1}),
                            "id 99 is not in the index");
        REQUIRE_THROWS_AS(flat.add_metadata_column("x", ColumnType::Int64, Ids{1, 1}, std::vector<int64_t>{1, 2}),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(flat.add_metadata_column("x", ColumnType::Int64, Ids{1, 2}, std::vector<int64_t>{1}),
                          std::invalid_argument);
        REQUIRE_THROWS_AS(flat.add_metadata_column("x", ColumnType::Int64, Ids{1}, std::vector<double>{1.5}),
                          std::invalid_argument);
        REQUIRE_THROWS_WITH(flat.add_metadata_column("tag", ColumnType::String, Ids{1}, std::vector<std::string>{"a"}),
                            "metadata column 'tag' already exists");
        REQUIRE(flat.metadata_schema() == schema);
        // all-null values are fine: the type is given
        flat.add_metadata_column("empty", ColumnType::Int64, Ids{1}, std::vector<int64_t>{0}, {0});
        REQUIRE(match(Expr::is_null("empty")).size() == 6);
    }

    SECTION("drop removes a column and its values") {
        flat.add_metadata_column("tag", ColumnType::String, Ids{1}, std::vector<std::string>{"gone"});
        flat.drop_metadata_column("tag");
        REQUIRE(flat.metadata_schema().size() == 1);
        REQUIRE_THROWS_AS(flat.query(q, 1, Expr::is_null("tag")), std::invalid_argument);
        REQUIRE_THROWS_WITH(flat.drop_metadata_column("tag"), "the index has no metadata column 'tag'");
        std::stringstream buf;
        flat.save(buf);
        REQUIRE(buf.str().find("gone") == std::string::npos);
        // the name can be reused, with another type
        flat.add_metadata_column("tag", ColumnType::Int64);
        const int64_t eight = 8;
        flat.add_batch(std::span<const float>(data.data(), dim), 1, &eight, {{"tag", std::vector<int64_t>{3}}});
        REQUIRE(match(f("tag", Op::Eq, int64_t{3})) == Ids{8});
    }
}

TEST_CASE("update_metadata and string cleanup on compact work on every index", "[metadata]") {
    const size_t n = 1000, dim = 8;
    const auto data = md_random(n, dim, 15);
    std::vector<std::string> tags(n, "common");
    tags[5] = "rare";

    auto flat = std::make_unique<FlatIndex>(dim);
    auto ivf = std::make_unique<IVFIndex>(dim, 100, 100);    // every list probed, so results are exact
    ivf->train(data, n, 25, 1);
    auto hnsw = std::make_unique<HNSWIndex>(dim, 16, 100, false, 1);
    auto base = std::make_unique<FlatIndex>(dim);
    auto refine = std::make_unique<RefineIndex>(*base, 4);
    std::vector<std::pair<std::string, Index*>> indexes = {
        {"flat", flat.get()}, {"ivf", ivf.get()}, {"hnsw", hnsw.get()}, {"refine", refine.get()}};

    using Ids = std::vector<int64_t>;
    const std::span<const float> q(data.data(), dim);
    for (auto& [name, index] : indexes) {
        INFO(name);
        index->add_batch(data, n, nullptr, {{"tag", tags}});
        index->update_metadata(Ids{5, 9}, {{"tag", std::vector<std::string>{"common", "fresh"}}});
        const auto fresh = index->query(q, n, f("tag", Op::Eq, std::string("fresh")));
        REQUIRE(fresh.size() == 1);
        REQUIRE(fresh[0].index == 9);
        REQUIRE(index->query(q, n, f("tag", Op::Eq, std::string("rare"))).empty());

        // nothing is deleted, so compact frees only the string no row uses
        REQUIRE(index->compact() == 0);
        std::stringstream buf;
        index->save(buf);
        REQUIRE(buf.str().find("rare") == std::string::npos);
        REQUIRE(buf.str().find("fresh") != std::string::npos);
        auto loaded = load_index(buf);
        const auto again = loaded->query(q, n, f("tag", Op::Eq, std::string("fresh")));
        REQUIRE(again.size() == 1);
        REQUIRE(again[0].index == 9);

        // a column left with no values survives compact
        index->add_metadata_column("empty", ColumnType::Int64);
        index->remove(Ids{0});
        index->compact();
        REQUIRE(index->metadata_schema().size() == 2);
    }
    REQUIRE(base->metadata_schema().empty());
}

TEST_CASE("add_batch returns the vectors' ids", "[metadata]") {
    const size_t dim = 4;
    const auto data = md_random(4, dim, 16);
    using Ids = std::vector<int64_t>;

    // auto ids continue after the highest ever assigned, not after size()
    FlatIndex flat(dim);
    REQUIRE(flat.add_batch(std::span<const float>(data.data(), 3 * dim), 3) == Ids{0, 1, 2});
    flat.remove(Ids{1, 2});
    flat.compact();
    REQUIRE(flat.size() == 1);
    REQUIRE(flat.add_batch(std::span<const float>(data.data(), dim), 1) == Ids{3});
    REQUIRE(flat.add_batch(std::span<const float>(), 0).empty());

    // custom ids come back as given
    FlatIndex custom(dim, 16, true);
    const Ids given = {42, 7};
    REQUIRE(custom.add_batch(std::span<const float>(data.data(), 2 * dim), 2, given.data()) == given);

    // a RefineIndex reports its own ids
    FlatIndex base(dim);
    RefineIndex refine(base, 2, true);
    const Ids rids = {500, 600};
    REQUIRE(refine.add_batch(std::span<const float>(data.data(), 2 * dim), 2, rids.data()) == rids);
    FlatIndex auto_base(dim);
    RefineIndex auto_refine(auto_base, 2);
    auto_refine.add_batch(std::span<const float>(data.data(), 2 * dim), 2);
    auto_refine.remove(Ids{0});
    auto_refine.compact();
    REQUIRE(auto_refine.add_batch(std::span<const float>(data.data(), dim), 1) == Ids{2});
}

TEST_CASE("add_batch takes null flags", "[metadata]") {
    const size_t dim = 4;
    const auto data = md_random(3, dim, 13);
    FlatIndex flat(dim);
    flat.add_batch(data, 3, nullptr, {{"tag", std::vector<std::string>{"x", "", "y"}, {1, 0, 1}}});
    const std::span<const float> q(data.data(), dim);
    REQUIRE(flat.query(q, 3, Expr::is_null("tag"))[0].index == 1);
    REQUIRE_THROWS_AS(flat.add_batch(std::span<const float>(data.data(), dim), 1, nullptr,
                                     {{"tag", std::vector<std::string>{"x"}, {1, 1}}}),
                      std::invalid_argument);    // flags must match the values
    REQUIRE_THROWS_AS(flat.add_batch(std::span<const float>(data.data(), dim), 1, nullptr,
                                     {{"new", std::vector<int64_t>{0}, {0}}}),
                      std::invalid_argument);    // a new all-null column has no type
    REQUIRE(flat.size() == 3);
}

TEST_CASE("Metadata filters at 100k rows match a reference evaluation", "[metadata]") {
    // 100003 rows: not a multiple of 64, so the last word's padding is exercised by ~
    const size_t n = 100003, dim = 4, nq = 5, k = 20;
    const auto data = md_random(n, dim, 9);
    const auto queries = md_random(nq, dim, 10);
    std::mt19937 rng(11);
    std::vector<int64_t> price(n);
    std::vector<double> weight(n);
    std::vector<std::string> color(n);
    for (size_t i = 0; i < n; ++i) {
        price[i] = static_cast<int64_t>(rng() % 1000);
        weight[i] = static_cast<double>(rng() % 10000) / 100.0;
        color[i] = kColors[rng() % 4];
    }
    FlatIndex flat(dim, n);
    flat.add_batch(std::span<const float>(data.data(), (n - 3) * dim), n - 3, nullptr,
                   {{"price", std::vector<int64_t>(price.begin(), price.end() - 3)},
                    {"weight", std::vector<double>(weight.begin(), weight.end() - 3)},
                    {"color", std::vector<std::string>(color.begin(), color.end() - 3)}});
    flat.add_batch(std::span<const float>(data.data() + (n - 3) * dim, 3 * dim), 3);  // last three null

    struct Case {
        Expr where;
        std::function<bool(size_t)> match;
    };
    auto has = [&](size_t i) { return i < n - 3; };
    const std::vector<Case> cases = {
        {f("price", Op::Lt, int64_t{10}), [&](size_t i) { return has(i) && price[i] < 10; }},
        {~f("price", Op::Lt, int64_t{990}), [&](size_t i) { return !(has(i) && price[i] < 990); }},
        {(f("weight", Op::Ge, 99.5) | Expr::is_null("color")) & ~f("color", Op::Eq, std::string("red")),
         [&](size_t i) { return (has(i) ? weight[i] >= 99.5 : true) && !(has(i) && color[i] == "red"); }},
        {Expr::isin("price", {int64_t{1}, int64_t{500}, 999.0}) & f("color", Op::Ge, std::string("blue")),
         [&](size_t i) { return has(i) && (price[i] == 1 || price[i] == 500 || price[i] == 999) && color[i] >= "blue"; }},
    };
    for (size_t c = 0; c < cases.size(); ++c) {
        CAPTURE(c);
        const auto ids = ids_where(n, cases[c].match);
        md_require_identical(flat.query_batch(queries, nq, k, cases[c].where), flat.query_batch(queries, nq, k, ids));
    }
}

TEST_CASE("String filters through per-string rows skip removed vectors, before and after compaction", "[metadata]") {
    // tag: 40% "common" (a bitmap), the rest spread over 1000 rare strings (lists)
    const size_t n = 20000, dim = 4, nq = 4, k = 50;
    const auto data = md_random(n + 500, dim, 31);
    const auto queries = md_random(nq, dim, 32);
    std::mt19937 rng(33);
    std::vector<std::string> tag(n + 500);
    for (auto& t : tag) t = rng() % 10 < 4 ? "common" : "r" + std::to_string(rng() % 1000);

    FlatIndex flat(dim);
    flat.add_batch(std::span<const float>(data.data(), n * dim), n, nullptr,
                   {{"tag", std::vector<std::string>(tag.begin(), tag.begin() + n)}});
    std::vector<uint8_t> removed(n + 500, 0);
    std::vector<int64_t> drop;
    for (size_t i = 0; i < n; ++i)
        if (rng() % 10 == 0 || tag[i] == "r5") drop.push_back(static_cast<int64_t>(i));
    for (int64_t id : drop) removed[id] = 1;
    flat.remove(drop);

    auto check = [&](size_t rows) {
        const std::vector<std::pair<Expr, std::set<std::string>>> cases = {
            {f("tag", Op::Eq, std::string("common")), {"common"}},
            {Expr::isin("tag", {std::string("r1"), std::string("r2")}), {"r1", "r2"}},
            {f("tag", Op::Eq, std::string("r5")), {"r5"}},
        };
        for (size_t c = 0; c < cases.size(); ++c) {
            CAPTURE(c, rows);
            const auto ids = ids_where(rows, [&](size_t i) { return !removed[i] && cases[c].second.count(tag[i]); });
            md_require_identical(flat.query_batch(queries, nq, k, cases[c].first), flat.query_batch(queries, nq, k, ids));
        }
    };
    check(n);
    flat.compact();
    check(n);
    flat.add_batch(std::span<const float>(data.data() + n * dim, 500 * dim), 500, nullptr,
                   {{"tag", std::vector<std::string>(tag.begin() + n, tag.end())}});
    check(n + 500);
}

TEST_CASE("Expr::parse builds the same tree as the operator form", "[metadata][parse]") {
    auto same = [](const std::string& sql, const Expr& e) {
        INFO(sql);
        REQUIRE(Expr::parse(sql).to_string() == e.to_string());
    };
    const auto i = [](int64_t v) { return Expr::Value{v}; };
    const auto s = [](const char* v) { return Expr::Value{std::string(v)}; };

    same("category = 'shoes' AND price < 50", f("category", Op::Eq, s("shoes")) & f("price", Op::Lt, i(50)));
    same("a = 1", f("a", Op::Eq, i(1)));
    same("a == 1", f("a", Op::Eq, i(1)));
    same("a != 1", f("a", Op::Ne, i(1)));
    same("a <> 1", f("a", Op::Ne, i(1)));
    same("a<1", f("a", Op::Lt, i(1)));
    same("a <= 1", f("a", Op::Le, i(1)));
    same("a > -1", f("a", Op::Gt, i(-1)));
    same("a >= +1", f("a", Op::Ge, i(1)));
    same("c IN ('a', 'b')", Expr::isin("c", {s("a"), s("b")}));
    same("c NOT IN (1)", ~Expr::isin("c", {i(1)}) & ~Expr::is_null("c"));
    same("c IS NULL", Expr::is_null("c"));
    same("c IS NOT NULL", ~Expr::is_null("c"));
    same("p BETWEEN 1 AND 5", f("p", Op::Ge, i(1)) & f("p", Op::Le, i(5)));
    same("p NOT BETWEEN 1 AND 5", f("p", Op::Lt, i(1)) | f("p", Op::Gt, i(5)));

    SECTION("NOT is pushed down to the leaves") {
        same("NOT p < 5", f("p", Op::Ge, i(5)));
        same("NOT p >= 5", f("p", Op::Lt, i(5)));
        same("NOT p = 5", f("p", Op::Ne, i(5)));
        same("NOT NOT p = 5", f("p", Op::Eq, i(5)));
        same("NOT (a = 1 AND b = 2)", f("a", Op::Ne, i(1)) | f("b", Op::Ne, i(2)));
        same("NOT (a = 1 OR b = 2)", f("a", Op::Ne, i(1)) & f("b", Op::Ne, i(2)));
        same("NOT a IS NULL", ~Expr::is_null("a"));
        same("NOT a IS NOT NULL", Expr::is_null("a"));
        same("NOT a NOT IN (1)", Expr::isin("a", {i(1)}));
        same("NOT a BETWEEN 1 AND 5", f("a", Op::Lt, i(1)) | f("a", Op::Gt, i(5)));
    }

    SECTION("precedence: comparisons, then NOT, AND, OR") {
        const Expr a = f("a", Op::Eq, i(1)), b = f("b", Op::Eq, i(2)), c = f("c", Op::Eq, i(3));
        same("a = 1 OR b = 2 AND c = 3", a | (b & c));
        same("a = 1 AND b = 2 OR c = 3", (a & b) | c);
        same("(a = 1 OR b = 2) AND c = 3", (a | b) & c);
        same("a = 1 AND b = 2 AND c = 3", (a & b) & c);
        same("NOT a = 1 AND b = 2", f("a", Op::Ne, i(1)) & b);
        same("((a = 1))", a);
        same("a BETWEEN 1 AND 2 AND b = 2", (f("a", Op::Ge, i(1)) & f("a", Op::Le, i(2))) & b);
    }

    SECTION("keywords are case-insensitive and only whole words") {
        same("a in (1) and b is not null Or c between 1 aNd 2",
             (Expr::isin("a", {i(1)}) & ~Expr::is_null("b")) | (f("c", Op::Ge, i(1)) & f("c", Op::Le, i(2))));
        same("notes = 1 AND android = 2 OR order_id IN (3)",
             (f("notes", Op::Eq, i(1)) & f("android", Op::Eq, i(2))) | Expr::isin("order_id", {i(3)}));
    }

    SECTION("quoting and values") {
        same("x = 'it''s'", f("x", Op::Eq, s("it's")));
        same("x = ''", f("x", Op::Eq, s("")));
        same("x = 'a AND b = 1'", f("x", Op::Eq, s("a AND b = 1")));
        same("\"and\" = 1", f("and", Op::Eq, i(1)));
        same("\"a\"\"b c\" = 1", f("a\"b c", Op::Eq, i(1)));
        same("user.id = 1", f("user.id", Op::Eq, i(1)));
        same("x = -3e2", f("x", Op::Eq, -300.0));
        same("x = 2.5", f("x", Op::Eq, 2.5));
        REQUIRE(std::holds_alternative<int64_t>(Expr::parse("x = 42").values()[0]));
        REQUIRE(std::holds_alternative<double>(Expr::parse("x = 42.0").values()[0]));
        REQUIRE(std::holds_alternative<double>(Expr::parse("x = 1e3").values()[0]));
        REQUIRE(std::get<int64_t>(Expr::parse("x = -9223372036854775808").values()[0]) == INT64_MIN);
        same("x = .5", f("x", Op::Eq, 0.5));
        same("x = -.5", f("x", Op::Eq, -0.5));
        same("x = 1.", f("x", Op::Eq, 1.0));
        same("x = 1e-400", f("x", Op::Eq, 0.0));    // underflow rounds to zero
        same("\"café\" = 'thé'", f("café", Op::Eq, s("thé")));
    }

    SECTION("BETWEEN is Expr::between") {
        same("x BETWEEN 1 AND 2.5", Expr::between("x", int64_t{1}, 2.5));
        REQUIRE(Expr::between("x", int64_t{1}, int64_t{2}).to_string() == "((x >= 1) & (x <= 2))");
    }
}

TEST_CASE("Expr::parse binds parameters without changing the structure", "[metadata][parse]") {
    using Params = std::vector<Expr::Value>;
    using Named = std::unordered_map<std::string, Expr::Value>;

    REQUIRE(Expr::parse("a = ? AND b IN (?, ?)", Params{int64_t{7}, std::string("x"), 2.5}).to_string() ==
            (f("a", Op::Eq, int64_t{7}) & Expr::isin("b", {std::string("x"), 2.5})).to_string());
    REQUIRE(Expr::parse("a = :uid AND b < :p OR c = :uid", Named{{"uid", int64_t{3}}, {"p", 9.5}}).to_string() ==
            ((f("a", Op::Eq, int64_t{3}) & f("b", Op::Lt, 9.5)) | f("c", Op::Eq, int64_t{3})).to_string());

    // an injected string stays one value
    const Expr e = Expr::parse("a = ?", Params{std::string("1 OR 1=1")});
    REQUIRE(e.op() == Op::Eq);
    REQUIRE(std::get<std::string>(e.values()[0]) == "1 OR 1=1");

    REQUIRE_THROWS_WITH(Expr::parse("a = ?"), "? placeholder, but no parameters were given at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a = ? AND b = ?", Params{int64_t{1}}), "more ? placeholders than parameters at column 15");
    REQUIRE_THROWS_WITH(Expr::parse("a = ?", Params{int64_t{1}, int64_t{2}}), "filter has 1 ? placeholders but 2 parameters were given");
    REQUIRE_THROWS_WITH(Expr::parse("a = 1", Params{int64_t{1}}), "filter has 0 ? placeholders but 1 parameters were given");
    REQUIRE_THROWS_WITH(Expr::parse("a = ?", Named{{"x", int64_t{1}}}), "? placeholder, but parameters were given by name at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a = :x", Params{int64_t{1}}), ":x placeholder, but parameters were given as a list at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a = :x"), ":x placeholder, but no parameters were given at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a = :y", Named{{"x", int64_t{1}}}), "no parameter named 'y' at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a = :", Named{{"x", int64_t{1}}}), "expected a parameter name after ':' at column 5");
}

TEST_CASE("Expr::parse reports where a filter is wrong", "[metadata][parse]") {
    REQUIRE_THROWS_WITH(Expr::parse(""), "expected a column name at end of filter");
    REQUIRE_THROWS_WITH(Expr::parse("a = 1 AND"), "expected a column name at end of filter");
    REQUIRE_THROWS_WITH(Expr::parse("and = 1"), "expected a column name at column 1");
    REQUIRE_THROWS_WITH(Expr::parse("a = 1 b = 2"), "unexpected 'b' at column 7");
    REQUIRE_THROWS_WITH(Expr::parse("(a = 1"), "expected ')' at end of filter");
    REQUIRE_THROWS_WITH(Expr::parse("a < 'x"), "unterminated string at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("\"a = 1"), "unterminated quoted name at column 1");
    REQUIRE_THROWS_WITH(Expr::parse("a IN 1"), "expected '(' at column 6");
    REQUIRE_THROWS_WITH(Expr::parse("a IN (1 2)"), "expected ',' or ')' at column 9");
    REQUIRE_THROWS_WITH(Expr::parse("a NOT = 1"), "expected IN or BETWEEN at column 7");
    REQUIRE_THROWS_WITH(Expr::parse("a BETWEEN 1 OR 2"), "expected AND at column 13");
    REQUIRE_THROWS_WITH(Expr::parse("a IS 1"), "expected NULL at column 6");
    REQUIRE_THROWS_WITH(Expr::parse("a ~ 1"), "expected a comparison, IN, BETWEEN or IS at column 3");
    REQUIRE_THROWS_WITH(Expr::parse("a = x"), "expected a value at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a = 1abc"), "expected a value at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a = 99999999999999999999"), "integer out of range at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a = -9223372036854775809"), "integer out of range at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a = 1e400"), "number out of range at column 5");
    REQUIRE_THROWS_WITH(Expr::parse("a IN (1, -1e400)"), "number out of range at column 10");
    REQUIRE_THROWS_WITH(Expr::parse("a IN ()"), "expected a value at column 7");
    REQUIRE_THROWS_WITH(Expr::parse("café = 1"), "a column name with non-ASCII characters needs double quotes at column 1");
    REQUIRE_THROWS_WITH(Expr::parse("a = 1 AND é = 1"), "a column name with non-ASCII characters needs double quotes at column 11");
    REQUIRE_THROWS_WITH(Expr::parse("a = 1 é"), "unexpected 'é' at column 7");
    REQUIRE_THROWS_AS(Expr::parse("a = 1 OR OR b = 2"), std::invalid_argument);
}

TEST_CASE("Expr::parse matches SQL's three-valued logic on rows with nulls (fuzz)", "[metadata][parse]") {
    // Columns a (int), café (float) and "and" (string): one bare, one that needs
    // quoting for its non-ASCII name, one for being a keyword. Batches leave
    // columns out so every combination of nulls appears.
    constexpr size_t kRows = 40;
    std::vector<std::optional<int64_t>> A(kRows);
    std::vector<std::optional<double>> B(kRows);
    std::vector<std::optional<std::string>> S(kRows);
    MetadataStore store;
    for (size_t batch = 0; batch < 4; ++batch) {
        const bool has_a = batch < 2, has_bs = batch == 0 || batch == 2;
        std::vector<int64_t> a;
        std::vector<double> b;
        std::vector<std::string> s;
        for (size_t r = batch * 10; r < batch * 10 + 10; ++r) {
            if (has_a) a.push_back(A[r].emplace(static_cast<int64_t>(r % 7)));
            if (has_bs) {
                b.push_back(B[r].emplace(static_cast<double>(r % 9) * 0.5));
                s.push_back(S[r].emplace(std::string(1, static_cast<char>('x' + r % 3))));
            }
        }
        Metadata md;
        if (has_a) md.push_back({"a", a});
        if (has_bs) md.push_back({"café", b}), md.push_back({"and", s});
        store.check(10, md);
        store.append(batch * 10, 10, md);
    }

    enum class T { F, U, Tr };
    const auto not3 = [](T x) { return x == T::U ? T::U : (x == T::Tr ? T::F : T::Tr); };
    const auto and3 = [](T x, T y) { return x == T::F || y == T::F ? T::F : (x == T::Tr && y == T::Tr ? T::Tr : T::U); };
    const auto or3 = [](T x, T y) { return x == T::Tr || y == T::Tr ? T::Tr : (x == T::F && y == T::F ? T::F : T::U); };
    const auto tri = [](bool b) { return b ? T::Tr : T::F; };

    struct Lit {
        std::string sql;
        double num = 0;
        std::string str;
    };
    struct Node {
        std::string sql;
        std::function<T(size_t)> eval;
    };
    std::mt19937 rng(46);
    auto pick = [&](size_t n) { return static_cast<size_t>(rng() % n); };
    auto kw = [&](const char* w) {
        std::string out = w;
        if (pick(2)) for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        return out;
    };
    auto show = [](const Expr::Value& v) {
        std::ostringstream o;
        std::visit([&](const auto& x) { o << x; }, v);
        return o.str();
    };
    // mode 0 writes every value inline; 1 and 2 also bind some as ? or :name
    // parameters (one kind per filter, as parse() takes either a list or a map)
    int mode = 0;
    std::vector<Expr::Value> list;
    std::unordered_map<std::string, Expr::Value> named;
    auto render = [&](Expr::Value v, std::string inline_sql) -> std::string {
        if (mode == 0 || pick(2)) return inline_sql;
        if (mode == 1) {
            list.push_back(std::move(v));
            return "?";
        }
        const std::string key = "p" + std::to_string(named.size());
        named.emplace(key, std::move(v));
        return ":" + key;
    };
    auto col_sql = [&](size_t c) -> std::string {
        if (c == 0) return pick(2) ? "a" : "\"a\"";
        return c == 1 ? "\"café\"" : "\"and\"";
    };
    // both number columns get ints and halves, so comparisons cross types;
    // strings include a quote and an injection attempt, escaped inline as ''
    static const char* strs[] = {"w", "x", "y", "z", "it's", "x' OR 'a' = 'a"};
    auto lit = [&](size_t c) {
        if (c != 2) {
            Expr::Value v;
            double num;
            if (pick(2)) {
                const int64_t i = static_cast<int64_t>(pick(8));
                v = i, num = static_cast<double>(i);
            } else {
                num = static_cast<double>(pick(10)) * 0.5;
                v = num;
            }
            return Lit{render(v, show(v)), num, ""};
        }
        const std::string v = strs[pick(6)];
        std::string quoted = "'";
        for (char ch : v) quoted += ch == '\'' ? std::string("''") : std::string(1, ch);
        return Lit{render(v, quoted + "'"), 0, v};
    };
    // sign of row r's value against l, or nullopt when the row is null
    auto cmp = [&](size_t c, size_t r, const Lit& l) -> std::optional<int> {
        if (c == 2) {
            if (!S[r]) return std::nullopt;
            return (*S[r] > l.str) - (*S[r] < l.str);
        }
        double x;
        if (c == 0) {
            if (!A[r]) return std::nullopt;
            x = static_cast<double>(*A[r]);
        } else {
            if (!B[r]) return std::nullopt;
            x = *B[r];
        }
        return (x > l.num) - (x < l.num);
    };

    std::function<Node(int)> gen = [&](int depth) -> Node {
        if (depth == 0 || pick(3) == 0) {
            const size_t c = pick(3);
            const std::string col = col_sql(c);
            switch (pick(6)) {
                case 0: {
                    static const char* ops[] = {"=", "==", "!=", "<>", "<", "<=", ">", ">="};
                    const size_t op = pick(8);
                    const Lit l = lit(c);
                    return {col + " " + ops[op] + " " + l.sql, [=](size_t r) {
                                const auto sg = cmp(c, r, l);
                                if (!sg) return T::U;
                                switch (op) {
                                    case 0: case 1: return tri(*sg == 0);
                                    case 2: case 3: return tri(*sg != 0);
                                    case 4: return tri(*sg < 0);
                                    case 5: return tri(*sg <= 0);
                                    case 6: return tri(*sg > 0);
                                    default: return tri(*sg >= 0);
                                }
                            }};
                }
                case 1: case 2: {
                    const bool negated = pick(2);
                    std::vector<Lit> ls(1 + pick(3));
                    std::string sql = col + (negated ? " " + kw("NOT") : "") + " " + kw("IN") + " (";
                    for (size_t k = 0; k < ls.size(); ++k) sql += (k ? ", " : "") + (ls[k] = lit(c)).sql;
                    return {sql + ")", [=](size_t r) {
                                if (!cmp(c, r, ls[0])) return T::U;
                                bool in = false;
                                for (const Lit& l : ls) in = in || *cmp(c, r, l) == 0;
                                return tri(in != negated);
                            }};
                }
                case 3: {
                    const bool negated = pick(2);
                    return {col + " " + kw("IS") + (negated ? " " + kw("NOT") : "") + " " + kw("NULL"), [=](size_t r) {
                                const bool null = c == 0 ? !A[r] : c == 1 ? !B[r] : !S[r];
                                return tri(null != negated);
                            }};
                }
                default: {
                    const bool negated = pick(2);
                    const Lit lo = lit(c), hi = lit(c);
                    return {col + (negated ? " " + kw("NOT") : "") + " " + kw("BETWEEN") + " " + lo.sql + " " + kw("AND") + " " + hi.sql,
                            [=](size_t r) {
                                const auto a = cmp(c, r, lo), b = cmp(c, r, hi);
                                if (!a) return T::U;
                                return tri((*a >= 0 && *b <= 0) != negated);
                            }};
                }
            }
        }
        switch (pick(3)) {
            case 0: {
                const Node x = gen(depth - 1), y = gen(depth - 1);
                return {"(" + x.sql + " " + kw("AND") + " " + y.sql + ")", [=](size_t r) { return and3(x.eval(r), y.eval(r)); }};
            }
            case 1: {
                const Node x = gen(depth - 1), y = gen(depth - 1);
                return {"(" + x.sql + " " + kw("OR") + " " + y.sql + ")", [=](size_t r) { return or3(x.eval(r), y.eval(r)); }};
            }
            default: {
                const Node x = gen(depth - 1);
                return {kw("NOT") + " (" + x.sql + ")", [=](size_t r) { return not3(x.eval(r)); }};
            }
        }
    };

    size_t bound = 0;
    for (int iter = 0; iter < 3000; ++iter) {
        mode = iter % 3;
        list.clear();
        named.clear();
        const Node n = gen(4);
        std::string params;
        for (const auto& v : list) params += " " + show(v);
        for (const auto& [k, v] : named) params += " :" + k + "=" + show(v);
        INFO(n.sql << " | params:" << params);
        bound += list.size() + named.size();
        const auto want = ids_where(kRows, [&](size_t r) { return n.eval(r) == T::Tr; });
        const Expr e = mode == 1 ? Expr::parse(n.sql, list) : mode == 2 ? Expr::parse(n.sql, named) : Expr::parse(n.sql);
        REQUIRE(bits_to_ids(store.evaluate(e, kRows), kRows) == want);
    }
    REQUIRE(bound > 2000);    // the parameter paths really ran
}

TEST_CASE("Expr::parse caps the number of conditions", "[metadata][parse]") {
    REQUIRE(Expr::max_conditions() == Expr::kDefaultMaxConditions);
    const auto chain = [](size_t n) {
        std::string sql = "a = 0";
        for (size_t i = 1; i < n; ++i) sql += " OR a = " + std::to_string(i);
        return sql;
    };
    REQUIRE(Expr::parse(chain(1024)).children().size() == 1024);
    const std::string over = chain(1025);
    REQUIRE_THROWS_WITH(Expr::parse(over), "filter has more than 1024 conditions; combine values with IN (...) or "
                                           "raise the limit at column " + std::to_string(over.rfind("a = ") + 1));

    // one IN list, BETWEEN or negated form is one condition, however it expands
    std::string in = "a IN (0";
    for (int i = 1; i < 5000; ++i) in += ", " + std::to_string(i);
    REQUIRE(Expr::parse(in + ")").values().size() == 5000);
    std::string mixed;
    for (int i = 0; i < 1024; ++i) mixed += std::string(i ? " AND " : "") + (i % 2 ? "a NOT BETWEEN 1 AND 2" : "a NOT IN (1, 2)");
    REQUIRE_NOTHROW(Expr::parse(mixed));
    REQUIRE_THROWS_AS(Expr::parse(mixed + " AND a IS NULL"), std::invalid_argument);

    // filters built in code aren't limited
    std::vector<Expr> terms;
    for (int64_t i = 0; i < 5000; ++i) terms.push_back(Expr::compare("a", Op::Eq, i));
    REQUIRE(Expr::join(Op::Or, terms).children().size() == 5000);

    {
        const ConditionLimit two(2);
        REQUIRE_NOTHROW(Expr::parse("a = 1 AND b = 2"));
        REQUIRE_THROWS_WITH(Expr::parse("a = 1 AND b = 2 AND c = 3"),
                            "filter has more than 2 conditions; combine values with IN (...) or raise the limit at column 21");
    }
    REQUIRE(Expr::max_conditions() == 1024);
    REQUIRE_THROWS_AS(Expr::set_max_conditions(0), std::invalid_argument);
    REQUIRE(Expr::max_conditions() == 1024);
}

TEST_CASE("Expr: And / Or chains are flattened, and nesting is capped", "[metadata][parse]") {
    const auto a = [](int64_t v) { return f("a", Op::Eq, v); };

    SECTION("chains are one node, from either form") {
        const Expr ops = (a(1) | a(2)) | (a(3) | a(4));
        REQUIRE(ops.op() == Op::Or);
        REQUIRE(ops.children().size() == 4);
        REQUIRE(ops.to_string() == "((a == 1) | (a == 2) | (a == 3) | (a == 4))");
        REQUIRE(Expr::parse("a = 1 OR a = 2 OR a = 3 OR a = 4").to_string() == ops.to_string());
        // different ops still nest
        REQUIRE((a(1) & (a(2) | a(3))).children().size() == 2);
        // NOT (x AND y AND z) is one Or of three
        REQUIRE(Expr::parse("NOT (a = 1 AND a = 2 AND a = 3)").children().size() == 3);
    }

    SECTION("a long chain evaluates and prints without deep recursion") {
        MetadataStore store;
        store.append(0, 10, {{"a", std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}});
        const ConditionLimit unlimited(1000000);
        std::string sql = "a = 100";
        std::vector<Expr> terms{a(100)};
        for (int64_t i = 0; i < 200000; ++i) {
            sql += " OR a = " + std::to_string(i % 3 == 0 ? 100 + i : i % 10);
            terms.push_back(a(i % 3 == 0 ? 100 + i : i % 10));
        }
        const Expr parsed = Expr::parse(sql);
        const Expr joined = Expr::join(Op::Or, terms);
        REQUIRE(parsed.children().size() == 200001);
        REQUIRE(joined.children().size() == 200001);
        // each | copies the left chain, so keep the operator form short
        Expr built = terms[0];
        for (size_t i = 1; i < 1000; ++i) built = built | terms[i];
        REQUIRE(built.children().size() == 1000);
        const auto all = std::vector<int64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        REQUIRE(bits_to_ids(store.evaluate(parsed, 10), 10) == all);
        REQUIRE(bits_to_ids(store.evaluate(joined, 10), 10) == all);
        REQUIRE(bits_to_ids(store.evaluate(built, 10), 10) == all);
        REQUIRE(parsed.to_string().size() > 200000);
    }

    SECTION("the parser rejects deep nesting with a column") {
        const auto parens = [](size_t n) { return std::string(n, '(') + "a = 1" + std::string(n, ')'); };
        REQUIRE(Expr::parse(parens(Expr::kMaxDepth - 1)).to_string() == "(a == 1)");
        REQUIRE_THROWS_WITH(Expr::parse(parens(Expr::kMaxDepth)), "filter nested more than 256 levels deep at column 257");
        REQUIRE_THROWS_WITH(Expr::parse(parens(100000)), "filter nested more than 256 levels deep at column 257");
        std::string nots;
        for (int i = 0; i < 100000; ++i) nots += "NOT ";
        REQUIRE_THROWS_AS(Expr::parse(nots + "a = 1"), std::invalid_argument);
    }

    SECTION("building a filter deeper than the cap throws") {
        Expr e = a(1);
        for (size_t i = 1; i < Expr::kMaxDepth; ++i) e = ~e;   // depth kMaxDepth
        REQUIRE_THROWS_WITH(~e, "filter nested more than 256 levels deep");
        Expr alternating = a(0);
        REQUIRE_THROWS_AS([&] {
            for (int64_t i = 1; i < 1000; ++i) alternating = i % 2 ? (alternating & a(i)) : (alternating | a(i));
        }(), std::invalid_argument);
        REQUIRE_THROWS_AS(Expr::join(Op::Eq, {a(1), a(2)}), std::invalid_argument);
        REQUIRE_THROWS_AS(Expr::join(Op::And, {}), std::invalid_argument);
        REQUIRE(Expr::join(Op::And, {a(1)}).to_string() == "(a == 1)");
    }
}
