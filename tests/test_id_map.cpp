#include <catch2/catch_test_macros.hpp>

#include "vecengine/id_map.hpp"
#include "serialize.hpp"

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using vecengine::detail::IdMap;
using namespace vecengine::detail;

TEST_CASE("IdMap: default mode uses slots as ids", "[idmap]") {
    IdMap map;
    map.add(5, nullptr);
    REQUIRE(map.slots() == 5);
    REQUIRE(map.live() == 5);
    REQUIRE(map.label(3) == 3);
    REQUIRE_FALSE(map.is_deleted(3));

    std::vector<uint32_t> removed;
    const std::vector<int64_t> ids{3, 3, 0, -1, 99};
    REQUIRE(map.remove(ids, &removed) == 2);
    REQUIRE(removed == std::vector<uint32_t>{3, 0});
    REQUIRE(map.is_deleted(3));
    REQUIRE(map.is_deleted(0));
    REQUIRE(map.live() == 3);
    REQUIRE(map.slots() == 5);

    // Deleting again is a no-op; new slots continue after the old ones.
    REQUIRE(map.remove(ids) == 0);
    map.add(2, nullptr);
    REQUIRE(map.label(6) == 6);
    REQUIRE(map.live() == 5);
}

TEST_CASE("IdMap: custom ids map slots to labels", "[idmap]") {
    IdMap map(true);
    const std::vector<int64_t> a{100, 7, 42};
    map.add(a.size(), a.data());
    REQUIRE(map.label(0) == 100);
    REQUIRE(map.label(1) == 7);
    REQUIRE(map.label(2) == 42);

    std::vector<uint32_t> removed;
    const std::vector<int64_t> del{7, 5, 42};
    REQUIRE(map.remove(del, &removed) == 2);
    REQUIRE(removed == std::vector<uint32_t>{1, 2});
    REQUIRE(map.live() == 1);

    // A deleted id can be added again; it gets a new slot.
    const std::vector<int64_t> again{42};
    map.add(1, again.data());
    REQUIRE(map.label(3) == 42);
    REQUIRE_FALSE(map.is_deleted(3));
    REQUIRE(map.is_deleted(2));

    // Removing it now deletes the new slot, not the old one.
    removed.clear();
    REQUIRE(map.remove(again, &removed) == 1);
    REQUIRE(removed == std::vector<uint32_t>{3});
}

TEST_CASE("IdMap: add rejects bad input without changing anything", "[idmap]") {
    IdMap custom(true);
    const std::vector<int64_t> first{1, 2, 3};
    custom.add(first.size(), first.data());

    const std::vector<int64_t> dup_in_batch{10, 11, 10};
    REQUIRE_THROWS_AS(custom.add(dup_in_batch.size(), dup_in_batch.data()), std::invalid_argument);
    const std::vector<int64_t> dup_existing{20, 2};
    REQUIRE_THROWS_AS(custom.add(dup_existing.size(), dup_existing.data()), std::invalid_argument);
    REQUIRE_THROWS_AS(custom.add(1, nullptr), std::invalid_argument);
    const std::vector<int64_t> negative{30, -1};
    REQUIRE_THROWS_AS(custom.add(negative.size(), negative.data()), std::invalid_argument);

    REQUIRE(custom.slots() == 3);
    REQUIRE(custom.live() == 3);
    // Ids from the rejected batches were never registered.
    const std::vector<int64_t> check{10, 20, 30};
    REQUIRE(custom.remove(check) == 0);

    IdMap plain;
    REQUIRE_THROWS_AS(plain.add(1, first.data()), std::invalid_argument);
    REQUIRE(plain.slots() == 0);
}

TEST_CASE("IdMap: save/load round-trip", "[idmap][serialize]") {
    for (bool custom : {false, true}) {
        IdMap map(custom);
        const std::vector<int64_t> ids{5, 6, 7, 8};
        map.add(ids.size(), custom ? ids.data() : nullptr);
        const std::vector<int64_t> del{custom ? int64_t{6} : int64_t{1}};
        map.remove(del);

        std::stringstream s;
        map.save(s);
        IdMap loaded = IdMap::load(s, 4);

        REQUIRE(loaded.custom_ids() == custom);
        REQUIRE(loaded.slots() == 4);
        REQUIRE(loaded.live() == 3);
        for (uint32_t slot = 0; slot < 4; ++slot) {
            REQUIRE(loaded.label(slot) == map.label(slot));
            REQUIRE(loaded.is_deleted(slot) == map.is_deleted(slot));
        }
        // The lookup table was rebuilt: live ids can be removed, deleted ones can't.
        REQUIRE(loaded.remove(del) == 0);
        const std::vector<int64_t> live{custom ? int64_t{7} : int64_t{2}};
        REQUIRE(loaded.remove(live) == 1);
    }
}

TEST_CASE("IdMap: load rejects corrupt data", "[idmap][serialize]") {
    auto body = [](uint8_t custom, uint64_t slots, std::vector<int64_t> labels, std::vector<uint8_t> deleted) {
        std::stringstream t;
        write_pod(t, custom);
        write_pod(t, slots);
        write_vec(t, labels);
        write_vec(t, deleted);
        return t;
    };

    SECTION("valid hand-written body loads") {
        auto t = body(1, 2, {5, 6}, {0, 0});
        REQUIRE(IdMap::load(t, 2).live() == 2);
    }
    SECTION("slot count differs from the index") {
        auto t = body(0, 3, {}, {0, 0, 0});
        REQUIRE_THROWS_AS(IdMap::load(t, 2), std::runtime_error);
    }
    SECTION("bad custom flag") {
        auto t = body(2, 1, {}, {0});
        REQUIRE_THROWS_AS(IdMap::load(t, 1), std::runtime_error);
    }
    SECTION("labels without custom ids") {
        auto t = body(0, 2, {5, 6}, {0, 0});
        REQUIRE_THROWS_AS(IdMap::load(t, 2), std::runtime_error);
    }
    SECTION("too few labels") {
        auto t = body(1, 2, {5}, {0, 0});
        REQUIRE_THROWS_AS(IdMap::load(t, 2), std::runtime_error);
    }
    SECTION("deleted flags wrong length or value") {
        auto short_flags = body(0, 2, {}, {0});
        REQUIRE_THROWS_AS(IdMap::load(short_flags, 2), std::runtime_error);
        auto bad_flag = body(0, 2, {}, {0, 2});
        REQUIRE_THROWS_AS(IdMap::load(bad_flag, 2), std::runtime_error);
    }
    SECTION("duplicate live ids") {
        auto t = body(1, 2, {5, 5}, {0, 0});
        REQUIRE_THROWS_AS(IdMap::load(t, 2), std::runtime_error);
    }
    SECTION("a deleted slot may share an id with a live one") {
        auto t = body(1, 2, {5, 5}, {1, 0});
        REQUIRE(IdMap::load(t, 2).live() == 1);
    }
    SECTION("truncated") {
        auto t = body(1, 2, {5, 6}, {0, 0});
        const std::string bytes = t.str();
        std::stringstream cut(bytes.substr(0, bytes.size() - 1));
        REQUIRE_THROWS_AS(IdMap::load(cut, 2), std::runtime_error);
    }
}

// ---------------------------------------------------------------------------
// FlatIndex with ids and deletion
// ---------------------------------------------------------------------------

#include "vecengine/flat_index.hpp"

#include <random>
#include <span>

using vecengine::FlatIndex;

static std::vector<float> random_matrix(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(n * dim);
    for (auto& x : out) x = dist(rng);
    return out;
}

static std::span<const float> row(const std::vector<float>& m, size_t i, size_t dim) {
    return {m.data() + i * dim, dim};
}

TEST_CASE("FlatIndex: custom ids are reported in results", "[flat][ids]") {
    const size_t n = 200, dim = 8;
    auto data = random_matrix(n, dim, 1);
    std::vector<int64_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = 1000 + 7 * static_cast<int64_t>(i);

    FlatIndex index(dim, 16, true);
    index.add_batch(data, n, ids.data());
    REQUIRE(index.size() == n);

    for (size_t i = 0; i < n; i += 37) {
        auto res = index.query(row(data, i, dim), 3);
        REQUIRE(res[0].index == static_cast<size_t>(ids[i]));
    }

    // Single-vector add and adds without ids are rejected and store nothing.
    REQUIRE_THROWS_AS(index.add(row(data, 0, dim)), std::invalid_argument);
    REQUIRE_THROWS_AS(index.add_batch(data, n), std::invalid_argument);
    const std::vector<int64_t> dup(n, 5);
    REQUIRE_THROWS_AS(index.add_batch(data, n, dup.data()), std::invalid_argument);
    REQUIRE(index.size() == n);
    REQUIRE(index.query(row(data, 3, dim), 1)[0].index == static_cast<size_t>(ids[3]));
}

TEST_CASE("FlatIndex: default index rejects ids", "[flat][ids]") {
    FlatIndex index(4);
    auto data = random_matrix(3, 4, 2);
    const std::vector<int64_t> ids{1, 2, 3};
    REQUIRE_THROWS_AS(index.add_batch(data, 3, ids.data()), std::invalid_argument);
    REQUIRE(index.size() == 0);
}

TEST_CASE("FlatIndex: deleted vectors are never returned", "[flat][ids]") {
    const size_t n = 300, dim = 8, k = 10;
    auto data = random_matrix(n, dim, 3);

    for (bool custom : {false, true}) {
        std::vector<int64_t> ids(n);
        for (size_t i = 0; i < n; ++i) ids[i] = custom ? 5000 + static_cast<int64_t>(i) : static_cast<int64_t>(i);

        FlatIndex index(dim, 16, custom);
        index.add_batch(data, n, custom ? ids.data() : nullptr);

        // Delete every third vector, then query with each deleted vector itself:
        // its own exact match must not come back, and k live results still do.
        std::vector<int64_t> del;
        for (size_t i = 0; i < n; i += 3) del.push_back(ids[i]);
        REQUIRE(index.remove(del) == del.size());
        REQUIRE(index.remove(del) == 0);
        REQUIRE(index.size() == n - del.size());

        for (size_t i = 0; i < n; i += 3) {
            auto res = index.query(row(data, i, dim), k);
            REQUIRE(res.size() == k);
            for (const auto& nb : res) REQUIRE((nb.index - static_cast<size_t>(ids[0])) % 3 != 0);
        }
    }
}

TEST_CASE("FlatIndex: deleted id can be re-added", "[flat][ids]") {
    const size_t dim = 4;
    auto data = random_matrix(3, dim, 4);
    FlatIndex index(dim, 16, true);
    const std::vector<int64_t> ids{10, 20, 30};
    index.add_batch(data, 3, ids.data());

    const std::vector<int64_t> del{20};
    REQUIRE(index.remove(del) == 1);
    REQUIRE(index.size() == 2);

    // Re-add id 20 with a different vector: queries find the new one.
    auto replacement = random_matrix(1, dim, 5);
    index.add_batch(replacement, 1, del.data());
    REQUIRE(index.size() == 3);
    auto res = index.query(row(replacement, 0, dim), 1);
    REQUIRE(res[0].index == 20);
    REQUIRE(res[0].distance == 0.0f);
    // The old vector for id 20 is gone.
    auto old = index.query(row(data, 1, dim), 3);
    for (const auto& nb : old) REQUIRE_FALSE((nb.index == 20 && nb.distance == 0.0f));
}

TEST_CASE("FlatIndex: query with fewer live vectors than k", "[flat][ids]") {
    const size_t dim = 4;
    auto data = random_matrix(5, dim, 6);
    FlatIndex index(dim);
    index.add_batch(data, 5);
    const std::vector<int64_t> del{0, 1, 2};
    index.remove(del);
    auto res = index.query(row(data, 0, dim), 10);
    REQUIRE(res.size() == 2);
    for (const auto& nb : res) REQUIRE(nb.index >= 3);
}

// ---------------------------------------------------------------------------
// IVFIndex with ids and deletion
// ---------------------------------------------------------------------------

#include "vecengine/ivf_index.hpp"

#include <memory>

using vecengine::IVFIndex;
using vecengine::PQDistance;

enum class IvfMode { Plain, ADC, SDC };

static std::unique_ptr<IVFIndex> make_ivf(IvfMode mode, size_t dim, const std::vector<float>& train, size_t n, bool custom) {
    // nprobe = nlist scans every list, so results don't depend on clustering.
    auto index = std::make_unique<IVFIndex>(dim, 100, 100, custom);
    if (mode != IvfMode::Plain) index->enable_pq(dim / 2);
    if (mode == IvfMode::SDC) index->set_pq_distance(PQDistance::SDC);
    index->train(train, n, 5, 1);
    return index;
}

TEST_CASE("IVFIndex: custom ids and deletion", "[ivf][ids]") {
    const size_t n = 1000, dim = 8, k = 10;
    auto data = random_matrix(n, dim, 7);

    for (IvfMode mode : {IvfMode::Plain, IvfMode::ADC, IvfMode::SDC}) {
        for (bool custom : {false, true}) {
            CAPTURE(static_cast<int>(mode), custom);
            std::vector<int64_t> ids(n);
            for (size_t i = 0; i < n; ++i) ids[i] = custom ? 9000 + static_cast<int64_t>(i) : static_cast<int64_t>(i);
            const size_t base = static_cast<size_t>(ids[0]);

            auto ptr = make_ivf(mode, dim, data, n, custom);
            IVFIndex& index = *ptr;
            index.add_batch(data, n, custom ? ids.data() : nullptr);
            REQUIRE(index.size() == n);

            // Results are reported as the user's ids.
            for (size_t i = 0; i < n; i += 97) {
                auto res = index.query(row(data, i, dim), k);
                REQUIRE(res.size() == k);
                for (const auto& nb : res) REQUIRE(nb.index - base < n);
            }

            // Delete every third vector: none of them come back, even when the
            // query is the deleted vector itself, and k live results still do.
            std::vector<int64_t> del;
            for (size_t i = 0; i < n; i += 3) del.push_back(ids[i]);
            REQUIRE(index.remove(del) == del.size());
            REQUIRE(index.remove(del) == 0);
            REQUIRE(index.size() == n - del.size());

            for (size_t i = 0; i < n; i += 3) {
                auto res = index.query(row(data, i, dim), k);
                REQUIRE(res.size() == k);
                for (const auto& nb : res) REQUIRE((nb.index - base) % 3 != 0);
            }
            auto batch = index.query_batch(std::span<const float>(data.data(), 30 * dim), 30, k);
            for (const auto& res : batch)
                for (const auto& nb : res) REQUIRE((nb.index - base) % 3 != 0);
        }
    }
}

TEST_CASE("IVFIndex: constructor checks nprobe", "[ivf]") {
    REQUIRE_THROWS_AS(IVFIndex(8, 100, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(IVFIndex(8, 100, 101), std::invalid_argument);
    REQUIRE(IVFIndex(8, 100, 100).nprobe() == 100);
}

TEST_CASE("IVFIndex: rejected adds store nothing; deleted ids can be re-added", "[ivf][ids]") {
    const size_t n = 500, dim = 8;
    auto data = random_matrix(n, dim, 8);
    std::vector<int64_t> ids(n);
    for (size_t i = 0; i < n; ++i) ids[i] = 100 + static_cast<int64_t>(i);

    auto ptr = make_ivf(IvfMode::Plain, dim, data, n, true);
    IVFIndex& index = *ptr;
    index.add_batch(data, n, ids.data());
    auto before = index.query(row(data, 5, dim), 5);

    std::vector<int64_t> dup(ids.begin(), ids.begin() + 10);
    REQUIRE_THROWS_AS(index.add_batch(std::span<const float>(data.data(), 10 * dim), 10, dup.data()), std::invalid_argument);
    REQUIRE_THROWS_AS(index.add_batch(std::span<const float>(data.data(), 10 * dim), 10), std::invalid_argument);
    REQUIRE_THROWS_AS(index.add(row(data, 0, dim)), std::invalid_argument);
    REQUIRE(index.size() == n);
    auto after = index.query(row(data, 5, dim), 5);
    REQUIRE(after.size() == before.size());
    for (size_t j = 0; j < after.size(); ++j) REQUIRE(after[j].index == before[j].index);

    // Re-add id 105 with a new vector.
    const std::vector<int64_t> del{105};
    REQUIRE(index.remove(del) == 1);
    auto replacement = random_matrix(1, dim, 9);
    index.add_batch(replacement, 1, del.data());
    REQUIRE(index.size() == n);
    auto res = index.query(row(replacement, 0, dim), 1);
    REQUIRE(res[0].index == 105);
    REQUIRE(res[0].distance == 0.0f);

    // enable_pq after an add is still refused.
    REQUIRE_THROWS_AS(index.enable_pq(4), std::logic_error);
}

// ---------------------------------------------------------------------------
// RefineIndex with ids and deletion
// ---------------------------------------------------------------------------

#include "vecengine/refine_index.hpp"

using vecengine::Index;
using vecengine::RefineIndex;

TEST_CASE("RefineIndex: custom ids and deletion", "[refine][ids]") {
    const size_t n = 800, dim = 8, k = 10;
    auto data = random_matrix(n, dim, 10);

    for (int base_kind = 0; base_kind < 3; ++base_kind) {
        for (bool custom : {false, true}) {
            CAPTURE(base_kind, custom);
            std::unique_ptr<Index> base;
            if (base_kind == 0) base = std::make_unique<FlatIndex>(dim);
            else base = make_ivf(base_kind == 1 ? IvfMode::ADC : IvfMode::SDC, dim, data, n, false);

            std::vector<int64_t> ids(n);
            for (size_t i = 0; i < n; ++i) ids[i] = custom ? 70000 + static_cast<int64_t>(i) : static_cast<int64_t>(i);
            const size_t first = static_cast<size_t>(ids[0]);

            RefineIndex refine(*base, 5, custom);
            refine.add_batch(data, n, custom ? ids.data() : nullptr);
            REQUIRE(refine.size() == n);
            REQUIRE(base->slots() == n);

            // Exact re-ranking finds each vector itself, reported by user id.
            for (size_t i = 0; i < n; i += 61) REQUIRE(refine.query(row(data, i, dim), 1)[0].index == static_cast<size_t>(ids[i]));

            std::vector<int64_t> del;
            for (size_t i = 0; i < n; i += 3) del.push_back(ids[i]);
            REQUIRE(refine.remove(del) == del.size());
            REQUIRE(refine.size() == n - del.size());
            // The deletion reached the base too (it counts slots, not user ids).
            REQUIRE(base->size() == n - del.size());

            for (size_t i = 0; i < n; i += 3) {
                auto res = refine.query(row(data, i, dim), k);
                REQUIRE(res.size() == k);
                for (const auto& nb : res) REQUIRE((nb.index - first) % 3 != 0);
                for (const auto& nb : base->query(row(data, i, dim), k)) REQUIRE(nb.index % 3 != 0);
            }
        }
    }
}

TEST_CASE("RefineIndex: failed adds keep refine and base in step", "[refine][ids]") {
    const size_t n = 300, dim = 8;
    auto data = random_matrix(n, dim, 11);

    SECTION("bad ids reject the batch before the base is touched") {
        FlatIndex base(dim);
        RefineIndex refine(base, 4, true);
        std::vector<int64_t> ids(n);
        for (size_t i = 0; i < n; ++i) ids[i] = 10 + static_cast<int64_t>(i);
        refine.add_batch(data, n, ids.data());

        std::vector<int64_t> dup{5000, 10};
        REQUIRE_THROWS_AS(refine.add_batch(std::span<const float>(data.data(), 2 * dim), 2, dup.data()), std::invalid_argument);
        REQUIRE_THROWS_AS(refine.add(row(data, 0, dim)), std::invalid_argument);
        REQUIRE(refine.slots() == n);
        REQUIRE(base.slots() == n);

        // Still aligned: a later valid add is found under its id.
        auto extra = random_matrix(1, dim, 12);
        const std::vector<int64_t> extra_id{99999};
        refine.add_batch(extra, 1, extra_id.data());
        REQUIRE(refine.query(row(extra, 0, dim), 1)[0].index == 99999);
    }

    SECTION("an untrained IVF base rejects the batch before ids are registered") {
        IVFIndex base(dim, 100);
        RefineIndex refine(base, 4, true);
        std::vector<int64_t> ids{1, 2};
        REQUIRE_THROWS_AS(refine.add_batch(std::span<const float>(data.data(), 2 * dim), 2, ids.data()), std::logic_error);
        REQUIRE(refine.slots() == 0);
        REQUIRE(base.slots() == 0);

        base.train(data, n, 5, 1);
        refine.add_batch(std::span<const float>(data.data(), 2 * dim), 2, ids.data());  // the same ids are still free
        REQUIRE(refine.size() == 2);
    }
}

TEST_CASE("RefineIndex: constructor rejects misaligned bases", "[refine][ids]") {
    const size_t dim = 4;
    auto data = random_matrix(3, dim, 13);

    FlatIndex custom_base(dim, 16, true);
    REQUIRE_THROWS_AS(RefineIndex(custom_base), std::invalid_argument);

    // Every vector deleted: size() is 0 but the slots are still taken.
    FlatIndex used_base(dim);
    used_base.add_batch(data, 3);
    const std::vector<int64_t> all{0, 1, 2};
    used_base.remove(all);
    REQUIRE(used_base.size() == 0);
    REQUIRE_THROWS_AS(RefineIndex(used_base), std::invalid_argument);
}

TEST_CASE("RefineIndex: query after deleting everything returns nothing", "[refine][ids]") {
    const size_t n = 300, dim = 8;
    auto data = random_matrix(n, dim, 14);
    auto base = make_ivf(IvfMode::ADC, dim, data, n, false);
    RefineIndex refine(*base, 4);
    refine.add_batch(data, n);

    std::vector<int64_t> all(n);
    for (size_t i = 0; i < n; ++i) all[i] = static_cast<int64_t>(i);
    REQUIRE(refine.remove(all) == n);
    REQUIRE(refine.size() == 0);
    REQUIRE(refine.query(row(data, 0, dim), 5).empty());
    REQUIRE(refine.query_batch(std::span<const float>(data.data(), 3 * dim), 3, 5)[2].empty());
}

// ---------------------------------------------------------------------------
// Saving and loading ids and deletions (format version 2)
// ---------------------------------------------------------------------------

#include "vecengine/index_factory.hpp"

#include <filesystem>
#include <sstream>

using vecengine::load_index;
using vecengine::Neighbor;

static std::unique_ptr<Index> roundtrip(const Index& index) {
    std::stringstream s;
    index.save(s);
    return load_index(s);
}

static void require_same(const Index& a, const Index& b, const std::vector<float>& queries, size_t nq, size_t k) {
    REQUIRE(a.size() == b.size());
    REQUIRE(a.slots() == b.slots());
    REQUIRE(a.custom_ids() == b.custom_ids());
    auto ra = a.query_batch(std::span<const float>(queries.data(), nq * a.dim()), nq, k);
    auto rb = b.query_batch(std::span<const float>(queries.data(), nq * b.dim()), nq, k);
    for (size_t q = 0; q < nq; ++q) {
        REQUIRE(ra[q].size() == rb[q].size());
        for (size_t j = 0; j < ra[q].size(); ++j) {
            REQUIRE(ra[q][j].index == rb[q][j].index);
            REQUIRE(ra[q][j].distance == rb[q][j].distance);
        }
    }
}

// Applies the same adds and deletes to two indexes, which must stay identical.
static void mutate(Index& index, size_t dim, int64_t next_id) {
    auto extra = random_matrix(5, dim, static_cast<unsigned>(next_id));
    std::vector<int64_t> ids(5);
    for (size_t i = 0; i < 5; ++i) ids[i] = next_id + static_cast<int64_t>(i);
    index.add_batch(extra, 5, index.custom_ids() ? ids.data() : nullptr);
    const std::vector<int64_t> del{index.custom_ids() ? next_id : static_cast<int64_t>(index.slots() - 1)};
    index.remove(del);
}

TEST_CASE("save/load keeps custom ids and deletions", "[serialize][ids]") {
    const size_t n = 600, dim = 8, nq = 40, k = 10;
    auto data = random_matrix(n, dim, 20);
    auto queries = random_matrix(nq, dim, 21);

    for (int kind = 0; kind < 5; ++kind) {
        for (bool custom : {false, true}) {
            CAPTURE(kind, custom);
            std::unique_ptr<Index> base;   // RefineIndex's base, kept alive alongside it
            std::unique_ptr<Index> index;
            if (kind == 0) index = std::make_unique<FlatIndex>(dim, 16, custom);
            else if (kind <= 3) {
                auto ivf = make_ivf(kind == 1 ? IvfMode::Plain : kind == 2 ? IvfMode::ADC : IvfMode::SDC, dim, data, n, false);
                if (custom) {
                    ivf = std::make_unique<IVFIndex>(dim, 100, 100, true);
                    if (kind != 1) ivf->enable_pq(dim / 2);
                    if (kind == 3) ivf->set_pq_distance(PQDistance::SDC);
                    ivf->train(data, n, 5, 1);
                }
                index = std::move(ivf);
            } else {
                base = make_ivf(IvfMode::ADC, dim, data, n, false);
                index = std::make_unique<RefineIndex>(*base, 4, custom);
            }

            std::vector<int64_t> ids(n);
            for (size_t i = 0; i < n; ++i) ids[i] = custom ? 300000 + 3 * static_cast<int64_t>(i) : static_cast<int64_t>(i);
            index->add_batch(data, n, custom ? ids.data() : nullptr);

            std::vector<int64_t> del;
            for (size_t i = 0; i < n; i += 4) del.push_back(ids[i]);
            index->remove(del);
            if (custom) {  // re-add a deleted id with a new vector
                auto replacement = random_matrix(1, dim, 22);
                index->add_batch(replacement, 1, &ids[0]);
            }

            auto loaded = roundtrip(*index);
            require_same(*index, *loaded, queries, nq, k);
            REQUIRE(loaded->remove(del) == (custom ? 1u : 0u));  // only the re-added id is live

            // Both keep behaving the same after further adds and deletes.
            std::unique_ptr<Index> loaded_again = roundtrip(*index);
            mutate(*index, dim, 900000);
            mutate(*loaded_again, dim, 900000);
            require_same(*index, *loaded_again, queries, nq, k);
        }
    }
}

TEST_CASE("files from format version 1 (0.1.0a4) still load", "[serialize][compat]") {
    const std::filesystem::path dir = QANN_TEST_DATA_DIR;
    for (const char* name : {"v1_flat.qann", "v1_ivf_pq.qann", "v1_refine.qann"}) {
        CAPTURE(name);
        auto index = load_index(dir / name);
        REQUIRE(index->size() == 2000);
        REQUIRE(index->slots() == 2000);
        REQUIRE_FALSE(index->custom_ids());
        REQUIRE(index->dim() == 16);

        auto query = random_matrix(1, 16, 23);
        auto res = index->query(query, 10);
        REQUIRE(res.size() == 10);
        for (const auto& nb : res) REQUIRE(nb.index < 2000);

        // Deleting works, and re-saving writes the current version.
        const std::vector<int64_t> del{static_cast<int64_t>(res[0].index)};
        REQUIRE(index->remove(del) == 1);
        std::stringstream s;
        index->save(s);
        REQUIRE(read_header(s).version == kFormatVersion);
        s.seekg(0);
        auto reloaded = load_index(s);
        REQUIRE(reloaded->size() == 1999);
    }
}

TEST_CASE("load rejects bad versions and misaligned id maps", "[serialize][ids]") {
    const size_t dim = 4;
    auto data = random_matrix(10, dim, 24);

    SECTION("newer format version") {
        FlatIndex flat(dim);
        std::stringstream s;
        flat.save(s);
        std::string bytes = s.str();
        bytes[4] = static_cast<char>(kFormatVersion + 1);
        std::stringstream t(bytes);
        REQUIRE_THROWS_AS(load_index(t), std::runtime_error);
    }

    SECTION("RefineIndex base written with a different version") {
        FlatIndex base(dim);
        base.add_batch(data, 10);
        std::stringstream t;
        write_header(t, IndexKind::Refine);
        write_pod<uint64_t>(t, 3);
        write_pod<uint64_t>(t, 10);
        std::stringstream inner;
        base.save(inner);
        std::string inner_bytes = inner.str();
        inner_bytes[4] = 1;   // nested header claims version 1 inside a version 2 file
        t << inner_bytes;
        REQUIRE_THROWS_AS(load_index(t), std::runtime_error);
    }

    SECTION("RefineIndex base with custom ids") {
        FlatIndex base(dim, 16, true);
        const std::vector<int64_t> ids{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        base.add_batch(data, 10, ids.data());
        std::stringstream t;
        write_header(t, IndexKind::Refine);
        write_pod<uint64_t>(t, 3);
        write_pod<uint64_t>(t, 10);
        base.save(t);
        write_vec(t, data);
        IdMap map;
        map.add(10, nullptr);
        map.save(t);
        REQUIRE_THROWS_AS(load_index(t), std::runtime_error);
    }

    SECTION("id map with a different slot count") {
        std::stringstream t;
        write_header(t, IndexKind::Flat);
        write_pod<uint64_t>(t, dim);
        write_pod<uint64_t>(t, 10);
        write_vec(t, data);
        IdMap map;
        map.add(9, nullptr);
        map.save(t);
        REQUIRE_THROWS_AS(load_index(t), std::runtime_error);
    }
}

TEST_CASE("RefineIndex marks its base as wrapped", "[refine][ids]") {
    const size_t dim = 4;
    auto data = random_matrix(10, dim, 30);
    FlatIndex base(dim);
    REQUIRE_FALSE(base.wrapped());
    {
        RefineIndex refine(base, 2);
        REQUIRE(base.wrapped());
        REQUIRE_FALSE(refine.wrapped());
        // A second RefineIndex over the same base would misalign both.
        REQUIRE_THROWS_AS(RefineIndex(base, 2), std::invalid_argument);
        refine.add_batch(data, 10);

        std::stringstream s;
        refine.save(s);
        auto loaded = load_index(s);
        REQUIRE(dynamic_cast<RefineIndex&>(*loaded).base().wrapped());
    }
    REQUIRE_FALSE(base.wrapped());  // cleared when the RefineIndex is destroyed
}
