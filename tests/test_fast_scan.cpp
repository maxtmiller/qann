#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "load_dense.hpp"

#include "fast_scan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

using namespace vecengine::detail;

TEST_CASE("fs_set/fs_get: every position and subspace round-trips", "[fastscan]") {
    const size_t m = 8;
    std::mt19937 rng(1);
    std::vector<uint8_t> block(fs_block_bytes(m), 0);
    std::vector<uint8_t> codes(kFsBlock * m);
    for (auto& c : codes) c = rng() & 0x0F;

    for (size_t v = 0; v < kFsBlock; ++v)
        for (size_t s = 0; s < m; ++s) fs_set(block.data(), v, s, codes[v * m + s]);
    for (size_t v = 0; v < kFsBlock; ++v)
        for (size_t s = 0; s < m; ++s) REQUIRE(fs_get(block.data(), v, s) == codes[v * m + s]);

    // Overwriting one code leaves its neighbor in the same byte alone.
    fs_set(block.data(), 3, 2, 0x0F);
    REQUIRE(fs_get(block.data(), 3, 2) == 0x0F);
    REQUIRE(fs_get(block.data(), 19, 2) == codes[19 * m + 2]);
}

TEST_CASE("fs_set: layout is low nibble for 0-15, high nibble for 16-31", "[fastscan]") {
    std::vector<uint8_t> block(fs_block_bytes(2), 0);
    fs_set(block.data(), 0, 0, 0x3);
    fs_set(block.data(), 16, 0, 0xA);
    fs_set(block.data(), 5, 1, 0x7);
    fs_set(block.data(), 31, 1, 0x1);
    REQUIRE(block[0] == 0xA3);           // subspace 0, byte 0: vectors 0 and 16
    REQUIRE(block[16 + 5] == 0x07);      // subspace 1 starts at byte 16
    REQUIRE(block[16 + 15] == 0x10);     // vector 31 shares byte 15 with vector 15
}

TEST_CASE("fast_scan_block_scalar sums the picked lut entries", "[fastscan]") {
    const size_t m = 4;
    std::vector<uint8_t> lut(m * 16);
    for (size_t i = 0; i < lut.size(); ++i) lut[i] = static_cast<uint8_t>(i);   // entry = s * 16 + j
    std::vector<uint8_t> block(fs_block_bytes(m), 0);
    for (size_t v = 0; v < kFsBlock; ++v)
        for (size_t s = 0; s < m; ++s) fs_set(block.data(), v, s, static_cast<uint8_t>((v + s) % 16));

    std::vector<uint16_t> out(kFsBlock);
    fast_scan_block_scalar(lut.data(), block.data(), m, out.data());
    for (size_t v = 0; v < kFsBlock; ++v) {
        unsigned expected = 0;
        for (size_t s = 0; s < m; ++s) expected += s * 16 + (v + s) % 16;
        REQUIRE(out[v] == expected);
    }
}

TEST_CASE("fast_scan_block matches the scalar version", "[fastscan]") {
    const size_t m = GENERATE(2, 16, 64, 256, 480, 1024);
    CAPTURE(m);
    std::mt19937 rng(static_cast<unsigned>(m));
    std::vector<uint8_t> lut(m * 16), block(fs_block_bytes(m));
    for (auto& x : block) x = rng() & 0xFF;

    SECTION("random tables") { for (auto& x : lut) x = rng() % (kLutMax + 1); }
    SECTION("all kLutMax, the largest sums") { std::fill(lut.begin(), lut.end(), kLutMax); }

    std::vector<uint16_t> simd(kFsBlock), scalar(kFsBlock);
    fast_scan_block(lut.data(), block.data(), m, simd.data());
    fast_scan_block_scalar(lut.data(), block.data(), m, scalar.data());
    REQUIRE(simd == scalar);
    if (lut[0] == kLutMax && lut[1] == kLutMax)
        for (uint16_t x : simd) REQUIRE(x == m * kLutMax);   // no uint16_t wraparound
}

TEST_CASE("quantize_lut keeps distances within its error bound", "[fastscan]") {
    const size_t m = 32;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(0.0f, 5000.0f);
    std::vector<float> table(m * 16);
    for (auto& x : table) x = dist(rng);
    // Rows with different minimums and ranges, like real residual tables.
    for (size_t s = 0; s < m; ++s)
        for (size_t j = 0; j < 16; ++j) table[s * 16 + j] = table[s * 16 + j] * (1 + s % 3) + 100.0f * s;

    std::vector<uint8_t> lut(m * 16);
    float scale = 0, offset = 0;
    quantize_lut(table.data(), m, lut.data(), scale, offset);
    REQUIRE(scale > 0);

    for (size_t s = 0; s < m; ++s) {
        const auto row = lut.begin() + s * 16;
        REQUIRE(*std::min_element(row, row + 16) == 0);   // each row's minimum maps to 0
    }
    REQUIRE(*std::max_element(lut.begin(), lut.end()) == kLutMax);  // the widest row spans 0..kLutMax

    const float bound = m / (2 * scale) + 1e-2f;
    for (int trial = 0; trial < 200; ++trial) {
        float exact = 0;
        unsigned sum = 0;
        for (size_t s = 0; s < m; ++s) {
            const size_t j = rng() % 16;
            exact += table[s * 16 + j];
            sum += lut[s * 16 + j];
        }
        REQUIRE(std::fabs(sum / scale + offset - exact) <= bound);
    }
}

TEST_CASE("quantize_lut handles constant rows", "[fastscan]") {
    const size_t m = 4;
    std::vector<float> table(m * 16, 3.0f);
    std::vector<uint8_t> lut(m * 16, 9);
    float scale = 0, offset = 0;
    quantize_lut(table.data(), m, lut.data(), scale, offset);
    REQUIRE(scale == 1.0f);
    REQUIRE(offset == 12.0f);
    for (uint8_t x : lut) REQUIRE(x == 0);
}

TEST_CASE("quantize_lut handles rows of negative values", "[fastscan]") {
    // With precomputed tables a row is B + C, and C = -2<q, y> is usually
    // negative, so whole rows can sit below zero.
    const size_t m = 2;
    std::vector<float> table(m * 16);
    for (size_t j = 0; j < 16; ++j) {
        table[j] = -500.0f + 10.0f * j;        // row 0: -500 .. -350, range 150
        table[16 + j] = -40.0f - 2.0f * j;     // row 1: -70 .. -40, range 30
    }
    std::vector<uint8_t> lut(m * 16);
    float scale = 0, offset = 0;
    quantize_lut(table.data(), m, lut.data(), scale, offset);
    REQUIRE(scale == kLutMax / 150.0f);
    REQUIRE(offset == -570.0f);
    REQUIRE(lut[0] == 0);
    REQUIRE(lut[15] == kLutMax);
    REQUIRE(lut[16 + 15] == 0);                // row 1's minimum is its last entry
    REQUIRE(lut[16] == std::lround(30.0f * scale));
}

// ---------------------------------------------------------------------------
// IVFIndex with fast-scan
// ---------------------------------------------------------------------------

#include "vecengine/flat_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"

#include <memory>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>

using vecengine::FlatIndex;
using vecengine::IVFIndex;
using vecengine::Neighbor;
using vecengine::PQDistance;
using vecengine::RefineIndex;

static std::vector<float> fs_random(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> out(n * dim);
    for (auto& x : out) x = d(rng);
    return out;
}

// An IVF index with 4-bit PQ; with the same seed, fast-scan on and off get
// identical centroids and codes, so only the scoring differs.
static std::unique_ptr<IVFIndex> fs_ivf(size_t dim, size_t m, bool fast, const std::vector<float>& data, size_t n,
                                        size_t nprobe = 10) {
    auto index = std::make_unique<IVFIndex>(dim, 100, nprobe);
    index->enable_pq(m, 16, fast);
    index->train(data, n, 10, 1);
    return index;
}

using Results = std::vector<std::vector<Neighbor>>;

// Fraction of result ids that appear in both result sets, query by query.
static double overlap(const Results& a, const Results& b) {
    size_t same = 0, total = 0;
    for (size_t q = 0; q < a.size(); ++q) {
        std::set<size_t> ids;
        for (const auto& nb : b[q]) ids.insert(nb.index);
        for (const auto& nb : a[q]) same += ids.count(nb.index);
        total += a[q].size();
    }
    return total ? static_cast<double>(same) / total : 1.0;
}

static void require_identical(const Results& a, const Results& b) {
    REQUIRE(a.size() == b.size());
    for (size_t q = 0; q < a.size(); ++q) {
        REQUIRE(a[q].size() == b[q].size());
        for (size_t i = 0; i < a[q].size(); ++i) {
            REQUIRE(a[q][i].index == b[q][i].index);
            REQUIRE(a[q][i].distance == b[q][i].distance);
        }
    }
}

TEST_CASE("fast-scan matches the float ADC scan on the same codes", "[fastscan][ivf]") {
    const auto [dim, m] = GENERATE(std::pair<size_t, size_t>{32, 16}, std::pair<size_t, size_t>{960, 480},
                                   std::pair<size_t, size_t>{1024, 1024});
    const size_t n = 5000, nq = 200, k = 10;
    CAPTURE(dim, m);
    auto data = fs_random(n, dim, 1);
    auto queries = fs_random(nq, dim, 2);
    const bool precomputed = GENERATE(true, false);
    CAPTURE(precomputed);

    auto fast = fs_ivf(dim, m, true, data, n);
    auto ref = fs_ivf(dim, m, false, data, n);
    fast->set_precomputed_tables(precomputed);
    ref->set_precomputed_tables(precomputed);
    fast->add_batch(data, n);
    ref->add_batch(data, n);
    REQUIRE(fast->fast_scan());

    const auto rf = fast->query_batch(queries, nq, k);
    const auto rr = ref->query_batch(queries, nq, k);
    REQUIRE(overlap(rf, rr) >= 0.95);

    // Matching ids get nearly the same distance: rounding the table to bytes
    // moves each by at most m / (2 * scale), and scale shrinks with kLutMax.
    const float slack = 255.0f / kLutMax;
    for (size_t q = 0; q < nq; ++q)
        for (const auto& a : rf[q])
            for (const auto& b : rr[q])
                if (a.index == b.index) REQUIRE(std::fabs(a.distance - b.distance) <= slack * (0.02f * b.distance + 0.05f));

    // Against exact search, fast-scan loses almost nothing next to the float scan.
    FlatIndex flat(dim);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);
    REQUIRE(overlap(rf, truth) >= overlap(rr, truth) - 0.02);

    std::stringstream s;
    fast->save(s);
    require_identical(load_dense(s)->query_batch(queries, nq, k), rf);
}

TEST_CASE("fast-scan: add and add_batch store the same codes", "[fastscan][ivf]") {
    const size_t dim = 16, m = 8, n = 2000, nq = 100;
    auto data = fs_random(n, dim, 3);
    auto queries = fs_random(nq, dim, 4);

    auto one = fs_ivf(dim, m, true, data, n);
    for (size_t i = 0; i < n; ++i) one->add(std::span<const float>(data.data() + i * dim, dim));
    auto batch = fs_ivf(dim, m, true, data, n);
    batch->add_batch(data, n);

    // add() and add_batch() can assign a near-tie to different lists, so allow
    // a few differences; wrongly packed codes would make this collapse.
    REQUIRE(overlap(one->query_batch(queries, nq, 10), batch->query_batch(queries, nq, 10)) >= 0.97);
}

TEST_CASE("fast-scan: query_batch matches query", "[fastscan][ivf]") {
    const size_t dim = 16, m = 8, n = 3000, nq = 1100, k = 10;
    auto data = fs_random(n, dim, 5);
    auto queries = fs_random(nq, dim, 6);
    auto index = fs_ivf(dim, m, true, data, n, 12);
    index->add_batch(data, n);

    const auto batch = index->query_batch(queries, nq, k);
    Results single(nq);
    for (size_t q = 0; q < nq; ++q) single[q] = index->query(std::span<const float>(queries.data() + q * dim, dim), k);
    REQUIRE(overlap(batch, single) >= 0.99);
}

TEST_CASE("fast-scan: deletions, compaction and save/load", "[fastscan][ivf][compact]") {
    const size_t dim = 16, m = 8, n = 3000, nq = 100, k = 10;
    auto data = fs_random(n, dim, 7);
    auto queries = fs_random(nq, dim, 8);
    auto index = fs_ivf(dim, m, true, data, n, 100);   // every list, so results don't depend on clustering
    index->add_batch(data, n);

    std::vector<int64_t> dead;
    for (int64_t id = 0; id < static_cast<int64_t>(n); id += 3) dead.push_back(id);
    index->remove(dead);
    const auto before = index->query_batch(queries, nq, k);
    for (const auto& r : before)
        for (const auto& nb : r) REQUIRE(nb.index % 3 != 0);

    REQUIRE(index->compact() == dead.size());
    require_identical(index->query_batch(queries, nq, k), before);

    std::stringstream s;
    index->save(s);
    auto loaded = load_dense(s);
    auto& ivf = dynamic_cast<IVFIndex&>(*loaded);
    REQUIRE(ivf.fast_scan());
    require_identical(ivf.query_batch(queries, nq, k), before);

    // Both keep accepting vectors and find them.
    std::span<const float> extra(queries.data(), dim);
    index->add(extra);
    ivf.add(extra);
    require_identical(ivf.query_batch(queries, nq, k), index->query_batch(queries, nq, k));
    REQUIRE(ivf.query(extra, 1)[0].index == n);
}

TEST_CASE("fast-scan under a RefineIndex recovers exact ranking", "[fastscan][ivf][refine]") {
    const size_t dim = 32, m = 16, n = 5000, nq = 200, k = 10;
    auto data = fs_random(n, dim, 9);
    auto queries = fs_random(nq, dim, 10);
    auto base = fs_ivf(dim, m, true, data, n, 100);
    RefineIndex refine(*base, 10);
    refine.add_batch(data, n);

    FlatIndex flat(dim);
    flat.add_batch(data, n);
    const auto truth = flat.query_batch(queries, nq, k);
    const auto got = refine.query_batch(queries, nq, k);
    REQUIRE(overlap(got, truth) >= 0.95);
    for (size_t q = 0; q < nq; ++q)   // re-ranked distances are exact
        for (const auto& a : got[q])
            for (const auto& t : truth[q])
                if (a.index == t.index) REQUIRE(a.distance == t.distance);
}

TEST_CASE("fast-scan settings are validated", "[fastscan][ivf]") {
    IVFIndex index(32, 100);
    REQUIRE_THROWS_AS(index.enable_pq(16, 256, true), std::invalid_argument);   // needs 16 centroids
    IVFIndex odd(30, 100);
    REQUIRE_THROWS_AS(odd.enable_pq(15, 16, true), std::invalid_argument);      // needs an even m
    IVFIndex wide(2052, 100);
    REQUIRE_THROWS_AS(wide.enable_pq(1026, 16, true), std::invalid_argument);   // at most kFsMaxSubspaces
    IVFIndex widest(1024, 100);
    widest.enable_pq(1024, 16, true);

    index.enable_pq(16, 16, true);
    REQUIRE_THROWS_AS(index.set_pq_distance(PQDistance::SDC), std::invalid_argument);
    REQUIRE(index.pq_distance() == PQDistance::ADC);

    IVFIndex sdc_first(32, 100);
    sdc_first.set_pq_distance(PQDistance::SDC);
    REQUIRE_THROWS_AS(sdc_first.enable_pq(16, 16, true), std::invalid_argument);
}

TEST_CASE("quantize_lut matches the scalar version exactly", "[fastscan]") {
    const size_t m = GENERATE(2, 16, 64, 256, 480, 1024);
    CAPTURE(m);
    std::mt19937 rng(static_cast<unsigned>(m) + 100);
    std::uniform_real_distribution<float> d(-3000.0f, 3000.0f);
    std::vector<float> table(m * 16);

    SECTION("random rows, including negative ones") { for (auto& x : table) x = d(rng); }
    SECTION("some flat rows") {
        for (auto& x : table) x = d(rng);
        for (size_t j = 0; j < 16; ++j) table[j] = 7.0f;
    }
    SECTION("every row flat") { std::fill(table.begin(), table.end(), -2.5f); }

    std::vector<uint8_t> a(m * 16), b(m * 16);
    float sa = 0, oa = 0, sb = 0, ob = 0;
    quantize_lut(table.data(), m, a.data(), sa, oa);
    quantize_lut_scalar(table.data(), m, b.data(), sb, ob);
    REQUIRE(a == b);
    REQUIRE(sa == sb);
    REQUIRE(oa == ob);
}

TEST_CASE("quantize_lut rounds halves away from zero like the scalar version", "[fastscan]") {
    // Row 0 spans 0..kLutMax, so scale is exactly 1 and every entry is its own
    // rounding input: exact halves, and the floats either side of 0.5.
    const size_t m = 2;
    std::vector<float> table(m * 16, 0.0f);
    table[1] = kLutMax;
    const float probes[] = {0.5f, 1.5f, 2.5f, 62.5f, std::nextafter(0.5f, 0.0f), std::nextafter(0.5f, 1.0f),
                            std::nextafter(1.5f, 0.0f), 0.25f, 3.0f};
    std::copy(std::begin(probes), std::end(probes), table.begin() + 16 + 1);   // row 1, min 0 at entry 0

    std::vector<uint8_t> a(m * 16), b(m * 16);
    float sa = 0, oa = 0, sb = 0, ob = 0;
    quantize_lut(table.data(), m, a.data(), sa, oa);
    quantize_lut_scalar(table.data(), m, b.data(), sb, ob);
    REQUIRE(sa == 1.0f);
    REQUIRE(a == b);
    REQUIRE(std::vector<uint8_t>(a.begin() + 17, a.begin() + 26) == std::vector<uint8_t>{1, 2, 3, 63, 0, 1, 1, 0, 3});
}

TEST_CASE("fast_scan_block_mask flags exactly the sums below the limit", "[fastscan]") {
    const size_t m = GENERATE(2, 16, 64, 256, 480, 1024);
    CAPTURE(m);
    std::mt19937 rng(static_cast<unsigned>(m) + 200);
    std::vector<uint8_t> lut(m * 16), block(fs_block_bytes(m));
    for (auto& x : lut) x = rng() % (kLutMax + 1);
    for (auto& x : block) x = rng() & 0xFF;

    std::vector<uint16_t> ref(kFsBlock);
    fast_scan_block_scalar(lut.data(), block.data(), m, ref.data());
    std::vector<uint16_t> sorted = ref;
    std::sort(sorted.begin(), sorted.end());

    // limits at the extremes, and exactly at sums so ties are covered
    for (uint16_t limit : {uint16_t{0}, uint16_t{65535}, sorted[0], sorted[10], sorted[31], uint16_t(sorted[20] + 1)}) {
        CAPTURE(limit);
        std::vector<uint16_t> sums(kFsBlock), scalar_sums(kFsBlock);
        const uint32_t mask = fast_scan_block_mask(lut.data(), block.data(), m, limit, sums.data());
        const uint32_t scalar = fast_scan_block_mask_scalar(lut.data(), block.data(), m, limit, scalar_sums.data());
        REQUIRE(sums == ref);
        REQUIRE(mask == scalar);
        for (size_t v = 0; v < kFsBlock; ++v) REQUIRE(((mask >> v) & 1u) == (ref[v] < limit ? 1u : 0u));
    }
}
