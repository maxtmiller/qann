#include <catch2/catch_test_macros.hpp>

#include "vecengine/flat_index.hpp"
#include "vecengine/id_map.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/pq.hpp"
#include "vecengine/refine_index.hpp"
#include "serialize.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vecengine;
using namespace vecengine::detail;

static std::vector<float> random_vectors(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(n * dim);
    for (auto& x : out) x = dist(rng);
    return out;
}

// ---------------------------------------------------------------------------
// serialize.hpp helpers
// ---------------------------------------------------------------------------

TEST_CASE("serialize: header, pod and vector round-trip", "[serialize]") {
    std::stringstream s;
    write_header(s, IndexKind::IVF);
    write_pod<uint64_t>(s, 42);
    write_vec(s, std::vector<float>{1.5f, 2.5f, 3.5f});
    write_vec(s, std::vector<uint8_t>{});

    REQUIRE(s.str().substr(0, 4) == "QANN");
    const Header header = read_header(s);
    REQUIRE(header.kind == IndexKind::IVF);
    REQUIRE(header.version == kFormatVersion);
    REQUIRE(read_pod<uint64_t>(s) == 42);
    REQUIRE(read_vec<float>(s, 3) == std::vector<float>{1.5f, 2.5f, 3.5f});
    REQUIRE(read_vec<uint8_t>(s, 0).empty());
}

TEST_CASE("serialize: read_header rejects bad input", "[serialize]") {
    std::stringstream good;
    write_header(good, IndexKind::Flat);
    const std::string bytes = good.str();

    auto expect_throw = [](std::string b) {
        std::stringstream s(b);
        REQUIRE_THROWS_AS(read_header(s), std::runtime_error);
    };

    std::string bad_magic = bytes;
    bad_magic[0] = 'X';
    expect_throw(bad_magic);

    std::string bad_version = bytes;
    bad_version[4] = 9;
    expect_throw(bad_version);

    std::string bad_kind = bytes;
    bad_kind[8] = 0;
    expect_throw(bad_kind);
    bad_kind[8] = 99;
    expect_throw(bad_kind);

    expect_throw(bytes.substr(0, 6));
}

TEST_CASE("serialize: read_vec rejects oversized and truncated vectors", "[serialize]") {
    std::stringstream s;
    write_vec(s, std::vector<float>{1.0f, 2.0f, 3.0f});
    const std::string bytes = s.str();

    std::stringstream too_long(bytes);
    REQUIRE_THROWS_AS(read_vec<float>(too_long, 2), std::runtime_error);

    std::stringstream truncated(bytes.substr(0, bytes.size() - 2));
    REQUIRE_THROWS_AS(read_vec<float>(truncated, 3), std::runtime_error);
}

// ---------------------------------------------------------------------------
// PQCodebook
// ---------------------------------------------------------------------------

TEST_CASE("PQCodebook: rejects zero dim or zero subspaces", "[pq][serialize]") {
    REQUIRE_THROWS_AS(PQCodebook(0, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(PQCodebook(32, 0), std::invalid_argument);
}

TEST_CASE("PQCodebook: save/load round-trip is bit-identical", "[pq][serialize]") {
    const size_t n = 2000, dim = 32, m = 8, k = 256;
    auto data = random_vectors(n, dim, 1);

    PQCodebook pq(dim, m, k);
    pq.train(data, n, 25, 1);

    std::stringstream s;
    pq.save(s);
    PQCodebook loaded = PQCodebook::load(s);

    REQUIRE(loaded.dim() == dim);
    REQUIRE(loaded.num_subspaces() == m);
    REQUIRE(loaded.centroids_per_subspace() == k);

    std::vector<float> table_a(m * k), table_b(m * k);
    for (size_t i = 0; i + 1 < 200; ++i) {
        std::span<const float> v(data.data() + i * dim, dim);
        std::span<const float> w(data.data() + (i + 1) * dim, dim);

        auto code_a = pq.encode(v);
        auto code_b = loaded.encode(v);
        REQUIRE(code_a == code_b);

        pq.compute_adc_table(v, table_a);
        loaded.compute_adc_table(v, table_b);
        REQUIRE(table_a == table_b);

        auto other = pq.encode(w);
        REQUIRE(pq.distance_sdc(code_a, other) == loaded.distance_sdc(code_b, other));
    }
}

TEST_CASE("PQCodebook: load rejects truncated and corrupt data", "[pq][serialize]") {
    const size_t n = 500, dim = 16, m = 4, k = 16;
    auto data = random_vectors(n, dim, 2);
    PQCodebook pq(dim, m, k);
    pq.train(data, n, 10, 1);

    std::stringstream s;
    pq.save(s);
    const std::string bytes = s.str();

    SECTION("truncated mid-centroids") {
        std::stringstream t(bytes.substr(0, bytes.size() / 2));
        REQUIRE_THROWS_AS(PQCodebook::load(t), std::runtime_error);
    }

    SECTION("zero subspaces") {
        std::stringstream t;
        write_pod<uint64_t>(t, dim);
        write_pod<uint64_t>(t, 0);
        write_pod<uint64_t>(t, k);
        REQUIRE_THROWS_AS(PQCodebook::load(t), std::invalid_argument);
    }

    SECTION("fewer centroids than the shape requires") {
        std::stringstream t;
        write_pod<uint64_t>(t, dim);
        write_pod<uint64_t>(t, m);
        write_pod<uint64_t>(t, k);
        write_vec(t, std::vector<float>(m * k * (dim / m) - 1));
        REQUIRE_THROWS_AS(PQCodebook::load(t), std::runtime_error);
    }
}

// ---------------------------------------------------------------------------
// FlatIndex
// ---------------------------------------------------------------------------

static std::unique_ptr<FlatIndex> flat_roundtrip(const FlatIndex& index) {
    std::stringstream s;
    index.save(s);
    const Header header = read_header(s);
    REQUIRE(header.kind == IndexKind::Flat);
    return FlatIndex::load_body(s, header.version);
}

TEST_CASE("FlatIndex: rejects zero dim", "[index][serialize]") {
    REQUIRE_THROWS_AS(FlatIndex(0), std::invalid_argument);
}

TEST_CASE("FlatIndex: save/load round-trip returns identical results", "[index][serialize]") {
    const size_t n = 1000, dim = 32, nq = 50, k = 10;
    auto data = random_vectors(n, dim, 3);
    auto queries = random_vectors(nq, dim, 4);

    FlatIndex index(dim);
    index.add_batch(data, n);
    auto loaded = flat_roundtrip(index);

    REQUIRE(loaded->size() == n);
    REQUIRE(loaded->dim() == dim);

    auto expected = index.query_batch(queries, nq, k);
    auto actual = loaded->query_batch(queries, nq, k);
    for (size_t q = 0; q < nq; ++q) {
        REQUIRE(actual[q].size() == expected[q].size());
        for (size_t j = 0; j < expected[q].size(); ++j) {
            REQUIRE(actual[q][j].index == expected[q][j].index);
            REQUIRE(actual[q][j].distance == expected[q][j].distance);
        }
    }

    // A loaded index keeps accepting vectors with consecutive ids.
    loaded->add(std::span<const float>(queries.data(), dim));
    REQUIRE(loaded->size() == n + 1);
    REQUIRE(loaded->query(std::span<const float>(queries.data(), dim), 1)[0].index == n);
}

TEST_CASE("FlatIndex: empty index round-trips", "[index][serialize]") {
    FlatIndex index(16);
    auto loaded = flat_roundtrip(index);
    REQUIRE(loaded->size() == 0);
    REQUIRE(loaded->dim() == 16);
}

TEST_CASE("FlatIndex: load rejects truncated and corrupt data", "[index][serialize]") {
    const size_t n = 100, dim = 8;
    auto data = random_vectors(n, dim, 5);
    FlatIndex index(dim);
    index.add_batch(data, n);

    std::stringstream s;
    index.save(s);
    read_header(s);
    const std::string body = s.str().substr(s.tellg());

    SECTION("truncated") {
        std::stringstream t(body.substr(0, body.size() / 2));
        REQUIRE_THROWS_AS(FlatIndex::load_body(t, kFormatVersion), std::runtime_error);
    }

    SECTION("zero dim") {
        std::stringstream t;
        write_pod<uint64_t>(t, 0);
        write_pod<uint64_t>(t, n);
        REQUIRE_THROWS_AS(FlatIndex::load_body(t, 1), std::runtime_error);
    }

    SECTION("count that overflows count * dim") {
        std::stringstream t;
        write_pod<uint64_t>(t, dim);
        write_pod<uint64_t>(t, UINT64_MAX / 2);
        REQUIRE_THROWS_AS(FlatIndex::load_body(t, 1), std::runtime_error);
    }

    SECTION("count does not match data length") {
        std::stringstream t;
        write_pod<uint64_t>(t, dim);
        write_pod<uint64_t>(t, n);
        write_vec(t, std::vector<float>((n - 1) * dim));
        REQUIRE_THROWS_AS(FlatIndex::load_body(t, 1), std::runtime_error);
    }
}

// ---------------------------------------------------------------------------
// IVFIndex
// ---------------------------------------------------------------------------

static std::unique_ptr<IVFIndex> ivf_roundtrip(const IVFIndex& index) {
    std::stringstream s;
    index.save(s);
    const Header header = read_header(s);
    REQUIRE(header.kind == IndexKind::IVF);
    return IVFIndex::load_body(s, header.version);
}

static void require_same_results(const Index& a, const Index& b, const std::vector<float>& queries, size_t nq, size_t k) {
    const std::span<const float> q(queries.data(), nq * a.dim());
    auto expected = a.query_batch(q, nq, k);
    auto actual = b.query_batch(q, nq, k);
    for (size_t q = 0; q < nq; ++q) {
        REQUIRE(actual[q].size() == expected[q].size());
        for (size_t j = 0; j < expected[q].size(); ++j) {
            REQUIRE(actual[q][j].index == expected[q][j].index);
            REQUIRE(actual[q][j].distance == expected[q][j].distance);
        }
    }
}

TEST_CASE("IVFIndex: rejects zero dim", "[ivf][serialize]") {
    REQUIRE_THROWS_AS(IVFIndex(0, 100), std::invalid_argument);
}

TEST_CASE("IVFIndex: save/load round-trip returns identical results", "[ivf][serialize]") {
    const size_t n = 3000, dim = 16, nlist = 100, nq = 50, k = 10;
    auto data = random_vectors(n, dim, 6);
    auto queries = random_vectors(nq, dim, 7);

    IVFIndex index(dim, nlist, 8);

    SECTION("plain") {}
    SECTION("PQ, ADC") { index.enable_pq(4); }
    SECTION("PQ, SDC") {
        index.enable_pq(4);
        index.set_pq_distance(PQDistance::SDC);
    }

    index.train(data, n, 10, 1);
    index.add_batch(data, n);
    auto loaded = ivf_roundtrip(index);

    REQUIRE(loaded->size() == n);
    REQUIRE(loaded->dim() == dim);
    REQUIRE(loaded->nprobe() == 8);
    REQUIRE(loaded->pq_distance() == index.pq_distance());
    REQUIRE(loaded->precomputed_tables() == index.precomputed_tables());
    require_same_results(index, *loaded, queries, nq, k);

    // Both keep accepting vectors and assign the same next id.
    std::span<const float> extra(queries.data(), dim);
    index.add(extra);
    loaded->add(extra);
    REQUIRE(loaded->size() == n + 1);
    require_same_results(index, *loaded, queries, nq, k);
}

TEST_CASE("IVFIndex: untrained and empty indexes round-trip", "[ivf][serialize]") {
    const size_t n = 500, dim = 8, nlist = 100;
    auto data = random_vectors(n, dim, 8);

    SECTION("untrained, with PQ enabled") {
        IVFIndex index(dim, nlist);
        index.enable_pq(2);
        auto loaded = ivf_roundtrip(index);
        REQUIRE(loaded->size() == 0);
        REQUIRE_THROWS_AS(loaded->add(std::span<const float>(data.data(), dim)), std::logic_error);

        // Still trainable after loading.
        loaded->train(data, n, 5, 1);
        loaded->add_batch(data, n);
        REQUIRE(loaded->size() == n);
    }

    SECTION("trained, nothing added") {
        IVFIndex index(dim, nlist);
        index.train(data, n, 5, 1);
        auto loaded = ivf_roundtrip(index);
        REQUIRE(loaded->size() == 0);
        loaded->add_batch(data, n);
        REQUIRE(loaded->query(std::span<const float>(data.data(), dim), 1)[0].index == 0);
    }
}

// Writes a version-1 IVF body (no id map) by hand: plain (no PQ), trained, all centroids zero,
// with list 0 holding `ids` and every other list empty.
static std::stringstream ivf_body(uint64_t n_total, const std::vector<uint32_t>& ids, uint64_t nprobe = 1) {
    const uint64_t dim = 2, nlist = 100;
    std::stringstream t;
    write_pod<uint64_t>(t, dim);
    write_pod<uint64_t>(t, nlist);
    write_pod<uint64_t>(t, nprobe);
    write_pod<uint64_t>(t, n_total);
    write_pod<uint8_t>(t, 1);
    write_pod<uint8_t>(t, 0);
    write_vec(t, std::vector<float>(nlist * dim));
    write_pod<uint8_t>(t, 0);
    write_vec(t, ids);
    write_vec(t, std::vector<float>(ids.size() * dim));
    for (uint64_t c = 1; c < nlist; ++c) {
        write_vec(t, std::vector<uint32_t>{});
        write_vec(t, std::vector<float>{});
    }
    return t;
}

TEST_CASE("IVFIndex: load rejects truncated and corrupt data", "[ivf][serialize]") {
    SECTION("hand-written body is valid") {
        auto t = ivf_body(2, {0, 1});
        REQUIRE(IVFIndex::load_body(t, 1)->size() == 2);
    }

    SECTION("truncated") {
        const size_t n = 500, dim = 8;
        auto data = random_vectors(n, dim, 9);
        IVFIndex index(dim, 100);
        index.enable_pq(2);
        index.train(data, n, 5, 1);
        index.add_batch(data, n);

        std::stringstream s;
        index.save(s);
        read_header(s);
        const std::string body = s.str().substr(s.tellg());
        for (size_t cut : {size_t{10}, body.size() / 3, body.size() - 1}) {
            std::stringstream t(body.substr(0, cut));
            REQUIRE_THROWS_AS(IVFIndex::load_body(t, kFormatVersion), std::runtime_error);
        }
    }

    SECTION("nprobe = 0") {
        auto t = ivf_body(1, {0}, 0);
        REQUIRE_THROWS_AS(IVFIndex::load_body(t, 1), std::invalid_argument);
    }

    SECTION("id >= n_total") {
        auto t = ivf_body(1, {5});
        REQUIRE_THROWS_AS(IVFIndex::load_body(t, 1), std::runtime_error);
    }

    SECTION("list sizes do not sum to n_total") {
        auto t = ivf_body(3, {0, 1});
        REQUIRE_THROWS_AS(IVFIndex::load_body(t, 1), std::runtime_error);
    }

    SECTION("PQ codebook dim does not match the index") {
        const uint64_t dim = 8, nlist = 100;
        std::stringstream t;
        write_pod<uint64_t>(t, dim);
        write_pod<uint64_t>(t, nlist);
        write_pod<uint64_t>(t, 1);
        write_pod<uint64_t>(t, 0);
        write_pod<uint8_t>(t, 1);
        write_pod<uint8_t>(t, 0);
        write_vec(t, std::vector<float>(nlist * dim));
        write_pod<uint8_t>(t, 1);
        PQCodebook(4, 2, 1).save(t);
        REQUIRE_THROWS_AS(IVFIndex::load_body(t, 1), std::runtime_error);
    }
}

// ---------------------------------------------------------------------------
// RefineIndex and load_index
// ---------------------------------------------------------------------------

static std::unique_ptr<Index> index_roundtrip(const Index& index) {
    std::stringstream s;
    index.save(s);
    return load_index(s);
}

TEST_CASE("load_index: returns the saved concrete type", "[serialize]") {
    const size_t n = 500, dim = 8;
    auto data = random_vectors(n, dim, 10);

    FlatIndex flat(dim);
    flat.add_batch(data, n);
    REQUIRE(dynamic_cast<FlatIndex*>(index_roundtrip(flat).get()) != nullptr);

    IVFIndex ivf(dim, 100);
    ivf.train(data, n, 5, 1);
    ivf.add_batch(data, n);
    REQUIRE(dynamic_cast<IVFIndex*>(index_roundtrip(ivf).get()) != nullptr);
}

TEST_CASE("RefineIndex: save/load round-trip returns identical results", "[refine][serialize]") {
    const size_t n = 3000, dim = 16, nq = 50, k = 10;
    auto data = random_vectors(n, dim, 11);
    auto queries = random_vectors(nq, dim, 12);

    SECTION("over IVF + PQ") {
        IVFIndex ivf(dim, 100, 8);
        ivf.enable_pq(4);
        ivf.train(data, n, 10, 1);
        RefineIndex refine(ivf, 5);
        refine.add_batch(data, n);

        auto loaded_base = index_roundtrip(refine);
        auto* loaded = dynamic_cast<RefineIndex*>(loaded_base.get());
        REQUIRE(loaded != nullptr);
        REQUIRE(loaded->size() == n);
        REQUIRE(loaded->dim() == dim);
        REQUIRE(loaded->k_factor() == 5);
        require_same_results(refine, *loaded, queries, nq, k);

        // The loaded base is reachable and tuning it takes effect.
        auto* loaded_ivf = dynamic_cast<IVFIndex*>(&loaded->base());
        REQUIRE(loaded_ivf != nullptr);
        ivf.set_nprobe(40);
        loaded_ivf->set_nprobe(40);
        require_same_results(refine, *loaded, queries, nq, k);

        // Adding after a load keeps ids and raw vectors in step.
        std::span<const float> extra(queries.data(), dim);
        refine.add(extra);
        loaded->add(extra);
        REQUIRE(loaded->size() == n + 1);
        require_same_results(refine, *loaded, queries, nq, k);
    }

    SECTION("over Flat") {
        FlatIndex flat(dim);
        RefineIndex refine(flat, 2);
        refine.add_batch(data, n);

        auto loaded = index_roundtrip(refine);
        REQUIRE(dynamic_cast<RefineIndex*>(loaded.get()) != nullptr);
        require_same_results(refine, *loaded, queries, nq, k);
    }
}

TEST_CASE("RefineIndex: load rejects truncated and corrupt data", "[refine][serialize]") {
    const size_t n = 20, dim = 4;
    auto data = random_vectors(n, dim, 13);
    FlatIndex flat(dim);
    flat.add_batch(data, n);

    auto body = [&](uint64_t k_factor, uint64_t count, size_t data_len) {
        std::stringstream t;
        write_pod<uint64_t>(t, k_factor);
        write_pod<uint64_t>(t, count);
        flat.save(t);
        write_vec(t, std::vector<float>(data_len));
        IdMap ids;
        ids.add(count, nullptr);
        ids.save(t);
        return t;
    };

    SECTION("hand-written body is valid") {
        auto t = body(3, n, n * dim);
        REQUIRE(RefineIndex::load_body(t, kFormatVersion)->size() == n);
    }

    SECTION("k_factor = 0") {
        auto t = body(0, n, n * dim);
        REQUIRE_THROWS_AS(RefineIndex::load_body(t, kFormatVersion), std::runtime_error);
    }

    SECTION("count does not match base size") {
        auto t = body(3, n + 1, (n + 1) * dim);
        REQUIRE_THROWS_AS(RefineIndex::load_body(t, kFormatVersion), std::runtime_error);
    }

    SECTION("data length mismatch") {
        auto t = body(3, n, n * dim - 1);
        REQUIRE_THROWS_AS(RefineIndex::load_body(t, kFormatVersion), std::runtime_error);
    }

    SECTION("truncated") {
        auto t = body(3, n, n * dim);
        const std::string bytes = t.str();
        std::stringstream cut(bytes.substr(0, bytes.size() - 1));
        REQUIRE_THROWS_AS(RefineIndex::load_body(cut, kFormatVersion), std::runtime_error);
    }

    SECTION("nested RefineIndex base is rejected") {
        FlatIndex inner_base(dim);
        RefineIndex inner(inner_base, 2);
        std::stringstream t;
        write_pod<uint64_t>(t, 3);
        write_pod<uint64_t>(t, 0);
        inner.save(t);
        write_vec(t, std::vector<float>{});
        REQUIRE_THROWS_AS(RefineIndex::load_body(t, kFormatVersion), std::runtime_error);
    }
}

// ---------------------------------------------------------------------------
// save_index / load_index(path)
// ---------------------------------------------------------------------------

namespace fs = std::filesystem;

// A fresh directory under the system temp dir, removed when the test ends.
struct TempDir {
    fs::path path;
    TempDir() : path(fs::temp_directory_path() / ("qann_test_" + std::to_string(std::random_device{}()))) {
        fs::create_directories(path);
    }
    ~TempDir() { fs::remove_all(path); }
};

TEST_CASE("save_index/load_index: file round-trip", "[serialize][file]") {
    TempDir dir;
    const size_t n = 500, dim = 8;
    auto data = random_vectors(n, dim, 14);

    IVFIndex ivf(dim, 100, 4);
    ivf.enable_pq(2);
    ivf.train(data, n, 5, 1);
    RefineIndex refine(ivf, 3);
    refine.add_batch(data, n);

    const fs::path file = dir.path / "index.qann";
    save_index(refine, file);
    REQUIRE(fs::exists(file));
    REQUIRE_FALSE(fs::exists(dir.path / "index.qann.tmp"));

    auto loaded = load_index(file);
    require_same_results(refine, *loaded, data, 20, 5);

    // Saving again overwrites the existing file.
    save_index(ivf, file);
    REQUIRE(dynamic_cast<IVFIndex*>(load_index(file).get()) != nullptr);
}

TEST_CASE("save_index/load_index: file errors", "[serialize][file]") {
    TempDir dir;
    FlatIndex flat(4);
    flat.add_batch(random_vectors(10, 4, 15), 10);
    const fs::path file = dir.path / "index.qann";
    save_index(flat, file);

    SECTION("missing file") {
        REQUIRE_THROWS_AS(load_index(dir.path / "missing.qann"), std::runtime_error);
    }

    SECTION("unwritable path") {
        REQUIRE_THROWS_AS(save_index(flat, dir.path / "no_such_dir" / "index.qann"), std::runtime_error);
    }

    SECTION("junk after the index") {
        std::ofstream(file, std::ios::binary | std::ios::app) << "junk";
        REQUIRE_THROWS_AS(load_index(file), std::runtime_error);
    }

    SECTION("out-of-range field reported as runtime_error") {
        IVFIndex ivf(4, 100);
        std::stringstream s;
        ivf.save(s);
        std::string bytes = s.str();
        bytes[9 + 16] = 0; // nprobe = 0
        std::ofstream(file, std::ios::binary | std::ios::trunc) << bytes;
        REQUIRE_THROWS_AS(load_index(file), std::runtime_error);
        try { load_index(file); } catch (const std::invalid_argument&) { FAIL("leaked invalid_argument"); } catch (const std::runtime_error&) {}
    }
}
