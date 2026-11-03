#include <catch2/catch_test_macros.hpp>

#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/vector_store.hpp"
#include "serialize.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vecengine;
namespace fs = std::filesystem;

namespace {

std::vector<float> random_vectors(size_t n, size_t dim, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist;
    std::vector<float> v(n * dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

struct TempDir {
    fs::path path;
    TempDir() : path(fs::temp_directory_path() / ("qann_mmap_" + std::to_string(std::random_device{}()))) {
        fs::create_directories(path);
    }
    ~TempDir() { fs::remove_all(path); }
};

std::unique_ptr<RefineIndex> load_refine(const fs::path& file, bool mmap) {
    auto loaded = load_index(file, mmap);
    REQUIRE(dynamic_cast<RefineIndex*>(loaded.get()) != nullptr);
    return std::unique_ptr<RefineIndex>(static_cast<RefineIndex*>(loaded.release()));
}

void require_same(const DenseIndex& a, const DenseIndex& b, const std::vector<float>& queries, size_t nq, size_t k) {
    const std::span<const float> q(queries.data(), nq * a.dim());
    auto expected = a.query_batch(q, nq, k);
    auto actual = b.query_batch(q, nq, k);
    for (size_t i = 0; i < nq; ++i) {
        REQUIRE(actual[i].size() == expected[i].size());
        for (size_t j = 0; j < expected[i].size(); ++j) {
            REQUIRE(actual[i][j].index == expected[i][j].index);
            REQUIRE(actual[i][j].distance == expected[i][j].distance);
        }
        auto single = b.query({queries.data() + i * a.dim(), a.dim()}, k);
        REQUIRE(single.size() == expected[i].size());
        for (size_t j = 0; j < single.size(); ++j) REQUIRE(single[j].index == expected[i][j].index);
    }
    const auto ids = a.ids();
    REQUIRE(b.ids() == ids);
    REQUIRE(b.get_vectors(ids) == a.get_vectors(ids));
}

} // namespace

TEST_CASE("mmap: a mapped RefineIndex matches one read into RAM", "[mmap][refine][file]") {
    TempDir dir;
    const size_t n = 4000, dim = 16, nq = 40, k = 10;
    const auto data = random_vectors(n, dim, 31);
    const auto queries = random_vectors(nq, dim, 32);
    const fs::path file = dir.path / "index.qann";

    SECTION("IVF-PQ base, custom ids") {
        IVFIndex ivf(dim, 100, 8);
        ivf.enable_pq(4);
        ivf.train(data, n, 5, 1);
        RefineIndex refine(ivf, 4, true);
        std::vector<int64_t> ids(n);
        std::iota(ids.begin(), ids.end(), 1000);
        refine.add_batch(data, n, ids.data());
        save_index(refine, file);
    }
    SECTION("HNSW base") {
        HNSWIndex hnsw(dim, 8, 40);
        RefineIndex refine(hnsw, 3);
        refine.add_batch(data, n);
        save_index(refine, file);
    }

    auto in_ram = load_refine(file, false);
    auto mapped = load_refine(file, true);
    REQUIRE_FALSE(in_ram->mapped());
    REQUIRE(mapped->mapped());
    require_same(*in_ram, *mapped, queries, nq, k);
}

TEST_CASE("mmap: adds, removes and compaction on a mapped index", "[mmap][refine][file]") {
    TempDir dir;
    const size_t n = 1500, more = 300, dim = 12, nq = 30, k = 10;
    const auto data = random_vectors(n + more, dim, 33);
    const auto queries = random_vectors(nq, dim, 34);
    const fs::path file = dir.path / "index.qann";
    {
        FlatIndex flat(dim);
        RefineIndex refine(flat, 2);
        refine.add_batch(std::span<const float>(data.data(), n * dim), n);
        save_index(refine, file);
    }
    auto in_ram = load_refine(file, false);
    auto mapped = load_refine(file, true);

    // The same operations on both: rows added to the RAM tail, removals spanning the
    // mapped rows and the tail, and compactions that remap a table more than once.
    auto apply = [&](RefineIndex& index) {
        index.add_batch(std::span<const float>(data.data() + n * dim, more * dim), more);
        std::vector<int64_t> gone;
        for (int64_t id = 0; id < static_cast<int64_t>(n + more); id += 3) gone.push_back(id);
        index.remove(gone);
        index.compact();
        index.add_batch(std::span<const float>(data.data(), 50 * dim), 50);
        index.remove(std::vector<int64_t>{1, 2, 4, 5, 1700});
        index.compact();
    };
    apply(*in_ram);
    apply(*mapped);
    REQUIRE(mapped->mapped());
    require_same(*in_ram, *mapped, queries, nq, k);

    SECTION("saving over the mapped file writes only live rows, in slot order") {
        const auto before = fs::file_size(file);
        save_index(*mapped, file);
        REQUIRE(fs::file_size(file) < before);
        require_same(*in_ram, *mapped, queries, nq, k);  // still reads the replaced file's pages
        require_same(*in_ram, *load_refine(file, true), queries, nq, k);
        require_same(*in_ram, *load_refine(file, false), queries, nq, k);
    }

    SECTION("compacting away every row, then adding again") {
        auto all = mapped->ids();
        mapped->remove(all);
        in_ram->remove(all);
        mapped->compact();
        in_ram->compact();
        REQUIRE_FALSE(mapped->mapped());
        mapped->add_batch(std::span<const float>(data.data(), 100 * dim), 100);
        in_ram->add_batch(std::span<const float>(data.data(), 100 * dim), 100);
        require_same(*in_ram, *mapped, queries, nq, k);
    }
}

TEST_CASE("mmap: files without a RefineIndex and older files", "[mmap][file]") {
    TempDir dir;
    const fs::path file = dir.path / "flat.qann";
    FlatIndex flat(4);
    flat.add_batch(random_vectors(10, 4, 35), 10);
    save_index(flat, file);
    REQUIRE(load_index(file, true)->size() == 10);

    // Format 9 has no padding before the vectors, so it can't be mapped.
    const fs::path v9 = fs::path(QANN_TEST_DATA_DIR) / "v9_refine_metadata.qann";
    REQUIRE(load_index(v9, false)->size() > 0);
    REQUIRE_THROWS_AS(load_index(v9, true), std::runtime_error);
}

TEST_CASE("VectorStore: rows start at a 64 KB file offset", "[mmap][serialize]") {
    for (size_t prefix : {0, 7, 65531, 65532, 65536, 100000}) {
        detail::VectorStore store(3);
        const auto rows = random_vectors(5, 3, 36);
        store.append(rows);
        std::stringstream out;
        out.write(std::string(prefix, 'x').data(), static_cast<std::streamsize>(prefix));
        store.save(out);
        const std::string bytes = out.str();
        const size_t start = bytes.size() - rows.size() * sizeof(float);
        REQUIRE(start % (64 * 1024) == 0);
        REQUIRE(std::memcmp(bytes.data() + start, rows.data(), rows.size() * sizeof(float)) == 0);

        std::stringstream in(bytes.substr(prefix));
        auto loaded = detail::VectorStore::load(in, detail::kFormatVersion, 5, 3, nullptr);
        REQUIRE(std::memcmp(loaded.row(4), rows.data() + 12, 3 * sizeof(float)) == 0);
    }
}
