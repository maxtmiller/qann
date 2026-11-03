#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "vecengine/flat_index.hpp"
#include "vecengine/hnsw_index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/ivf_index.hpp"
#include "vecengine/refine_index.hpp"
#include "vecengine/inverted_index.hpp"
#include "vecengine/sparse_approx_index.hpp"
#include "vecengine/vector_store.hpp"
#include "serialize.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
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
    if (a.stores_vectors()) REQUIRE(b.get_vectors(ids) == a.get_vectors(ids));
}

std::unique_ptr<IVFIndex> load_ivf(const fs::path& file, bool mmap) {
    auto loaded = load_index(file, mmap);
    REQUIRE(dynamic_cast<IVFIndex*>(loaded.get()) != nullptr);
    return std::unique_ptr<IVFIndex>(static_cast<IVFIndex*>(loaded.release()));
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

TEST_CASE("mmap: IVF float and SQ lists mapped match ones read into RAM", "[mmap][ivf][file]") {
    TempDir dir;
    const size_t n = 3000, more = 400, dim = 13, nq = 30, k = 10;   // odd dim: SQ4 rows end in half a byte
    const auto data = random_vectors(n + more, dim, 41);
    const auto queries = random_vectors(nq, dim, 42);
    const fs::path file = dir.path / "ivf.qann";
    const int bits = GENERATE(0, 8, 4);
    CAPTURE(bits);
    {
        IVFIndex ivf(dim, 100, 12);
        if (bits) ivf.enable_sq(bits);
        ivf.train(std::span<const float>(data.data(), n * dim), n, 5, 1);
        ivf.add_batch(std::span<const float>(data.data(), n * dim), n);
        save_index(ivf, file);
    }
    auto in_ram = load_ivf(file, false);
    auto mapped = load_ivf(file, true);
    REQUIRE_FALSE(in_ram->mapped());
    REQUIRE(mapped->mapped());
    require_same(*in_ram, *mapped, queries, nq, k);

    // rows added to the RAM tails, removals across mapped and added rows, compaction twice
    auto apply = [&](IVFIndex& index) {
        index.add_batch(std::span<const float>(data.data() + n * dim, more * dim), more);
        std::vector<int64_t> gone;
        for (int64_t id = 0; id < static_cast<int64_t>(n + more); id += 3) gone.push_back(id);
        index.remove(gone);
        index.compact();
        index.add_batch(std::span<const float>(data.data(), 50 * dim), 50);
        index.remove(std::vector<int64_t>{1, 2, 4, 5, 3200});
        index.compact();
    };
    apply(*in_ram);
    apply(*mapped);
    REQUIRE(mapped->mapped());
    require_same(*in_ram, *mapped, queries, nq, k);

    SECTION("saving over the mapped file writes only live rows") {
        save_index(*in_ram, dir.path / "compacted.qann");
        save_index(*mapped, file);
        auto bytes = [](const fs::path& p) {
            std::ifstream f(p, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(f), {});
        };
        REQUIRE(bytes(file) == bytes(dir.path / "compacted.qann"));   // identical to compacting in RAM
        require_same(*in_ram, *mapped, queries, nq, k);
        require_same(*in_ram, *load_ivf(file, true), queries, nq, k);
        require_same(*in_ram, *load_ivf(file, false), queries, nq, k);
    }

    SECTION("compacting away every vector, then adding again") {
        auto all = mapped->ids();
        mapped->remove(all);
        in_ram->remove(all);
        mapped->compact();
        in_ram->compact();
        REQUIRE_FALSE(mapped->mapped());   // no list points into the file any more
        mapped->add_batch(std::span<const float>(data.data(), 200 * dim), 200);
        in_ram->add_batch(std::span<const float>(data.data(), 200 * dim), 200);
        require_same(*in_ram, *mapped, queries, nq, k);
    }
}

TEST_CASE("mmap: an IVF float base under a RefineIndex maps both", "[mmap][ivf][refine][file]") {
    TempDir dir;
    const size_t n = 2000, dim = 16, nq = 20, k = 10;
    const auto data = random_vectors(n, dim, 43);
    const auto queries = random_vectors(nq, dim, 44);
    const fs::path file = dir.path / "refine.qann";
    {
        IVFIndex ivf(dim, 100, 8);
        ivf.train(data, n, 5, 1);
        RefineIndex refine(ivf, 3);
        refine.add_batch(data, n);
        save_index(refine, file);
    }
    auto in_ram = load_refine(file, false);
    auto mapped = load_refine(file, true);
    REQUIRE(mapped->mapped());
    REQUIRE(dynamic_cast<const IVFIndex&>(mapped->base()).mapped());
    require_same(*in_ram, *mapped, queries, nq, k);
}

TEST_CASE("mmap: IVF lists the file can't map are read into RAM", "[mmap][ivf][file]") {
    TempDir dir;
    const size_t n = 1000, dim = 8;
    const auto data = random_vectors(n, dim, 45);
    const fs::path file = dir.path / "pq.qann";
    IVFIndex ivf(dim, 100, 8);
    ivf.enable_pq(4);
    ivf.train(data, n, 5, 1);
    ivf.add_batch(data, n);
    save_index(ivf, file);
    REQUIRE_FALSE(load_ivf(file, true)->mapped());   // PQ codes always stay in RAM

    // before format 11 float lists had no padding; a pre-11 IVF file loads read in
    const fs::path v1 = fs::path(QANN_TEST_DATA_DIR) / "v1_ivf_pq.qann";
    REQUIRE_FALSE(load_ivf(v1, true)->mapped());
}

namespace {

using Row = std::vector<std::pair<uint32_t, float>>;

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

std::vector<Row> sparse_rows(size_t n, uint32_t vocab, size_t nnz, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint32_t> term(0, vocab - 1);
    std::uniform_real_distribution<float> weight(0.1f, 2.0f);
    std::vector<Row> rows(n);
    for (Row& row : rows) {
        std::map<uint32_t, float> terms;
        for (size_t j = 0; j < nnz; ++j) terms[term(rng) % (1 + term(rng))] = weight(rng);  // low terms common
        row.assign(terms.begin(), terms.end());
    }
    return rows;
}

std::unique_ptr<SparseIndex> load_sparse(const fs::path& file, bool mmap) {
    auto loaded = load_index(file, mmap);
    REQUIRE(dynamic_cast<SparseIndex*>(loaded.get()) != nullptr);
    return std::unique_ptr<SparseIndex>(static_cast<SparseIndex*>(loaded.release()));
}

bool sparse_mapped(const SparseIndex& index) {
    if (auto* inv = dynamic_cast<const InvertedIndex*>(&index)) return inv->mapped();
    return dynamic_cast<const SparseApproxIndex&>(index).mapped();
}

void require_same_sparse(const SparseIndex& a, const SparseIndex& b, const std::vector<Row>& queries, size_t k) {
    const Csr q(queries);
    const auto expected = a.query_batch(q.view(), k);
    const auto actual = b.query_batch(q.view(), k);
    for (size_t i = 0; i < queries.size(); ++i) {
        REQUIRE(actual[i].size() == expected[i].size());
        for (size_t j = 0; j < expected[i].size(); ++j) {
            REQUIRE(actual[i][j].index == expected[i][j].index);
            REQUIRE(actual[i][j].distance == expected[i][j].distance);
        }
    }
    REQUIRE(a.ids() == b.ids());
}

std::string file_bytes(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), {});
}

} // namespace

TEST_CASE("mmap: sparse posting lists mapped match ones read into RAM", "[mmap][sparse][file]") {
    TempDir dir;
    const auto docs = sparse_rows(3000, 400, 20, 51);
    const auto queries = sparse_rows(30, 400, 12, 52);
    const fs::path file = dir.path / "sparse.qann";
    const bool approx = GENERATE(false, true);
    CAPTURE(approx);
    {
        std::unique_ptr<SparseIndex> index;
        if (approx) index = std::make_unique<SparseApproxIndex>(true, 6, 40, 5);  // reads only part of each list
        else index = std::make_unique<InvertedIndex>(true);
        std::vector<int64_t> ids(docs.size());
        std::iota(ids.begin(), ids.end(), 500);
        index->add_batch(Csr(docs).view(), ids.data());
        save_index(*index, file);
    }
    auto in_ram = load_sparse(file, false);
    auto mapped = load_sparse(file, true);
    REQUIRE_FALSE(sparse_mapped(*in_ram));
    REQUIRE(sparse_mapped(*mapped));
    require_same_sparse(*in_ram, *mapped, queries, 10);

    SECTION("pruning on and off agree") {
        if (!approx) {
            static_cast<InvertedIndex&>(*mapped).set_pruning(false);
            require_same_sparse(*in_ram, *mapped, queries, 10);
        }
    }
}

TEST_CASE("mmap: adds, removes and compaction on mapped sparse lists", "[mmap][sparse][file]") {
    TempDir dir;
    const auto docs = sparse_rows(3000, 400, 20, 53);
    const auto more = sparse_rows(500, 500, 20, 54);  // also terms the file doesn't have
    const auto queries = sparse_rows(30, 500, 12, 55);
    const fs::path file = dir.path / "sparse.qann";
    const bool approx = GENERATE(false, true);
    CAPTURE(approx);
    {
        // list_depth 0 reads whole lists, so results don't depend on which entries
        // are sorted (a mapped list's new entries stay unsorted until a save)
        std::unique_ptr<SparseIndex> index;
        if (approx) index = std::make_unique<SparseApproxIndex>(false, 0, 0, 1000);
        else index = std::make_unique<InvertedIndex>();
        index->add_batch(Csr(docs).view());
        save_index(*index, file);
    }
    auto in_ram = load_sparse(file, false);
    auto mapped = load_sparse(file, true);

    auto apply = [&](SparseIndex& index) {
        index.add_batch(Csr(more).view());
        std::vector<int64_t> gone;
        for (int64_t id = 0; id < 3500; id += 3) gone.push_back(id);
        index.remove(gone);
        index.compact();
        index.add_batch(Csr(std::vector<Row>(docs.begin(), docs.begin() + 100)).view());
        index.remove(std::vector<int64_t>{1, 2, 4, 3400});
        index.compact();
    };
    apply(*in_ram);
    apply(*mapped);
    REQUIRE(sparse_mapped(*mapped));  // compaction maps a new file
    require_same_sparse(*in_ram, *mapped, queries, 10);

    SECTION("saving over the mapped file, then loading it both ways") {
        save_index(*in_ram, dir.path / "compacted.qann");
        save_index(*mapped, file);
        if (!approx) REQUIRE(file_bytes(file) == file_bytes(dir.path / "compacted.qann"));
        require_same_sparse(*in_ram, *mapped, queries, 10);
        require_same_sparse(*in_ram, *load_sparse(file, true), queries, 10);
        require_same_sparse(*in_ram, *load_sparse(file, false), queries, 10);
    }

    SECTION("compacting away every vector, then adding again") {
        auto all = mapped->ids();
        mapped->remove(all);
        in_ram->remove(all);
        mapped->compact();
        in_ram->compact();
        REQUIRE_FALSE(sparse_mapped(*mapped));
        mapped->add_batch(Csr(more).view());
        in_ram->add_batch(Csr(more).view());
        require_same_sparse(*in_ram, *mapped, queries, 10);
    }

    SECTION("a compaction that can't write its file changes nothing") {
        mapped->remove(std::vector<int64_t>{5, 7, 3401});
        in_ram->remove(std::vector<int64_t>{5, 7, 3401});
        fs::permissions(dir.path, fs::perms::owner_read | fs::perms::owner_exec);
        const bool refused = [&] {
            try {
                mapped->compact();
                return false;
            } catch (const std::runtime_error&) {
                return true;
            }
        }();
        fs::permissions(dir.path, fs::perms::owner_all);
        if (refused) {   // (a root user can write anyway)
            require_same_sparse(*in_ram, *mapped, queries, 10);
            REQUIRE(mapped->size() == in_ram->size());
            mapped->compact();
            in_ram->compact();
            require_same_sparse(*in_ram, *mapped, queries, 10);
        }
    }
}

TEST_CASE("mmap: sparse files before format 11 load read in", "[mmap][sparse][file]") {
    const fs::path dir = QANN_TEST_DATA_DIR;
    for (const char* name : {"v9_sparse.qann", "v9_sparse_approx.qann"}) {
        auto index = load_sparse(dir / name, true);
        REQUIRE_FALSE(sparse_mapped(*index));
    }
}
