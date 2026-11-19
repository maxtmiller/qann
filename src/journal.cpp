#include "vecengine/journal.hpp"

#include "vecengine/hybrid_index.hpp"
#include "vecengine/index.hpp"
#include "vecengine/index_factory.hpp"
#include "vecengine/sparse_index.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#if defined(__SSE4_2__)
#  include <nmmintrin.h>
#elif defined(__ARM_FEATURE_CRC32)
#  include <arm_acle.h>
#endif
#include <cerrno>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <type_traits>

namespace vecengine {

namespace fs = std::filesystem;

namespace {

// "QWAL" and "QMAN" read as little-endian uint32s: the log and MANIFEST files.
constexpr uint32_t kLogMagic = 0x4C415751;
constexpr uint32_t kManifestMagic = 0x4E414D51;
constexpr uint32_t kJournalVersion = 1;
constexpr uint64_t kLogHeaderBytes = 16;  // magic, version, generation
constexpr uint64_t kMinCheckpointLog = uint64_t{64} << 20;

// Record types. Values are part of the log format: only append.
enum class Op : uint8_t {
    Add = 1,
    AddSparse = 2,
    AddHybrid = 3,
    Remove = 4,
    UpdateMetadata = 5,
    AddColumn = 6,
    AddColumnValues = 7,
    DropColumn = 8,
};

// CRC-32C (Castagnoli). With the CPU's CRC32C instruction where the build
// enables it (SSE4.2, implied by the x86 build's -mavx2; the ARM CRC extension,
// on by default for Apple arm64), else slicing-by-8 tables. A table read per
// byte cost more than the add itself on 768-d batches.
uint32_t crc32c(const char* data, size_t n) {
    const auto* p = reinterpret_cast<const uint8_t*>(data);
    uint32_t crc = 0xFFFFFFFFu;
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
    for (; n >= 8; n -= 8, p += 8) {
        uint64_t word;
        std::memcpy(&word, p, 8);
#  if defined(__SSE4_2__)
        crc = static_cast<uint32_t>(_mm_crc32_u64(crc, word));
#  else
        crc = __crc32cd(crc, word);
#  endif
    }
    for (; n > 0; --n, ++p) {
#  if defined(__SSE4_2__)
        crc = _mm_crc32_u8(crc, *p);
#  else
        crc = __crc32cb(crc, *p);
#  endif
    }
#else
    static const auto tables = [] {
        std::array<std::array<uint32_t, 256>, 8> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? (c >> 1) ^ 0x82F63B78u : c >> 1;
            t[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; ++i)
            for (int s = 1; s < 8; ++s) t[s][i] = (t[s - 1][i] >> 8) ^ t[0][t[s - 1][i] & 0xFF];
        return t;
    }();
    for (; n >= 8; n -= 8, p += 8) {
        uint64_t word;
        std::memcpy(&word, p, 8);
        word ^= crc;
        crc = tables[7][word & 0xFF] ^ tables[6][(word >> 8) & 0xFF] ^ tables[5][(word >> 16) & 0xFF] ^
              tables[4][(word >> 24) & 0xFF] ^ tables[3][(word >> 32) & 0xFF] ^ tables[2][(word >> 40) & 0xFF] ^
              tables[1][(word >> 48) & 0xFF] ^ tables[0][word >> 56];
    }
    for (; n > 0; --n, ++p) crc = tables[0][(crc ^ *p) & 0xFF] ^ (crc >> 8);
#endif
    return crc ^ 0xFFFFFFFFu;
}

[[noreturn]] void throw_errno(const std::string& what, const fs::path& path) {
    throw std::runtime_error(what + " " + path.string() + ": " + std::strerror(errno));
}

void write_all(int fd, const char* data, size_t n, const fs::path& path) {
    while (n > 0) {
        const ssize_t w = ::write(fd, data, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            throw_errno("cannot write", path);
        }
        data += w;
        n -= static_cast<size_t>(w);
    }
}

int sync_fd(int fd) {
#ifdef __APPLE__
    return ::fsync(fd);
#else
    return ::fdatasync(fd);
#endif
}

void sync_path(const fs::path& path, bool directory) {
    const int fd = ::open(path.c_str(), directory ? O_RDONLY | O_DIRECTORY : O_RDONLY);
    if (fd < 0) throw_errno("cannot open", path);
    const int r = ::fsync(fd);
    ::close(fd);
    if (r != 0) throw_errno("cannot sync", path);
}

fs::path index_path(const fs::path& dir, uint64_t gen) { return dir / ("index-" + std::to_string(gen) + ".qann"); }
fs::path log_path(const fs::path& dir, uint64_t gen) { return dir / ("wal-" + std::to_string(gen) + ".log"); }

// Payload encoding: the same raw little-endian layout as index files.
struct Writer {
    std::vector<char> buf;

    template <class T>
    void pod(const T& v) {
        static_assert(std::is_trivially_copyable_v<T>);
        const char* p = reinterpret_cast<const char*>(&v);
        buf.insert(buf.end(), p, p + sizeof(T));
    }
    template <class T>
    void vec(std::span<const T> v) {
        pod<uint64_t>(v.size());
        const char* p = reinterpret_cast<const char*>(v.data());
        buf.insert(buf.end(), p, p + v.size_bytes());
    }
    void str(const std::string& s) { vec(std::span<const char>(s.data(), s.size())); }

    void values(const MetadataColumn::Values& values) {
        pod<uint8_t>(static_cast<uint8_t>(values.index()));
        if (auto* ints = std::get_if<vector<int64_t>>(&values)) vec<int64_t>(*ints);
        else if (auto* floats = std::get_if<vector<double>>(&values)) vec<double>(*floats);
        else {
            const auto& strings = std::get<vector<std::string>>(values);
            pod<uint64_t>(strings.size());
            for (const auto& s : strings) str(s);
        }
    }
    void metadata(const Metadata& md) {
        pod<uint64_t>(md.size());
        for (const auto& col : md) {
            str(col.name);
            values(col.values);
            vec<uint8_t>(col.valid);
        }
    }
    void sparse(const SparseRows& rows) {
        vec<uint64_t>(rows.indptr);
        vec<uint32_t>(rows.indices);
        vec<float>(rows.values);
    }
};

struct Reader {
    const char* p;
    const char* end;

    template <class T>
    T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        if (static_cast<size_t>(end - p) < sizeof(T)) throw std::runtime_error("corrupt log record");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    template <class T>
    vector<T> vec() {
        const uint64_t n = pod<uint64_t>();
        if (n > static_cast<uint64_t>(end - p) / sizeof(T)) throw std::runtime_error("corrupt log record");
        vector<T> v(n);
        if (n) std::memcpy(v.data(), p, n * sizeof(T));
        p += n * sizeof(T);
        return v;
    }
    std::string str() {
        const vector<char> c = vec<char>();
        return std::string(c.begin(), c.end());
    }
    MetadataColumn::Values values() {
        switch (pod<uint8_t>()) {
            case 0: return vec<int64_t>();
            case 1: return vec<double>();
            case 2: {
                const uint64_t n = pod<uint64_t>();
                if (n > static_cast<uint64_t>(end - p) / sizeof(uint64_t)) throw std::runtime_error("corrupt log record");
                vector<std::string> strings;
                strings.reserve(n);
                for (uint64_t i = 0; i < n; ++i) strings.push_back(str());
                return strings;
            }
            default: throw std::runtime_error("corrupt log record");
        }
    }
    Metadata metadata() {
        const uint64_t n = pod<uint64_t>();
        Metadata md;
        for (uint64_t i = 0; i < n; ++i) {
            std::string name = str();
            MetadataColumn::Values v = values();
            md.emplace_back(std::move(name), std::move(v), vec<uint8_t>());
        }
        return md;
    }
};

struct OwnedSparse {
    vector<uint64_t> indptr;
    vector<uint32_t> indices;
    vector<float> values;

    explicit OwnedSparse(Reader& r) : indptr(r.vec<uint64_t>()), indices(r.vec<uint32_t>()), values(r.vec<float>()) {}
    SparseRows view() const { return {indptr, indices, values}; }
};

ColumnType read_type(Reader& r) {
    const auto t = r.pod<uint8_t>();
    if (t > static_cast<uint8_t>(ColumnType::String)) throw std::runtime_error("corrupt log record");
    return static_cast<ColumnType>(t);
}

template <class T>
T& index_as(IndexBase& index) {
    auto* typed = dynamic_cast<T*>(&index);
    if (!typed) throw std::runtime_error("log record does not match the index type");
    return *typed;
}

void check_ids(const vector<int64_t>& added, const vector<int64_t>& logged) {
    if (added != logged) throw std::runtime_error("log replay assigned different ids than the log recorded");
}

// Applies one record to index, as the original write did.
void apply(IndexBase& index, Reader r) {
    const auto op = static_cast<Op>(r.pod<uint8_t>());
    switch (op) {
        case Op::Add: {
            auto& dense = index_as<DenseIndex>(index);
            const uint64_t n = r.pod<uint64_t>();
            const vector<float> vecs = r.vec<float>();
            const vector<int64_t> ids = r.vec<int64_t>();
            const Metadata md = r.metadata();
            check_ids(dense.add_batch(vecs, n, index.custom_ids() ? ids.data() : nullptr, md), ids);
            break;
        }
        case Op::AddSparse: {
            auto& sparse = index_as<SparseIndex>(index);
            const OwnedSparse rows(r);
            const vector<int64_t> ids = r.vec<int64_t>();
            const Metadata md = r.metadata();
            check_ids(sparse.add_batch(rows.view(), index.custom_ids() ? ids.data() : nullptr, md), ids);
            break;
        }
        case Op::AddHybrid: {
            auto& hybrid = index_as<HybridIndex>(index);
            const vector<float> dense = r.vec<float>();
            const OwnedSparse rows(r);
            const vector<int64_t> ids = r.vec<int64_t>();
            const Metadata md = r.metadata();
            check_ids(hybrid.add_batch(dense, rows.view(), index.custom_ids() ? ids.data() : nullptr, md), ids);
            break;
        }
        case Op::Remove:
            index.remove(r.vec<int64_t>());
            break;
        case Op::UpdateMetadata: {
            const vector<int64_t> ids = r.vec<int64_t>();
            index.update_metadata(ids, r.metadata());
            break;
        }
        case Op::AddColumn: {
            std::string name = r.str();
            const ColumnType type = read_type(r);
            std::optional<Expr::Value> value;
            switch (r.pod<uint8_t>()) {
                case 0: break;
                case 1: value = r.pod<int64_t>(); break;
                case 2: value = r.pod<double>(); break;
                case 3: value = r.str(); break;
                default: throw std::runtime_error("corrupt log record");
            }
            index.add_metadata_column(std::move(name), type, std::move(value));
            break;
        }
        case Op::AddColumnValues: {
            std::string name = r.str();
            const ColumnType type = read_type(r);
            const vector<int64_t> ids = r.vec<int64_t>();
            MetadataColumn::Values values = r.values();
            index.add_metadata_column(std::move(name), type, ids, std::move(values), r.vec<uint8_t>());
            break;
        }
        case Op::DropColumn:
            index.drop_metadata_column(r.str());
            break;
        default:
            throw std::runtime_error("corrupt log record: unknown type");
    }
    if (r.p != r.end) throw std::runtime_error("corrupt log record: unexpected trailing bytes");
}

uint64_t read_manifest(const fs::path& dir) {
    std::ifstream in(dir / "MANIFEST", std::ios::binary);
    if (!in) throw std::runtime_error("no qann store in " + dir.string() + " (MANIFEST missing)");
    char buf[20];
    in.read(buf, sizeof(buf));
    if (!in) throw std::runtime_error("corrupt MANIFEST in " + dir.string());
    uint32_t magic, version, crc;
    uint64_t gen;
    std::memcpy(&magic, buf, 4);
    std::memcpy(&version, buf + 4, 4);
    std::memcpy(&gen, buf + 8, 8);
    std::memcpy(&crc, buf + 16, 4);
    if (magic != kManifestMagic || crc != crc32c(buf, 16)) throw std::runtime_error("corrupt MANIFEST in " + dir.string());
    if (version != kJournalVersion) throw std::runtime_error("unsupported store version in " + dir.string());
    return gen;
}

// Points the MANIFEST at gen: written to a temporary file, synced, renamed
// over the old one, and the directory synced, so a crash leaves either one.
void write_manifest(const fs::path& dir, uint64_t gen) {
    char buf[20];
    std::memcpy(buf, &kManifestMagic, 4);
    std::memcpy(buf + 4, &kJournalVersion, 4);
    std::memcpy(buf + 8, &gen, 8);
    const uint32_t crc = crc32c(buf, 16);
    std::memcpy(buf + 16, &crc, 4);

    const fs::path tmp = dir / "MANIFEST.tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) throw_errno("cannot create", tmp);
    try {
        write_all(fd, buf, sizeof(buf), tmp);
        if (::fsync(fd) != 0) throw_errno("cannot sync", tmp);
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
    fs::rename(tmp, dir / "MANIFEST");
    sync_path(dir, true);
}

// Saves index as generation gen's checkpoint, synced to disk.
void write_checkpoint(const fs::path& dir, uint64_t gen, const IndexBase& index) {
    const fs::path path = index_path(dir, gen);
    save_index(index, path);
    sync_path(path, false);
}

// Replays the log at path onto index. A record that is cut short or fails its
// CRC ends the log: the file is truncated there, since only a crash during an
// append leaves one. Returns the log's size afterwards.
uint64_t replay(const fs::path& path, uint64_t gen, IndexBase& index) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    char header[kLogHeaderBytes];
    in.read(header, sizeof(header));
    uint32_t magic, version;
    uint64_t file_gen;
    std::memcpy(&magic, header, 4);
    std::memcpy(&version, header + 4, 4);
    std::memcpy(&file_gen, header + 8, 8);
    if (!in || magic != kLogMagic || version != kJournalVersion || file_gen != gen)
        throw std::runtime_error("corrupt log " + path.string());

    uint64_t good = kLogHeaderBytes;
    vector<char> payload;
    while (true) {
        char frame[8];
        in.read(frame, sizeof(frame));
        if (!in) break;
        uint32_t len, crc;
        std::memcpy(&len, frame, 4);
        std::memcpy(&crc, frame + 4, 4);
        payload.resize(len);
        in.read(payload.data(), len);
        if (!in || crc32c(payload.data(), len) != crc) break;
        apply(index, Reader{payload.data(), payload.data() + len});
        good += sizeof(frame) + len;
    }
    in.close();
    if (fs::file_size(path) != good) fs::resize_file(path, good);
    return good;
}

void remove_other_generations(const fs::path& dir, uint64_t gen) {
    const std::string keep_index = index_path(dir, gen).filename().string();
    const std::string keep_log = log_path(dir, gen).filename().string();
    for (const auto& entry : fs::directory_iterator(dir)) {
        const std::string name = entry.path().filename().string();
        const bool ours = (name.starts_with("index-") && (name.ends_with(".qann") || name.ends_with(".qann.tmp"))) ||
                          (name.starts_with("wal-") && name.ends_with(".log")) || name == "MANIFEST.tmp";
        if (ours && name != keep_index && name != keep_log) {
            std::error_code ignored;
            fs::remove(entry.path(), ignored);
        }
    }
}

} // namespace

Journal::Journal(fs::path dir, uint64_t gen, bool sync_writes)
    : dir_(std::move(dir)), gen_(gen), sync_writes_(sync_writes) {}

Journal::~Journal() {
    if (fd_ >= 0) {
        if (sync_writes_) sync_fd(fd_);
        ::close(fd_);
    }
}

// Opens the current generation's log for appending; with create, makes it
// first, holding just its header, synced.
void Journal::open_log(bool create) {
    const fs::path path = log_path(dir_, gen_);
    const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND | (create ? O_CREAT | O_EXCL : 0), 0644);
    if (fd < 0) throw_errno("cannot open", path);
    if (create) {
        char header[kLogHeaderBytes];
        std::memcpy(header, &kLogMagic, 4);
        std::memcpy(header + 4, &kJournalVersion, 4);
        std::memcpy(header + 8, &gen_, 8);
        try {
            write_all(fd, header, sizeof(header), path);
            if (::fsync(fd) != 0) throw_errno("cannot sync", path);
        } catch (...) {
            ::close(fd);
            throw;
        }
        log_bytes_ = kLogHeaderBytes;
    }
    fd_ = fd;
}

std::shared_ptr<Journal> Journal::create(const fs::path& dir, const IndexBase& index, bool sync_writes) {
    fs::create_directories(dir);
    if (!fs::is_empty(dir)) throw std::invalid_argument("cannot persist to " + dir.string() + ": the directory is not empty");
    std::shared_ptr<Journal> j(new Journal(dir, 1, sync_writes));
    write_checkpoint(dir, 1, index);
    j->open_log(true);
    write_manifest(dir, 1);
    j->checkpoint_bytes_ = fs::file_size(index_path(dir, 1));
    return j;
}

std::pair<std::unique_ptr<IndexBase>, std::shared_ptr<Journal>> Journal::open(const fs::path& dir, bool mmap,
                                                                              bool sync_writes) {
    const uint64_t gen = read_manifest(dir);
    remove_other_generations(dir, gen);
    std::unique_ptr<IndexBase> index = load_index(index_path(dir, gen), mmap);
    std::shared_ptr<Journal> j(new Journal(dir, gen, sync_writes));
    try {
        j->log_bytes_ = replay(log_path(dir, gen), gen, *index);
    } catch (const std::invalid_argument& e) {
        throw std::runtime_error(std::string("log replay failed: ") + e.what());
    }
    j->open_log(false);
    j->checkpoint_bytes_ = fs::file_size(index_path(dir, gen));
    return {std::move(index), std::move(j)};
}

void Journal::check() const {
    std::lock_guard lk(mu_);
    if (failed_)
        throw std::runtime_error("the store at " + dir_.string() + " failed to write its log; reopen it with qann.open");
}

uint64_t Journal::append(const std::vector<char>& payload) {
    std::lock_guard lk(mu_);
    if (failed_) throw std::runtime_error("the store at " + dir_.string() + " failed to write its log; reopen it with qann.open");
    if (payload.size() > UINT32_MAX) {
        failed_ = true;
        throw std::runtime_error("write too large for one log record (4 GiB); add in smaller batches");
    }
    const uint32_t len = static_cast<uint32_t>(payload.size());
    const uint32_t crc = crc32c(payload.data(), payload.size());
    char frame[8];
    std::memcpy(frame, &len, 4);
    std::memcpy(frame + 4, &crc, 4);
    try {
        const fs::path path = log_path(dir_, gen_);
        write_all(fd_, frame, sizeof(frame), path);
        write_all(fd_, payload.data(), payload.size(), path);
    } catch (...) {
        failed_ = true;
        throw;
    }
    log_bytes_ += sizeof(frame) + payload.size();
    return ++appended_;
}

void Journal::sync(uint64_t seq) {
    if (!sync_writes_) return;
    std::unique_lock lk(mu_);
    while (synced_ < seq) {
        if (failed_) throw std::runtime_error("the store at " + dir_.string() + " failed to sync its log; reopen it with qann.open");
        if (syncing_) {
            synced_cv_.wait(lk);
            continue;
        }
        syncing_ = true;
        const uint64_t target = appended_;
        const int fd = fd_;
        lk.unlock();
        const int r = sync_fd(fd);
        lk.lock();
        syncing_ = false;
        if (r != 0) failed_ = true;
        else synced_ = std::max(synced_, target);
        synced_cv_.notify_all();
    }
}

bool Journal::checkpoint_due() const {
    std::lock_guard lk(mu_);
    return log_bytes_ > kMinCheckpointLog && log_bytes_ > checkpoint_bytes_ / 4;
}

void Journal::checkpoint(const IndexBase& index) {
    std::lock_guard ck(checkpoint_mu_);
    check();
    const uint64_t next = gen_ + 1;
    const uint64_t prev = gen_;
    try {
        write_checkpoint(dir_, next, index);
        const fs::path next_log = log_path(dir_, next);
        const int fd = ::open(next_log.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_EXCL, 0644);
        if (fd < 0) throw_errno("cannot create", next_log);
        char header[kLogHeaderBytes];
        std::memcpy(header, &kLogMagic, 4);
        std::memcpy(header + 4, &kJournalVersion, 4);
        std::memcpy(header + 8, &next, 8);
        try {
            write_all(fd, header, sizeof(header), next_log);
            if (::fsync(fd) != 0) throw_errno("cannot sync", next_log);
            write_manifest(dir_, next);
        } catch (...) {
            ::close(fd);
            throw;
        }
        // The MANIFEST now names the new generation: switch to its log. A sync
        // in flight still uses the old descriptor, so wait for it first.
        std::unique_lock lk(mu_);
        synced_cv_.wait(lk, [&] { return !syncing_; });
        ::close(fd_);
        fd_ = fd;
        gen_ = next;
        log_bytes_ = kLogHeaderBytes;
        synced_ = appended_;  // the checkpoint holds every write so far, synced
        synced_cv_.notify_all();
    } catch (...) {
        std::error_code ignored;
        if (gen_ == prev) {
            fs::remove(index_path(dir_, next), ignored);
            fs::remove(log_path(dir_, next), ignored);
        }
        throw;
    }
    const uint64_t size = fs::file_size(index_path(dir_, next));
    {
        std::lock_guard lk(mu_);
        checkpoint_bytes_ = size;
    }
    std::error_code ignored;
    fs::remove(index_path(dir_, prev), ignored);
    fs::remove(log_path(dir_, prev), ignored);
}

uint64_t Journal::log_add(std::span<const float> vecs, size_t n, std::span<const int64_t> ids, const Metadata& metadata) {
    Writer w;
    w.buf.reserve(64 + vecs.size_bytes() + ids.size_bytes());
    w.pod(Op::Add);
    w.pod<uint64_t>(n);
    w.vec(vecs);
    w.vec(ids);
    w.metadata(metadata);
    return append(w.buf);
}

uint64_t Journal::log_add_sparse(const SparseRows& rows, std::span<const int64_t> ids, const Metadata& metadata) {
    Writer w;
    w.pod(Op::AddSparse);
    w.sparse(rows);
    w.vec(ids);
    w.metadata(metadata);
    return append(w.buf);
}

uint64_t Journal::log_add_hybrid(std::span<const float> dense, const SparseRows& sparse, std::span<const int64_t> ids,
                                 const Metadata& metadata) {
    Writer w;
    w.buf.reserve(64 + dense.size_bytes() + sparse.indices.size_bytes() * 2 + ids.size_bytes());
    w.pod(Op::AddHybrid);
    w.vec(dense);
    w.sparse(sparse);
    w.vec(ids);
    w.metadata(metadata);
    return append(w.buf);
}

uint64_t Journal::log_remove(std::span<const int64_t> ids) {
    Writer w;
    w.pod(Op::Remove);
    w.vec(ids);
    return append(w.buf);
}

uint64_t Journal::log_update_metadata(std::span<const int64_t> ids, const Metadata& metadata) {
    Writer w;
    w.pod(Op::UpdateMetadata);
    w.vec(ids);
    w.metadata(metadata);
    return append(w.buf);
}

uint64_t Journal::log_add_column(const std::string& name, ColumnType type, const std::optional<Expr::Value>& value) {
    Writer w;
    w.pod(Op::AddColumn);
    w.str(name);
    w.pod(static_cast<uint8_t>(type));
    if (!value) w.pod<uint8_t>(0);
    else if (auto* i = std::get_if<int64_t>(&*value)) w.pod<uint8_t>(1), w.pod(*i);
    else if (auto* d = std::get_if<double>(&*value)) w.pod<uint8_t>(2), w.pod(*d);
    else w.pod<uint8_t>(3), w.str(std::get<std::string>(*value));
    return append(w.buf);
}

uint64_t Journal::log_add_column(const std::string& name, ColumnType type, std::span<const int64_t> ids,
                                 const MetadataColumn::Values& values, std::span<const uint8_t> valid) {
    Writer w;
    w.pod(Op::AddColumnValues);
    w.str(name);
    w.pod(static_cast<uint8_t>(type));
    w.vec(ids);
    w.values(values);
    w.vec(valid);
    return append(w.buf);
}

uint64_t Journal::log_drop_column(const std::string& name) {
    Writer w;
    w.pod(Op::DropColumn);
    w.str(name);
    return append(w.buf);
}

} // namespace vecengine
