#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "vecengine/metadata.hpp"

namespace vecengine {

class IndexBase;
struct SparseRows;

// Durable writes for one index (issue #145, stage 1): a directory holding the
// last checkpoint (an ordinary save_index file) and a write-ahead log of every
// change since it.
//
//   MANIFEST        the current generation g
//   index-g.qann    the checkpoint
//   wal-g.log       records of the writes made after it
//
// A write is applied to the index first and then appended to the log, both
// under the index's exclusive lock, so the log holds exactly the writes that
// succeeded, in order. sync() then makes the record durable (fsync), shared by
// writers that wait at the same time. checkpoint() saves the index as
// generation g + 1 and switches the MANIFEST to it. Opening replays the log on
// top of the checkpoint; a record cut off by a crash, or failing its CRC, ends
// the log there.
//
// Not covered by the log: training, enable_pq / enable_sq, compact() and query
// settings. The bindings checkpoint after the first three; settings are saved
// by the next checkpoint.
class Journal {
public:
    // Starts a store in dir (created if missing, otherwise it must be empty):
    // saves index as generation 1 and opens an empty log. sync_writes: whether
    // sync() fsyncs.
    static std::shared_ptr<Journal> create(const std::filesystem::path& dir, const IndexBase& index, bool sync_writes);

    // Opens the store in dir: loads its checkpoint (mapped with mmap, as
    // load_index does), replays its log, and returns both. Files of other
    // generations, left by a checkpoint a crash interrupted, are deleted.
    // Throws std::runtime_error on a missing or corrupt MANIFEST or checkpoint,
    // or a log record the index rejects.
    static std::pair<std::unique_ptr<IndexBase>, std::shared_ptr<Journal>> open(const std::filesystem::path& dir,
                                                                                bool mmap, bool sync_writes);

    ~Journal();
    Journal(const Journal&) = delete;
    Journal& operator=(const Journal&) = delete;

    // Each log_* appends one record for a write that just succeeded on the
    // index and returns its sequence number for sync(). Call with the index
    // locked exclusively. ids are the ids the write reported (an add's
    // assigned ids). Throws std::runtime_error if the store has failed.
    uint64_t log_add(std::span<const float> vecs, size_t n, std::span<const int64_t> ids, const Metadata& metadata);
    uint64_t log_add_sparse(const SparseRows& rows, std::span<const int64_t> ids, const Metadata& metadata);
    uint64_t log_add_hybrid(std::span<const float> dense, const SparseRows& sparse, std::span<const int64_t> ids,
                            const Metadata& metadata);
    uint64_t log_remove(std::span<const int64_t> ids);
    uint64_t log_update_metadata(std::span<const int64_t> ids, const Metadata& metadata);
    uint64_t log_add_column(const std::string& name, ColumnType type, const std::optional<Expr::Value>& value);
    uint64_t log_add_column(const std::string& name, ColumnType type, std::span<const int64_t> ids,
                            const MetadataColumn::Values& values, std::span<const uint8_t> valid);
    uint64_t log_drop_column(const std::string& name);

    // Throws std::runtime_error if an earlier append or sync failed: the index
    // then holds a write the log may lack, so the store refuses further writes
    // until it is reopened. Call before applying a write.
    void check() const;

    // Returns once record seq is on disk (immediately without sync_writes).
    // Call without the index lock: one waiting writer fsyncs for everyone
    // waiting.
    void sync(uint64_t seq);

    // Whether the log has grown enough to checkpoint: past 64 MiB and a
    // quarter of the checkpoint's size.
    bool checkpoint_due() const;

    // Saves index as the next generation, switches the MANIFEST and the log to
    // it, and deletes the previous generation. Call with the index locked
    // (shared is enough: it blocks writes, which take it exclusively).
    void checkpoint(const IndexBase& index);

    const std::filesystem::path& dir() const noexcept { return dir_; }

private:
    Journal(std::filesystem::path dir, uint64_t gen, bool sync_writes);

    uint64_t append(const std::vector<char>& payload);
    void open_log(bool create);

    std::filesystem::path dir_;
    uint64_t gen_;
    bool sync_writes_;
    int fd_ = -1;                 // the current log, opened for appending
    uint64_t log_bytes_ = 0;      // bytes in the current log
    uint64_t checkpoint_bytes_ = 0;  // size of the current checkpoint file

    mutable std::mutex mu_;       // guards everything below and fd_ / log_bytes_
    std::condition_variable synced_cv_;
    uint64_t appended_ = 0;       // sequence number of the last appended record
    uint64_t synced_ = 0;         // all records up to this one are on disk
    bool syncing_ = false;
    bool failed_ = false;
    std::mutex checkpoint_mu_;
};

} // namespace vecengine
