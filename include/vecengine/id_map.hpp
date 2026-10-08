#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <span>
#include <unordered_map>
#include <vector>

namespace vecengine::detail {

using std::size_t;
using std::span;
using std::vector;

// Maps between slots and the ids users see, and tracks deleted slots.
//
// A slot is a vector's position in insertion order (0, 1, 2, ...); it is what
// indexes store internally and never changes. With custom ids off, a slot's
// id is the slot itself. With custom ids on, every add supplies one int64 id
// (>= 0) per vector. Deleting only marks a slot; its storage stays until a
// future compaction, and queries must skip it.
class IdMap {
public:
    // An empty map. custom_ids fixes whether add() takes user ids.
    explicit IdMap(bool custom_ids = false) : custom_ids_(custom_ids) {}

    // Registers n new slots. ids must point to n ids when custom ids are on and
    // be nullptr when they are off. Throws std::invalid_argument on a mode
    // mismatch, a negative id (-1 marks "no result" in padded query output) or
    // a duplicate id (within ids, or against an id that is still live) before
    // changing anything. A deleted id may be added again.
    void add(size_t n, const int64_t* ids);

    // Marks the slots of the given ids deleted. Unknown and already deleted ids
    // are skipped. Each newly deleted slot is appended to removed_slots when it
    // is not nullptr (RefineIndex uses this to delete the same slots in its
    // base). Returns how many ids were deleted.
    size_t remove(span<const int64_t> ids, vector<uint32_t>* removed_slots = nullptr);

    // Runs add()'s validation without changing anything, so a caller that must
    // keep two indexes in step (RefineIndex and its base) can reject bad ids
    // before touching either.
    void check(size_t n, const int64_t* ids) const;

    // The id reported for a slot. Called for every query result, so it stays
    // inline.
    int64_t label(uint32_t slot) const noexcept {
        if (custom_ids_) return labels_[slot];
        return slot;
    }

    // Whether a slot was deleted. Called for every candidate a query scans, so
    // it stays inline.
    bool is_deleted(uint32_t slot) const noexcept {
        return deleted_[slot];
    }

    // Whether add() takes user ids.
    bool custom_ids() const noexcept { return custom_ids_; }
    size_t slots() const noexcept { return slots_; }                  // ever added, including deleted
    size_t live() const noexcept { return slots_ - deleted_count_; }  // what Index::size() reports

    // Writes the custom_ids flag, the labels (when custom ids are on) and the
    // deleted flags. No header: it is part of each index's body.
    void save(std::ostream& out) const;

    // Reads what save() wrote for an index with `slots` slots. Throws
    // std::runtime_error unless the labels are empty (custom ids off) or exactly
    // `slots` long with no duplicates among live slots, and the deleted flags
    // are exactly `slots` long.
    static IdMap load(std::istream& in, uint64_t slots);

private:
    bool custom_ids_;
    size_t slots_ = 0;
    size_t deleted_count_ = 0;
    vector<int64_t> labels_;                    // slot -> id; empty unless custom_ids_
    std::unordered_map<int64_t, uint32_t> slot_of_;  // live id -> slot; empty unless custom_ids_
    vector<bool> deleted_;                      // one flag per slot
};

} // namespace vecengine::detail
