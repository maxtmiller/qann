#pragma once

#include <bit>
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

// The live slots a filtered query may return: one bit per slot. Built once
// per query call by IdMap::filter(), so deleted slots are already left out
// and scans test only this.
struct SlotFilter {
    vector<uint64_t> words;  // bit s % 64 of words[s / 64] is set when slot s is allowed
    size_t count = 0;        // set bits

    // Whether a query may return this slot. Called per candidate, so inline.
    bool test(uint32_t slot) const noexcept { return (words[slot >> 6] >> (slot & 63)) & 1; }

    // Calls f(slot) for every allowed slot, in increasing order.
    template <class F>
    void for_each(F&& f) const {
        for (size_t w = 0; w < words.size(); ++w) {
            for (uint64_t bits = words[w]; bits != 0; bits &= bits - 1)
                f(static_cast<uint32_t>(w * 64 + static_cast<size_t>(std::countr_zero(bits))));
        }
    }
};

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

    // The slots of the live vectors among ids. Unknown, deleted and repeated
    // ids are ignored.
    SlotFilter filter(span<const int64_t> ids) const;

    // The slot of each id, in order. Throws std::invalid_argument on the first
    // unknown, deleted or repeated id.
    vector<uint32_t> slots_of(span<const int64_t> ids) const;

    // A filter from raw bits (one per slot, as MetadataStore::evaluate makes
    // them) with deleted slots cleared and the set bits counted.
    SlotFilter live(vector<uint64_t> words) const;

    // Runs add()'s validation without changing anything, so a caller that must
    // keep two indexes in step (RefineIndex and its base) can reject bad ids
    // before touching either.
    void check(size_t n, const int64_t* ids) const;

    // The id reported for a slot. Called for every query result, so it stays
    // inline.
    int64_t label(uint32_t slot) const noexcept {
        return labels_.empty() ? slot : labels_[slot];
    }

    // Whether a slot was deleted. Called for every candidate a query scans, so
    // it stays inline.
    bool is_deleted(uint32_t slot) const noexcept {
        return deleted_[slot];
    }

    // constant for determining if a vector was moved
    static constexpr uint32_t kRemoved = UINT32_MAX;

    // reassigns slots to ids to compact vector store 
    vector<uint32_t> compact(bool keep_ids);

    // Whether every slot's id is the slot itself, which a RefineIndex base
    // needs: auto ids, never compacted with keep_ids, no gaps in the ids.
    bool ids_are_slots() const noexcept {
        return !custom_ids_ && labels_.empty() && static_cast<size_t>(next_id_) == slots_;
    }

    // Whether add() takes user ids.
    bool custom_ids() const noexcept { return custom_ids_; }
    size_t slots() const noexcept { return slots_; }                  // current slots, including deleted
    size_t live() const noexcept { return slots_ - deleted_count_; }  // what IndexBase::size() reports

    // Writes the custom_ids flag, the slot count, the next auto id, the labels
    // (when ids differ from slots) and the deleted flags. No header: it is part
    // of each index's body.
    void save(std::ostream& out) const;

    // Reads what save() wrote, in format `version`, for an index with `slots`
    // slots. Version 2 has no next auto id (it equals the slot count). Throws
    // std::runtime_error unless the deleted flags are exactly `slots` long and
    // the labels are either `slots` long (custom ids: no duplicates among live
    // slots; auto ids: strictly increasing, >= 0 and below the next id) or, for
    // auto ids, empty with ids equal to slots.
    static IdMap load(std::istream& in, uint64_t slots, uint32_t version);

private:
    bool custom_ids_;
    size_t slots_ = 0;
    int64_t next_id_ = 0; // for auto ids next id
    size_t deleted_count_ = 0;
    vector<int64_t> labels_; // slot -> id; empty while ids equal slots
    std::unordered_map<int64_t, uint32_t> slot_of_;  // live id -> slot; empty unless custom_ids_
    vector<bool> deleted_; // one flag per slot
};

} // namespace vecengine::detail
