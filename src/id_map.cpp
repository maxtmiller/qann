#include "vecengine/id_map.hpp"
#include "serialize.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <numeric>
#include <algorithm>

namespace vecengine::detail {

void IdMap::check(size_t n, const int64_t* ids) const {
    if (n > UINT32_MAX - slots_) throw std::invalid_argument("an index holds at most 2^32 - 1 vectors");
    if (custom_ids_ && ids == nullptr)
        throw std::invalid_argument("ids are required: this index was created with custom ids");
    if (!custom_ids_ && ids != nullptr)
        throw std::invalid_argument("ids are not accepted: this index was created without custom ids");
    if (!custom_ids_) return;

    std::unordered_set<int64_t> seen;
    for (const int64_t* ptr = ids; ptr < ids + n; ++ptr) {
        const int64_t id = *ptr;
        if (id < 0) throw std::invalid_argument("ids must be >= 0, got " + std::to_string(id));
        if (!seen.insert(id).second) throw std::invalid_argument("duplicate id " + std::to_string(id) + " in this batch");
        if (slot_of_.count(id) != 0) throw std::invalid_argument("id " + std::to_string(id) + " is already in the index");
    }
}

void IdMap::add(size_t n, const int64_t* ids) {
    check(n, ids);
    if (!custom_ids_) {
        if (labels_.empty() && static_cast<size_t>(next_id_) == slots_) {
            deleted_.resize(slots_ + n, false);
            slots_ += n;
            next_id_ += n;
        } else {
            for (size_t i = 0; i < n; ++i) {
                labels_.emplace_back(next_id_++);
                ++slots_;
                deleted_.emplace_back(false);
            }
        }
        return;
    }

    for (const int64_t* ptr = ids; ptr < ids + n; ++ptr) {
        int64_t id = *ptr;
        labels_.emplace_back(id);
        slot_of_[id] = static_cast<uint32_t>(slots_);
        deleted_.emplace_back(false);
        ++slots_;
    }
}

size_t IdMap::remove(span<const int64_t> ids, vector<uint32_t>* removed_slots) {
    size_t counter = 0;

    for (int64_t id : ids) {
        uint32_t slot;
        if (custom_ids_) {
            auto it = slot_of_.find(id);
            if (it == slot_of_.end()) continue;

            slot = it->second;
            slot_of_.erase(it);
        } else if (labels_.empty()) {
            if (id < 0 || static_cast<uint64_t>(id) >= slots_ || deleted_[id]) continue;
            slot = static_cast<uint32_t>(id);
        } else {
            auto it = std::lower_bound(labels_.begin(), labels_.end(), id);
            if (it == labels_.end() || *it != id) continue;
            slot = static_cast<uint32_t>(it - labels_.begin());
            if (deleted_[slot]) continue;
        }

        deleted_[slot] = true;
        ++deleted_count_;
        ++counter;

        if (removed_slots) removed_slots->emplace_back(slot);
    }

    return counter;
}

vector<uint32_t> IdMap::compact(bool keep_ids) {
    if (deleted_count_ == 0) return {};

    vector<uint32_t> remap(slots_, kRemoved);
    uint32_t next = 0;
    for (size_t old = 0; old < slots_; ++old)
        if (!deleted_[old]) remap[old] = next++;

    if (keep_ids && !custom_ids_ && labels_.empty()) {
        labels_.resize(slots_);
        std::iota(labels_.begin(), labels_.end(), int64_t{0});
    }
    if (!labels_.empty()) {
        for (size_t old = 0; old < slots_; ++old)
            if (remap[old] != kRemoved) labels_[remap[old]] = labels_[old];
        labels_.resize(next);
    }
    for (auto& [id, slot] : slot_of_) slot = remap[slot];

    if (!keep_ids) next_id_ = next;
    deleted_.assign(next, false);
    slots_ = next;
    deleted_count_ = 0;

    return remap;

}

void IdMap::save(std::ostream& out) const {
    write_pod(out, static_cast<uint8_t>(custom_ids_));
    write_pod(out, static_cast<uint64_t>(slots_));
    write_pod(out, next_id_);
    write_vec(out, labels_);

    vector<uint8_t> deleted(deleted_.begin(), deleted_.end());
    write_vec(out, deleted);
}

IdMap IdMap::load(std::istream& in, uint64_t slots, uint32_t version) {
    const auto custom_ids = read_pod<uint8_t>(in);
    const auto slot_num = read_pod<uint64_t>(in);
    const auto next_id = version >= 3 ? read_pod<int64_t>(in) : static_cast<int64_t>(slot_num);

    auto labels = read_vec<int64_t>(in, slot_num);
    if (slots != slot_num || slots > UINT32_MAX) throw std::runtime_error("corrupt file: id map slot count mismatch");
    if (custom_ids > 1 || next_id < 0) throw std::runtime_error("corrupt file: invalid id map shape");
    if (custom_ids) {
        if (labels.size() != slots) throw std::runtime_error("corrupt file: invalid id map shape");
    } else if (labels.empty()) {
        // ids equal slots; an index compacted down to nothing keeps its next id
        if (next_id != static_cast<int64_t>(slots) && slots != 0) throw std::runtime_error("corrupt file: invalid id map shape");
    } else {
        if (version < 3 || labels.size() != slots || labels[0] < 0 || labels.back() >= next_id)
            throw std::runtime_error("corrupt file: invalid id map shape");
        for (size_t i = 1; i < labels.size(); ++i)
            if (labels[i] <= labels[i - 1]) throw std::runtime_error("corrupt file: auto ids out of order");
    }

    auto deleted = read_vec<uint8_t>(in, slot_num);
    if (deleted.size() != slots) throw std::runtime_error("corrupt file: invalid deleted flags length");

    for (size_t i = 0; i < deleted.size(); ++i) {
        if (deleted[i] != 0 && deleted[i] != 1) throw std::runtime_error("corrupt file: invalid deleted flag value");
    }

    IdMap map(custom_ids == 1);
    map.slots_ = slots;
    map.next_id_ = next_id;
    map.labels_ = std::move(labels);
    map.deleted_.assign(deleted.begin(), deleted.end());

    for (uint64_t s = 0; s < slots; ++s) {
        if (map.deleted_[s]) { ++map.deleted_count_; continue; }
        if (custom_ids && !map.slot_of_.emplace(map.labels_[s], static_cast<uint32_t>(s)).second)
            throw std::runtime_error("corrupt file: duplicate live id");
    }

    return map;
}

} // namespace vecengine::detail
