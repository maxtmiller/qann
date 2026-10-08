#include "vecengine/id_map.hpp"
#include "serialize.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>

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
        deleted_.resize(slots_ + n, false);
        slots_ += n;
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
        } else {
            if (id < 0 || static_cast<uint64_t>(id) >= slots_ || deleted_[id]) continue;
            slot = static_cast<uint32_t>(id);
        }

        deleted_[slot] = true;
        ++deleted_count_;
        ++counter;

        if (removed_slots) removed_slots->emplace_back(slot);
    }

    return counter;
}

void IdMap::save(std::ostream& out) const {
    write_pod(out, static_cast<uint8_t>(custom_ids_));
    write_pod(out, static_cast<uint64_t>(slots_));
    write_vec(out, labels_);

    vector<uint8_t> deleted(deleted_.begin(), deleted_.end());
    write_vec(out, deleted);
}

IdMap IdMap::load(std::istream& in, uint64_t slots) {
    const auto custom_ids = read_pod<uint8_t>(in);
    const auto slot_num = read_pod<uint64_t>(in);

    auto labels = read_vec<int64_t>(in, slot_num);
    if (slots != slot_num || slots > UINT32_MAX) throw std::runtime_error("corrupt file: id map slot count mismatch");
    if (custom_ids > 1 || labels.size() != (custom_ids ? slots : 0))
        throw std::runtime_error("corrupt file: invalid id map shape");

    auto deleted = read_vec<uint8_t>(in, slot_num);
    if (deleted.size() != slots) throw std::runtime_error("corrupt file: invalid deleted flags length");

    for (size_t i = 0; i < deleted.size(); ++i) {
        if (deleted[i] != 0 && deleted[i] != 1) throw std::runtime_error("corrupt file: invalid deleted flag value");
    }

    IdMap map(custom_ids == 1);
    map.slots_ = slots;
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
