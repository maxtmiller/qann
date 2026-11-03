#pragma once

// Format-11 posting lists, shared by InvertedIndex and SparseApproxIndex: a directory (per
// list, in increasing term order: the term, its length and the index's own fields), then every
// list's entries as one block at a 64-byte file offset, so a load with mmap can point into it.

#include "vecengine/sparse_index.hpp"
#include "vecengine/vector_store.hpp"
#include "serialize.hpp"

#include <algorithm>
#include <cstdint>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace vecengine::detail {

inline constexpr uint64_t kPostingAlign = 64;

// The terms of lists in increasing order, so the same index always writes the same bytes.
template <class List>
std::vector<uint32_t> sorted_terms(const std::unordered_map<uint32_t, List>& lists) {
    std::vector<uint32_t> terms;
    terms.reserve(lists.size());
    for (const auto& [term, list] : lists) terms.push_back(term);
    std::sort(terms.begin(), terms.end());
    return terms;
}

// Writes the bytes of trivially copyable values.
template <class T>
void write_span(std::ostream& out, std::span<const T> values) {
    if (!values.empty()) out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size_bytes()));
}

// Writes lists as described above; extra(out, list) writes a list's own directory fields.
template <class List, class Extra>
void save_posting_lists(std::ostream& out, const std::unordered_map<uint32_t, List>& lists, Extra extra) {
    const std::vector<uint32_t> terms = sorted_terms(lists);
    write_pod(out, static_cast<uint64_t>(terms.size()));
    uint64_t total = 0;
    for (uint32_t term : terms) {
        const List& list = lists.at(term);
        write_pod(out, term);
        write_pod(out, static_cast<uint64_t>(list.size()));
        extra(out, list);
        total += list.size();
    }
    write_pod(out, total);
    write_pad(out, kPostingAlign);
    for (uint32_t term : terms) {
        const List& list = lists.at(term);
        write_span(out, list.mapped);
        write_span(out, std::span<const SparsePosting>(list.postings));
    }
}

// Reads what save_posting_lists wrote into lists (empty), for an index of `count` slots; a list
// holds a slot at most once, so a length over count is corrupt. read_extra(in, list) reads a
// list's own fields. With ctx->file the entries are mapped and file is set; they are read
// otherwise. The caller still checks every list's entries. Throws std::runtime_error on corrupt data.
template <class List, class ReadExtra>
void load_posting_lists(std::istream& in, uint64_t count, const LoadContext* ctx,
                        std::unordered_map<uint32_t, List>& lists, std::shared_ptr<const MappedFile>& file,
                        ReadExtra read_extra) {
    const auto nlists = read_pod<uint64_t>(in);
    std::vector<std::pair<List*, uint64_t>> order;  // in file order; grows as entries are read, so a corrupt nlists runs out of file
    uint64_t total = 0;
    for (uint64_t l = 0; l < nlists; ++l) {
        const auto term = read_pod<uint32_t>(in);
        const auto length = read_pod<uint64_t>(in);
        if (length == 0 || length > count) throw std::runtime_error("corrupt file: invalid sparse posting list length");
        auto [it, added] = lists.try_emplace(term);
        if (!added) throw std::runtime_error("corrupt file: sparse posting list term appears twice");
        read_extra(in, it->second);
        order.emplace_back(&it->second, length);
        total += length;
        if (total > UINT64_MAX / sizeof(SparsePosting)) throw std::runtime_error("corrupt file: invalid sparse posting count");
    }
    if (read_pod<uint64_t>(in) != total) throw std::runtime_error("corrupt file: sparse posting count mismatch");
    read_pad(in, kPostingAlign);

    if (ctx && ctx->file && total > 0) {
        const auto* at = reinterpret_cast<const SparsePosting*>(
            map_next(in, *ctx->file, total * sizeof(SparsePosting), alignof(SparsePosting)));
        for (auto [list, length] : order) {
            list->mapped = std::span<const SparsePosting>(at, length);
            at += length;
        }
        file = ctx->file;
        return;
    }
    for (auto [list, length] : order) {
        list->postings.resize(length);
        in.read(reinterpret_cast<char*>(list->postings.data()), static_cast<std::streamsize>(length * sizeof(SparsePosting)));
        if (!in) throw std::runtime_error("truncated file");
    }
}

} // namespace vecengine::detail
