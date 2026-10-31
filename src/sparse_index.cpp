#include "vecengine/sparse_index.hpp"
#include "vecengine/id_map.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <stdexcept>

namespace vecengine {

namespace {

// Throws std::invalid_argument unless rows is well-formed CSR (see SparseRows).
void check_rows(const SparseRows& rows) {
    if (rows.indptr.empty()) throw std::invalid_argument("sparse rows need indptr with at least one entry");
    if (rows.indptr[0] != 0) throw std::invalid_argument("sparse indptr must start at 0");
    for (size_t i = 0; i < rows.rows(); ++i)
        if (rows.indptr[i + 1] < rows.indptr[i]) throw std::invalid_argument("sparse indptr must be non-decreasing");
    if (rows.indptr.back() != rows.indices.size() || rows.values.size() != rows.indices.size())
        throw std::invalid_argument("sparse indptr, indices and values must agree in length");
}

} // namespace

void SparseIndex::canonical_row(const SparseRows& rows, size_t i, Terms& out) {
    out.clear();
    for (uint64_t j = rows.indptr[i]; j < rows.indptr[i + 1]; ++j) out.emplace_back(rows.indices[j], rows.values[j]);
    std::stable_sort(out.begin(), out.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    size_t w = 0;
    for (size_t r = 0; r < out.size(); ++r) {
        if (w > 0 && out[w - 1].first == out[r].first) out[w - 1].second += out[r].second;
        else out[w++] = out[r];
    }
    out.resize(w);
    std::erase_if(out, [](const auto& t) { return t.second == 0.0f; });
}

vector<int64_t> SparseIndex::add_batch(const SparseRows& rows, const int64_t* ids, const Metadata& metadata) {
    check_rows(rows);
    const size_t n = rows.rows();
    // validate metadata first and append it only once the vectors are in, so a
    // throw anywhere leaves both unchanged
    metadata_.check(n, metadata);
    const size_t first = slots();
    add_batch_impl(rows, ids);
    metadata_.append(first, n, metadata);

    vector<int64_t> added(n);
    for (size_t i = 0; i < n; ++i) added[i] = id_map().label(static_cast<uint32_t>(first + i));
    return added;
}

vector<vector<Neighbor>> SparseIndex::query_batch(const SparseRows& queries, size_t k) const {
    return search(queries, k, nullptr);
}

vector<vector<Neighbor>> SparseIndex::query_batch(const SparseRows& queries, size_t k, span<const int64_t> allowed) const {
    const detail::SlotFilter filter = id_map().filter(allowed);
    return search(queries, k, &filter);
}

vector<vector<Neighbor>> SparseIndex::query_batch(const SparseRows& queries, size_t k, const Expr& where) const {
    const detail::SlotFilter filter = where_filter(where, nullptr);
    return search(queries, k, &filter);
}

vector<vector<Neighbor>> SparseIndex::query_batch(const SparseRows& queries, size_t k, const Expr& where,
                                                  span<const int64_t> allowed) const {
    const detail::SlotFilter filter = where_filter(where, &allowed);
    return search(queries, k, &filter);
}

vector<vector<Neighbor>> SparseIndex::search(const SparseRows& queries, size_t k, const detail::SlotFilter* filter) const {
    check_rows(queries);
    const size_t n = queries.rows();
    vector<vector<Neighbor>> results(n);
    if (k == 0 || (filter && filter->count == 0)) return results;

    detail::parallel_for(n, [&](size_t i) {
        thread_local Terms terms;
        canonical_row(queries, i, terms);
        results[i] = query_impl(terms, k, filter);
    });
    return results;
}

} // namespace vecengine
