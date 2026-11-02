#include "vecengine/fusion.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace vecengine {

static void check_args(size_t num_lists, span<const float> weights, float k) {
    if (num_lists == 0) throw std::invalid_argument("fusion needs at least one result list");
    if (!weights.empty() && weights.size() != num_lists)
        throw std::invalid_argument("weights must have one value per result list (" + std::to_string(num_lists) +
                                    "), got " + std::to_string(weights.size()));
    for (float w : weights)
        if (!std::isfinite(w) || w < 0.0f) throw std::invalid_argument("weights must be finite and >= 0");
    if (!std::isfinite(k) || k <= 0.0f) throw std::invalid_argument("k must be finite and > 0");
}

// rrf_fuse() without the argument checks.
static vector<FusedHit> fuse_one(span<const span<const int64_t>> lists, span<const float> weights, float k,
                                 size_t limit) {
    size_t total = 0;
    for (auto list : lists) total += list.size();
    // score, and the last list that added to it, so a repeat within a list is skipped
    std::unordered_map<int64_t, std::pair<double, size_t>> scores;
    scores.reserve(total);
    for (size_t i = 0; i < lists.size(); ++i) {
        const double w = weights.empty() ? 1.0 : weights[i];
        if (w == 0.0) continue;
        for (size_t r = 0; r < lists[i].size(); ++r) {
            const int64_t id = lists[i][r];
            if (id < 0) continue;
            auto [it, added] = scores.try_emplace(id, 0.0, i);
            if (!added && it->second.second == i) continue;
            it->second.first += w / (static_cast<double>(k) + static_cast<double>(r + 1));
            it->second.second = i;
        }
    }

    // ranked by the double scores, so ties are exact ties
    vector<std::pair<double, int64_t>> ranked;
    ranked.reserve(scores.size());
    for (const auto& [id, s] : scores) ranked.emplace_back(s.first, id);
    const auto end = ranked.begin() + static_cast<std::ptrdiff_t>(std::min(limit, ranked.size()));
    std::partial_sort(ranked.begin(), end, ranked.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    vector<FusedHit> out;
    out.reserve(static_cast<size_t>(end - ranked.begin()));
    for (auto it = ranked.begin(); it != end; ++it) out.push_back({it->second, static_cast<float>(it->first)});
    return out;
}

vector<FusedHit> rrf_fuse(span<const span<const int64_t>> lists, span<const float> weights, float k, size_t limit) {
    check_args(lists.size(), weights, k);
    return fuse_one(lists, weights, k, limit);
}

vector<vector<FusedHit>> rrf_fuse_batch(span<const span<const int64_t>> lists, span<const size_t> widths,
                                        size_t num_queries, span<const float> weights, float k, size_t limit) {
    check_args(lists.size(), weights, k);
    if (widths.size() != lists.size()) throw std::invalid_argument("widths must have one value per result list");
    for (size_t i = 0; i < lists.size(); ++i)
        if (lists[i].size() != num_queries * widths[i])
            throw std::invalid_argument("result list " + std::to_string(i) + " must hold " +
                                        std::to_string(num_queries) + " rows of " + std::to_string(widths[i]) + " ids");

    vector<vector<FusedHit>> out(num_queries);
    detail::parallel_for(num_queries, [&](size_t q) {
        vector<span<const int64_t>> rows(lists.size());
        for (size_t i = 0; i < lists.size(); ++i) rows[i] = lists[i].subspan(q * widths[i], widths[i]);
        out[q] = fuse_one(rows, weights, k, limit);
    });
    return out;
}

} // namespace vecengine
