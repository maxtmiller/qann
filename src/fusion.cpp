#include "vecengine/fusion.hpp"
#include "parallel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace vecengine {

static void check_args(FusionMethod method, size_t num_lists, size_t num_dists, span<const float> weights, float k) {
    if (num_lists == 0) throw std::invalid_argument("fusion needs at least one result list");
    if (method == FusionMethod::Linear && num_dists != num_lists)
        throw std::invalid_argument("linear fusion needs the distances of every result list");
    if (!weights.empty() && weights.size() != num_lists)
        throw std::invalid_argument("weights must have one value per result list (" + std::to_string(num_lists) +
                                    "), got " + std::to_string(weights.size()));
    for (float w : weights)
        if (!std::isfinite(w) || w < 0.0f) throw std::invalid_argument("weights must be finite and >= 0");
    if (method == FusionMethod::RRF && (!std::isfinite(k) || k <= 0.0f))
        throw std::invalid_argument("k must be finite and > 0");
}

// fuse() without the argument checks.
static vector<FusedHit> fuse_one(FusionMethod method, span<const span<const int64_t>> ids,
                                 span<const span<const float>> dists, span<const float> weights, float k,
                                 size_t limit) {
    size_t total = 0;
    for (auto list : ids) total += list.size();
    // score, and the last list that added to it, so a repeat within a list is skipped
    std::unordered_map<int64_t, std::pair<double, size_t>> scores;
    scores.reserve(total);
    for (size_t i = 0; i < ids.size(); ++i) {
        const double w = weights.empty() ? 1.0 : weights[i];
        if (w == 0.0) continue;
        const span<const int64_t> list = ids[i];
        const bool linear = method == FusionMethod::Linear;
        auto valid = [&](size_t r) { return list[r] >= 0 && (!linear || std::isfinite(dists[i][r])); };

        double lo = std::numeric_limits<double>::infinity(), hi = -lo;
        if (linear)
            for (size_t r = 0; r < list.size(); ++r)
                if (valid(r)) {
                    lo = std::min(lo, static_cast<double>(dists[i][r]));
                    hi = std::max(hi, static_cast<double>(dists[i][r]));
                }
        for (size_t r = 0; r < list.size(); ++r) {
            if (!valid(r)) continue;
            auto [it, added] = scores.try_emplace(list[r], 0.0, i);
            if (!added && it->second.second == i) continue;
            const double part = !linear  ? 1.0 / (static_cast<double>(k) + static_cast<double>(r + 1))
                              : hi > lo ? (hi - static_cast<double>(dists[i][r])) / (hi - lo)
                                        : 1.0;
            it->second.first += w * part;
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

vector<FusedHit> fuse(FusionMethod method, span<const span<const int64_t>> ids, span<const span<const float>> dists,
                      span<const float> weights, float k, size_t limit) {
    check_args(method, ids.size(), dists.size(), weights, k);
    if (method == FusionMethod::Linear)
        for (size_t i = 0; i < ids.size(); ++i)
            if (dists[i].size() != ids[i].size())
                throw std::invalid_argument("result list " + std::to_string(i) + " needs one distance per id");
    return fuse_one(method, ids, dists, weights, k, limit);
}

vector<vector<FusedHit>> fuse_batch(FusionMethod method, span<const span<const int64_t>> ids,
                                    span<const span<const float>> dists, span<const size_t> widths,
                                    size_t num_queries, span<const float> weights, float k, size_t limit) {
    check_args(method, ids.size(), dists.size(), weights, k);
    if (widths.size() != ids.size()) throw std::invalid_argument("widths must have one value per result list");
    for (size_t i = 0; i < ids.size(); ++i) {
        const size_t size = num_queries * widths[i];
        if (ids[i].size() != size || (method == FusionMethod::Linear && dists[i].size() != size))
            throw std::invalid_argument("result list " + std::to_string(i) + " must hold " +
                                        std::to_string(num_queries) + " rows of " + std::to_string(widths[i]) +
                                        " ids" + (method == FusionMethod::Linear ? " and distances" : ""));
    }

    const bool linear = method == FusionMethod::Linear;
    vector<vector<FusedHit>> out(num_queries);
    detail::parallel_for(num_queries, [&](size_t q) {
        vector<span<const int64_t>> id_rows(ids.size());
        vector<span<const float>> dist_rows(linear ? ids.size() : 0);
        for (size_t i = 0; i < ids.size(); ++i) {
            id_rows[i] = ids[i].subspan(q * widths[i], widths[i]);
            if (linear) dist_rows[i] = dists[i].subspan(q * widths[i], widths[i]);
        }
        out[q] = fuse_one(method, id_rows, dist_rows, weights, k, limit);
    });
    return out;
}

} // namespace vecengine
