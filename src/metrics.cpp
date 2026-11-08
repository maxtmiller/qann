#include "vecengine/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

namespace vecengine {

void check_relevance(Relevance& labels) {
    std::sort(labels.begin(), labels.end());
    bool any_relevant = false;
    for (size_t i = 0; i < labels.size(); ++i) {
        if (!std::isfinite(labels[i].second)) throw std::invalid_argument("relevance grades must be finite");
        if (i > 0 && labels[i].first == labels[i - 1].first)
            throw std::invalid_argument("a query's relevant ids must not repeat");
        any_relevant |= labels[i].second > 0.0f;
    }
    if (!any_relevant) throw std::invalid_argument("every query needs at least one relevant id (grade > 0)");
}

double rank_score(RankMetric metric, span<const int64_t> ranked, const Relevance& labels, size_t k) {
    if (k == 0) throw std::invalid_argument("k must be at least 1");
    const size_t n = std::min(k, ranked.size());
    vector<int64_t> seen;
    double sum = 0.0;
    for (size_t r = 0; r < n; ++r) {
        const int64_t id = ranked[r];
        if (id < 0 || std::find(seen.begin(), seen.end(), id) != seen.end()) continue;
        seen.push_back(id);
        const auto it = std::lower_bound(labels.begin(), labels.end(), std::make_pair(id, -INFINITY));
        if (it == labels.end() || it->first != id || it->second <= 0.0f) continue;
        if (metric == RankMetric::MRR) return 1.0 / static_cast<double>(r + 1);
        sum += metric == RankMetric::NDCG ? it->second / std::log2(static_cast<double>(r + 2)) : 1.0;
    }
    if (metric == RankMetric::MRR) return 0.0;

    vector<float> grades;
    for (const auto& [id, grade] : labels)
        if (grade > 0.0f) grades.push_back(grade);
    if (metric == RankMetric::Recall) return sum / static_cast<double>(grades.size());
    std::sort(grades.begin(), grades.end(), std::greater<>());
    double ideal = 0.0;
    for (size_t r = 0; r < std::min(k, grades.size()); ++r) ideal += grades[r] / std::log2(static_cast<double>(r + 2));
    return sum / ideal;
}

} // namespace vecengine
