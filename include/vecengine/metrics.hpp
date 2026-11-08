#pragma once

// Ranking metrics (issue #119): score a ranked list of ids against the ids
// known to be relevant to its query, e.g. to tune HybridIndex weights.

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace vecengine {

using std::size_t;
using std::span;
using std::vector;

// Recall: share of the relevant ids found. NDCG: discounted cumulative gain
// (gain = grade, discount log2(rank + 1)) over the best possible order of the
// relevant ids. MRR: 1 / rank of the first relevant id, 0 if none.
enum class RankMetric { Recall, NDCG, MRR };

// One query's relevance labels: (id, grade) pairs, sorted by id with no
// repeats; a grade > 0 marks the id relevant, and NDCG uses it as the gain.
using Relevance = vector<std::pair<int64_t, float>>;

// Sorts labels by id and checks them: throws std::invalid_argument on a
// repeated id, a non-finite grade, or no grade > 0.
void check_relevance(Relevance& labels);

// metric@k of one ranked list (best first), labels as check_relevance()
// leaves them. Only the first k ids count; negative ids (padding) and repeats
// of an earlier id score nothing. Throws std::invalid_argument if k is 0.
double rank_score(RankMetric metric, span<const int64_t> ranked, const Relevance& labels, size_t k);

} // namespace vecengine
