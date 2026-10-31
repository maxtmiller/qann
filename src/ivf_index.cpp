#include "vecengine/ivf_index.hpp"
#include "vecengine/distance.hpp"
#include "vecengine/pq.hpp"
#include "kmeans.hpp"
#include "parallel.hpp"
#include "profile.hpp"
#include "topk.hpp"
#include "serialize.hpp"
#include "fast_scan.hpp"
#include "rotation.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <cassert>
#include <queue>
#include <utility>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <bit>

#ifdef VECENGINE_USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;
using std::pair;

IVFIndex::IVFIndex(size_t dim, size_t nlist, size_t nprobe, bool custom_ids, Metric metric): DenseIndex(metric), dim_(dim), nlist_(nlist), nprobe_(nprobe), id_map_(custom_ids)  {
    if (nlist < 100 || nlist > 65535) throw std::invalid_argument("nlist must be in [100, 65535]");
    if (dim == 0) throw std::invalid_argument("dim must be >= 1");
    set_nprobe(nprobe);
    coarse_centroids_.reserve(dim_ * nlist_);
}

void IVFIndex::set_nprobe(size_t nprobe) {
    if (nprobe == 0 || nprobe > nlist_) throw std::invalid_argument("nprobe must be in [1, nlist]");
    nprobe_ = nprobe;
}

void IVFIndex::set_precomputed_tables(bool enabled) {
    use_precomputed_ = enabled;
    if (enabled) build_precomputed();
    else vector<float>().swap(precomputed_);
}

void IVFIndex::build_precomputed() {
    precomputed_.clear();
    // InnerProduct needs none: its table doesn't depend on the list (see scan)
    if (!use_precomputed_ || !trained_ || !pq_enabled() || metric_ == Metric::InnerProduct) return;

    size_t list_size = pq_->num_subspaces() * pq_->centroids_per_subspace();
    if (nlist_ * list_size * sizeof(float) > kPrecomputedMaxBytes) return;
    
    precomputed_.resize(nlist_ * list_size);
    detail::parallel_for(nlist_, [&](size_t i) {
        pq_->compute_list_term({coarse_centroids_.data() + i * dim_, dim_}, {precomputed_.data() + i * list_size, list_size});
    });
}

void IVFIndex::enable_pq(size_t num_subspaces, size_t centroids_per_subspace, bool fast_scan, bool rotate) {
    if (id_map_.slots() > 0 || trained_)
        throw std::logic_error("enable_pq must be called before train()/add()");
    if (sq_bits_) throw std::logic_error("enable_pq cannot be combined with enable_sq");
    if (fast_scan && (centroids_per_subspace != 16 || num_subspaces % 2 != 0 || num_subspaces > detail::kFsMaxSubspaces))
        throw std::invalid_argument("fast_scan needs 16 centroids per subspace and an even num_subspaces <= 1024");
    if (fast_scan && pq_distance_ == PQDistance::SDC)
        throw std::invalid_argument("cannot use SDC with fast-scan");
    fast_scan_= fast_scan;
    rotate_ = rotate;
    pq_ = std::make_unique<PQCodebook>(dim_, num_subspaces, centroids_per_subspace);
}

vector<float> IVFIndex::rotated(span<const float> vecs, size_t n) const {
    assert(rotation_panels_.size() == dim_ * dim_ && vecs.size() == n * dim_);
    vector<float> out(n * dim_);
    detail::rotate(rotation_panels_, vecs.data(), n, dim_, out.data());
    return out;
}

void IVFIndex::enable_sq(int bits) {
    if (bits != 4 && bits != 8) throw std::invalid_argument("bits must be 4 or 8");
    if (id_map_.slots() > 0 || trained_) throw std::logic_error("enable_sq must be called before train()/add()");
    if (pq_enabled()) throw std::logic_error("enable_sq cannot be combined with enable_pq");
    sq_bits_ = bits;
    sq_.bits = bits;
}

void IVFIndex::add_batch_impl(span<const float> vecs, size_t n, const int64_t* ids) {
    assert(vecs.size() == n * dim());
    if (!trained_) throw std::logic_error("train() has not be run yet");
    vector<float> rot;
    if (rotate_) {
        rot = rotated(vecs, n);
        vecs = rot;
    }

    const uint32_t first_slot = static_cast<uint32_t>(id_map_.slots());
    id_map_.add(n, ids);
    const bool lookup = !pq_enabled();
    if (lookup) {
        slot_list_.resize(first_slot + n);
        slot_pos_.resize(first_slot + n);
    }

    vector<uint32_t> assign(n);
    detail::assign_nearest(vecs.data(), n, dim_, coarse_centroids_.data(), nlist_, dim_, assign.data());

    const bool use_pq = pq_enabled();
    const size_t code_bytes = use_pq ? pq_->num_subspaces() : 0;
    
    vector<uint8_t> codes(use_pq ? n * code_bytes : 0);

    if (sq_bits_) {
        codes.resize(n * sq_code_bytes());
        detail::parallel_for(n, [&](size_t i) { sq_.encode(vecs.data() + i * dim_, codes.data() + i * sq_code_bytes()); });
    } else if (use_pq) {
        detail::parallel_for(n, [&](size_t i) {
            const float* src_vec = vecs.data() + i * dim_;
            const float* centroid = coarse_centroids_.data() + assign[i] * dim_;

            vector<float> residual(dim_);
            for (size_t d = 0; d < dim_; ++d) residual[d] = src_vec[d] - centroid[d];

            auto code = pq_->encode(residual);
            std::copy_n(code.data(), code_bytes, codes.data() + i * code_bytes);
        });
    }

    for (size_t i = 0; i < n; ++i) {
        InvertedList& list = data_[static_cast<uint16_t>(assign[i])];
        list.ids.push_back(first_slot + static_cast<uint32_t>(i));
        if (lookup) {
            slot_list_[first_slot + i] = static_cast<uint16_t>(assign[i]);
            slot_pos_[first_slot + i] = static_cast<uint32_t>(list.ids.size() - 1);
        }

        if (sq_bits_) {
            const size_t w = sq_code_bytes();
            list.codes.insert(list.codes.end(), codes.begin() + i * w, codes.begin() + (i + 1) * w);
        } else if (use_pq) {
            if (fast_scan_) {
            const size_t m = pq_->num_subspaces(), pos = list.ids.size() - 1;
            if (pos % 32 == 0) list.codes.resize(list.codes.size() + m * 16, 0);
            uint8_t* block = list.codes.data() + (pos / 32) * m * 16;
            for (size_t s = 0; s < m; ++s) detail::fs_set(block, pos % 32, s, codes[i * code_bytes + s]);
            } else {
                list.codes.insert(list.codes.end(), codes.begin() + i * code_bytes, codes.begin() + (i + 1) * code_bytes);
            }
        } else {
            const float* src = vecs.data() + i * dim_;
            list.vecs.insert(list.vecs.end(), src, src + dim_);
        }
    }
}

vector<Neighbor> IVFIndex::query_impl(span<const float> vec, size_t k, const detail::SlotFilter* filter) const {
    assert(vec.size() == dim_);
    if (!trained_) throw std::logic_error("train() has not be run yet");
    vector<float> rot;
    if (rotate_) {
        rot = rotated(vec, 1);
        vec = rot;
    }

    if (use_direct(k, filter)) return direct_scan(vec, k, *filter);

    VECENGINE_PROF_START(t_coarse);
    vector<Probe> probes = filter ? coarse_all(vec) : coarse(vec);
    VECENGINE_PROF_STOP(t_coarse, kProfCoarse);

    return scan(vec, probes, k, filter);
}

bool IVFIndex::use_direct(size_t k, const detail::SlotFilter* filter) const {
    // Scoring every allowed vector is exact for plain IVF, and while few are
    // allowed it is faster than probing enough lists to find them: on SIFT1M
    // (nlist 1000, nprobe 16) it ran at 98k QPS against 44k for the scaled
    // probing in scan() at 2,000 allowed, 33k against 49k at 5,000, a
    // crossover near 4,000; SQ's direct scoring is somewhat slower, so this
    // switches at 3 * nlist. PQ isn't switched: each allowed vector needs its
    // own list's table there, which made it 2-50x slower; a RefineIndex over PQ
    // makes this switch instead.
    return filter && k > 0 && !pq_enabled() && filter->count <= 3 * nlist_;
}

void IVFIndex::rebuild_slot_lookup() {
    slot_list_.clear();
    slot_pos_.clear();
    if (pq_enabled()) return;
    slot_list_.resize(id_map_.slots());
    slot_pos_.resize(id_map_.slots());
    for (const auto& [l, list] : data_)
        for (size_t j = 0; j < list.ids.size(); ++j) {
            slot_list_[list.ids[j]] = l;
            slot_pos_[list.ids[j]] = static_cast<uint32_t>(j);
        }
}

vector<Neighbor> IVFIndex::direct_scan(span<const float> vec, size_t k, const detail::SlotFilter& filter) const {
    detail::TopK top(k);
    filter.for_each([&](uint32_t slot) {
        const InvertedList& list = data_.at(slot_list_[slot]);
        const size_t j = slot_pos_[slot];
        float dist;
        if (sq_bits_) {
            const uint8_t* code = list.codes.data() + j * sq_code_bytes();
            dist = metric_ == Metric::InnerProduct ? -sq_.inner_product(vec.data(), code) : sq_.distance(vec.data(), code);
        } else {
            dist = exact_score(vec.data(), list.vecs.data() + j * dim_, dim_);
        }
        if (dist < top.threshold()) top.push(dist, id_map_.label(slot));
    });
    return top.take_sorted();
}

vector<vector<Neighbor>> IVFIndex::query_batch_impl(span<const float> queries, size_t num_queries, size_t k,
                                                    const detail::SlotFilter* filter) const {
    assert(queries.size() == num_queries * dim_);
    if (!trained_) throw std::logic_error("train() has not be run yet");

#ifndef VECENGINE_USE_BLAS
    return DenseIndex::query_batch_impl(queries, num_queries, k, filter);
#else
    // a filtered query may probe every list, which the batched coarse step doesn't rank
    if (filter) return DenseIndex::query_batch_impl(queries, num_queries, k, filter);

    // after the early returns above, which go through query_impl and rotate there
    vector<float> rot;
    if (rotate_) {
        rot = rotated(queries, num_queries);
        queries = rot;
    }

    // coarse_batch holds block * nlist_ scores; cap them at 32 MB
    const size_t block = std::clamp<size_t>((size_t{32} << 20) / (nlist_ * sizeof(float)), 1, 1024);

    vector<vector<Neighbor>> results(num_queries);
    vector<Probe> probes(block * nprobe_);
    for (size_t start = 0; start < num_queries; start += block) {
        const size_t b = std::min(block, num_queries - start);
        const float* q = queries.data() + start * dim_;

        VECENGINE_PROF_START(t_coarse);
        coarse_batch(q, b, probes.data());
        VECENGINE_PROF_STOP(t_coarse, kProfCoarse);

        detail::parallel_for(b, [&](size_t i) {
            results[start + i] = scan({q + i * dim_, dim_}, {probes.data() + i * nprobe_, nprobe_}, k, nullptr);
        });
    }
    return results;
#endif
}

void IVFIndex::update_centroid_sqnorms() {
    centroid_sqnorms_.resize(coarse_centroids_.size() / dim_);
    for (size_t c = 0; c < centroid_sqnorms_.size(); ++c) {
        const float* centroid = coarse_centroids_.data() + c * dim_;
        centroid_sqnorms_[c] = inner_product(centroid, centroid, dim_);
    }
}

vector<IVFIndex::Probe> IVFIndex::coarse(span<const float> vec) const {
    // Accelerate only: OpenBLAS's single-vector sgemv hasn't been measured against the loop
#if defined(VECENGINE_USE_BLAS) && defined(__APPLE__)
    if (dim_ >= kBlasCoarseMinDim) {
        // One sgemv scores every centroid by ||c||^2 - 2 q.c (-q.c for InnerProduct),
        // as coarse_batch does; the scores lose precision to cancellation, so the
        // nprobe picked are recomputed exactly
        const bool ip = metric_ == Metric::InnerProduct;
        vector<float> score = ip ? vector<float>(nlist_, 0.0f) : centroid_sqnorms_;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(nlist_), static_cast<int>(dim_), ip ? -1.0f : -2.0f,
                    coarse_centroids_.data(), static_cast<int>(dim_), vec.data(), 1, 1.0f, score.data(), 1);
        vector<pair<float, uint16_t>> scored(nlist_);
        for (size_t c = 0; c < nlist_; ++c) scored[c] = {score[c], static_cast<uint16_t>(c)};
        std::nth_element(scored.begin(), scored.begin() + (nprobe_ - 1), scored.end());
        vector<Probe> probes(nprobe_);
        for (size_t j = 0; j < nprobe_; ++j) {
            const uint16_t list = scored[j].second;
            probes[j] = {exact_score(vec.data(), coarse_centroids_.data() + list * dim_, dim_), list};
        }
        std::sort(probes.begin(), probes.end());
        return probes;
    }
#endif
    // find the closest nprobe coarse centroids
    std::priority_queue<pair<float, uint16_t>, vector<pair<float, uint16_t>>, std::less<pair<float, uint16_t>>> maxCentroidHeap;
    for (size_t i = 0; i < nlist_; ++i) {
        const float* centroid_ptr = coarse_centroids_.data() + i * dim_;
        float dist = exact_score(vec.data(), centroid_ptr, dim_);

        if (maxCentroidHeap.size() < nprobe_) {
            maxCentroidHeap.emplace(dist, static_cast<uint16_t>(i));
        } else if (dist < maxCentroidHeap.top().first) {
            maxCentroidHeap.pop();
            maxCentroidHeap.emplace(dist, static_cast<uint16_t>(i));
        }
    }

    // the heap pops farthest first, so fill from the back: lists are scanned
    // nearest first, which tightens the top-k threshold sooner
    size_t i = maxCentroidHeap.size() - 1;
    vector<Probe> probes(i + 1);
    while (!maxCentroidHeap.empty()) {
        probes[i] = {maxCentroidHeap.top().first, maxCentroidHeap.top().second};
        maxCentroidHeap.pop();
        --i;
    }
    return probes;
}

bool IVFIndex::coarse_all_approx() const noexcept {
#if defined(VECENGINE_USE_BLAS) && defined(__APPLE__)
    return dim_ >= kBlasCoarseMinDim;
#else
    return false;
#endif
}

vector<IVFIndex::Probe> IVFIndex::coarse_all(span<const float> vec) const {
    vector<Probe> probes(nlist_);
#if defined(VECENGINE_USE_BLAS) && defined(__APPLE__)
    if (coarse_all_approx()) {
        // as in coarse(): ||c||^2 - 2 q.c (-q.c for InnerProduct) from one sgemv; adding
        // ||q||^2 keeps the ranking and puts dist on the exact scale, which scan()
        // restores precisely for the lists it probes
        const bool ip = metric_ == Metric::InnerProduct;
        vector<float> score = ip ? vector<float>(nlist_, 0.0f) : centroid_sqnorms_;
        cblas_sgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(nlist_), static_cast<int>(dim_), ip ? -1.0f : -2.0f,
                    coarse_centroids_.data(), static_cast<int>(dim_), vec.data(), 1, 1.0f, score.data(), 1);
        const float qq = ip ? 0.0f : cblas_sdot(static_cast<int>(dim_), vec.data(), 1, vec.data(), 1);
        for (size_t i = 0; i < nlist_; ++i) probes[i] = {score[i] + qq, static_cast<uint16_t>(i)};
        std::partial_sort(probes.begin(), probes.begin() + nprobe_, probes.end());
        return probes;
    }
#endif
    for (size_t i = 0; i < nlist_; ++i)
        probes[i] = {exact_score(vec.data(), coarse_centroids_.data() + i * dim_, dim_), static_cast<uint16_t>(i)};
    std::partial_sort(probes.begin(), probes.begin() + nprobe_, probes.end());
    return probes;
}

#ifdef VECENGINE_USE_BLAS
void IVFIndex::coarse_batch(const float* queries, size_t n, Probe* out) const {
    const vector<float>& norms = centroid_sqnorms_;

    // dots[i * nlist_ + c] = q_i . c for the whole block in one call
    vector<float> dots(n * nlist_);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(n), static_cast<int>(nlist_), static_cast<int>(dim_),
                1.0f, queries, static_cast<int>(dim_),
                coarse_centroids_.data(), static_cast<int>(dim_),
                0.0f, dots.data(), static_cast<int>(nlist_));

    const bool ip = metric_ == Metric::InnerProduct;
    detail::parallel_for(n, [&](size_t i) {
        // ||q - c||^2 = ||q||^2 + ||c||^2 - 2 q.c, and ||q||^2 is the same for
        // every list, so ||c||^2 - 2 q.c ranks the lists (-q.c for InnerProduct)
        const float* row = dots.data() + i * nlist_;
        vector<pair<float, uint16_t>> scored(nlist_);
        for (size_t c = 0; c < nlist_; ++c)
            scored[c] = {ip ? -row[c] : norms[c] - 2.0f * row[c], static_cast<uint16_t>(c)};
        std::nth_element(scored.begin(), scored.begin() + (nprobe_ - 1), scored.end());

        // the scores lose precision to cancellation, so A is recomputed exactly
        const float* q = queries + i * dim_;
        Probe* probes = out + i * nprobe_;
        for (size_t j = 0; j < nprobe_; ++j) {
            const uint16_t list = scored[j].second;
            probes[j] = {exact_score(q, coarse_centroids_.data() + list * dim_, dim_), list};
        }

        // nearest first, ordered by exact distance like coarse()
        std::sort(probes, probes + nprobe_);
    });
}
#endif

vector<Neighbor> IVFIndex::scan(span<const float> vec, span<Probe> probes, size_t k, const detail::SlotFilter* filter) const {
    VECENGINE_PROF_START(t_setup);
    // intialize vars for pq, reuse allocated memory for adc table
    const size_t m = pq_enabled() ? pq_->num_subspaces() : 0;
    const size_t K = pq_enabled() ? pq_->centroids_per_subspace() : 0;
    vector<float> table(pq_enabled() ? m * K : 0);
    vector<float> dists;
    detail::TopK top(k);

    const bool use_pre = pq_enabled() && pq_distance_ == PQDistance::ADC && precomputed_tables();
    vector<float> C(use_pre ? m * K : 0);
    vector<uint8_t> lut(fast_scan_ ? m * K : 0);
    VECENGINE_PROF_STOP(t_setup, kProfSetup);
    VECENGINE_PROF_START(t_inner);
    if (use_pre) pq_->compute_inner_table(vec, C);
    VECENGINE_PROF_STOP(t_inner, kProfInner);

    // InnerProduct: -(q . x) = -(q . c) - sum_s q_s . r_s, for residual r = x - c.
    // The table of -(q_s . y_sj) is the same for every list, so it is built
    // (and for fast-scan rounded to bytes) once per query, and each list adds
    // its -(q . c), the probe's dist.
    const bool ip_pq = pq_enabled() && metric_ == Metric::InnerProduct;
    float ip_scale = 1.0f, ip_offset = 0.0f;
    if (ip_pq) {
        pq_->compute_inner_table(vec, table);
        for (float& t : table) t *= 0.5f;
        if (fast_scan_) detail::quantize_lut(table.data(), m, lut.data(), ip_scale, ip_offset);
    }

    // A block of 32 is skipped only when none of it passes, (1 - s)^32 of blocks
    // at selectivity s, while building its mask costs about 12% of the kernel's
    // time (measured on SIFT1M), so below 1/16 allowed the skips win. Above it,
    // push() tests the filter as for an unfiltered query's deleted vectors.
    const bool block_filter = filter && filter->count * 16 < id_map_.slots();

    // scores one fast-scan list with lut (scale and offset as quantize_lut set
    // them, plus anything added to every distance in the list)
    auto scan_fast = [&](const InvertedList& list, size_t listSize, float scale, float offset, auto&& push) {
        const size_t block_bytes = detail::fs_block_bytes(m);
        uint16_t sums[detail::kFsBlock];

        for (size_t start = 0; start < listSize; start += detail::kFsBlock) {
            const size_t count = std::min(detail::kFsBlock, listSize - start);
            // a selective filter skips blocks it rejects entirely, before the kernel
            uint32_t allowed = count < detail::kFsBlock ? (1u << count) - 1 : ~0u;
            if (block_filter) {
                allowed = 0;
                for (size_t v = 0; v < count; ++v) allowed |= uint32_t{filter->test(list.ids[start + v])} << v;
                if (allowed == 0) continue;
            }

            // the current k-th best distance in the table's integer scale, recomputed
            // per block as it shrinks; sums are whole, so sum < limit <=> sum < ceil(limit)
            const float limit = (top.threshold() - offset) * scale;
            const uint16_t limit16 = limit >= 65535.0f ? 65535
                                   : limit <= 0.0f    ? 0
                                                      : static_cast<uint16_t>(std::ceil(limit));

            // bit v is set when vector v can beat the k-th best; padding bits are dropped
            uint32_t mask = detail::fast_scan_block_mask(
                lut.data(), list.codes.data() + (start / detail::kFsBlock) * block_bytes, m, limit16, sums);
            mask &= allowed;    // drops padding bits, and with a filter the vectors it rejects

            // visit only the set bits, lowest first
            while (mask != 0) {
                const unsigned v = static_cast<unsigned>(std::countr_zero(mask));
                mask &= mask - 1;
                push(sums[v] / scale + offset, start + v);
            }
        }
    };

    // a filtered query keeps probing past nprobe lists until it has this many results
    const size_t wanted = filter ? std::min(k, filter->count) : 0;

    // With a fraction s of vectors allowed, the filtered top k sit about as deep
    // as the unfiltered top k / s, so nprobe lists hold fewer of them. Probing
    // 0.75 * nprobe * s^(-1/3) lists kept recall@10 at 0.92-0.96 from 0.2% to
    // 20% allowed on SIFT1M (nprobe 16: unfiltered 0.93), where stopping at
    // the first k allowed vectors gave 0.60-0.88. nprobe / s overshot: recall
    // 1.0 at many times the cost.
    size_t min_probes = nprobe_;
    if (filter && filter_probe_scaling_ && filter->count > 0) {
        const double want = std::ceil(0.75 * double(nprobe_) * std::cbrt(double(id_map_.live()) / double(filter->count)));
        min_probes = std::clamp<size_t>(static_cast<size_t>(std::min(want, double(nlist_))), nprobe_, nlist_);
    }

    // find top-k vectors from the inverted lists associated with the nprobe closest coarse centroids
    for (size_t p = 0; p < probes.size(); ++p) {
        if (p >= nprobe_) {
            if (p >= min_probes && top.size() >= wanted) break;
            if (p == nprobe_) std::sort(probes.begin() + p, probes.end());
        }
        const Probe& probe = probes[p];
        VECENGINE_PROF_START(t_list);
        // a filtered query's probes may hold coarse_all()'s approximate distances
        const float A = filter && coarse_all_approx()
                            ? exact_score(vec.data(), coarse_centroids_.data() + size_t{probe.list} * dim_, dim_)
                            : probe.dist;
        auto it = data_.find(probe.list);
        if (it == data_.end()) continue;

        const InvertedList& list = it->second;
        size_t listSize = list.ids.size();
        dists.resize(listSize);
        VECENGINE_PROF_STOP(t_list, kProfListSetup);

        // adds vector j to the top k if it beats the k-th best and may be returned
        auto push = [&](float dist, size_t j) {
            const uint32_t slot = list.ids[j];
            if (dist < top.threshold() && (filter ? filter->test(slot) : !id_map_.is_deleted(slot)))
                top.push(dist, id_map_.label(slot));
        };
        // with a filter, whether vector j passes it, checked before scoring so a
        // rejected vector costs no distance; unfiltered queries score every vector
        auto passes = [&](size_t j) { return !filter || filter->test(list.ids[j]); };

        if (sq_bits_) {
            VECENGINE_PROF_START(t_raw);
            const size_t w = sq_code_bytes();
            if (metric_ == Metric::InnerProduct) {
                for (size_t j = 0; j < listSize; ++j)
                    if (passes(j)) push(-sq_.inner_product(vec.data(), list.codes.data() + j * w), j);
            } else {
                for (size_t j = 0; j < listSize; ++j)
                    if (passes(j)) push(sq_.distance(vec.data(), list.codes.data() + j * w), j);
            }
            VECENGINE_PROF_STOP(t_raw, kProfRawScan);
            continue;
        }

        if (!pq_enabled()) {
            VECENGINE_PROF_START(t_raw);
            for (size_t j = 0; j < listSize; ++j)
                if (passes(j)) push(exact_score(vec.data(), list.vecs.data() + j * dim_, dim_), j);
            VECENGINE_PROF_STOP(t_raw, kProfRawScan);
            continue;
        }

        if (ip_pq) {
            VECENGINE_PROF_START(t_scan);
            if (fast_scan_) {
                scan_fast(list, listSize, ip_scale, ip_offset + A, push);
            } else {
                pq_->distances_adc(table, list.codes.data(), listSize, dists.data());
                for (size_t j = 0; j < listSize; ++j) push(dists[j] + A, j);
            }
            VECENGINE_PROF_STOP(t_scan, kProfCodeScan);
            continue;
        }

        VECENGINE_PROF_START(t_table);
        vector<float> residual(use_pre ? 0 : dim_);
        if (use_pre) {
            const float* B = precomputed_.data() + it->first * m * K;
            for (size_t t = 0; t < m * K; ++t) table[t] = B[t] + C[t];
            for (size_t j = 0; j < K; ++j) table[j] += A;
        } else {
            // calculates residuals 
            const float* centroid_ptr = coarse_centroids_.data() + it->first * dim_;
            for (size_t l = 0; l < dim_; ++l) {
                residual[l] = vec[l] - centroid_ptr[l];
            }
        }

        // calculate distance with ADC or SDC
        if (pq_distance_ == PQDistance::ADC) {
            if (!use_pre) pq_->compute_adc_table(residual, table);
            VECENGINE_PROF_STOP(t_table, kProfTable);
            VECENGINE_PROF_START(t_scan);
            if (fast_scan_) {
                // round the table to bytes, then score 32 vectors per kernel call
                float scale, offset;
                detail::quantize_lut(table.data(), m, lut.data(), scale, offset);
                scan_fast(list, listSize, scale, offset, push);
            } else {
                pq_->distances_adc(table, list.codes.data(), listSize, dists.data());
                for (size_t j = 0; j < listSize; ++j) {
                    push(dists[j], j);
                }
            }
            VECENGINE_PROF_STOP(t_scan, kProfCodeScan);
        } else {
            vector<uint8_t> query_code = pq_->encode(residual);
            for (size_t j = 0; j < listSize; ++j) {
                span<const uint8_t> code(list.codes.data() + j * m, m);
                push(pq_->distance_sdc(query_code, code), j);
            }
        }
    }

    VECENGINE_PROF_START(t_drain);
    vector<Neighbor> results = top.take_sorted();
    VECENGINE_PROF_STOP(t_drain, kProfDrain);

    return results;
}

size_t IVFIndex::remove(span<const int64_t> ids) {
    return id_map_.remove(ids);
}

size_t IVFIndex::compact() { return compact_impl(true); }

size_t IVFIndex::compact_as_base() { return compact_impl(false); }

size_t IVFIndex::compact_impl(bool keep_ids) {
    auto remap = id_map_.compact(keep_ids);
    if (remap.empty()) {
        metadata_.drop_unused_strings();
        return 0;
    }
    compact_storage(remap);
    rebuild_slot_lookup();
    metadata_.compact(remap);
    return remap.size() - id_map_.slots();
}

void IVFIndex::compact_storage(const vector<uint32_t>& remap) {
    // bytes per vector in list.codes (PQ without fast-scan, or SQ)
    const size_t m = pq_enabled() ? pq_->num_subspaces() : sq_bits_ ? sq_code_bytes() : 0;
    for (uint16_t i = 0; i < nlist_; ++i) {
        size_t write_slot = 0;

        auto it = data_.find(i);
        if (it == data_.end()) continue;
        InvertedList& list = it->second;
        
        for (size_t j = 0; j < list.ids.size(); ++j) {
            uint32_t new_slot = remap[list.ids[j]];
            if (new_slot == detail::IdMap::kRemoved) continue;

            list.ids[write_slot] = new_slot;
            if (j != write_slot) {
                if (fast_scan_) {
                    // w < j, so writing entry w never clobbers an entry still to be read
                    const uint8_t* src = list.codes.data() + (j / detail::kFsBlock) * detail::fs_block_bytes(m);
                    uint8_t* dst = list.codes.data() + (write_slot / detail::kFsBlock) * detail::fs_block_bytes(m);
                    for (size_t s = 0; s < m; ++s)
                        detail::fs_set(dst, write_slot % detail::kFsBlock, s, detail::fs_get(src, j % detail::kFsBlock, s));
                } else if (m > 0) {
                    uint8_t* src = list.codes.data() + j * m;
                    uint8_t* dst = list.codes.data() + write_slot * m;
                    std::copy_n(src, m, dst);
                } else {
                    float* src = list.vecs.data() + j * dim();
                    float* dst = list.vecs.data() + write_slot * dim();
                    std::copy_n(src, dim(), dst);
                }
            }
            ++write_slot;
        }
        if (write_slot == 0) { data_.erase(it); continue; }

        list.ids.resize(write_slot);
        list.ids.shrink_to_fit();
        if (m > 0) {
            const size_t blocks = (write_slot + detail::kFsBlock - 1) / detail::kFsBlock;
            list.codes.resize(fast_scan_ ? blocks * detail::fs_block_bytes(m) : write_slot * m);
            list.codes.shrink_to_fit();
        } else {
            list.vecs.resize(write_slot * dim());
            list.vecs.shrink_to_fit();
        }
    }
    
}

void IVFIndex::train(span<const float> vectors, size_t num_vectors, size_t max_iters, std::optional<uint32_t> seed) {
    if (id_map_.slots() > 0) throw std::logic_error("train() cannot be called after add()");
    if (trained_) throw std::logic_error("train() has already been called");
    if (num_vectors < nlist_) throw std::invalid_argument("training set must have at least nlist vectors");

    // learn the clusters on the same unit vectors that adds and queries will use
    vector<float> unit;
    if (metric_ == Metric::Cosine) {
        unit = normalize_rows(vectors, num_vectors);
        vectors = unit;
    }

    // learn the rotation, then the clusters and codebook on the rotated sample
    vector<float> rot;
    if (rotate_) {
        rotation_ = detail::balanced_pca_rotation(vectors, num_vectors, dim_, pq_->num_subspaces());
        rotation_panels_ = detail::pack_rotation(rotation_, dim_);
        rot = rotated(vectors, num_vectors);
        vectors = rot;
    }

    // run kmeans
    auto res = vecengine::detail::kmeans(vectors.data(), num_vectors, dim_, dim_, nlist_, max_iters, seed);
    coarse_centroids_ = std::move(res.centroids);
    update_centroid_sqnorms();

    // calculate residuals and run kmeans again on each subspace
    const vector<uint32_t>& assignments = res.assignments;
    if (pq_enabled()) {
        vector<float> residuals(num_vectors * dim_);
        for (size_t i = 0; i < num_vectors; ++i) {
            const float* centroid_ptr = coarse_centroids_.data() + assignments[i] * dim_;
            const float* vec_ptr = vectors.data() + i * dim_;
            for (size_t j = 0; j < dim_; ++j) {
                residuals[i * dim_ +j] = vec_ptr[j] - centroid_ptr[j];
            }
        }

        pq_->train(residuals, num_vectors, max_iters, seed ? std::optional<uint32_t>(*seed + 1) : std::nullopt);
    }
    if (sq_bits_) sq_.train(vectors, num_vectors, dim_);
    trained_ = true;
    build_precomputed();
}

void IVFIndex::save(std::ostream& out) const {
    using namespace detail;

    write_header(out, IndexKind::IVF);
    write_pod(out, static_cast<uint64_t>(dim_));
    write_pod(out, static_cast<uint64_t>(nlist_));
    write_pod(out, static_cast<uint64_t>(nprobe_));
    write_pod(out, static_cast<uint64_t>(id_map_.slots()));
    write_pod(out, static_cast<uint8_t>(trained_));
    write_pod(out, static_cast<uint8_t>(pq_distance_));
    write_vec(out, coarse_centroids_);
    write_pod(out, static_cast<uint8_t>(pq_ != nullptr));
    if (pq_) {
        pq_->save(out);
        write_pod(out, static_cast<uint8_t>(fast_scan_));
        // version 8: the rotate setting, then the rotation (empty until trained)
        write_pod(out, static_cast<uint8_t>(rotate_));
        write_vec(out, rotation_);
    }
    // version 7: SQ bits per dimension (0 = off), then the ranges (empty until trained)
    write_pod(out, static_cast<uint8_t>(sq_bits_));
    if (sq_bits_) {
        write_vec(out, sq_.vmin);
        write_vec(out, sq_.step);
    }

    static const InvertedList empty;
    for (uint16_t i = 0; i < nlist_; ++i) {
        auto it = data_.find(i);
        const InvertedList& list = it == data_.end() ? empty : it->second;
        write_vec(out, list.ids);
        if (pq_ || sq_bits_) write_vec(out, list.codes); else write_vec(out, list.vecs);
    }
    id_map_.save(out);
    write_pod(out, static_cast<uint8_t>(use_precomputed_));
    write_pod(out, static_cast<uint8_t>(metric_));
    metadata_.save(out);
}

std::unique_ptr<IVFIndex> IVFIndex::load_body(std::istream& in, uint32_t version) {
    using namespace detail;

    const auto dim = read_pod<uint64_t>(in);
    const auto n_list = read_pod<uint64_t>(in);
    const auto n_probe = read_pod<uint64_t>(in);
    const auto n_total = read_pod<uint64_t>(in);
    const auto trained = read_pod<uint8_t>(in);
    const auto pq_distance = read_pod<uint8_t>(in);

    if (dim == 0 || n_total > UINT32_MAX || trained > 1) throw std::runtime_error("corrupt file: invalid IVFIndex shape");

    auto index = std::make_unique<IVFIndex>(dim, n_list, 1);
    if (dim > UINT64_MAX / n_list) throw std::runtime_error("corrupt file: invalid IVFIndex shape");
    index->set_nprobe(n_probe);

    auto coarse_centroids = read_vec<float>(in, n_list * dim);
    index->coarse_centroids_ = std::move(coarse_centroids);
    index->update_centroid_sqnorms();

    if (pq_distance > static_cast<uint8_t>(PQDistance::SDC)) throw std::runtime_error("corrupt file: invalid IVFIndex shape");
    if (!trained && n_total > 0) throw std::runtime_error("corrupt file: invalid IVFIndex shape");
    if ((trained && index->coarse_centroids_.size() != n_list * dim) || (!trained && index->coarse_centroids_.size() != 0)) throw std::runtime_error("corrupt file: invalid IVFIndex shape");

    auto pq_enabled = read_pod<uint8_t>(in);
    if (pq_enabled) {
        index->pq_ = std::make_unique<PQCodebook>(PQCodebook::load(in));
        if (index->pq_->dim() != dim) throw std::runtime_error("corrupt file: invalid IVFIndex shape");
        if (version >= 4) {
            const auto fast_scan = read_pod<uint8_t>(in);
            const size_t m = index->pq_->num_subspaces();
            if (fast_scan > 1 || (fast_scan && (index->pq_->centroids_per_subspace() != 16 || m % 2 != 0 || m > detail::kFsMaxSubspaces ||
                                                pq_distance != static_cast<uint8_t>(PQDistance::ADC))))
                throw std::runtime_error("corrupt file: invalid IVFIndex fast-scan settings");
            index->fast_scan_ = fast_scan;
        }
        if (version >= 8) {
            const auto rotate = read_pod<uint8_t>(in);
            if (rotate > 1 || dim > UINT32_MAX) throw std::runtime_error("corrupt file: invalid IVFIndex rotation");
            index->rotate_ = rotate;
            index->rotation_ = read_vec<float>(in, dim * dim);
            if (index->rotation_.size() != (rotate && trained ? dim * dim : 0))
                throw std::runtime_error("corrupt file: IVFIndex rotation length mismatch");
            if (!index->rotation_.empty()) {
                if (!detail::is_orthogonal(index->rotation_, dim))
                    throw std::runtime_error("corrupt file: IVFIndex rotation is not orthogonal");
                index->rotation_panels_ = detail::pack_rotation(index->rotation_, dim);
            }
        }
    }

    if (version >= 7) {
        const auto bits = read_pod<uint8_t>(in);
        if ((bits != 0 && bits != 4 && bits != 8) || (bits && index->pq_))
            throw std::runtime_error("corrupt file: invalid IVFIndex SQ settings");
        if (bits) {
            index->enable_sq(bits);
            index->sq_.vmin = read_vec<float>(in, dim);
            index->sq_.step = read_vec<float>(in, dim);
            const size_t ranges = trained ? dim : 0;
            if (index->sq_.vmin.size() != ranges || index->sq_.step.size() != ranges)
                throw std::runtime_error("corrupt file: IVF SQ range length mismatch");
        }
    }

    const size_t num_subspaces = index->pq_ ? index->pq_->num_subspaces() : 0;
    uint64_t seen = 0;
    for (size_t i = 0; i < n_list; ++i) {
        InvertedList list;
        list.ids = read_vec<uint32_t>(in, n_total);

        if (index->sq_bits_) {
            const uint64_t expected = list.ids.size() * index->sq_code_bytes();
            list.codes = read_vec<uint8_t>(in, expected);
            if (list.codes.size() != expected) throw std::runtime_error("corrupt file: IVF code length mismatch");
        } else if (index->pq_) {
            const uint64_t blocks = (list.ids.size() + detail::kFsBlock - 1) / detail::kFsBlock;
            const uint64_t expected = index->fast_scan_ ? blocks * detail::fs_block_bytes(num_subspaces)
                                                        : list.ids.size() * num_subspaces;
            list.codes = read_vec<uint8_t>(in, expected);
            if (list.codes.size() != expected) throw std::runtime_error("corrupt file: IVF code length mismatch");
        } else {
            const uint64_t expected = list.ids.size() * dim;
            list.vecs = read_vec<float>(in, expected);
            if (list.vecs.size() != expected) throw std::runtime_error("corrupt file: IVF vector length mismatch");
        }

        for (uint32_t id : list.ids)
            if (id >= n_total) throw std::runtime_error("corrupt file: IVF id out of range");

        seen += list.ids.size();
        if (!list.ids.empty()) index->data_.emplace(static_cast<uint16_t>(i), std::move(list));
    }
    if (seen != n_total) throw std::runtime_error("corrupt file: invalid IVFIndex shape");

    index->trained_ = trained;
    index->pq_distance_ = static_cast<PQDistance>(pq_distance);

    if (version >= 2) index->id_map_ = detail::IdMap::load(in, n_total, version);
    else index->id_map_.add(n_total, nullptr);

    if (version >= 3) {
        const auto use_precomputed = read_pod<uint8_t>(in);
        if (use_precomputed > 1) throw std::runtime_error("corrupt file: invalid IVFIndex shape");
        index->use_precomputed_ = use_precomputed;
    }
    if (version >= 5) index->metric_ = read_metric(in);
    if (version >= 9) index->metadata_ = MetadataStore::load(in, n_total);
    index->build_precomputed();
    index->rebuild_slot_lookup();

    return index;
}

}
