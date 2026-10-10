#include "vecengine/ivf_index.hpp"
#include "vecengine/distance.hpp"
#include "vecengine/pq.hpp"
#include "kmeans.hpp"
#include "parallel.hpp"
#include "profile.hpp"
#include "topk.hpp"
#include "serialize.hpp"

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

IVFIndex::IVFIndex(size_t dim, size_t nlist, size_t nprobe, bool custom_ids): dim_(dim), nlist_(nlist), nprobe_(nprobe), id_map_(custom_ids)  {
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
    if (!use_precomputed_ || !trained_ || !pq_enabled()) return;

    size_t list_size = pq_->num_subspaces() * pq_->centroids_per_subspace();
    if (nlist_ * list_size * sizeof(float) > kPrecomputedMaxBytes) return;
    
    precomputed_.resize(nlist_ * list_size);
    detail::parallel_for(nlist_, [&](size_t i) {
        pq_->compute_list_term({coarse_centroids_.data() + i * dim_, dim_}, {precomputed_.data() + i * list_size, list_size});
    });
}

void IVFIndex::enable_pq(size_t num_subspaces, size_t centroids_per_subspace) {
    if (id_map_.slots() > 0 || trained_)
        throw std::logic_error("enable_pq must be called before train()/add()");
    pq_ = std::make_unique<PQCodebook>(dim_, num_subspaces, centroids_per_subspace);
}

void IVFIndex::add(span<const float> vec) {
    assert(vec.size() == dim_);
    if (!trained_) throw std::logic_error("train() has not be run yet");

    // add a vector to the inverted list of the closest coarse centroid
    size_t closest_index = vecengine::detail::nearest_centroid(vec.data(), coarse_centroids_.data(), nlist_, dim_);

    uint16_t cluster_id = static_cast<uint16_t>(closest_index);
    const uint32_t slot = static_cast<uint32_t>(id_map_.slots());
    id_map_.add(1, nullptr);

    InvertedList& list = data_[cluster_id];
    
    list.ids.emplace_back(slot);

    // adds quantized vectors if pq enabled else full dimensional vectors 
    if (pq_enabled()) {
        vector<float> residual(dim_);
        const float* centroid_ptr = coarse_centroids_.data() + cluster_id * dim_;
        for (size_t i = 0; i < dim_; ++i) {
            residual[i] = vec[i] - centroid_ptr[i];
        }

        auto code = pq_->encode(residual);
        list.codes.insert(list.codes.end(), code.begin(), code.end());
    } else {
        list.vecs.insert(list.vecs.end(), vec.begin(), vec.end());
    }
}

void IVFIndex::add_batch(span<const float> vecs, size_t n, const int64_t* ids) {
    assert(vecs.size() == n * dim());
    if (!trained_) throw std::logic_error("train() has not be run yet");

    const uint32_t first_slot = static_cast<uint32_t>(id_map_.slots());
    id_map_.add(n, ids);

    vector<uint32_t> assign(n);
    detail::assign_nearest(vecs.data(), n, dim_, coarse_centroids_.data(), nlist_, dim_, assign.data());

    const bool use_pq = pq_enabled();
    const size_t code_bytes = use_pq ? pq_->num_subspaces() : 0;
    
    vector<uint8_t> codes(use_pq ? n * code_bytes : 0);

    if (use_pq) {
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

        if (use_pq) {
            list.codes.insert(list.codes.end(), codes.begin() + i * code_bytes, codes.begin() + (i + 1) * code_bytes);
        } else {
            const float* src = vecs.data() + i * dim_;
            list.vecs.insert(list.vecs.end(), src, src + dim_);
        }
    }
}

vector<Neighbor> IVFIndex::query(span<const float> vec, size_t k) const {
    assert(vec.size() == dim_);
    if (!trained_) throw std::logic_error("train() has not be run yet");

    VECENGINE_PROF_START(t_coarse);
    vector<Probe> probes = coarse(vec);
    VECENGINE_PROF_STOP(t_coarse, kProfCoarse);

    return scan(vec, probes, k);
}

vector<vector<Neighbor>> IVFIndex::query_batch(span<const float> queries, size_t num_queries, size_t k) const {
    assert(queries.size() == num_queries * dim_);
    if (!trained_) throw std::logic_error("train() has not be run yet");

#ifndef VECENGINE_USE_BLAS
    return Index::query_batch(queries, num_queries, k);
#else
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
            results[start + i] = scan({q + i * dim_, dim_}, {probes.data() + i * nprobe_, nprobe_}, k);
        });
    }
    return results;
#endif
}

vector<IVFIndex::Probe> IVFIndex::coarse(span<const float> vec) const {
    // find the closest nprobe coarse centroids
    std::priority_queue<pair<float, uint16_t>, vector<pair<float, uint16_t>>, std::less<pair<float, uint16_t>>> maxCentroidHeap;
    for (size_t i = 0; i < nlist_; ++i) {
        const float* centroid_ptr = coarse_centroids_.data() + i * dim_;
        float dist = l2_distance(vec.data(), centroid_ptr, dim_);

        if (maxCentroidHeap.size() < nprobe_) {
            maxCentroidHeap.emplace(dist, static_cast<uint16_t>(i));
        } else if (dist < maxCentroidHeap.top().first) {
            maxCentroidHeap.pop();
            maxCentroidHeap.emplace(dist, static_cast<uint16_t>(i));
        }
    }

    vector<Probe> probes;
    probes.reserve(maxCentroidHeap.size());
    while (!maxCentroidHeap.empty()) {
        probes.push_back({maxCentroidHeap.top().first, maxCentroidHeap.top().second});
        maxCentroidHeap.pop();
    }
    return probes;
}

#ifdef VECENGINE_USE_BLAS
void IVFIndex::coarse_batch(const float* queries, size_t n, Probe* out) const {
    vector<float> norms(nlist_);
    for (size_t c = 0; c < nlist_; ++c) {
        const float* centroid = coarse_centroids_.data() + c * dim_;
        float sum = 0.0f;
        for (size_t d = 0; d < dim_; ++d) sum += centroid[d] * centroid[d];
        norms[c] = sum;
    }

    // dots[i * nlist_ + c] = q_i . c for the whole block in one call
    vector<float> dots(n * nlist_);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(n), static_cast<int>(nlist_), static_cast<int>(dim_),
                1.0f, queries, static_cast<int>(dim_),
                coarse_centroids_.data(), static_cast<int>(dim_),
                0.0f, dots.data(), static_cast<int>(nlist_));

    detail::parallel_for(n, [&](size_t i) {
        // ||q - c||^2 = ||q||^2 + ||c||^2 - 2 q.c, and ||q||^2 is the same for
        // every list, so ||c||^2 - 2 q.c ranks the lists
        const float* row = dots.data() + i * nlist_;
        vector<pair<float, uint16_t>> scored(nlist_);
        for (size_t c = 0; c < nlist_; ++c) scored[c] = {norms[c] - 2.0f * row[c], static_cast<uint16_t>(c)};
        std::nth_element(scored.begin(), scored.begin() + (nprobe_ - 1), scored.end());

        // the scores lose precision to cancellation, so A is recomputed exactly
        const float* q = queries + i * dim_;
        for (size_t j = 0; j < nprobe_; ++j) {
            const uint16_t list = scored[j].second;
            out[i * nprobe_ + j] = {l2_distance(q, coarse_centroids_.data() + list * dim_, dim_), list};
        }
    });
}
#endif

vector<Neighbor> IVFIndex::scan(span<const float> vec, span<const Probe> probes, size_t k) const {
    // intialize vars for pq, reuse allocated memory for adc table
    const size_t m = pq_enabled() ? pq_->num_subspaces() : 0;
    const size_t K = pq_enabled() ? pq_->centroids_per_subspace() : 0;
    vector<float> table(pq_enabled() ? m * K : 0);
    vector<float> dists;
    detail::TopK top(k);

    const bool use_pre = pq_enabled() && pq_distance_ == PQDistance::ADC && precomputed_tables();
    vector<float> C(use_pre ? m * K : 0);
    if (use_pre) pq_->compute_inner_table(vec, C);

    // find top-k vectors from the inverted lists associated with the nprobe closest coarse centroids
    for (const Probe& probe : probes) {
        const float A = probe.dist;
        auto it = data_.find(probe.list);
        if (it == data_.end()) continue;

        const InvertedList& list = it->second;
        size_t listSize = list.ids.size();
        dists.resize(listSize);

        // lamdba function for adding to heap
        auto push = [&](float dist, size_t j) {
            if (dist < top.threshold() && !id_map_.is_deleted(list.ids[j]))
                top.push(dist, id_map_.label(list.ids[j]));
        };

        if (!pq_enabled()) {
            VECENGINE_PROF_START(t_raw);
            for (size_t j = 0; j < listSize; ++j) {
                const float* start = list.vecs.data() + j * dim_;
                push(l2_distance(vec.data(), start, dim_), j);
            }
            VECENGINE_PROF_STOP(t_raw, kProfRawScan);
            continue;
        }

        vector<float> residual(use_pre ? 0 : dim_);
        VECENGINE_PROF_START(t_table);
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
            pq_->distances_adc(table, list.codes.data(), listSize, dists.data());
            for (size_t j = 0; j < listSize; ++j) {
                push(dists[j], j);
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

void IVFIndex::train(span<const float> vectors, size_t num_vectors, size_t max_iters, std::optional<uint32_t> seed) {
    if (id_map_.slots() > 0) throw std::logic_error("train() cannot be called after add()");
    if (trained_) throw std::logic_error("train() has already been called");
    if (num_vectors < nlist_) throw std::invalid_argument("training set must have at least nlist vectors");

    // run kmeans
    auto res = vecengine::detail::kmeans(vectors.data(), num_vectors, dim_, dim_, nlist_, max_iters, seed);
    coarse_centroids_ = std::move(res.centroids);

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
    if (pq_) pq_->save(out);

    static const InvertedList empty;
    for (uint16_t i = 0; i < nlist_; ++i) {
        auto it = data_.find(i);
        const InvertedList& list = it == data_.end() ? empty : it->second;
        write_vec(out, list.ids);
        if (pq_) write_vec(out, list.codes); else write_vec(out, list.vecs);
    }
    id_map_.save(out);
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

    if (pq_distance > static_cast<uint8_t>(PQDistance::SDC)) throw std::runtime_error("corrupt file: invalid IVFIndex shape");
    if (!trained && n_total > 0) throw std::runtime_error("corrupt file: invalid IVFIndex shape");
    if ((trained && index->coarse_centroids_.size() != n_list * dim) || (!trained && index->coarse_centroids_.size() != 0)) throw std::runtime_error("corrupt file: invalid IVFIndex shape");

    auto pq_enabled = read_pod<uint8_t>(in);
    if (pq_enabled) {
        index->pq_ = std::make_unique<PQCodebook>(PQCodebook::load(in));
        if (index->pq_->dim() != dim) throw std::runtime_error("corrupt file: invalid IVFIndex shape");
    }

    const size_t num_subspaces = index->pq_ ? index->pq_->num_subspaces() : 0;
    uint64_t seen = 0;
    for (size_t i = 0; i < n_list; ++i) {
        InvertedList list;
        list.ids = read_vec<uint32_t>(in, n_total);

        if (index->pq_) {
            const uint64_t expected = list.ids.size() * num_subspaces;
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

    if (version >= 2) index->id_map_ = detail::IdMap::load(in, n_total);
    else index->id_map_.add(n_total, nullptr);

    index->build_precomputed();

    return index;
}

}
