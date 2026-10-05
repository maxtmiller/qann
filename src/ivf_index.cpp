#include "vecengine/ivf_index.hpp"
#include "vecengine/distance.hpp"
#include "vecengine/pq.hpp"
#include "kmeans.hpp"
#include "parallel.hpp"
#include "profile.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>
#include <span>
#include <cassert>
#include <queue>
#include <utility>
#include <stdexcept>
#include <cmath>

namespace vecengine {

using std::size_t;
using std::vector;
using std::span;
using std::pair;

IVFIndex::IVFIndex(size_t dim, size_t nlist, size_t nprobe): dim_(dim), nlist_(nlist), nprobe_(nprobe)  {
    if (nlist < 100 || nlist > 65535) throw std::invalid_argument("nlist must be in [100, 65535]");
    coarse_centroids_.reserve(dim_ * nlist_);
}

void IVFIndex::set_nprobe(size_t nprobe) {
    if (nprobe == 0 || nprobe > nlist_) throw std::invalid_argument("nprobe must be in [1, nlist]");
    nprobe_ = nprobe;
}

void IVFIndex::enable_pq(size_t num_subspaces, size_t centroids_per_subspace) {
    if (n_total_ > 0 || trained_)
        throw std::logic_error("enable_pq must be called before train()/add()");
    pq_ = std::make_unique<PQCodebook>(dim_, num_subspaces, centroids_per_subspace);
}

void IVFIndex::add(span<const float> vec) {
    assert(vec.size() == dim_);
    if (!trained_) throw std::logic_error("train() has not be run yet");

    // add a vector to the inverted list of the closest coarse centroid
    size_t closest_index = vecengine::detail::nearest_centroid(vec.data(), coarse_centroids_.data(), nlist_, dim_);

    uint16_t cluster_id = static_cast<uint16_t>(closest_index);
    uint32_t new_id = n_total_++;

    InvertedList& list = data_[cluster_id];
    
    list.ids.emplace_back(new_id);

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

void IVFIndex::add_batch(span<const float> vecs, size_t n) {
    assert(vecs.size() == n * dim());
    if (!trained_) throw std::logic_error("train() has not be run yet");

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
        list.ids.push_back(n_total_++);

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
    assert(k <= n_total_);
    if (!trained_) throw std::logic_error("train() has not be run yet");

    VECENGINE_PROF_START(t_coarse);
    // find the closest nprobe coarse centroids
    std::priority_queue<pair<float, uint16_t>, vector<pair<float, uint16_t>>, std::less<pair<float, uint16_t>>> maxCentroidHeap;
    for (size_t i = 0; i < nlist_; ++i) {
        const float* centroid_ptr = coarse_centroids_.data() + i * dim_;
        float dist = l2_distance(vec.data(), centroid_ptr, dim_);
        
        if (maxCentroidHeap.size() < nprobe_) {
            maxCentroidHeap.emplace(dist, i);
        } else if (dist < maxCentroidHeap.top().first) {
            maxCentroidHeap.pop();
            maxCentroidHeap.emplace(dist, static_cast<uint16_t>(i));
        }
    }

    VECENGINE_PROF_STOP(t_coarse, kProfCoarse);

    // intialize vars for pq, reuse allocated memory for adc table
    const size_t m = pq_enabled() ? pq_->num_subspaces() : 0;
    vector<float> table(pq_enabled() ? m * pq_->centroids_per_subspace() : 0);
    vector<float> dists;

    // find top-k vectors from the inverted lists associated with the nprobe closest coarse centroids
    const size_t n1 = maxCentroidHeap.size();
    std::priority_queue<pair<float, size_t>, vector<pair<float, size_t>>, std::less<pair<float, size_t>>> maxVectorHeap;
    for (size_t i = 0; i < n1; ++i) {
        auto it = data_.find(maxCentroidHeap.top().second);
        maxCentroidHeap.pop();
        if (it == data_.end()) continue;

        const InvertedList& list = it->second;
        size_t listSize = list.ids.size();
        dists.resize(listSize);

        // lamdba function for adding to heap
        auto push = [&](float dist, size_t j) {
            if (maxVectorHeap.size() < k) {
                maxVectorHeap.emplace(dist, list.ids[j]);
            } else if (dist < maxVectorHeap.top().first) {
                maxVectorHeap.pop();
                maxVectorHeap.emplace(dist, list.ids[j]);
            }
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

        VECENGINE_PROF_START(t_table);
        // calculates residuals 
        vector<float> residual(dim_);
        const float* centroid_ptr = coarse_centroids_.data() + it->first * dim_;
        for (size_t l = 0; l < dim_; ++l) {
            residual[l] = vec[l] - centroid_ptr[l];
        }

        // calculate distance with ADC or SDC
        if (pq_distance_ == PQDistance::ADC) {
            pq_->compute_adc_table(residual, table);
            VECENGINE_PROF_STOP(t_table, kProfTable);
            VECENGINE_PROF_START(t_scan);
            pq_->distances_adc(table, list.codes.data(), listSize, dists.data());
            float worst = maxVectorHeap.size() < k ? INFINITY : maxVectorHeap.top().first;
            for (size_t j = 0; j < listSize; ++j) {
                if (dists[j] >= worst) continue;
                push(dists[j], j);
                worst = maxVectorHeap.size() < k ? INFINITY : maxVectorHeap.top().first;
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
    // add top-k vectors to results array
    const size_t n2 = maxVectorHeap.size();
    vector<Neighbor> results(n2);
    for (int i = static_cast<int>(n2) - 1; i >= 0; --i) {
        auto [dist, idx] = maxVectorHeap.top();
        results[i] = {idx,dist};
        maxVectorHeap.pop();
    }
    VECENGINE_PROF_STOP(t_drain, kProfDrain);

    return results;
}

void IVFIndex::train(span<const float> vectors, size_t num_vectors, size_t max_iters, std::optional<uint32_t> seed) {
    if (n_total_ > 0) throw std::logic_error("train() cannot be called after add()");
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
}


}
