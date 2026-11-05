#pragma once
/*
 * bme_branch_lengths_gpu.cuh
 *
 * O(N) GPU Balanced Minimum Evolution (FastME / Pauplin 2000 / Desper & Gascuel 2002)
 * branch-length estimation via Exact Clade Condensation (Theorem 1) and bottom-up /
 * top-down dyadic leaf-weight profiles.
 *
 * Total profile-pair evaluations for the ENTIRE N-leaf tree: 3*(N - 2) pairs = O(N).
 * Uses a fast 1-to-C parallel path when either profile is a singleton leaf (covering all N
 * pendant edges) and a shared-memory staged 4x4 register-tiled kernel when both profiles
 * have >= 2 leaves.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

namespace p2v_bme_bl {

struct LeafWeight {
    int32_t leaf;
    float   w;
};

inline int32_t merge_profiles_capped(
    const LeafWeight* __restrict__ p1, int32_t n1,
    const LeafWeight* __restrict__ p2, int32_t n2,
    LeafWeight* __restrict__ out, int32_t cap
) {
    int32_t i = 0, j = 0, k = 0;
    double wsum = 0.0;
    while (k < cap && (i < n1 || j < n2)) {
        bool take1 = (i < n1 && j < n2) ? (p1[i].w >= p2[j].w) : (i < n1);
        if (take1) {
            float nw = 0.5f * p1[i].w;
            out[k++] = {p1[i].leaf, nw};
            wsum += nw;
            ++i;
        } else {
            float nw = 0.5f * p2[j].w;
            out[k++] = {p2[j].leaf, nw};
            wsum += nw;
            ++j;
        }
    }
    if (k > 0 && wsum > 0.0) {
        float inv = (float)(1.0 / wsum);
        for (int32_t m = 0; m < k; ++m) out[m].w *= inv;
    }
    return k;
}

__device__ __forceinline__ float dev_jc69_from_nv_nm(int32_t n_valid, int32_t n_match) {
    if (n_valid <= 0) return 1.5f;
    float p = (float)(n_valid - n_match) / (float)n_valid;
    if (p >= 0.74f) p = 0.74f;
    float d = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
    return (d > 0.0f) ? d : 0.0f;
}

__device__ __forceinline__ float dev_jc69_bl_popc(
    const uint64_t* __restrict__ d_packed_seqs,
    int32_t W, int32_t leaf_u, int32_t leaf_v
) {
    if (leaf_u == leaf_v) return 0.0f;
    const uint64_t* su = d_packed_seqs + (size_t)leaf_u * 5 * W;
    const uint64_t* sv = d_packed_seqs + (size_t)leaf_v * 5 * W;
    int32_t n_valid = 0;
    int32_t n_match = 0;
    for (int32_t w = 0; w < W; ++w) {
        uint64_t vu = su[4 * W + w];
        uint64_t vv = sv[4 * W + w];
        n_valid += __popcll(vu & vv);
        uint64_t m = (su[0 * W + w] & sv[0 * W + w])
                   | (su[1 * W + w] & sv[1 * W + w])
                   | (su[2 * W + w] & sv[2 * W + w])
                   | (su[3 * W + w] & sv[3 * W + w]);
        n_match += __popcll(m);
    }
    return dev_jc69_from_nv_nm(n_valid, n_match);
}

constexpr int32_t BL_TILE = 32;
constexpr int32_t BL_CW   = 4;

/* 64 threads per block (8x8 grid of 4x4 register tiles covering 32x32 tiles).
 * One block evaluates one (local_i, kind) job in [0, 3 * n_int_chunk).
 *   - Fast path when len1 == 1 or len2 == 1: 64 threads evaluate up to 64 leaf pairs in 1 step.
 *   - Tiled shared-memory path when len1 > 1 and len2 > 1: stages sequences into sA/sB in
 *     chunks of BL_CW = 4 words and accumulates 4x4 popcount tiles in registers. */
__global__ __launch_bounds__(64) void k_eval_internal_node_triples_tiled(
    int32_t n_int_chunk,
    int32_t N,
    int32_t W,
    int32_t cap,
    const int32_t* __restrict__ d_c1,
    const int32_t* __restrict__ d_c2,
    const int32_t* __restrict__ d_in_len,
    const int32_t* __restrict__ d_in_leaves,
    const float*   __restrict__ d_in_weights,
    const int32_t* __restrict__ d_out_len,
    const int32_t* __restrict__ d_out_leaves,
    const float*   __restrict__ d_out_weights,
    const uint64_t* __restrict__ d_packed_seqs,
    float* __restrict__ d_dAB,
    float* __restrict__ d_dAO,
    float* __restrict__ d_dBO
) {
    __shared__ uint64_t sA[5 * BL_CW * BL_TILE];
    __shared__ uint64_t sB[5 * BL_CW * BL_TILE];
    __shared__ int32_t  s_la[BL_TILE], s_lb[BL_TILE];
    __shared__ float    s_wa[BL_TILE], s_wb[BL_TILE];
    __shared__ float    s_warp_sum[2];

    const int32_t job_id = (int32_t)blockIdx.x;
    if (job_id >= 3 * n_int_chunk) return;

    const int32_t local_i = job_id / 3;
    const int32_t kind    = job_id - local_i * 3; /* 0: AB, 1: AO, 2: BO */
    const int32_t tid     = (int32_t)threadIdx.x;
    const int32_t tx      = tid & 7;
    const int32_t ty      = tid >> 3;

    const int32_t u_left  = d_c1[local_i];
    const int32_t u_right = d_c2[local_i];

    int32_t len1 = 0, len2 = 0;
    int32_t single_leaf1 = -1, single_leaf2 = -1;
    const int32_t* leaves1  = nullptr;
    const float*   weights1 = nullptr;
    const int32_t* leaves2  = nullptr;
    const float*   weights2 = nullptr;

    int32_t node1 = (kind == 2) ? u_right : u_left;
    if (node1 < N) {
        len1 = 1;
        single_leaf1 = node1;
    } else {
        int32_t idx1 = node1 - N;
        len1 = d_in_len[idx1];
        leaves1  = d_in_leaves  + (size_t)idx1 * cap;
        weights1 = d_in_weights + (size_t)idx1 * cap;
    }

    if (kind == 0) {
        if (u_right < N) {
            len2 = 1;
            single_leaf2 = u_right;
        } else {
            int32_t idx2 = u_right - N;
            len2 = d_in_len[idx2];
            leaves2  = d_in_leaves  + (size_t)idx2 * cap;
            weights2 = d_in_weights + (size_t)idx2 * cap;
        }
    } else {
        len2 = d_out_len[local_i];
        leaves2  = d_out_leaves  + (size_t)local_i * cap;
        weights2 = d_out_weights + (size_t)local_i * cap;
    }

    float thread_sum = 0.0f;

    /* Fast path: at least one side has length <= 2 (covers all N pendant leaf edges!) */
    if (len1 <= 2 || len2 <= 2) {
        const int32_t total = len1 * len2;
        for (int32_t idx = tid; idx < total; idx += 64) {
            const int32_t i = idx / len2;
            const int32_t j = idx - i * len2;
            const int32_t lu = (single_leaf1 >= 0) ? single_leaf1 : leaves1[i];
            const float   wu = (single_leaf1 >= 0) ? 1.0f         : weights1[i];
            const int32_t lv = (single_leaf2 >= 0) ? single_leaf2 : leaves2[j];
            const float   wv = (single_leaf2 >= 0) ? 1.0f         : weights2[j];
            thread_sum += wu * wv * dev_jc69_bl_popc(d_packed_seqs, W, lu, lv);
        }
    } else {
        /* Shared-memory staged 32x32 tiled path with 4x4 register tiles per thread */
        for (int32_t pa0 = 0; pa0 < len1; pa0 += BL_TILE) {
            const int32_t na = min(BL_TILE, len1 - pa0);
            if (tid < na) {
                s_la[tid] = leaves1[pa0 + tid];
                s_wa[tid] = weights1[pa0 + tid];
            }
            for (int32_t qb0 = 0; qb0 < len2; qb0 += BL_TILE) {
                const int32_t nb = min(BL_TILE, len2 - qb0);
                if (tid < nb) {
                    s_lb[tid] = leaves2[qb0 + tid];
                    s_wb[tid] = weights2[qb0 + tid];
                }
                __syncthreads();

                int32_t nv[4][4], nm[4][4];
                #pragma unroll
                for (int32_t i = 0; i < 4; ++i)
                    #pragma unroll
                    for (int32_t j = 0; j < 4; ++j) { nv[i][j] = 0; nm[i][j] = 0; }

                for (int32_t w0 = 0; w0 < W; w0 += BL_CW) {
                    for (int32_t e = tid; e < 5 * BL_CW * BL_TILE; e += 64) {
                        const int32_t cw = e % BL_CW;
                        const int32_t ch = (e / BL_CW) % 5;
                        const int32_t r  = e / (5 * BL_CW);
                        const int32_t s_idx = (ch * BL_CW + cw) * BL_TILE + r;
                        const int32_t w = w0 + cw;
                        const size_t g_off = (size_t)ch * W + w;
                        sA[s_idx] = (r < na && w < W) ? d_packed_seqs[(size_t)s_la[r] * 5 * W + g_off] : 0ULL;
                        sB[s_idx] = (r < nb && w < W) ? d_packed_seqs[(size_t)s_lb[r] * 5 * W + g_off] : 0ULL;
                    }
                    __syncthreads();

                    #pragma unroll
                    for (int32_t cw = 0; cw < BL_CW; ++cw) {
                        uint64_t ra[4], rb[4], m[4][4];
                        #pragma unroll
                        for (int32_t i = 0; i < 4; ++i) ra[i] = sA[(4 * BL_CW + cw) * BL_TILE + ty + 8 * i];
                        #pragma unroll
                        for (int32_t j = 0; j < 4; ++j) rb[j] = sB[(4 * BL_CW + cw) * BL_TILE + tx + 8 * j];
                        #pragma unroll
                        for (int32_t i = 0; i < 4; ++i)
                            #pragma unroll
                            for (int32_t j = 0; j < 4; ++j) nv[i][j] += __popcll(ra[i] & rb[j]);

                        #pragma unroll
                        for (int32_t c = 0; c < 4; ++c) {
                            #pragma unroll
                            for (int32_t i = 0; i < 4; ++i) ra[i] = sA[(c * BL_CW + cw) * BL_TILE + ty + 8 * i];
                            #pragma unroll
                            for (int32_t j = 0; j < 4; ++j) rb[j] = sB[(c * BL_CW + cw) * BL_TILE + tx + 8 * j];
                            #pragma unroll
                            for (int32_t i = 0; i < 4; ++i)
                                #pragma unroll
                                for (int32_t j = 0; j < 4; ++j)
                                    m[i][j] = (c == 0) ? (ra[i] & rb[j]) : (m[i][j] | (ra[i] & rb[j]));
                        }
                        #pragma unroll
                        for (int32_t i = 0; i < 4; ++i)
                            #pragma unroll
                            for (int32_t j = 0; j < 4; ++j) nm[i][j] += __popcll(m[i][j]);
                    }
                    __syncthreads();
                }

                #pragma unroll
                for (int32_t i = 0; i < 4; ++i) {
                    const int32_t p = ty + 8 * i;
                    if (p < na) {
                        const int32_t lp = s_la[p];
                        const float   wp = s_wa[p];
                        #pragma unroll
                        for (int32_t j = 0; j < 4; ++j) {
                            const int32_t q = tx + 8 * j;
                            if (q < nb) {
                                const int32_t lq = s_lb[q];
                                const float   wq = s_wb[q];
                                const float d = (lp == lq) ? 0.0f : dev_jc69_from_nv_nm(nv[i][j], nm[i][j]);
                                thread_sum += wp * wq * d;
                            }
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    /* Reduce across the 64 threads (2 warps) */
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffffu, thread_sum, offset);
    }
    if ((tid & 31) == 0) {
        s_warp_sum[tid >> 5] = thread_sum;
    }
    __syncthreads();
    if (tid == 0) {
        float total_sum = s_warp_sum[0] + s_warp_sum[1];
        if (kind == 0)      d_dAB[local_i] = total_sum;
        else if (kind == 1) d_dAO[local_i] = total_sum;
        else                d_dBO[local_i] = total_sum;
    }
}

inline double compute_bme_branch_lengths_gpu(
    int32_t N,
    int32_t W,
    const int32_t* h_c1,
    const int32_t* h_c2,
    const uint64_t* const* d_packed_seqs_per_gpu,
    int32_t n_gpus,
    std::vector<float>& h_branch_lengths,
    int32_t cap = 32,
    double* out_profile_sec = nullptr,
    double* out_kernel_sec = nullptr
) {
    const double t0 = omp_get_wtime();
    const int32_t root = 2 * N - 2;
    const int32_t n_int = N - 2; /* internal nodes N .. 2N-3 */
    h_branch_lengths.assign(2 * N - 2, 0.0f);
    if (N <= 2) return 0.0;

    /* 1. Build post-order traversal of internal nodes N .. 2N-2 and parent pointers */
    std::vector<int32_t> parent(2 * N - 1, -1);
    std::vector<int32_t> postorder;
    postorder.reserve(N - 1);
    {
        std::vector<int32_t> stack;
        stack.reserve(2 * N - 1);
        stack.push_back(root);
        while (!stack.empty()) {
            int32_t u = stack.back();
            stack.pop_back();
            if (u >= N) {
                postorder.push_back(u);
                int32_t l = h_c1[u], r = h_c2[u];
                parent[l] = u;
                parent[r] = u;
                stack.push_back(l);
                stack.push_back(r);
            }
        }
        std::reverse(postorder.begin(), postorder.end());
    }

    /* 2. Allocate flat inside and outside profiles for the N - 2 internal nodes N .. 2N-3 */
    std::vector<int32_t> in_len(n_int, 0), out_len(n_int, 0);
    std::vector<int32_t> in_leaves((size_t)n_int * cap, 0), out_leaves((size_t)n_int * cap, 0);
    std::vector<float>   in_weights((size_t)n_int * cap, 0.0f), out_weights((size_t)n_int * cap, 0.0f);

    std::vector<LeafWeight> tmp_a(cap), tmp_b(cap), tmp_out(cap);

    auto get_in_profile = [&](int32_t u, LeafWeight* buf) -> int32_t {
        if (u < N) {
            buf[0] = {u, 1.0f};
            return 1;
        }
        int32_t idx = u - N;
        int32_t l = in_len[idx];
        const int32_t* lv = &in_leaves[(size_t)idx * cap];
        const float*   wv = &in_weights[(size_t)idx * cap];
        for (int32_t k = 0; k < l; ++k) buf[k] = {lv[k], wv[k]};
        return l;
    };

    for (int32_t u : postorder) {
        if (u == root) continue;
        int32_t idx = u - N;
        int32_t na = get_in_profile(h_c1[u], tmp_a.data());
        int32_t nb = get_in_profile(h_c2[u], tmp_b.data());
        int32_t nm = merge_profiles_capped(tmp_a.data(), na, tmp_b.data(), nb, tmp_out.data(), cap);
        in_len[idx] = nm;
        int32_t* lv = &in_leaves[(size_t)idx * cap];
        float*   wv = &in_weights[(size_t)idx * cap];
        for (int32_t k = 0; k < nm; ++k) {
            lv[k] = tmp_out[k].leaf;
            wv[k] = tmp_out[k].w;
        }
    }

    const int32_t r_left  = h_c1[root];
    const int32_t r_right = h_c2[root];
    auto set_out_profile = [&](int32_t u, const LeafWeight* buf, int32_t l) {
        if (u < N) return;
        int32_t idx = u - N;
        out_len[idx] = l;
        int32_t* lv = &out_leaves[(size_t)idx * cap];
        float*   wv = &out_weights[(size_t)idx * cap];
        for (int32_t k = 0; k < l; ++k) {
            lv[k] = buf[k].leaf;
            wv[k] = buf[k].w;
        }
    };

    {
        int32_t nr = get_in_profile(r_right, tmp_a.data());
        set_out_profile(r_left, tmp_a.data(), nr);
        int32_t nl = get_in_profile(r_left, tmp_b.data());
        set_out_profile(r_right, tmp_b.data(), nl);
    }

    for (int32_t pi = (int32_t)postorder.size() - 1; pi >= 0; --pi) {
        int32_t u = postorder[pi];
        if (u == root) continue;
        int32_t idx = u - N;
        int32_t no = out_len[idx];
        for (int32_t k = 0; k < no; ++k) {
            tmp_a[k] = {out_leaves[(size_t)idx * cap + k], out_weights[(size_t)idx * cap + k]};
        }
        int32_t c_left  = h_c1[u];
        int32_t c_right = h_c2[u];
        if (c_left >= N) {
            int32_t nb = get_in_profile(c_right, tmp_b.data());
            int32_t nm = merge_profiles_capped(tmp_a.data(), no, tmp_b.data(), nb, tmp_out.data(), cap);
            set_out_profile(c_left, tmp_out.data(), nm);
        }
        if (c_right >= N) {
            int32_t na = get_in_profile(c_left, tmp_b.data());
            int32_t nm = merge_profiles_capped(tmp_a.data(), no, tmp_b.data(), na, tmp_out.data(), cap);
            set_out_profile(c_right, tmp_out.data(), nm);
        }
    }

    const double t1 = omp_get_wtime();
    if (out_profile_sec) *out_profile_sec = t1 - t0;

    /* 3. Multi-GPU tiled evaluation of d_AB(p), d_AO(p), d_BO(p) for all p in 0..n_int-1 */
    std::vector<float> h_dAB(n_int, 0.0f), h_dAO(n_int, 0.0f), h_dBO(n_int, 0.0f);
    int32_t active_gpus = std::max(1, std::min(n_gpus, n_int));

    #pragma omp parallel num_threads(active_gpus)
    {
        int g = omp_get_thread_num();
        cudaSetDevice(g);

        int32_t chunk = (n_int + active_gpus - 1) / active_gpus;
        int32_t start = g * chunk;
        int32_t end   = std::min(n_int, start + chunk);
        int32_t count = std::max(0, end - start);

        if (count > 0) {
            int32_t* d_c1_g = nullptr;
            int32_t* d_c2_g = nullptr;
            int32_t* d_in_len_g = nullptr;
            int32_t* d_in_leaves_g = nullptr;
            float*   d_in_weights_g = nullptr;
            int32_t* d_out_len_g = nullptr;
            int32_t* d_out_leaves_g = nullptr;
            float*   d_out_weights_g = nullptr;
            float*   d_dAB_g = nullptr;
            float*   d_dAO_g = nullptr;
            float*   d_dBO_g = nullptr;

            cudaMalloc(&d_c1_g, (size_t)count * sizeof(int32_t));
            cudaMalloc(&d_c2_g, (size_t)count * sizeof(int32_t));
            cudaMalloc(&d_in_len_g, (size_t)n_int * sizeof(int32_t));
            cudaMalloc(&d_in_leaves_g, (size_t)n_int * cap * sizeof(int32_t));
            cudaMalloc(&d_in_weights_g, (size_t)n_int * cap * sizeof(float));
            cudaMalloc(&d_out_len_g, (size_t)count * sizeof(int32_t));
            cudaMalloc(&d_out_leaves_g, (size_t)count * cap * sizeof(int32_t));
            cudaMalloc(&d_out_weights_g, (size_t)count * cap * sizeof(float));
            cudaMalloc(&d_dAB_g, (size_t)count * sizeof(float));
            cudaMalloc(&d_dAO_g, (size_t)count * sizeof(float));
            cudaMalloc(&d_dBO_g, (size_t)count * sizeof(float));

            cudaMemcpy(d_c1_g, h_c1 + N + start, (size_t)count * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_c2_g, h_c2 + N + start, (size_t)count * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_in_len_g, in_len.data(), (size_t)n_int * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_in_leaves_g, in_leaves.data(), (size_t)n_int * cap * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_in_weights_g, in_weights.data(), (size_t)n_int * cap * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_len_g, out_len.data() + start, (size_t)count * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_leaves_g, out_leaves.data() + (size_t)start * cap, (size_t)count * cap * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_out_weights_g, out_weights.data() + (size_t)start * cap, (size_t)count * cap * sizeof(float), cudaMemcpyHostToDevice);

            int32_t blocks = 3 * count;
            k_eval_internal_node_triples_tiled<<<blocks, 64>>>(
                count, N, W, cap,
                d_c1_g, d_c2_g,
                d_in_len_g, d_in_leaves_g, d_in_weights_g,
                d_out_len_g, d_out_leaves_g, d_out_weights_g,
                d_packed_seqs_per_gpu[g],
                d_dAB_g, d_dAO_g, d_dBO_g
            );
            cudaStreamSynchronize(0);
            cudaMemcpy(h_dAB.data() + start, d_dAB_g, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_dAO.data() + start, d_dAO_g, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_dBO.data() + start, d_dBO_g, (size_t)count * sizeof(float), cudaMemcpyDeviceToHost);

            cudaFree(d_c1_g);
            cudaFree(d_c2_g);
            cudaFree(d_in_len_g);
            cudaFree(d_in_leaves_g);
            cudaFree(d_in_weights_g);
            cudaFree(d_out_len_g);
            cudaFree(d_out_leaves_g);
            cudaFree(d_out_weights_g);
            cudaFree(d_dAB_g);
            cudaFree(d_dAO_g);
            cudaFree(d_dBO_g);
        }
    }

    /* 4. Assemble all 2N - 3 unrooted FastME/BME branch lengths in O(N) */
    double total_bme_len = 0.0;
    for (int32_t u = 0; u < 2 * N - 2; ++u) {
        if (u == r_right) {
            h_branch_lengths[u] = 0.0f;
            continue;
        }
        if (u == r_left) {
            float cross_l = (r_left >= N)
                ? 0.5f * (h_dAO[r_left - N] + h_dBO[r_left - N])
                : 0.5f * (h_dAO[r_right - N] + h_dBO[r_right - N]);
            float cross_r = (r_right >= N)
                ? 0.5f * (h_dAO[r_right - N] + h_dBO[r_right - N])
                : cross_l;
            float div_l = (r_left >= N)  ? h_dAB[r_left - N]  : 0.0f;
            float div_r = (r_right >= N) ? h_dAB[r_right - N] : 0.0f;
            float bl = 0.5f * (cross_l + cross_r) - 0.5f * (div_l + div_r);
            h_branch_lengths[u] = bl;
            total_bme_len += bl;
            continue;
        }

        int32_t p = parent[u];
        int32_t p_idx = p - N;
        bool is_left = (h_c1[p] == u);
        float d_ab_p = h_dAB[p_idx];
        float d_child_out_p = is_left ? h_dAO[p_idx] : h_dBO[p_idx];
        float d_sib_out_p   = is_left ? h_dBO[p_idx] : h_dAO[p_idx];

        float bl;
        if (u < N) {
            bl = 0.5f * (d_ab_p + d_child_out_p - d_sib_out_p);
        } else {
            int32_t u_idx = u - N;
            float cross_from_p = 0.5f * (d_ab_p + d_child_out_p);
            float cross_from_u = 0.5f * (h_dAO[u_idx] + h_dBO[u_idx]);
            bl = 0.5f * (cross_from_p + cross_from_u) - 0.5f * (d_sib_out_p + h_dAB[u_idx]);
        }
        h_branch_lengths[u] = bl;
        total_bme_len += bl;
    }

    const double t2 = omp_get_wtime();
    if (out_kernel_sec) *out_kernel_sec = t2 - t1;
    return total_bme_len;
}

} // namespace p2v_bme_bl
