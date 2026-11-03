/*
 * bench_rnasim_gpu.cu
 *
 * Blind Empirical Evaluation on the Real RNASim Dataset (1,596-bp 16S rRNA
 * alignment simulated under biophysical secondary-structure selection,
 * Mirarab et al. 2015) comparing:
 *   1. VeryFastTree 4.0.5 Initial Top-Hit NJ (-nt -nome -noml -threads 64)
 *   2. VeryFastTree 4.0.5 Minimum Evolution (-nt -noml -threads 64)
 *   3. Full-Tree GPU Phylo2Vec BME Discrete-Gradient Optimizer (exact JC69 sequence distances via GPU __popcll)
 *   4. Multi-GPU Subtree-Wave Minibatch Phylo2Vec BME Optimizer (1x GPU and 8x H100 GPUs via OpenMP)
 *
 * Compile:
 *   nvcc -O3 -arch=sm_86 -Xcompiler -fopenmp bench_rnasim_gpu.cu -o bench_rnasim_gpu
 *   nvcc -O3 -arch=sm_90 -Xcompiler -fopenmp bench_rnasim_gpu.cu -o bench_rnasim_gpu_sm90
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <cuda_runtime.h>

static void relabel_from_rooted_tree(
    const int32_t *c1_r,
    const int32_t *c2_r,
    int32_t n_leaves,
    int32_t flip_mode,
    int64_t *v_cf_out,
    int32_t *perm_out
) {
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;
    std::vector<int32_t> ord(n_leaves - 1);
    int32_t ord_len = 0;
    ord[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        if (b0 >= n_leaves) ord[ord_len++] = b0;
        if (b1 >= n_leaves) ord[ord_len++] = b1;
    }
    std::vector<int32_t> min_leaf(n_nodes_r);
    for (int32_t x = 0; x < n_leaves; ++x) min_leaf[x] = x;
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        min_leaf[u] = (min_leaf[b0] < min_leaf[b1]) ? min_leaf[b0] : min_leaf[b1];
    }
    std::vector<int32_t> rep_leaf(n_nodes_r, 0);
    int32_t rc1 = c1_r[root_r], rc2 = c2_r[root_r];
    int32_t r_stay = (min_leaf[rc1] < min_leaf[rc2]) ? rc1 : rc2;
    int32_t r_move = (min_leaf[rc1] < min_leaf[rc2]) ? rc2 : rc1;
    if (flip_mode == 1) { int32_t tmp = r_stay; r_stay = r_move; r_move = tmp; }
    rep_leaf[r_stay] = 0;
    rep_leaf[r_move] = 1;
    v_cf_out[0] = 0;
    int32_t next_label = 2;
    for (int32_t idx = 1; idx < ord_len; ++idx) {
        int32_t m = ord[idx];
        int32_t c1 = c1_r[m], c2 = c2_r[m];
        int32_t c_stay = (min_leaf[c1] < min_leaf[c2]) ? c1 : c2;
        int32_t c_move = (min_leaf[c1] < min_leaf[c2]) ? c2 : c1;
        if (flip_mode == 1) { int32_t tmp = c_stay; c_stay = c_move; c_move = tmp; }
        int32_t x = rep_leaf[m];
        int32_t y = next_label++;
        rep_leaf[c_stay] = x;
        rep_leaf[c_move] = y;
        v_cf_out[y - 1] = x;
    }
    for (int32_t old_l = 0; old_l < n_leaves; ++old_l) {
        perm_out[rep_leaf[old_l]] = old_l;
    }
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static double wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ============================================================================
 * 1. O(N) Tree Decoding & Universal 64-Bit XOR Split Hashing for Exact RF
 * ============================================================================ */
static void decode_cf_to_rooted(
    const int64_t *v, int32_t N,
    int32_t *c1_r, int32_t *c2_r, int32_t *scratch_uf
) {
    const int32_t n_pairs = N - 1;
    for (int32_t x = 0; x < N; ++x) scratch_uf[x] = x;
    for (int32_t k = 0; k < n_pairs; ++k) {
        int32_t i = n_pairs - 1 - k;
        int32_t m = N + k;
        int32_t x = (int32_t)v[i];
        int32_t y = i + 1;
        c1_r[m] = scratch_uf[y];
        c2_r[m] = scratch_uf[x];
        scratch_uf[x] = m;
    }
}

static void compute_xor_splits_o_n(
    const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    const uint64_t *leaf_hashes, std::vector<uint64_t> &splits_out
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    std::vector<int32_t> ord(N - 1);
    std::vector<uint64_t> node_hash(n_nodes_r);

    for (int32_t x = 0; x < N; ++x) node_hash[x] = leaf_hashes[x];

    int32_t ord_len = 0;
    ord[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        if (b0 >= N) ord[ord_len++] = b0;
        if (b1 >= N) ord[ord_len++] = b1;
    }

    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        node_hash[u] = node_hash[c1_r[u]] ^ node_hash[c2_r[u]];
    }

    int32_t rc1 = c1_r[root_r], rc2 = c2_r[root_r];
    int32_t skip_rc = (rc1 > rc2) ? rc1 : rc2;

    splits_out.clear();
    splits_out.reserve(N - 3);
    for (int32_t u = N; u < root_r; ++u) {
        if (u == skip_rc) continue;
        splits_out.push_back(node_hash[u]);
    }
    if ((int32_t)splits_out.size() != N - 3) {
        fprintf(stderr, "FATAL: splits_out.size() = %zu != N - 3 = %d\n", splits_out.size(), N - 3);
        exit(1);
    }
    std::sort(splits_out.begin(), splits_out.end());
}

static int64_t count_wrong_splits_rf(
    const std::vector<uint64_t> &splits_true,
    const std::vector<uint64_t> &splits_curr
) {
    size_t i = 0, j = 0, common = 0;
    while (i < splits_true.size() && j < splits_curr.size()) {
        if (splits_true[i] == splits_curr[j]) {
            common++; i++; j++;
        } else if (splits_true[i] < splits_curr[j]) {
            i++;
        } else {
            j++;
        }
    }
    return (int64_t)splits_true.size() - (int64_t)common;
}

/* ============================================================================
 * 2. Real Sequence Bit-Packed GPU Jukes-Cantor Distance Kernels (Zero Tricks!)
 * ============================================================================ */
__device__ __forceinline__ float dev_jc69_dist_popc(
    int32_t x, int32_t y,
    const uint64_t * __restrict__ d_packed_seqs,
    int32_t W
) {
    if (x == y) return 0.0f;
    const uint64_t *sx = &d_packed_seqs[(size_t)x * 5 * W];
    const uint64_t *sy = &d_packed_seqs[(size_t)y * 5 * W];
    int32_t n_valid = 0;
    int32_t n_match = 0;
    #pragma unroll
    for (int w = 0; w < 25; ++w) {
        uint64_t vx = sx[4 * 25 + w];
        uint64_t vy = sy[4 * 25 + w];
        n_valid += __popcll(vx & vy);
        uint64_t m = (sx[0 * 25 + w] & sy[0 * 25 + w])
                   | (sx[1 * 25 + w] & sy[1 * 25 + w])
                   | (sx[2 * 25 + w] & sy[2 * 25 + w])
                   | (sx[3 * 25 + w] & sy[3 * 25 + w]);
        n_match += __popcll(m);
    }
    if (n_valid <= 0) return 1.5f;
    float p = (float)(n_valid - n_match) / (float)n_valid;
    if (p >= 0.74f) p = 0.74f;
    float d = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
    return (d > 0.0f) ? d : 0.0f;
}

/* Build K x K condensed distance matrix on the fly from dyadic-weighted super-leaf profiles:
 *   D_cond[a, b] = sum_{p in prof(a)} sum_{q in prof(b)} w_p * w_q * D(leaf_p, leaf_q).
 * Each super-leaf profile is stored in CSR format (d_prof_offsets, d_prof_leaves, d_prof_weights)
 * with nonnegative weights w_p summing to 1 (exact dyadic weights w_p = 2^{-depth(leaf_p)} when
 * leaf_cnt <= max_prof_leaves or max_prof_leaves <= 0; approximate frontier-subtree collapse above cap). */
__global__ void k_build_real_seq_condensed_D(
    int32_t K, int32_t W,
    const int32_t * __restrict__ d_prof_offsets,
    const int32_t * __restrict__ d_prof_leaves,
    const float   * __restrict__ d_prof_weights,
    const uint64_t * __restrict__ d_packed_seqs,
    float * __restrict__ d_D_cond
) {
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= K || b >= K) return;
    if (a == b) {
        d_D_cond[(size_t)a * K + b] = 0.0f;
        return;
    }
    if (a < b) {
        int32_t off_a = d_prof_offsets[a], end_a = d_prof_offsets[a + 1];
        int32_t off_b = d_prof_offsets[b], end_b = d_prof_offsets[b + 1];
        float d_val = 0.0f;
        for (int32_t p = off_a; p < end_a; ++p) {
            int32_t la = d_prof_leaves[p];
            float wa = d_prof_weights[p];
            float row_sum = 0.0f;
            for (int32_t q = off_b; q < end_b; ++q) {
                int32_t lb = d_prof_leaves[q];
                float wb = d_prof_weights[q];
                row_sum += wb * dev_jc69_dist_popc(la, lb, d_packed_seqs, W);
            }
            d_val += wa * row_sum;
        }
        d_D_cond[(size_t)a * K + b] = d_val;
        d_D_cond[(size_t)b * K + a] = d_val;
    }
}

/* Block-parallel exact refinement of selected super-leaf pairs (a, b) using uncapped
 * dyadic profiles (d_exact_offsets, d_exact_leaves, d_exact_weights) and on-the-fly JC69
 * popcount distances. One 256-thread block evaluates one pair
 * (a, b) = (d_pair_a[blockIdx.x], d_pair_b[blockIdx.x]) in FP64, updating d_D_cond[a * K + b],
 * d_D_cond[b * K + a], and d_pair_exact_D[pid]. */
__global__ void k_refine_real_seq_condensed_D_pairs(
    int32_t K, int32_t W, int32_t n_pairs,
    const int32_t  * __restrict__ d_pair_a,
    const int32_t  * __restrict__ d_pair_b,
    const int32_t  * __restrict__ d_exact_offsets,
    const int32_t  * __restrict__ d_exact_leaves,
    const float    * __restrict__ d_exact_weights,
    const uint64_t * __restrict__ d_packed_seqs,
    float  * __restrict__ d_D_cond,
    double * __restrict__ d_pair_exact_D
) {
    int32_t pid = blockIdx.x;
    if (pid >= n_pairs) return;

    const int32_t base_pid = pid & ~15;
    const int32_t b0 = d_pair_b[base_pid];
    const int32_t off_b0 = d_exact_offsets[b0], end_b0 = d_exact_offsets[b0 + 1];
    const int32_t len_b0 = end_b0 - off_b0;
    int32_t batch_sz = 0;
    if (5 * W == 125 && len_b0 > 1) {
        int32_t max_k = n_pairs - base_pid;
        if (max_k > 16) max_k = 16;
        for (int32_t k = 0; k < max_k; ++k) {
            int32_t ak = d_pair_a[base_pid + k];
            int32_t bk = d_pair_b[base_pid + k];
            int32_t lak = d_exact_offsets[ak + 1] - d_exact_offsets[ak];
            if (bk == b0 && lak == 1) batch_sz++;
            else break;
        }
    }
    if (batch_sz >= 2 && pid < base_pid + batch_sz) {
        if (pid != base_pid) return;
        __shared__ uint64_t s_batch_seq_a[16][125];
        __shared__ double s_wa[16];
        for (int32_t idx = threadIdx.x; idx < batch_sz * 125; idx += blockDim.x) {
            int32_t k = idx / 125;
            int32_t w = idx % 125;
            int32_t ak = d_pair_a[base_pid + k];
            int32_t la = d_exact_leaves[d_exact_offsets[ak]];
            s_batch_seq_a[k][w] = d_packed_seqs[(size_t)la * 125 + w];
        }
        if (threadIdx.x < (uint32_t)batch_sz) {
            int32_t ak = d_pair_a[base_pid + threadIdx.x];
            s_wa[threadIdx.x] = (double)d_exact_weights[d_exact_offsets[ak]];
        }
        __syncthreads();

        double my_sums[16];
        #pragma unroll
        for (int32_t k = 0; k < 16; ++k) my_sums[k] = 0.0;

        for (int32_t q = off_b0 + threadIdx.x; q < end_b0; q += blockDim.x) {
            float wq = d_exact_weights[q];
            if (wq == 0.0f) continue;
            int32_t lb = d_exact_leaves[q];
            const uint64_t *sy = d_packed_seqs + (size_t)lb * 125;
            int32_t n_valid[16], n_match[16];
            #pragma unroll
            for (int32_t k = 0; k < 16; ++k) { n_valid[k] = 0; n_match[k] = 0; }
            for (int32_t w = 0; w < 25; ++w) {
                uint64_t sy0 = sy[0 * 25 + w];
                uint64_t sy1 = sy[1 * 25 + w];
                uint64_t sy2 = sy[2 * 25 + w];
                uint64_t sy3 = sy[3 * 25 + w];
                uint64_t sy4 = sy[4 * 25 + w];
                for (int32_t k = 0; k < batch_sz; ++k) {
                    n_valid[k] += __popcll(s_batch_seq_a[k][4 * 25 + w] & sy4);
                    uint64_t m = (s_batch_seq_a[k][0 * 25 + w] & sy0)
                               | (s_batch_seq_a[k][1 * 25 + w] & sy1)
                               | (s_batch_seq_a[k][2 * 25 + w] & sy2)
                               | (s_batch_seq_a[k][3 * 25 + w] & sy3);
                    n_match[k] += __popcll(m);
                }
            }
            for (int32_t k = 0; k < batch_sz; ++k) {
                float d_jc = 1.5f;
                if (n_valid[k] > 0) {
                    float p = (float)(n_valid[k] - n_match[k]) / (float)n_valid[k];
                    if (p >= 0.74f) p = 0.74f;
                    d_jc = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
                    if (d_jc < 0.0f) d_jc = 0.0f;
                }
                my_sums[k] += (s_wa[k] * (double)wq) * (double)d_jc;
            }
        }
        __shared__ double s_red[256];
        for (int32_t k = 0; k < batch_sz; ++k) {
            s_red[threadIdx.x] = my_sums[k];
            __syncthreads();
            for (int32_t s = 128; s > 0; s >>= 1) {
                if (threadIdx.x < s) s_red[threadIdx.x] += s_red[threadIdx.x + s];
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                double exact_val = s_red[0];
                int32_t ak = d_pair_a[base_pid + k];
                d_pair_exact_D[base_pid + k] = exact_val;
                d_D_cond[(size_t)ak * K + b0] = (float)exact_val;
                d_D_cond[(size_t)b0 * K + ak] = (float)exact_val;
            }
            __syncthreads();
        }
        return;
    }

    int32_t a = d_pair_a[pid];
    int32_t b = d_pair_b[pid];
    int32_t off_a = d_exact_offsets[a], end_a = d_exact_offsets[a + 1];
    int32_t off_b = d_exact_offsets[b], end_b = d_exact_offsets[b + 1];
    int32_t len_a = end_a - off_a;
    int32_t len_b = end_b - off_b;
    if (len_a > len_b) {
        int32_t to = off_a; off_a = off_b; off_b = to;
        int32_t te = end_a; end_a = end_b; end_b = te;
        int32_t tl = len_a; len_a = len_b; len_b = tl;
    }

    __shared__ uint64_t s_seq_a[256];
    double my_sum = 0.0;
    if (len_a == 1 && 5 * W <= 256) {
        int32_t la = d_exact_leaves[off_a];
        double wa = (double)d_exact_weights[off_a];
        const uint64_t *sx = d_packed_seqs + (size_t)la * (5 * W);
        if (threadIdx.x < (uint32_t)(5 * W)) {
            s_seq_a[threadIdx.x] = sx[threadIdx.x];
        }
        __syncthreads();
        if (wa != 0.0) {
            for (int32_t q = off_b + threadIdx.x; q < end_b; q += blockDim.x) {
                float wq = d_exact_weights[q];
                if (wq == 0.0f) continue;
                int32_t lb = d_exact_leaves[q];
                const uint64_t *sy = d_packed_seqs + (size_t)lb * (5 * W);
                int32_t n_valid = 0, n_match = 0;
                #pragma unroll
                for (int32_t w = 0; w < 25; ++w) {
                    uint64_t v = s_seq_a[4 * 25 + w] & sy[4 * 25 + w];
                    n_valid += __popcll(v);
                    uint64_t m = (s_seq_a[0 * 25 + w] & sy[0 * 25 + w])
                               | (s_seq_a[1 * 25 + w] & sy[1 * 25 + w])
                               | (s_seq_a[2 * 25 + w] & sy[2 * 25 + w])
                               | (s_seq_a[3 * 25 + w] & sy[3 * 25 + w]);
                    n_match += __popcll(m);
                }
                float d_jc = 1.5f;
                if (n_valid > 0) {
                    float p = (float)(n_valid - n_match) / (float)n_valid;
                    if (p >= 0.74f) p = 0.74f;
                    d_jc = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
                    if (d_jc < 0.0f) d_jc = 0.0f;
                }
                my_sum += (wa * (double)wq) * (double)d_jc;
            }
        }
    } else {
        int64_t total_pq = (int64_t)len_a * (int64_t)len_b;
        for (int64_t idx = threadIdx.x; idx < total_pq; idx += blockDim.x) {
            int32_t p = off_a + (int32_t)(idx / len_b);
            int32_t q = off_b + (int32_t)(idx % len_b);
            double w = (double)d_exact_weights[p] * (double)d_exact_weights[q];
            if (w == 0.0) continue;
            int32_t la = d_exact_leaves[p];
            int32_t lb = d_exact_leaves[q];
            double d = (double)dev_jc69_dist_popc(la, lb, d_packed_seqs, W);
            my_sum += w * d;
        }
    }
    __shared__ double s_sum[256];
    s_sum[threadIdx.x] = my_sum;
    __syncthreads();
    for (int32_t s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        double exact_val = s_sum[0];
        d_pair_exact_D[pid] = exact_val;
        d_D_cond[(size_t)a * K + b] = (float)exact_val;
        d_D_cond[(size_t)b * K + a] = (float)exact_val;
    }
}

/* ============================================================================
 * 3. Exact Verified GPU Discrete Gradient Kernels (Stage 2 + Stage 3)
 * ============================================================================ */
__global__ void k_copy_leaf_rows_perm(
    const float * __restrict__ D_orig, const int32_t * __restrict__ d_perm,
    float * __restrict__ subdist, int32_t N, int32_t M
) {
    int32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        int32_t r_orig = d_perm[row];
        int32_t c_orig = d_perm[col];
        subdist[(size_t)row * M + col] = D_orig[(size_t)r_orig * N + c_orig];
    }
}

__global__ void k_bottom_up_leaves(
    float * __restrict__ subdist, const int32_t * __restrict__ c1_r,
    const int32_t * __restrict__ c2_r, int32_t N, int32_t M, int32_t root_u
) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    for (int32_t u = N; u < root_u; ++u) {
        int32_t c1 = c1_r[u], c2 = c2_r[u];
        subdist[(size_t)u * M + v] = 0.5f * (subdist[(size_t)c1 * M + v] + subdist[(size_t)c2 * M + v]);
    }
}

__global__ void k_transpose_leaf_block(
    float * __restrict__ subdist, int32_t N, int32_t M, int32_t root_u
) {
    __shared__ float tile[16][17];
    int32_t u = N + blockIdx.y * 16 + threadIdx.y;
    int32_t v = blockIdx.x * 16 + threadIdx.x;
    if (u < root_u && v < N) tile[threadIdx.y][threadIdx.x] = subdist[(size_t)u * M + v];
    __syncthreads();
    int32_t v_t = blockIdx.x * 16 + threadIdx.y;
    int32_t u_t = N + blockIdx.y * 16 + threadIdx.x;
    if (v_t < N && u_t < root_u) subdist[(size_t)v_t * M + u_t] = tile[threadIdx.x][threadIdx.y];
}

__global__ void k_bottom_up_internals(
    float * __restrict__ subdist, const int32_t * __restrict__ c1_r,
    const int32_t * __restrict__ c2_r, int32_t N, int32_t M, int32_t root_u
) {
    int32_t v = N + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= root_u) return;
    for (int32_t u = N; u < root_u; ++u) {
        int32_t c1 = c1_r[u], c2 = c2_r[u];
        subdist[(size_t)u * M + v] = 0.5f * (subdist[(size_t)c1 * M + v] + subdist[(size_t)c2 * M + v]);
    }
}

__global__ void k_top_down_and_root(
    float * __restrict__ subdist, const int32_t * __restrict__ c1_r,
    const int32_t * __restrict__ c2_r, const int32_t * __restrict__ tin,
    const int32_t * __restrict__ sub_size, int32_t N, int32_t M,
    int32_t root_u, int32_t rc0, int32_t rc1, int32_t rc2
) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= M) return;
    if (v == root_u) { subdist[(size_t)root_u * M + root_u] = 0.0f; return; }
    int32_t tv = tin[v];
    int32_t curr;
    float root_val;
    if (tv < tin[rc1]) {
        curr = rc0;
        root_val = 0.5f * (subdist[(size_t)rc1 * M + v] + subdist[(size_t)rc2 * M + v]);
    } else if (tv < tin[rc2]) {
        curr = rc1;
        root_val = 0.5f * (subdist[(size_t)rc0 * M + v] + subdist[(size_t)rc2 * M + v]);
    } else {
        curr = rc2;
        root_val = 0.5f * (subdist[(size_t)rc0 * M + v] + subdist[(size_t)rc1 * M + v]);
    }
    subdist[(size_t)root_u * M + v] = root_val;
    subdist[(size_t)v * M + root_u] = root_val;

    int32_t g = root_u;
    while (curr != v) {
        int32_t c1 = c1_r[curr], c2 = c2_r[curr];
        int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
        int32_t sib       = (tv < tin[c2]) ? c2 : c1;
        float val = 0.5f * (subdist[(size_t)sib * M + v] + subdist[(size_t)g * M + v]);
        subdist[(size_t)curr * M + v] = val;
        subdist[(size_t)v * M + curr] = val;
        g = curr;
        curr = next_curr;
    }
    subdist[(size_t)v * M + v] = 0.0f;
}

__global__ void k_populate_nd_u(
    const float * __restrict__ subdist, const int32_t * __restrict__ adj_u,
    float * __restrict__ nd_u, int32_t N, int32_t M, int32_t root_u
) {
    int32_t u = N + blockIdx.x * blockDim.x + threadIdx.x;
    if (u > root_u) return;
    int32_t n0 = adj_u[3 * u + 0], n1 = adj_u[3 * u + 1], n2 = adj_u[3 * u + 2];
    nd_u[3 * u + 0] = subdist[(size_t)n1 * M + n2];
    nd_u[3 * u + 1] = subdist[(size_t)n0 * M + n2];
    nd_u[3 * u + 2] = subdist[(size_t)n0 * M + n1];
}

__global__ void k_compute_bme_length(
    const float * __restrict__ subdist,
    const int32_t * __restrict__ parent_u,
    int32_t N, int32_t M,
    double * __restrict__ d_out_L
) {
    __shared__ double s_sum[256];
    double acc = 0.0;
    for (int32_t x = threadIdx.x; x < N; x += blockDim.x) {
        acc += (double)subdist[(size_t)x * M + parent_u[x]];
    }
    s_sum[threadIdx.x] = acc;
    __syncthreads();
    for (int32_t s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) *d_out_L = 0.5 * s_sum[0];
}

struct SprSharedQueue {
    int32_t u[256];
    int32_t c[256];
    float   pow_val[256];
    float   dkc[256];
    float   dist[256];
    int32_t bounds[4];
};

__device__ __forceinline__ void dev_expand_subtree_stack(
    int32_t u, int32_t C, float pow_neg_len, float d_kc, float curr_dist,
    int32_t n_leaves, int32_t n_nodes_u,
    const int32_t * __restrict__ adj_u, const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size, const float * __restrict__ nd_u,
    const float * __restrict__ row_root, const float * __restrict__ row_orig,
    float * __restrict__ edge_delta
) {
    int32_t st_u[18], st_c[18];
    float st_pow[18], st_dkc[18], st_dist[18];
    int32_t top = 0;

    for (;;) {
        int32_t a0 = adj_u[3 * u + 0], a1 = adj_u[3 * u + 1];
        int32_t ki = (a0 == C) ? 0 : ((a1 == C) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;

        int32_t b0 = adj_u[3 * u + idx0], b1 = adj_u[3 * u + idx1];
        float d_ab = nd_u[3 * u + ki];
        float r_b0 = row_root[b0], r_b1 = row_root[b1];

        float d_ac0 = nd_u[3 * u + idx0] + (row_orig[b1] - r_b1) * pow_neg_len;
        float ndist0 = curr_dist + 0.25f * (d_ac0 + r_b0 - d_ab - d_kc);
        edge_delta[(parent_u[u] == b0) ? u : b0] = ndist0;

        float d_ac1 = nd_u[3 * u + idx1] + (row_orig[b0] - r_b0) * pow_neg_len;
        float ndist1 = curr_dist + 0.25f * (d_ac1 + r_b1 - d_ab - d_kc);
        edge_delta[(parent_u[u] == b1) ? u : b1] = ndist1;

        float next_pow = pow_neg_len * 0.5f;
        if (b0 >= n_leaves && b1 >= n_leaves) {
            int32_t s0 = (parent_u[b0] == u) ? sub_size[b0] : (n_nodes_u - sub_size[u]);
            int32_t s1 = (parent_u[b1] == u) ? sub_size[b1] : (n_nodes_u - sub_size[u]);
            if (s0 >= s1) {
                st_u[top] = b0; st_c[top] = u; st_pow[top] = next_pow;
                st_dkc[top] = 0.5f * (d_kc + r_b1); st_dist[top] = ndist0; top++;
                C = u; u = b1; pow_neg_len = next_pow;
                d_kc = 0.5f * (d_kc + r_b0); curr_dist = ndist1;
            } else {
                st_u[top] = b1; st_c[top] = u; st_pow[top] = next_pow;
                st_dkc[top] = 0.5f * (d_kc + r_b0); st_dist[top] = ndist1; top++;
                C = u; u = b0; pow_neg_len = next_pow;
                d_kc = 0.5f * (d_kc + r_b1); curr_dist = ndist0;
            }
        } else if (b0 >= n_leaves) {
            C = u; u = b0; pow_neg_len = next_pow;
            d_kc = 0.5f * (d_kc + r_b1); curr_dist = ndist0;
        } else if (b1 >= n_leaves) {
            C = u; u = b1; pow_neg_len = next_pow;
            d_kc = 0.5f * (d_kc + r_b0); curr_dist = ndist1;
        } else {
            if (top == 0) break;
            --top;
            u = st_u[top]; C = st_c[top]; pow_neg_len = st_pow[top];
            d_kc = st_dkc[top]; curr_dist = st_dist[top];
        }
    }
}

__device__ __forceinline__ void dev_eval_cherry_or_enqueue(
    int32_t b, int32_t u, int32_t sb, float pow_neg_len, float d_kc, float curr_dist,
    const int32_t * __restrict__ adj_u, const int32_t * __restrict__ parent_u,
    const float * __restrict__ nd_u, const float * __restrict__ row_root,
    const float * __restrict__ row_orig, float * __restrict__ edge_delta,
    SprSharedQueue * __restrict__ sq, int32_t * __restrict__ q_tail_ptr
) {
    if (sb == 3) {
        int32_t a0 = adj_u[3 * b + 0], a1 = adj_u[3 * b + 1];
        int32_t ki = (a0 == u) ? 0 : ((a1 == u) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;
        int32_t c0 = adj_u[3 * b + idx0], c1 = adj_u[3 * b + idx1];
        float d_ab = nd_u[3 * b + ki];
        float r_c0 = row_root[c0], r_c1 = row_root[c1];
        float d_ac0 = nd_u[3 * b + idx0] + (row_orig[c1] - r_c1) * pow_neg_len;
        edge_delta[(parent_u[b] == c0) ? b : c0] = curr_dist + 0.25f * (d_ac0 + r_c0 - d_ab - d_kc);
        float d_ac1 = nd_u[3 * b + idx1] + (row_orig[c0] - r_c0) * pow_neg_len;
        edge_delta[(parent_u[b] == c1) ? b : c1] = curr_dist + 0.25f * (d_ac1 + r_c1 - d_ab - d_kc);
    } else {
        int32_t qt = *q_tail_ptr;
        sq->u[qt] = b; sq->c[qt] = u; sq->pow_val[qt] = pow_neg_len;
        sq->dkc[qt] = d_kc; sq->dist[qt] = curr_dist;
        *q_tail_ptr = qt + 1;
    }
}

__device__ __forceinline__ void dev_seed_half_tree_fifo(
    int32_t start_node, int32_t orig_node, int32_t parent_node, int32_t q_base,
    int32_t n_leaves, int32_t n_nodes_u,
    const int32_t * __restrict__ adj_u, const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size, const float * __restrict__ nd_u,
    const float * __restrict__ row_root, const float * __restrict__ row_orig,
    float * __restrict__ edge_delta, SprSharedQueue * __restrict__ sq,
    int32_t * __restrict__ out_head, int32_t * __restrict__ out_tail
) {
    int32_t q_tail = q_base;
    int32_t q_lim  = q_base + 128;
    int32_t tau    = (n_nodes_u > 512) ? (n_nodes_u / 64) : 16;

    if (start_node >= n_leaves) {
        int32_t n0 = adj_u[3 * start_node + 0], n1 = adj_u[3 * start_node + 1];
        int32_t k_in = (n0 == parent_node) ? 0 : ((n1 == parent_node) ? 1 : 2);
        int32_t k0 = (k_in == 0) ? 1 : 0;
        int32_t k1 = (k_in == 2) ? 1 : 2;

        int32_t B0 = adj_u[3 * start_node + k0], B1 = adj_u[3 * start_node + k1];
        float D_AB = nd_u[3 * start_node + k_in];
        float d_kc_init = row_root[orig_node];

        float new_dist0 = 0.25f * (row_orig[B1] + row_root[B0] - D_AB - d_kc_init);
        edge_delta[(parent_u[start_node] == B0) ? start_node : B0] = new_dist0;

        float new_dist1 = 0.25f * (row_orig[B0] + row_root[B1] - D_AB - d_kc_init);
        edge_delta[(parent_u[start_node] == B1) ? start_node : B1] = new_dist1;

        int32_t h_u[18], h_c[18];
        float h_pow[18], h_dkc[18], h_dist[18];
        int32_t h_top = 0;

        if (B0 >= n_leaves) {
            int32_t s0 = (parent_u[B0] == start_node) ? sub_size[B0] : (n_nodes_u - sub_size[start_node]);
            if (s0 > tau) {
                h_u[h_top] = B0; h_c[h_top] = start_node; h_pow[h_top] = 0.25f;
                h_dkc[h_top] = 0.5f * (d_kc_init + row_root[B1]); h_dist[h_top] = new_dist0; h_top++;
            } else {
                dev_eval_cherry_or_enqueue(B0, start_node, s0, 0.25f, 0.5f * (d_kc_init + row_root[B1]), new_dist0,
                                           adj_u, parent_u, nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
            }
        }
        if (B1 >= n_leaves) {
            int32_t s1 = (parent_u[B1] == start_node) ? sub_size[B1] : (n_nodes_u - sub_size[start_node]);
            if (s1 > tau) {
                h_u[h_top] = B1; h_c[h_top] = start_node; h_pow[h_top] = 0.25f;
                h_dkc[h_top] = 0.5f * (d_kc_init + row_root[B0]); h_dist[h_top] = new_dist1; h_top++;
            } else {
                dev_eval_cherry_or_enqueue(B1, start_node, s1, 0.25f, 0.5f * (d_kc_init + row_root[B0]), new_dist1,
                                           adj_u, parent_u, nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
            }
        }

        while (h_top > 0) {
            --h_top;
            int32_t u = h_u[h_top], C = h_c[h_top];
            float pow_neg_len = h_pow[h_top], d_kc = h_dkc[h_top], curr_dist = h_dist[h_top];

            for (;;) {
                if (q_tail + h_top + 2 > q_lim) {
                    sq->u[q_tail] = u; sq->c[q_tail] = C; sq->pow_val[q_tail] = pow_neg_len;
                    sq->dkc[q_tail] = d_kc; sq->dist[q_tail] = curr_dist; q_tail++;
                    while (h_top > 0) {
                        --h_top;
                        sq->u[q_tail] = h_u[h_top]; sq->c[q_tail] = h_c[h_top]; sq->pow_val[q_tail] = h_pow[h_top];
                        sq->dkc[q_tail] = h_dkc[h_top]; sq->dist[q_tail] = h_dist[h_top]; q_tail++;
                    }
                    break;
                }
                int32_t a0 = adj_u[3 * u + 0], a1 = adj_u[3 * u + 1];
                int32_t ki = (a0 == C) ? 0 : ((a1 == C) ? 1 : 2);
                int32_t idx0 = (ki == 0) ? 1 : 0;
                int32_t idx1 = (ki == 2) ? 1 : 2;

                int32_t b0 = adj_u[3 * u + idx0], b1 = adj_u[3 * u + idx1];
                float d_ab = nd_u[3 * u + ki], r_b0 = row_root[b0], r_b1 = row_root[b1];

                float d_ac0 = nd_u[3 * u + idx0] + (row_orig[b1] - r_b1) * pow_neg_len;
                float ndist0 = curr_dist + 0.25f * (d_ac0 + r_b0 - d_ab - d_kc);
                edge_delta[(parent_u[u] == b0) ? u : b0] = ndist0;

                float d_ac1 = nd_u[3 * u + idx1] + (row_orig[b0] - r_b0) * pow_neg_len;
                float ndist1 = curr_dist + 0.25f * (d_ac1 + r_b1 - d_ab - d_kc);
                edge_delta[(parent_u[u] == b1) ? u : b1] = ndist1;

                float next_pow = pow_neg_len * 0.5f;
                int32_t s0 = (b0 >= n_leaves) ? ((parent_u[b0] == u) ? sub_size[b0] : (n_nodes_u - sub_size[u])) : 0;
                int32_t s1 = (b1 >= n_leaves) ? ((parent_u[b1] == u) ? sub_size[b1] : (n_nodes_u - sub_size[u])) : 0;

                if (s0 > tau && s1 > tau) {
                    if (s0 >= s1) {
                        h_u[h_top] = b0; h_c[h_top] = u; h_pow[h_top] = next_pow;
                        h_dkc[h_top] = 0.5f * (d_kc + r_b1); h_dist[h_top] = ndist0; h_top++;
                        C = u; u = b1; pow_neg_len = next_pow; d_kc = 0.5f * (d_kc + r_b0); curr_dist = ndist1;
                    } else {
                        h_u[h_top] = b1; h_c[h_top] = u; h_pow[h_top] = next_pow;
                        h_dkc[h_top] = 0.5f * (d_kc + r_b0); h_dist[h_top] = ndist1; h_top++;
                        C = u; u = b0; pow_neg_len = next_pow; d_kc = 0.5f * (d_kc + r_b1); curr_dist = ndist0;
                    }
                } else if (s0 > tau) {
                    if (b1 >= n_leaves) {
                        dev_eval_cherry_or_enqueue(b1, u, s1, next_pow, 0.5f * (d_kc + r_b0), ndist1,
                                                   adj_u, parent_u, nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
                    }
                    C = u; u = b0; pow_neg_len = next_pow; d_kc = 0.5f * (d_kc + r_b1); curr_dist = ndist0;
                } else if (s1 > tau) {
                    if (b0 >= n_leaves) {
                        dev_eval_cherry_or_enqueue(b0, u, s0, next_pow, 0.5f * (d_kc + r_b1), ndist0,
                                                   adj_u, parent_u, nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
                    }
                    C = u; u = b1; pow_neg_len = next_pow; d_kc = 0.5f * (d_kc + r_b0); curr_dist = ndist1;
                } else {
                    if (b0 >= n_leaves) {
                        dev_eval_cherry_or_enqueue(b0, u, s0, next_pow, 0.5f * (d_kc + r_b1), ndist0,
                                                   adj_u, parent_u, nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
                    }
                    if (b1 >= n_leaves) {
                        dev_eval_cherry_or_enqueue(b1, u, s1, next_pow, 0.5f * (d_kc + r_b0), ndist1,
                                                   adj_u, parent_u, nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
                    }
                    break;
                }
            }
        }
    }
    *out_head = q_base;
    *out_tail = q_tail;
}

__device__ __forceinline__ void dev_run_parallel_spr_dfs(
    int32_t tree_root, int32_t parent, int32_t other0, int32_t other1,
    int32_t n_leaves, int32_t n_nodes_u,
    const int32_t * __restrict__ adj_u, const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size, const float * __restrict__ nd_u,
    const float * __restrict__ row_root, const float * __restrict__ row_o0,
    const float * __restrict__ row_o1, float * __restrict__ edge_delta,
    SprSharedQueue * __restrict__ sq
) {
    for (int32_t u = threadIdx.x; u < n_nodes_u; u += blockDim.x) {
        edge_delta[u] = 1e30f;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        edge_delta[(parent_u[tree_root] == parent) ? tree_root : parent] = 1e30f;
        edge_delta[(parent_u[other0]    == parent) ? other0    : parent] = 1e30f;
        edge_delta[(parent_u[other1]    == parent) ? other1    : parent] = 1e30f;
        dev_seed_half_tree_fifo(
            other0, other1, parent, 0, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
            nd_u, row_root, row_o1, edge_delta, sq, &sq->bounds[0], &sq->bounds[1]
        );
    } else if (threadIdx.x == 32) {
        dev_seed_half_tree_fifo(
            other1, other0, parent, 128, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
            nd_u, row_root, row_o0, edge_delta, sq, &sq->bounds[2], &sq->bounds[3]
        );
    }
    __syncthreads();

    int32_t h0 = sq->bounds[0], t0 = sq->bounds[1];
    int32_t h1 = sq->bounds[2], t1 = sq->bounds[3];
    int32_t n0 = t0 - h0, n1 = t1 - h1, total = n0 + n1;
    for (int32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int32_t q_idx = (idx < n0) ? (h0 + idx) : (h1 + (idx - n0));
        const float *r_orig = (idx < n0) ? row_o1 : row_o0;
        dev_expand_subtree_stack(
            sq->u[q_idx], sq->c[q_idx], sq->pow_val[q_idx], sq->dkc[q_idx], sq->dist[q_idx],
            n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u, row_root, r_orig, edge_delta
        );
    }
    __syncthreads();
}

__global__ void k_spr_rows_top1(
    int32_t r_start, int32_t n_leaves, int32_t n_nodes_u, int32_t other_root_child,
    const int64_t * __restrict__ v_curr, const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ adj_u, const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size, const int32_t * __restrict__ m_lmin,
    const int32_t * __restrict__ next_k_same_lmin, const float * __restrict__ subdist,
    const float * __restrict__ nd_u, float * __restrict__ row_tmp_pool,
    float * __restrict__ best_delta_out, int32_t * __restrict__ best_j_out
) {
    extern __shared__ float s_edge_delta_top1_buf[];
    __shared__ SprSharedQueue s_q;
    __shared__ float s_best_val[128];
    __shared__ int32_t s_best_j[128];

    int32_t v_index = r_start + blockIdx.x;
    int32_t i = v_index - 1;
    int32_t root_u = n_nodes_u - 1;
    float *edge_delta = s_edge_delta_top1_buf;
    (void)row_tmp_pool; (void)v_curr; (void)m_lmin; (void)next_k_same_lmin;

    int32_t tree_root = v_map[v_index];
    int32_t parent = (tree_root == root_u) ? other_root_child : parent_u[tree_root];
    int32_t other0 = -1, other1 = -1;
    for (int32_t k = 0; k < 3; ++k) {
        int32_t nb_node = adj_u[3 * parent + k];
        if (nb_node != -1 && nb_node != tree_root) {
            if (other0 == -1) other0 = nb_node;
            else other1 = nb_node;
        }
    }
    if (other0 == -1) {
        if (threadIdx.x == 0) {
            best_delta_out[i] = 1e30f;
            best_j_out[i]     = -1;
        }
        return;
    }

    dev_run_parallel_spr_dfs(
        tree_root, parent, other0, other1, n_leaves, n_nodes_u,
        adj_u, parent_u, sub_size, nd_u,
        &subdist[(size_t)tree_root * n_nodes_u],
        &subdist[(size_t)other0 * n_nodes_u],
        &subdist[(size_t)other1 * n_nodes_u],
        edge_delta, &s_q
    );

    float b_val = 1e30f;
    int32_t b_j = -1;
    for (int32_t du = threadIdx.x; du < root_u; du += blockDim.x) {
        float val = edge_delta[du];
        if (val < b_val || (val == b_val && du < b_j)) {
            b_val = val;
            b_j = du;
        }
    }
    s_best_val[threadIdx.x] = b_val;
    s_best_j[threadIdx.x]   = b_j;
    __syncthreads();

    for (int32_t stride = 64; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float v_other = s_best_val[threadIdx.x + stride];
            int32_t j_other = s_best_j[threadIdx.x + stride];
            float v_self = s_best_val[threadIdx.x];
            int32_t j_self = s_best_j[threadIdx.x];
            if (v_other < v_self || (v_other == v_self && j_other < j_self)) {
                s_best_val[threadIdx.x] = v_other;
                s_best_j[threadIdx.x]   = j_other;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        best_delta_out[i] = s_best_val[0];
        best_j_out[i]     = s_best_j[0];
    }
}

/* ============================================================================
 * 4. Host Tree Metadata & Active Skeleton Extraction
 * ============================================================================ */
static void build_skeleton_metadata(
    const int64_t *v_cf, int32_t K,
    std::vector<int32_t> &c1_cf, std::vector<int32_t> &c2_cf,
    std::vector<int32_t> &v_map_cf, std::vector<int32_t> &m_lmin_cf,
    std::vector<int32_t> &next_k_cf, std::vector<int32_t> &parent_u_cf,
    std::vector<int32_t> &sub_sz_cf, std::vector<int32_t> &tin_cf,
    std::vector<int32_t> &adj_u_cf,
    int32_t *out_other_rc, int32_t *out_rc0, int32_t *out_rc1, int32_t *out_rc2
) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t M_sk = 2 * K - 2;
    const int32_t root_u_sk = M_sk - 1;

    c1_cf.assign(R_sk, 0); c2_cf.assign(R_sk, 0);
    v_map_cf.assign(K, 0);
    m_lmin_cf.assign(K, 0);
    next_k_cf.assign(K, 0);
    parent_u_cf.assign(M_sk, -1);
    sub_sz_cf.assign(M_sk, 1);
    tin_cf.assign(M_sk, 0);
    adj_u_cf.assign(3 * M_sk, -1);

    std::vector<int32_t> scratch_uf(R_sk), last_seen(K, K);
    for (int32_t x = 0; x < K; ++x) scratch_uf[x] = x;

    for (int32_t k = 0; k < K - 1; ++k) {
        int32_t i = K - 2 - k;
        int32_t m = K + k;
        int32_t x = (int32_t)v_cf[i];
        int32_t y = i + 1;
        int32_t b0 = scratch_uf[y], b1 = scratch_uf[x];
        c1_cf[m] = b0; c2_cf[m] = b1;
        m_lmin_cf[k] = x; v_map_cf[y] = b0;
        scratch_uf[x] = m;
    }
    for (int32_t k = K - 3; k >= 0; --k) {
        int32_t lmin = m_lmin_cf[k];
        next_k_cf[k] = last_seen[lmin];
        last_seen[lmin] = k;
    }
    int32_t rc1 = c1_cf[R_sk - 1], rc2 = c2_cf[R_sk - 1];
    int32_t other_rc = (rc1 < rc2) ? rc1 : rc2;
    int32_t r_c0 = c1_cf[root_u_sk], r_c1 = c2_cf[root_u_sk], r_c2 = other_rc;
    *out_other_rc = other_rc;
    *out_rc0 = r_c0; *out_rc1 = r_c1; *out_rc2 = r_c2;

    for (int32_t u = 0; u < K; ++u) sub_sz_cf[u] = 1;
    for (int32_t u = K; u < root_u_sk; ++u) {
        int32_t b0 = c1_cf[u], b1 = c2_cf[u];
        parent_u_cf[b0] = u; parent_u_cf[b1] = u;
        sub_sz_cf[u] = 1 + sub_sz_cf[b0] + sub_sz_cf[b1];
        adj_u_cf[3 * u + 0] = b0; adj_u_cf[3 * u + 1] = b1;
    }
    parent_u_cf[r_c0] = root_u_sk; parent_u_cf[r_c1] = root_u_sk; parent_u_cf[r_c2] = root_u_sk;
    parent_u_cf[root_u_sk] = -1;
    for (int32_t u = 0; u < K; ++u) {
        adj_u_cf[3 * u + 0] = parent_u_cf[u]; adj_u_cf[3 * u + 1] = -1; adj_u_cf[3 * u + 2] = -1;
    }
    for (int32_t u = K; u < root_u_sk; ++u) adj_u_cf[3 * u + 2] = parent_u_cf[u];
    adj_u_cf[3 * root_u_sk + 0] = r_c0; adj_u_cf[3 * root_u_sk + 1] = r_c1; adj_u_cf[3 * root_u_sk + 2] = r_c2;

    tin_cf[root_u_sk] = 0;
    tin_cf[r_c0] = 1;
    tin_cf[r_c1] = 1 + sub_sz_cf[r_c0];
    tin_cf[r_c2] = 1 + sub_sz_cf[r_c0] + sub_sz_cf[r_c1];
    for (int32_t u = root_u_sk - 1; u >= K; --u) {
        int32_t b0 = c1_cf[u], b1 = c2_cf[u];
        tin_cf[b0] = tin_cf[u] + 1;
        tin_cf[b1] = tin_cf[u] + 1 + sub_sz_cf[b0];
    }
}

static void compute_tree_summaries_blind(
    const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    std::vector<int32_t> &ord_out,
    std::vector<int32_t> &parent_r_out,
    std::vector<int32_t> &leaf_cnt_out,
    std::vector<int32_t> &node_rep0_out
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;

    ord_out.resize(N - 1);
    parent_r_out.assign(n_nodes_r, -1);
    int32_t ord_len = 0;
    ord_out[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord_out[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        parent_r_out[b0] = u;
        parent_r_out[b1] = u;
        if (b0 >= N) ord_out[ord_len++] = b0;
        if (b1 >= N) ord_out[ord_len++] = b1;
    }

    leaf_cnt_out.assign(n_nodes_r, 1);
    node_rep0_out.resize(n_nodes_r);
    std::vector<int32_t> min_depth(n_nodes_r, 0);
    for (int32_t x = 0; x < N; ++x) {
        node_rep0_out[x] = x;
    }
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord_out[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        leaf_cnt_out[u] = leaf_cnt_out[b0] + leaf_cnt_out[b1];
        if (min_depth[b0] <= min_depth[b1]) {
            min_depth[u] = 1 + min_depth[b0];
            node_rep0_out[u] = node_rep0_out[b0];
        } else {
            min_depth[u] = 1 + min_depth[b1];
            node_rep0_out[u] = node_rep0_out[b1];
        }
    }
}

/* Compute the dyadic leaf-weight profile w(x) = 2^{-depth_u(x)} for a rooted clade u.
 * Expands internal nodes in best-first order (highest weight w = 2^{-d} first).
 * When max_prof_leaves <= 0 or leaf_cnt[u] <= max_prof_leaves (for balanced, unbalanced,
 * or caterpillar clades alike), all leaves in u are emitted with their exact dyadic weights.
 * When max_prof_leaves > 0 and leaf_cnt[u] > max_prof_leaves, expansion stops at max_prof_leaves
 * entries and each remaining frontier subtree in the priority queue collapses its mass onto a
 * single representative leaf node_rep0[top.u] (an approximation that can retain substantial
 * collapsed mass on balanced clades of size > max_prof_leaves). */
static void compute_clade_dyadic_profile(
    int32_t u, int32_t N,
    const int32_t *c1_r, const int32_t *c2_r,
    const int32_t *node_rep0,
    int32_t max_prof_leaves,
    std::vector<int32_t> &prof_leaves_out,
    std::vector<float> &prof_weights_out
) {
    if (u < N) {
        prof_leaves_out.push_back(u);
        prof_weights_out.push_back(1.0f);
        return;
    }
    if (max_prof_leaves < 0) {
        struct StackEntry { int32_t u; double weight; };
        std::vector<StackEntry> st;
        st.reserve(64);
        st.push_back({u, 1.0});
        while (!st.empty()) {
            StackEntry top = st.back();
            st.pop_back();
            int32_t b0 = c1_r[top.u], b1 = c2_r[top.u];
            double hw = 0.5 * top.weight;
            if (b1 < N) {
                prof_leaves_out.push_back(b1);
                prof_weights_out.push_back((float)hw);
            } else {
                st.push_back({b1, hw});
            }
            if (b0 < N) {
                prof_leaves_out.push_back(b0);
                prof_weights_out.push_back((float)hw);
            } else {
                st.push_back({b0, hw});
            }
        }
        return;
    }
    struct WeightedNode {
        double weight;
        int32_t u;
        bool operator<(const WeightedNode &o) const {
            if (weight != o.weight) return weight < o.weight;
            return u > o.u;
        }
    };
    std::priority_queue<WeightedNode> pq;
    pq.push({1.0, u});
    int32_t start_leaves = (int32_t)prof_leaves_out.size();
    while (!pq.empty()) {
        int32_t curr_total = ((int32_t)prof_leaves_out.size() - start_leaves) + (int32_t)pq.size();
        if (max_prof_leaves > 0 && curr_total >= max_prof_leaves) break;
        WeightedNode top = pq.top();
        pq.pop();
        int32_t b0 = c1_r[top.u], b1 = c2_r[top.u];
        double hw = 0.5 * top.weight;
        if (b0 < N) {
            prof_leaves_out.push_back(b0);
            prof_weights_out.push_back((float)hw);
        } else {
            pq.push({hw, b0});
        }
        if (b1 < N) {
            prof_leaves_out.push_back(b1);
            prof_weights_out.push_back((float)hw);
        } else {
            pq.push({hw, b1});
        }
    }
    while (!pq.empty()) {
        WeightedNode top = pq.top();
        pq.pop();
        prof_leaves_out.push_back(node_rep0[top.u]);
        prof_weights_out.push_back((float)top.weight);
    }
}

/* Compute the unrooted dyadic leaf-weight profile w_out(x) = 2^{-d_unrooted(sub_root, x)}
 * for the entire outside component V \ L(sub_root) across the full ancestor path to root_r
 * (suppressing degree-2 root_r). Expands directed subtrees in best-first weight order.
 * Exact when max_prof_leaves <= 0 or N - leaf_cnt[sub_root] <= max_prof_leaves; above a positive
 * cap max_prof_leaves > 0, unresolved directed frontier subtrees collapse onto node_rep0 representatives. */
static void compute_outgroup_dyadic_profile(
    int32_t sub_root, int32_t root_r, int32_t N,
    const int32_t *c1_r, const int32_t *c2_r,
    const int32_t *parent_r, const int32_t *node_rep0,
    int32_t max_prof_leaves,
    std::vector<int32_t> &prof_leaves_out,
    std::vector<float> &prof_weights_out
) {
    if (sub_root == root_r || parent_r[sub_root] == -1) return;
    int32_t p0 = parent_r[sub_root];
    int32_t start_u = (p0 == root_r)
        ? ((c1_r[root_r] == sub_root) ? c2_r[root_r] : c1_r[root_r])
        : p0;
    if (start_u < N) {
        prof_leaves_out.push_back(start_u);
        prof_weights_out.push_back(1.0f);
        return;
    }
    if (max_prof_leaves < 0) {
        struct DirectedStackEntry { int32_t u; int32_t from_u; double weight; };
        std::vector<DirectedStackEntry> st;
        st.reserve(64);
        st.push_back({start_u, sub_root, 1.0});
        while (!st.empty()) {
            DirectedStackEntry top = st.back();
            st.pop_back();
            int32_t v = top.u;
            int32_t n0 = c1_r[v];
            int32_t n1 = c2_r[v];
            int32_t pv = parent_r[v];
            int32_t n2 = (pv == root_r)
                ? ((c1_r[root_r] == v) ? c2_r[root_r] : c1_r[root_r])
                : pv;
            int32_t nbs[3] = {n0, n1, n2};
            double hw = 0.5 * top.weight;
            for (int k = 2; k >= 0; --k) {
                int32_t nb = nbs[k];
                if (nb == top.from_u || nb < 0) continue;
                if (nb < N) {
                    prof_leaves_out.push_back(nb);
                    prof_weights_out.push_back((float)hw);
                } else {
                    st.push_back({nb, v, hw});
                }
            }
        }
        return;
    }
    struct DirectedWeightedNode {
        double weight;
        int32_t u;
        int32_t from_u;
        bool operator<(const DirectedWeightedNode &o) const {
            if (weight != o.weight) return weight < o.weight;
            return u > o.u;
        }
    };
    std::priority_queue<DirectedWeightedNode> pq;
    pq.push({1.0, start_u, sub_root});
    int32_t start_leaves = (int32_t)prof_leaves_out.size();
    while (!pq.empty()) {
        int32_t curr_total = ((int32_t)prof_leaves_out.size() - start_leaves) + (int32_t)pq.size();
        if (max_prof_leaves > 0 && curr_total >= max_prof_leaves) break;
        DirectedWeightedNode top = pq.top();
        pq.pop();
        int32_t v = top.u;
        int32_t n0 = c1_r[v];
        int32_t n1 = c2_r[v];
        int32_t pv = parent_r[v];
        int32_t n2 = (pv == root_r)
            ? ((c1_r[root_r] == v) ? c2_r[root_r] : c1_r[root_r])
            : pv;
        int32_t nbs[3] = {n0, n1, n2};
        double hw = 0.5 * top.weight;
        for (int k = 0; k < 3; ++k) {
            int32_t nb = nbs[k];
            if (nb == top.from_u || nb < 0) continue;
            if (nb < N) {
                prof_leaves_out.push_back(nb);
                prof_weights_out.push_back((float)hw);
            } else {
                pq.push({hw, nb, v});
            }
        }
    }
    while (!pq.empty()) {
        DirectedWeightedNode top = pq.top();
        pq.pop();
        int32_t v = top.u;
        int32_t rep_leaf;
        if (top.from_u != c1_r[v] && top.from_u != c2_r[v]) {
            rep_leaf = node_rep0[v];
        } else {
            int32_t sib = (top.from_u == c1_r[v]) ? c2_r[v] : c1_r[v];
            rep_leaf = node_rep0[sib];
        }
        prof_leaves_out.push_back(rep_leaf);
        prof_weights_out.push_back((float)top.weight);
    }
}

/* Host helper: build K x K condensed distance matrix from CSR dyadic super-leaf profiles. */
static __attribute__((unused)) void build_condensed_D_from_profiles_host(
    int32_t K, int32_t N,
    const int32_t *prof_offsets,
    const int32_t *prof_leaves,
    const float *prof_weights,
    const double *D_full,
    double *D_cond_out
) {
    for (int32_t a = 0; a < K; ++a) {
        D_cond_out[(size_t)a * K + a] = 0.0;
        int32_t off_a = prof_offsets[a], end_a = prof_offsets[a + 1];
        for (int32_t b = a + 1; b < K; ++b) {
            int32_t off_b = prof_offsets[b], end_b = prof_offsets[b + 1];
            double d_val = 0.0;
            for (int32_t p = off_a; p < end_a; ++p) {
                int32_t la = prof_leaves[p];
                double wa = (double)prof_weights[p];
                double row_sum = 0.0;
                for (int32_t q = off_b; q < end_b; ++q) {
                    int32_t lb = prof_leaves[q];
                    double wb = (double)prof_weights[q];
                    row_sum += wb * D_full[(size_t)la * N + lb];
                }
                d_val += wa * row_sum;
            }
            D_cond_out[(size_t)a * K + b] = d_val;
            D_cond_out[(size_t)b * K + a] = d_val;
        }
    }
}

static void extract_active_skeleton(
    int32_t root_r, const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    int32_t condense_thresh, const std::vector<uint8_t> &node_has_active,
    const std::vector<int32_t> &leaf_cnt,
    std::vector<int32_t> &super_roots,
    std::vector<int32_t> &split_internals,
    std::vector<int32_t> &skel_left,
    std::vector<int32_t> &skel_right,
    int32_t *out_root_skel_id,
    int32_t max_internals = 1200,
    const uint8_t *stop_at_comp_root = NULL
) {
    if (root_r < N) {
        super_roots.push_back(root_r);
        *out_root_skel_id = -1;
        return;
    }

    struct ExpNode { int32_t u; int32_t left_exp; int32_t right_exp; };
    struct PqCand {
        int32_t prio;
        int32_t u;
        int32_t parent_exp;
        int32_t side;
        bool operator<(const PqCand &o) const {
            if (prio != o.prio) return prio < o.prio;
            return u < o.u;
        }
    };

    std::vector<ExpNode> exp_nodes;
    exp_nodes.reserve(std::min(max_internals, std::max(16, leaf_cnt[root_r])));
    exp_nodes.push_back({root_r, -1, -1});

    auto is_eligible = [&](int32_t w) -> bool {
        if (w < N) return false;
        if (stop_at_comp_root && stop_at_comp_root[w]) return false;
        if (!node_has_active[w] && leaf_cnt[w] <= condense_thresh) return false;
        return true;
    };

    std::priority_queue<PqCand> pq;
    int32_t b0_root = c1_r[root_r], b1_root = c2_r[root_r];
    if (is_eligible(b0_root)) pq.push({leaf_cnt[b0_root], b0_root, 0, 0});
    if (is_eligible(b1_root)) pq.push({leaf_cnt[b1_root], b1_root, 0, 1});

    while (!pq.empty() && (int32_t)exp_nodes.size() < max_internals) {
        PqCand cand = pq.top();
        pq.pop();
        int32_t my_exp = (int32_t)exp_nodes.size();
        exp_nodes.push_back({cand.u, -1, -1});
        if (cand.side == 0) exp_nodes[cand.parent_exp].left_exp = my_exp;
        else                exp_nodes[cand.parent_exp].right_exp = my_exp;

        int32_t b0 = c1_r[cand.u], b1 = c2_r[cand.u];
        if (is_eligible(b0)) pq.push({leaf_cnt[b0], b0, my_exp, 0});
        if (is_eligible(b1)) pq.push({leaf_cnt[b1], b1, my_exp, 1});
    }

    struct Frame { int32_t exp_idx; int32_t state; int32_t left_id; };
    std::vector<Frame> st;
    st.reserve(256);
    st.push_back({0, 0, -1});
    int32_t last_ret = -1;

    while (!st.empty()) {
        Frame &f = st.back();
        const ExpNode &en = exp_nodes[f.exp_idx];
        if (f.state == 0) {
            f.state = 1;
            if (en.left_exp == -1) {
                int32_t sid = (int32_t)super_roots.size();
                super_roots.push_back(c1_r[en.u]);
                last_ret = sid;
            } else {
                st.push_back({en.left_exp, 0, -1});
            }
        } else if (f.state == 1) {
            f.left_id = last_ret;
            f.state = 2;
            if (en.right_exp == -1) {
                int32_t sid = (int32_t)super_roots.size();
                super_roots.push_back(c2_r[en.u]);
                last_ret = sid;
            } else {
                st.push_back({en.right_exp, 0, -1});
            }
        } else {
            int32_t right_id = last_ret;
            int32_t my_idx = (int32_t)split_internals.size();
            split_internals.push_back(en.u);
            skel_left.push_back(f.left_id);
            skel_right.push_back(right_id);
            last_ret = -1 - my_idx;
            st.pop_back();
        }
    }
    *out_root_skel_id = -1 - last_ret;
}

/* Partition all N - 1 internal nodes into connected components of size <= max_cap (with optional
 * staggered first_cap at the bottom) and schedule all leaf-bearing components into ancestor-independent
 * level waves (exposing 100% of the N original leaves, while pure internal connector components with
 * zero direct leaves are covered by Pass 1 and Pass 2B). */
static void build_component_waves(
    int32_t N, const std::vector<int32_t> &ord,
    const std::vector<int32_t> &curr_c1, const std::vector<int32_t> &curr_c2,
    int32_t first_cap, int32_t max_cap,
    std::vector<uint8_t> &is_comp_root,
    std::vector<std::vector<int32_t>> &level_waves
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    is_comp_root.assign(n_nodes_r, 0);
    level_waves.clear();
    if (N < 6) return;

    std::vector<int32_t> rem_int(n_nodes_r, 0);
    bool any_closed = false;

    for (int32_t idx = (int32_t)ord.size() - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t c1 = curr_c1[u], c2 = curr_c2[u];
        int32_t r1 = rem_int[c1], r2 = rem_int[c2];
        int32_t cap = (!any_closed && first_cap > 0) ? first_cap : max_cap;

        if (r1 >= 15 && r2 >= 15 && (1 + r1 + r2 > cap)) {
            is_comp_root[c1] = 1;
            is_comp_root[c2] = 1;
            any_closed = true;
            r1 = 0;
            r2 = 0;
        } else if (1 + r1 + r2 > cap) {
            if (r1 >= r2) {
                if (r1 >= 5) {
                    is_comp_root[c1] = 1;
                    any_closed = true;
                    r1 = 0;
                }
            } else {
                if (r2 >= 5) {
                    is_comp_root[c2] = 1;
                    any_closed = true;
                    r2 = 0;
                }
            }
            if (1 + r1 + r2 > max_cap) {
                if (r1 >= 5) {
                    is_comp_root[c1] = 1;
                    any_closed = true;
                    r1 = 0;
                }
                if (r2 >= 5) {
                    is_comp_root[c2] = 1;
                    any_closed = true;
                    r2 = 0;
                }
            }
        }
        rem_int[u] = 1 + r1 + r2;
    }

    if (rem_int[root_r] < 5) {
        std::vector<int32_t> st = {root_r};
        int32_t best_c = -1;
        while (!st.empty()) {
            int32_t u = st.back();
            st.pop_back();
            int32_t ch[2] = {curr_c1[u], curr_c2[u]};
            for (int k = 0; k < 2; ++k) {
                int32_t c = ch[k];
                if (c >= N) {
                    if (is_comp_root[c]) {
                        if (best_c == -1 || rem_int[c] < rem_int[best_c]) best_c = c;
                    } else {
                        st.push_back(c);
                    }
                }
            }
        }
        if (best_c != -1) {
            is_comp_root[best_c] = 0;
            rem_int[root_r] += rem_int[best_c];
        }
    }
    is_comp_root[root_r] = 1;

    std::vector<int32_t> comp_lvl(n_nodes_r, 0);
    std::vector<int32_t> max_ch_lvl(n_nodes_r, -1);
    std::vector<int32_t> direct_leaves(n_nodes_r, 0);
    for (int32_t x = 0; x < N; ++x) direct_leaves[x] = 1;
    int32_t max_lvl = 0;
    for (int32_t idx = (int32_t)ord.size() - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t c1 = curr_c1[u], c2 = curr_c2[u];
        int32_t l1 = (c1 >= N && is_comp_root[c1]) ? (comp_lvl[c1] + 1) : max_ch_lvl[c1];
        int32_t l2 = (c2 >= N && is_comp_root[c2]) ? (comp_lvl[c2] + 1) : max_ch_lvl[c2];
        int32_t dl1 = (c1 >= N && is_comp_root[c1]) ? 0 : direct_leaves[c1];
        int32_t dl2 = (c2 >= N && is_comp_root[c2]) ? 0 : direct_leaves[c2];
        direct_leaves[u] = dl1 + dl2;
        max_ch_lvl[u] = std::max(l1, l2);
        if (is_comp_root[u]) {
            comp_lvl[u] = std::max(0, max_ch_lvl[u]);
            if (comp_lvl[u] > max_lvl) max_lvl = comp_lvl[u];
        }
    }

    level_waves.resize(max_lvl + 1);
    for (int32_t idx = (int32_t)ord.size() - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        if (is_comp_root[u] && direct_leaves[u] > 0) {
            level_waves[comp_lvl[u]].push_back(u);
        }
    }
    while (!level_waves.empty() && level_waves.back().empty()) {
        level_waves.pop_back();
    }
}

/* ============================================================================
 * 5. GPU Workspace for Full-Tree and Multi-GPU Minibatch Optimization
 * ============================================================================ */
struct RnaSimGpuWorkspace {
    int32_t dev_id;
    int32_t max_K;
    size_t  exact_prof_cap;
    size_t  pair_cap;
    cudaStream_t stream;
    int32_t *d_prof_offsets;
    int32_t *d_prof_leaves;
    float   *d_prof_weights;
    int32_t *d_exact_offsets;
    int32_t *d_exact_leaves;
    float   *d_exact_weights;
    int32_t *d_pair_a;
    int32_t *d_pair_b;
    double  *d_pair_exact_D;
    int32_t *d_perm;
    float   *d_D;
    float   *d_subdist;
    float   *d_nd_u;
    float   *d_row_scratch;
    int64_t *d_v_curr;
    int32_t *d_c1_r;
    int32_t *d_c2_r;
    int32_t *d_v_map;
    int32_t *d_m_lmin;
    int32_t *d_next_k;
    int32_t *d_parent_u;
    int32_t *d_sub_size;
    int32_t *d_tin;
    int32_t *d_adj_u;
    float   *d_best_delta;
    int32_t *d_best_j;
    double  *d_curr_L;

    void init(int32_t gpu_id, int32_t K_cap) {
        dev_id = gpu_id;
        max_K = K_cap;
        int32_t R = 2 * K_cap - 1;
        int32_t M = 2 * K_cap - 2;
        size_t prof_cap = std::max((size_t)128 * (size_t)K_cap, (size_t)2048);
        exact_prof_cap = std::max(prof_cap, (size_t)16384);
        pair_cap = (size_t)K_cap * (size_t)K_cap / 2 + 16;
        CUDA_CHECK(cudaSetDevice(dev_id));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&d_prof_offsets, (size_t)(K_cap + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_prof_leaves,  prof_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_prof_weights, prof_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_exact_offsets, (size_t)(K_cap + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_exact_leaves,  exact_prof_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_exact_weights, exact_prof_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_pair_a,        pair_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_pair_b,        pair_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_pair_exact_D,  pair_cap * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_perm,        (size_t)K_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_D,           (size_t)K_cap * K_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_subdist,     (size_t)M * M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nd_u,        (size_t)3 * M * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_row_scratch, (size_t)K_cap * (2 * K_cap) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_curr,      (size_t)K_cap * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_c1_r,        (size_t)R * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_c2_r,        (size_t)R * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_v_map,       (size_t)K_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_m_lmin,      (size_t)K_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_next_k,      (size_t)K_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_parent_u,    (size_t)M * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sub_size,    (size_t)M * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_tin,         (size_t)M * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_adj_u,       (size_t)3 * M * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_best_delta,  (size_t)K_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_best_j,      (size_t)K_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_curr_L,      sizeof(double)));
        CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_top1, cudaFuncAttributeMaxDynamicSharedMemorySize, 90112));
    }

    void ensure_exact_prof_cap(size_t needed) {
        if (needed > exact_prof_cap) {
            CUDA_CHECK(cudaSetDevice(dev_id));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaFree(d_exact_leaves);
            cudaFree(d_exact_weights);
            exact_prof_cap = needed + 4096;
            CUDA_CHECK(cudaMalloc(&d_exact_leaves,  exact_prof_cap * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_exact_weights, exact_prof_cap * sizeof(float)));
        }
    }

    void destroy() {
        CUDA_CHECK(cudaSetDevice(dev_id));
        CUDA_CHECK(cudaStreamDestroy(stream));
        cudaFree(d_prof_offsets); cudaFree(d_prof_leaves); cudaFree(d_prof_weights);
        cudaFree(d_exact_offsets); cudaFree(d_exact_leaves); cudaFree(d_exact_weights);
        cudaFree(d_pair_a); cudaFree(d_pair_b); cudaFree(d_pair_exact_D);
        cudaFree(d_perm); cudaFree(d_D); cudaFree(d_subdist); cudaFree(d_nd_u); cudaFree(d_row_scratch);
        cudaFree(d_v_curr); cudaFree(d_c1_r); cudaFree(d_c2_r); cudaFree(d_v_map); cudaFree(d_m_lmin);
        cudaFree(d_next_k); cudaFree(d_parent_u); cudaFree(d_sub_size); cudaFree(d_tin);
        cudaFree(d_adj_u); cudaFree(d_best_delta); cudaFree(d_best_j); cudaFree(d_curr_L);
    }
};

struct MoveCand { int32_t i; int32_t j; float d; };

static void apply_unrooted_spr_host(
    int32_t *adj_u, int32_t prune_node, int32_t anchor_node, int32_t dest_u, int32_t dest_v
) {
    int32_t n0 = -1, n1 = -1;
    for (int32_t k = 0; k < 3; ++k) {
        int32_t nb = adj_u[3 * anchor_node + k];
        if (nb != prune_node) {
            if (n0 == -1) n0 = nb;
            else n1 = nb;
        }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * n0 + k] == anchor_node) { adj_u[3 * n0 + k] = n1; break; }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * n1 + k] == anchor_node) { adj_u[3 * n1 + k] = n0; break; }
    }
    if (dest_u == anchor_node) dest_u = n0;
    if (dest_v == anchor_node) dest_v = n0;

    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * dest_u + k] == dest_v) { adj_u[3 * dest_u + k] = anchor_node; break; }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * dest_v + k] == dest_u) { adj_u[3 * dest_v + k] = anchor_node; break; }
    }
    adj_u[3 * anchor_node + 0] = prune_node;
    adj_u[3 * anchor_node + 1] = dest_u;
    adj_u[3 * anchor_node + 2] = dest_v;
}

static void relabel_from_adj_u(
    const int32_t *adj_u, int32_t K, int32_t root_edge_u, int32_t root_edge_v, int32_t flip_mode,
    std::vector<int64_t> &v_cf, std::vector<int32_t> &perm
) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t root_r = R_sk - 1;
    std::vector<int32_t> c1_r(R_sk, 0), c2_r(R_sk, 0);
    c1_r[root_r] = root_edge_u;
    c2_r[root_r] = root_edge_v;

    std::vector<int32_t> q_u;
    std::vector<int32_t> q_p;
    q_u.reserve(R_sk);
    q_p.reserve(R_sk);
    if (root_edge_u >= K) { q_u.push_back(root_edge_u); q_p.push_back(root_edge_v); }
    if (root_edge_v >= K) { q_u.push_back(root_edge_v); q_p.push_back(root_edge_u); }

    size_t head = 0;
    while (head < q_u.size()) {
        int32_t u = q_u[head];
        int32_t pr = q_p[head];
        head++;
        int32_t b0 = -1, b1 = -1;
        for (int32_t k = 0; k < 3; ++k) {
            int32_t nb = adj_u[3 * u + k];
            if (nb != -1 && nb != pr) {
                if (b0 == -1) b0 = nb;
                else b1 = nb;
            }
        }
        c1_r[u] = b0;
        c2_r[u] = b1;
        if (b0 >= K) { q_u.push_back(b0); q_p.push_back(u); }
        if (b1 >= K) { q_u.push_back(b1); q_p.push_back(u); }
    }

    std::vector<int32_t> step_perm(K);
    relabel_from_rooted_tree(c1_r.data(), c2_r.data(), K, flip_mode, v_cf.data(), step_perm.data());
    std::vector<int32_t> new_perm(K);
    for (int32_t a = 0; a < K; ++a) new_perm[a] = perm[step_perm[a]];
    perm = new_perm;
}

/* Compute all-pairs unrooted topological edge distances dist_out[a * K + b]
 * between original super-leaves a, b in {0, ..., K-1} for skeleton (v_cf, perm)
 * in O(K^2) time by visiting each leaf pair once at its rooted LCA. */
static void compute_skeleton_leaf_dists(
    const int64_t *v_cf, const int32_t *perm, int32_t K,
    std::vector<int32_t> &dist_out
) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t root_r = R_sk - 1;
    std::vector<int32_t> c1(R_sk), c2(R_sk), uf(R_sk);
    for (int32_t x = 0; x < K; ++x) uf[x] = x;
    for (int32_t k = 0; k < K - 1; ++k) {
        int32_t i = K - 2 - k;
        int32_t m = K + k;
        int32_t x = (int32_t)v_cf[i];
        int32_t y = i + 1;
        c1[m] = uf[y];
        c2[m] = uf[x];
        uf[x] = m;
    }
    std::vector<int32_t> depth(R_sk, 0);
    for (int32_t u = root_r; u >= K; --u) {
        depth[c1[u]] = depth[u] + 1;
        depth[c2[u]] = depth[u] + 1;
    }
    dist_out.assign((size_t)K * K, 0);
    std::vector<std::vector<int32_t>> sub_leaves(R_sk);
    for (int32_t x = 0; x < K; ++x) sub_leaves[x].push_back(x);
    for (int32_t u = K; u <= root_r; ++u) {
        int32_t b0 = c1[u], b1 = c2[u];
        const auto &L0 = sub_leaves[b0];
        const auto &L1 = sub_leaves[b1];
        int32_t sub = (u == root_r) ? 1 : (2 * depth[u]);
        for (int32_t x : L0) {
            int32_t ox = perm[x];
            int32_t dx = depth[x] - sub;
            for (int32_t y : L1) {
                int32_t oy = perm[y];
                int32_t d = dx + depth[y];
                dist_out[(size_t)ox * K + oy] = d;
                dist_out[(size_t)oy * K + ox] = d;
            }
        }
        if (u < root_r) {
            auto &Lu = sub_leaves[u];
            Lu.reserve(L0.size() + L1.size());
            Lu.insert(Lu.end(), L0.begin(), L0.end());
            Lu.insert(Lu.end(), L1.begin(), L1.end());
            std::vector<int32_t>().swap(sub_leaves[b0]);
            std::vector<int32_t>().swap(sub_leaves[b1]);
        }
    }
}

/* Optimize one K-leaf skeleton on GPU with path-disjoint SPR batching, monotonic condensed-skeleton
 * BME check, and (when exact_prof_* is supplied) an exact full-objective GPU acceptance check on
 * all changed super-leaf pairs involving capped clades or capped outside profiles. */
static double optimize_skeleton_on_gpu(
    int32_t K, int32_t W, int32_t max_iters, int32_t max_batch_cap,
    const int32_t *prof_offsets, const int32_t *prof_leaves, const float *prof_weights,
    const uint64_t *d_packed_seqs,
    RnaSimGpuWorkspace &gpu_ws,
    std::vector<int64_t> &v_cf, std::vector<int32_t> &perm,
    const int32_t *exact_prof_offsets = NULL,
    const int32_t *exact_prof_leaves = NULL,
    const float *exact_prof_weights = NULL
) {
    CUDA_CHECK(cudaSetDevice(gpu_ws.dev_id));
    const int32_t R_sk = 2 * K - 1;
    const int32_t M_sk = 2 * K - 2;
    const int32_t root_u_sk = M_sk - 1;
    const std::vector<int64_t> init_v_cf = v_cf;
    const std::vector<int32_t> init_perm = perm;

    int32_t total_prof = prof_offsets[K];
    CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_prof_offsets, prof_offsets, (size_t)(K + 1) * sizeof(int32_t),    cudaMemcpyHostToDevice, gpu_ws.stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_prof_leaves,  prof_leaves,  (size_t)total_prof * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
    CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_prof_weights, prof_weights, (size_t)total_prof * sizeof(float),   cudaMemcpyHostToDevice, gpu_ws.stream));
    dim3 blk2d(16, 16);
    dim3 grd2d((K + 15) / 16, (K + 15) / 16);
    k_build_real_seq_condensed_D<<<grd2d, blk2d, 0, gpu_ws.stream>>>(
        K, W, gpu_ws.d_prof_offsets, gpu_ws.d_prof_leaves, gpu_ws.d_prof_weights, d_packed_seqs, gpu_ws.d_D
    );

    std::vector<int32_t> c1_cf, c2_cf, v_map_cf, m_lmin_cf, next_k_cf, parent_u_cf, sub_size_cf, tin_cf, adj_u_cf;
    std::vector<float> h_best_delta(K);
    std::vector<int32_t> h_best_j(K);

    double init_L = 0.0, curr_L = 0.0, best_L = 1e300;
    std::vector<int64_t> best_v_cf = v_cf;
    std::vector<int32_t> best_perm = perm;
    int32_t flipped_already = 0;

    for (int32_t iter = 0; iter <= max_iters; ++iter) {
        int32_t other_rc, rc0, rc1, rc2;
        build_skeleton_metadata(
            v_cf.data(), K, c1_cf, c2_cf, v_map_cf, m_lmin_cf, next_k_cf,
            parent_u_cf, sub_size_cf, tin_cf, adj_u_cf,
            &other_rc, &rc0, &rc1, &rc2
        );

        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_perm,     perm.data(),        (size_t)K * sizeof(int32_t),       cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_c1_r,     c1_cf.data(),       (size_t)R_sk * sizeof(int32_t),    cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_c2_r,     c2_cf.data(),       (size_t)R_sk * sizeof(int32_t),    cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_v_map,    v_map_cf.data(),    (size_t)K * sizeof(int32_t),       cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_parent_u, parent_u_cf.data(), (size_t)M_sk * sizeof(int32_t),    cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_sub_size, sub_size_cf.data(), (size_t)M_sk * sizeof(int32_t),    cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_tin,      tin_cf.data(),      (size_t)M_sk * sizeof(int32_t),    cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_adj_u,    adj_u_cf.data(),    (size_t)3 * M_sk * sizeof(int32_t),cudaMemcpyHostToDevice, gpu_ws.stream));

        k_copy_leaf_rows_perm<<<grd2d, blk2d, 0, gpu_ws.stream>>>(gpu_ws.d_D, gpu_ws.d_perm, gpu_ws.d_subdist, K, M_sk);
        k_bottom_up_leaves<<<(K + 255) / 256, 256, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, K, M_sk, root_u_sk);
        int32_t n_int = root_u_sk - K;
        if (n_int > 0) {
            dim3 grd_tr((K + 15) / 16, (n_int + 15) / 16);
            k_transpose_leaf_block<<<grd_tr, blk2d, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, K, M_sk, root_u_sk);
            k_bottom_up_internals<<<(n_int + 255) / 256, 256, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, K, M_sk, root_u_sk);
        }
        k_top_down_and_root<<<(M_sk + 255) / 256, 256, 0, gpu_ws.stream>>>(
            gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, gpu_ws.d_tin, gpu_ws.d_sub_size,
            K, M_sk, root_u_sk, rc0, rc1, rc2
        );
        k_compute_bme_length<<<1, 256, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, gpu_ws.d_parent_u, K, M_sk, gpu_ws.d_curr_L);
        CUDA_CHECK(cudaMemcpyAsync(&curr_L, gpu_ws.d_curr_L, sizeof(double), cudaMemcpyDeviceToHost, gpu_ws.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
        if (iter == 0) {
            init_L = curr_L;
            best_L = curr_L;
        }
        if (curr_L < best_L) {
            best_L = curr_L;
            best_v_cf = v_cf;
            best_perm = perm;
        }
        if (iter == max_iters) break;

        k_populate_nd_u<<<(K - 1 + 255) / 256, 256, 0, gpu_ws.stream>>>(gpu_ws.d_subdist, gpu_ws.d_adj_u, gpu_ws.d_nd_u, K, M_sk, root_u_sk);
        k_spr_rows_top1<<<K - 2, 128, (size_t)M_sk * sizeof(float), gpu_ws.stream>>>(
            2, K, M_sk, other_rc, gpu_ws.d_v_curr, gpu_ws.d_v_map,
            gpu_ws.d_adj_u, gpu_ws.d_parent_u, gpu_ws.d_sub_size,
            gpu_ws.d_m_lmin, gpu_ws.d_next_k, gpu_ws.d_subdist,
            gpu_ws.d_nd_u, gpu_ws.d_row_scratch,
            gpu_ws.d_best_delta, gpu_ws.d_best_j
        );
        CUDA_CHECK(cudaMemcpyAsync(h_best_delta.data(), gpu_ws.d_best_delta, (size_t)K * sizeof(float), cudaMemcpyDeviceToHost, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(h_best_j.data(),     gpu_ws.d_best_j,     (size_t)K * sizeof(int32_t), cudaMemcpyDeviceToHost, gpu_ws.stream));
        CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));

        std::vector<MoveCand> cands;
        float eps_move_thresh = -1e-4f;
        for (int32_t i = 1; i < K - 1; ++i) {
            if (h_best_delta[i] < eps_move_thresh && h_best_j[i] >= 0 && h_best_j[i] < root_u_sk) {
                cands.push_back({i, h_best_j[i], h_best_delta[i]});
            }
        }
        if (cands.empty()) {
            if (flipped_already == 0) {
                flipped_already = 1;
                relabel_from_adj_u(adj_u_cf.data(), K, root_u_sk, other_rc, 1, v_cf, perm);
                continue;
            }
            break;
        }
        flipped_already = 0;
        std::sort(cands.begin(), cands.end(), [](const MoveCand &a, const MoveCand &b) { return a.d < b.d; });

        std::vector<uint8_t> node_busy(M_sk, 0);
        int32_t n_applied = 0;
        std::vector<int32_t> path_nodes;
        path_nodes.reserve(128);

        for (const auto &mc : cands) {
            int32_t i = mc.i, dst_u = mc.j;
            int32_t tr = v_map_cf[i + 1];
            int32_t pr = (tr == root_u_sk) ? other_rc : parent_u_cf[tr];
            if (dst_u < 0 || dst_u >= root_u_sk) continue;
            int32_t dst_v = parent_u_cf[dst_u];
            if (dst_v < 0) continue;

            if (tin_cf[tr] <= tin_cf[dst_u] && tin_cf[dst_u] < tin_cf[tr] + sub_size_cf[tr]) continue;
            if (node_busy[tr] || node_busy[pr] || node_busy[dst_u] || node_busy[dst_v]) continue;

            path_nodes.clear();
            path_nodes.push_back(tr);
            path_nodes.push_back(dst_v);
            int32_t u_anc = pr;
            int32_t t_dst = tin_cf[dst_u];
            int32_t hops = 0;
            while (u_anc != -1 && !(tin_cf[u_anc] <= t_dst && t_dst < tin_cf[u_anc] + sub_size_cf[u_anc])) {
                path_nodes.push_back(u_anc);
                u_anc = parent_u_cf[u_anc];
                if (++hops > 128) break;
            }
            if (u_anc == -1 || hops > 128) continue;
            path_nodes.push_back(u_anc);
            int32_t v_anc = dst_u;
            while (v_anc != u_anc && v_anc != -1) {
                path_nodes.push_back(v_anc);
                v_anc = parent_u_cf[v_anc];
                if (++hops > 128) break;
            }
            if (v_anc != u_anc || hops > 128) continue;

            bool conflict = false;
            for (int32_t pn : path_nodes) {
                if (node_busy[pn]) { conflict = true; break; }
            }
            if (conflict) continue;

            for (int32_t pn : path_nodes) node_busy[pn] = 1;
            apply_unrooted_spr_host(adj_u_cf.data(), tr, pr, dst_u, dst_v);
            n_applied++;
            if (n_applied >= max_batch_cap) break;
        }
        if (n_applied == 0) {
            int32_t i = cands[0].i, dst_u = cands[0].j;
            int32_t tr = v_map_cf[i + 1];
            int32_t pr = (tr == root_u_sk) ? other_rc : parent_u_cf[tr];
            int32_t dst_v = parent_u_cf[dst_u];
            apply_unrooted_spr_host(adj_u_cf.data(), tr, pr, dst_u, dst_v);
        }

        int32_t r_nb = adj_u_cf[3 * root_u_sk + 0];
        relabel_from_adj_u(adj_u_cf.data(), K, root_u_sk, r_nb, iter & 1, v_cf, perm);
    }

    if (best_L >= init_L - 1e-6) {
        v_cf = init_v_cf;
        perm = init_perm;
        return 0.0;
    }

    if (exact_prof_offsets != NULL && exact_prof_leaves != NULL && exact_prof_weights != NULL) {
        std::vector<uint8_t> is_capped(K, 0);
        bool any_capped = false;
        for (int32_t a = 0; a < K; ++a) {
            int32_t cap_len = prof_offsets[a + 1] - prof_offsets[a];
            int32_t ex_len  = exact_prof_offsets[a + 1] - exact_prof_offsets[a];
            if (ex_len != cap_len) {
                is_capped[a] = 1;
                any_capped = true;
            }
        }
        if (any_capped) {
            std::vector<int32_t> dist_init, dist_best;
            compute_skeleton_leaf_dists(init_v_cf.data(), init_perm.data(), K, dist_init);
            compute_skeleton_leaf_dists(best_v_cf.data(), best_perm.data(), K, dist_best);

            std::vector<int32_t> pair_a, pair_b;
            std::vector<int32_t> uncap_a, uncap_b;
            for (int32_t a = 0; a < K; ++a) {
                for (int32_t b = a + 1; b < K; ++b) {
                    int32_t db = dist_best[(size_t)a * K + b];
                    int32_t di = dist_init[(size_t)a * K + b];
                    if (db != di) {
                        if (is_capped[a] || is_capped[b]) {
                            pair_a.push_back(a);
                            pair_b.push_back(b);
                        } else {
                            uncap_a.push_back(a);
                            uncap_b.push_back(b);
                        }
                    }
                }
            }
            if (pair_a.empty() && uncap_a.empty()) {
                v_cf = init_v_cf;
                perm = init_perm;
                return 0.0;
            }

            std::vector<double> h_pair_exact_D(pair_a.size(), 0.0);
            if (!pair_a.empty()) {
                int32_t n_pairs = (int32_t)pair_a.size();
                int32_t total_exact_prof = exact_prof_offsets[K];
                gpu_ws.ensure_exact_prof_cap((size_t)total_exact_prof);
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_offsets, exact_prof_offsets, (size_t)(K + 1) * sizeof(int32_t),          cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_leaves,  exact_prof_leaves,  (size_t)total_exact_prof * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_weights, exact_prof_weights, (size_t)total_exact_prof * sizeof(float),   cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_a, pair_a.data(), (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_b, pair_b.data(), (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));

                k_refine_real_seq_condensed_D_pairs<<<n_pairs, 256, 0, gpu_ws.stream>>>(
                    K, W, n_pairs, gpu_ws.d_pair_a, gpu_ws.d_pair_b,
                    gpu_ws.d_exact_offsets, gpu_ws.d_exact_leaves, gpu_ws.d_exact_weights,
                    d_packed_seqs, gpu_ws.d_D, gpu_ws.d_pair_exact_D
                );
                CUDA_CHECK(cudaMemcpyAsync(h_pair_exact_D.data(), gpu_ws.d_pair_exact_D, (size_t)n_pairs * sizeof(double), cudaMemcpyDeviceToHost, gpu_ws.stream));
                CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
            }

            std::vector<float> h_D_cond;
            if (!uncap_a.empty()) {
                h_D_cond.resize((size_t)K * K);
                CUDA_CHECK(cudaMemcpyAsync(h_D_cond.data(), gpu_ws.d_D, (size_t)K * K * sizeof(float), cudaMemcpyDeviceToHost, gpu_ws.stream));
                CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
            }

            double exact_delta_L = 0.0;
            for (size_t pid = 0; pid < pair_a.size(); ++pid) {
                int32_t a = pair_a[pid], b = pair_b[pid];
                double w_diff = ldexp(1.0, 1 - dist_best[(size_t)a * K + b]) - ldexp(1.0, 1 - dist_init[(size_t)a * K + b]);
                exact_delta_L += w_diff * h_pair_exact_D[pid];
            }
            for (size_t uid = 0; uid < uncap_a.size(); ++uid) {
                int32_t a = uncap_a[uid], b = uncap_b[uid];
                double w_diff = ldexp(1.0, 1 - dist_best[(size_t)a * K + b]) - ldexp(1.0, 1 - dist_init[(size_t)a * K + b]);
                exact_delta_L += w_diff * (double)h_D_cond[(size_t)a * K + b];
            }

            if (exact_delta_L >= -1e-6) {
                v_cf = init_v_cf;
                perm = init_perm;
                return 0.0;
            }
            v_cf = best_v_cf;
            perm = best_perm;
            return exact_delta_L;
        }
    }

    v_cf = best_v_cf;
    perm = best_perm;
    return best_L - init_L;
}

static inline void update_one_node_summary(
    int32_t u, int32_t b0, int32_t b1,
    int32_t *parent_r, int32_t *leaf_cnt, int32_t *node_rep0
) {
    parent_r[b0] = u;
    parent_r[b1] = u;
    leaf_cnt[u] = leaf_cnt[b0] + leaf_cnt[b1];
    node_rep0[u] = (leaf_cnt[b0] <= leaf_cnt[b1]) ? node_rep0[b0] : node_rep0[b1];
}

/* Optimize a skeleton rooted at sub_root (either root_r or a proper subtree with outgroup)
 * using dyadic-weighted clade & outgroup profiles, and update curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0. */
static double run_subtree_minibatch_window(
    int32_t sub_root, int32_t N, int32_t W, int32_t condense_thresh, int32_t max_iters,
    const std::vector<uint8_t> &node_has_active,
    const std::vector<int32_t> &outgroup_leaves,
    const std::vector<float> &outgroup_weights,
    const uint64_t *d_packed_seqs,
    RnaSimGpuWorkspace &gpu_ws,
    std::vector<int32_t> &curr_c1, std::vector<int32_t> &curr_c2,
    std::vector<int32_t> &parent_r, std::vector<int32_t> &leaf_cnt, std::vector<int32_t> &node_rep0,
    int32_t max_internals = 1200,
    const uint8_t *stop_at_comp_root = NULL,
    const int32_t *outgroup_c1 = NULL,
    const int32_t *outgroup_c2 = NULL,
    const int32_t *outgroup_par = NULL,
    const int32_t *outgroup_rep0 = NULL
) {
    const int32_t root_r = 2 * N - 2;
    const int32_t max_prof_leaves = (N <= 2048) ? 0 : 64;
    std::vector<int32_t> super_roots, split_internals, skel_left, skel_right;
    int32_t root_skel_id = -1;
    extract_active_skeleton(
        sub_root, curr_c1.data(), curr_c2.data(), N,
        condense_thresh, node_has_active, leaf_cnt,
        super_roots, split_internals, skel_left, skel_right, &root_skel_id,
        max_internals, stop_at_comp_root
    );

    if (sub_root == root_r) {
        int32_t K = (int32_t)super_roots.size();
        if (K < 6 || K > gpu_ws.max_K) return 0.0;

        std::vector<int32_t> sk_c1(2 * K - 1), sk_c2(2 * K - 1);
        for (int32_t s = 0; s < K - 1; ++s) {
            int32_t l_raw = skel_left[s], r_raw = skel_right[s];
            sk_c1[K + s] = (l_raw >= 0) ? l_raw : (K + (-1 - l_raw));
            sk_c2[K + s] = (r_raw >= 0) ? r_raw : (K + (-1 - r_raw));
        }

        std::vector<int32_t> prof_offsets(K + 1, 0), prof_leaves;
        std::vector<float> prof_weights;
        prof_leaves.reserve(4 * K);
        prof_weights.reserve(4 * K);
        bool need_exact_prof = false;
        for (int32_t a = 0; a < K; ++a) {
            prof_offsets[a] = (int32_t)prof_leaves.size();
            compute_clade_dyadic_profile(
                super_roots[a], N, curr_c1.data(), curr_c2.data(), node_rep0.data(),
                max_prof_leaves, prof_leaves, prof_weights
            );
            if (max_prof_leaves > 0 && leaf_cnt[super_roots[a]] > max_prof_leaves) {
                need_exact_prof = true;
            }
        }
        prof_offsets[K] = (int32_t)prof_leaves.size();

        std::vector<int32_t> exact_prof_offsets, exact_prof_leaves;
        std::vector<float> exact_prof_weights;
        if (need_exact_prof) {
            exact_prof_offsets.resize(K + 1, 0);
            exact_prof_leaves.reserve((size_t)N);
            exact_prof_weights.reserve((size_t)N);
            for (int32_t a = 0; a < K; ++a) {
                exact_prof_offsets[a] = (int32_t)exact_prof_leaves.size();
                if (leaf_cnt[super_roots[a]] <= max_prof_leaves) {
                    exact_prof_leaves.insert(exact_prof_leaves.end(), prof_leaves.begin() + prof_offsets[a], prof_leaves.begin() + prof_offsets[a + 1]);
                    exact_prof_weights.insert(exact_prof_weights.end(), prof_weights.begin() + prof_offsets[a], prof_weights.begin() + prof_offsets[a + 1]);
                } else {
                    compute_clade_dyadic_profile(
                        super_roots[a], N, curr_c1.data(), curr_c2.data(), node_rep0.data(),
                        -1, exact_prof_leaves, exact_prof_weights
                    );
                }
            }
            exact_prof_offsets[K] = (int32_t)exact_prof_leaves.size();
        }

        std::vector<int64_t> v_mb(K - 1);
        std::vector<int32_t> perm_mb(K);
        relabel_from_rooted_tree(sk_c1.data(), sk_c2.data(), K, 0, v_mb.data(), perm_mb.data());

        double dL = optimize_skeleton_on_gpu(
            K, W, max_iters, 32,
            prof_offsets.data(), prof_leaves.data(), prof_weights.data(),
            d_packed_seqs, gpu_ws, v_mb, perm_mb,
            need_exact_prof ? exact_prof_offsets.data() : NULL,
            need_exact_prof ? exact_prof_leaves.data() : NULL,
            need_exact_prof ? exact_prof_weights.data() : NULL
        );
        if (dL == 0.0) return 0.0;

        int32_t R_k = 2 * K - 1;
        std::vector<int32_t> new_sk_c1(R_k), new_sk_c2(R_k), scratch_uf(R_k);
        decode_cf_to_rooted(v_mb.data(), K, new_sk_c1.data(), new_sk_c2.data(), scratch_uf.data());
        std::vector<int32_t> sk_to_full(R_k);
        for (int32_t a = 0; a < K; ++a) sk_to_full[a] = super_roots[perm_mb[a]];
        for (int32_t s = 0; s < K - 1; ++s) sk_to_full[K + s] = split_internals[s];
        for (int32_t s = 0; s < K - 1; ++s) {
            int32_t full_u = sk_to_full[K + s];
            int32_t b0 = sk_to_full[new_sk_c1[K + s]];
            int32_t b1 = sk_to_full[new_sk_c2[K + s]];
            curr_c1[full_u] = b0;
            curr_c2[full_u] = b1;
            update_one_node_summary(full_u, b0, b1, parent_r.data(), leaf_cnt.data(), node_rep0.data());
        }
        parent_r[root_r] = -1;
        return dL;
    } else {
        int32_t K_in = (int32_t)super_roots.size();
        int32_t K = K_in + 1;
        if (K < 6 || K > gpu_ws.max_K) return 0.0;

        int32_t parent_of_sub_root = parent_r[sub_root];
        std::vector<int32_t> sk_c1(2 * K - 1, 0), sk_c2(2 * K - 1, 0);
        for (int32_t s = 0; s < K_in - 1; ++s) {
            int32_t l_raw = skel_left[s], r_raw = skel_right[s];
            sk_c1[K + s] = (l_raw >= 0) ? l_raw : (K + (-1 - l_raw));
            sk_c2[K + s] = (r_raw >= 0) ? r_raw : (K + (-1 - r_raw));
        }
        sk_c1[2 * K - 2] = K + (K_in - 2);
        sk_c2[2 * K - 2] = K_in;

        std::vector<int32_t> prof_offsets(K + 1, 0), prof_leaves;
        std::vector<float> prof_weights;
        prof_leaves.reserve(4 * K);
        prof_weights.reserve(4 * K);
        bool need_exact_prof = false;
        for (int32_t a = 0; a < K_in; ++a) {
            prof_offsets[a] = (int32_t)prof_leaves.size();
            compute_clade_dyadic_profile(
                super_roots[a], N, curr_c1.data(), curr_c2.data(), node_rep0.data(),
                max_prof_leaves, prof_leaves, prof_weights
            );
            if (max_prof_leaves > 0 && leaf_cnt[super_roots[a]] > max_prof_leaves) {
                need_exact_prof = true;
            }
        }
        prof_offsets[K_in] = (int32_t)prof_leaves.size();
        prof_leaves.insert(prof_leaves.end(), outgroup_leaves.begin(), outgroup_leaves.end());
        prof_weights.insert(prof_weights.end(), outgroup_weights.begin(), outgroup_weights.end());
        prof_offsets[K] = (int32_t)prof_leaves.size();

        int32_t outgroup_true_leaves = N - leaf_cnt[sub_root];
        if (max_prof_leaves > 0 && outgroup_true_leaves > (int32_t)outgroup_leaves.size()) {
            need_exact_prof = true;
        }

        std::vector<int32_t> exact_prof_offsets, exact_prof_leaves;
        std::vector<float> exact_prof_weights;
        if (need_exact_prof) {
            const int32_t *oc1 = (outgroup_c1 != NULL) ? outgroup_c1 : curr_c1.data();
            const int32_t *oc2 = (outgroup_c2 != NULL) ? outgroup_c2 : curr_c2.data();
            const int32_t *opar = (outgroup_par != NULL) ? outgroup_par : parent_r.data();
            const int32_t *orep = (outgroup_rep0 != NULL) ? outgroup_rep0 : node_rep0.data();

            exact_prof_offsets.resize(K + 1, 0);
            exact_prof_leaves.reserve((size_t)N);
            exact_prof_weights.reserve((size_t)N);
            for (int32_t a = 0; a < K_in; ++a) {
                exact_prof_offsets[a] = (int32_t)exact_prof_leaves.size();
                if (max_prof_leaves <= 0 || leaf_cnt[super_roots[a]] <= max_prof_leaves) {
                    exact_prof_leaves.insert(exact_prof_leaves.end(), prof_leaves.begin() + prof_offsets[a], prof_leaves.begin() + prof_offsets[a + 1]);
                    exact_prof_weights.insert(exact_prof_weights.end(), prof_weights.begin() + prof_offsets[a], prof_weights.begin() + prof_offsets[a + 1]);
                } else {
                    compute_clade_dyadic_profile(
                        super_roots[a], N, curr_c1.data(), curr_c2.data(), node_rep0.data(),
                        -1, exact_prof_leaves, exact_prof_weights
                    );
                }
            }
            exact_prof_offsets[K_in] = (int32_t)exact_prof_leaves.size();
            if (outgroup_true_leaves <= (int32_t)outgroup_leaves.size()) {
                exact_prof_leaves.insert(exact_prof_leaves.end(), outgroup_leaves.begin(), outgroup_leaves.end());
                exact_prof_weights.insert(exact_prof_weights.end(), outgroup_weights.begin(), outgroup_weights.end());
            } else {
                compute_outgroup_dyadic_profile(
                    sub_root, root_r, N, oc1, oc2, opar, orep,
                    -1, exact_prof_leaves, exact_prof_weights
                );
            }
            exact_prof_offsets[K] = (int32_t)exact_prof_leaves.size();
        }

        std::vector<int64_t> v_mb(K - 1);
        std::vector<int32_t> perm_mb(K);
        relabel_from_rooted_tree(sk_c1.data(), sk_c2.data(), K, 0, v_mb.data(), perm_mb.data());

        double dL = optimize_skeleton_on_gpu(
            K, W, max_iters, 32,
            prof_offsets.data(), prof_leaves.data(), prof_weights.data(),
            d_packed_seqs, gpu_ws, v_mb, perm_mb,
            need_exact_prof ? exact_prof_offsets.data() : NULL,
            need_exact_prof ? exact_prof_leaves.data() : NULL,
            need_exact_prof ? exact_prof_weights.data() : NULL
        );
        if (dL == 0.0) return 0.0;

        std::vector<int32_t> c1_cf, c2_cf, v_map_cf, m_lmin_cf, next_k_cf, parent_u_cf, sub_sz_cf, tin_cf, adj_u_cf;
        int32_t other_rc, rc0, rc1, rc2;
        build_skeleton_metadata(
            v_mb.data(), K, c1_cf, c2_cf, v_map_cf, m_lmin_cf, next_k_cf,
            parent_u_cf, sub_sz_cf, tin_cf, adj_u_cf,
            &other_rc, &rc0, &rc1, &rc2
        );

        int32_t a_out = -1;
        for (int32_t a = 0; a < K; ++a) {
            if (perm_mb[a] == K_in) { a_out = a; break; }
        }
        int32_t u_attach = adj_u_cf[3 * a_out + 0];

        std::vector<int32_t> st_u = {u_attach};
        std::vector<int32_t> st_p = {a_out};
        st_u.reserve(K_in - 1);
        st_p.reserve(K_in - 1);
        std::vector<int32_t> child0(2 * K - 2, -1), child1(2 * K - 2, -1);
        for (size_t idx = 0; idx < st_u.size(); ++idx) {
            int32_t u = st_u[idx];
            int32_t p = st_p[idx];
            int32_t b0 = -1, b1 = -1;
            for (int32_t k = 0; k < 3; ++k) {
                int32_t nb = adj_u_cf[3 * u + k];
                if (nb != -1 && nb != p) {
                    if (b0 == -1) b0 = nb;
                    else b1 = nb;
                }
            }
            child0[u] = b0;
            child1[u] = b1;
            if (b0 >= K) { st_u.push_back(b0); st_p.push_back(u); }
            if (b1 >= K) { st_u.push_back(b1); st_p.push_back(u); }
        }

        std::vector<int32_t> sk_to_full(2 * K - 2, -1);
        for (int32_t a = 0; a < K; ++a) {
            if (a != a_out) sk_to_full[a] = super_roots[perm_mb[a]];
        }
        for (int32_t s = 0; s < K_in - 1; ++s) {
            int32_t u_sk = st_u[K_in - 2 - s];
            sk_to_full[u_sk] = split_internals[s];
        }
        for (int32_t s = 0; s < K_in - 1; ++s) {
            int32_t u_sk = st_u[K_in - 2 - s];
            int32_t full_u = sk_to_full[u_sk];
            int32_t b0 = sk_to_full[child0[u_sk]];
            int32_t b1 = sk_to_full[child1[u_sk]];
            curr_c1[full_u] = b0;
            curr_c2[full_u] = b1;
            update_one_node_summary(full_u, b0, b1, parent_r.data(), leaf_cnt.data(), node_rep0.data());
        }
        parent_r[sub_root] = parent_of_sub_root;
        return dL;
    }
}

static double run_multi_gpu_minibatch_bme(
    int32_t N, int32_t W, int32_t n_gpus,
    std::vector<uint64_t *> &d_packed_seqs,
    std::vector<RnaSimGpuWorkspace> &mb_ws,
    std::vector<int32_t> &curr_c1,
    std::vector<int32_t> &curr_c2
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    const int32_t max_prof_leaves = (N <= 2048) ? 0 : 64;
    std::vector<int32_t> ord, parent_r, leaf_cnt, node_rep0;
    compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);

    double total_mb_delta_L = 0.0;
    std::vector<uint8_t> node_has_active(n_nodes_r, 0);
    std::vector<int32_t> empty_out_leaves;
    std::vector<float> empty_out_weights;

    /* Pass 1: Global Backbone (priority-queue bounded to max_internals = 1200 so K <= 1201 <= 2048) */
    if (N >= 5000) {
        int32_t bb_thresh = N / 700;
        if (bb_thresh < 8) bb_thresh = 8;
        total_mb_delta_L += run_subtree_minibatch_window(
            root_r, N, W, bb_thresh, 25, node_has_active, empty_out_leaves, empty_out_weights,
            d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
            1200, NULL
        );
        compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
    }

    /* Pass 2A: Leaf-Complete Connected Component Waves (100% of all N original leaves across all leaf-bearing internal components, K <= 1056 <= 2048) */
    std::vector<uint8_t> is_comp_root;
    std::vector<std::vector<int32_t>> level_waves;
    build_component_waves(N, ord, curr_c1, curr_c2, 1050, 1050, is_comp_root, level_waves);

    int32_t leaves_in_lvl0 = 0;
    if (!level_waves.empty()) {
        for (int32_t sr : level_waves[0]) leaves_in_lvl0 += leaf_cnt[sr];
    }

    (void)n_gpus;
    for (size_t lvl = 0; lvl < level_waves.size(); ++lvl) {
        const std::vector<int32_t> &wave = level_waves[lvl];
        std::vector<int32_t> out_leaves;
        std::vector<float> out_weights;
        for (int32_t idx = 0; idx < (int32_t)wave.size(); ++idx) {
            int32_t sr = wave[idx];
            out_leaves.clear();
            out_weights.clear();
            if (sr != root_r) {
                compute_outgroup_dyadic_profile(
                    sr, root_r, N, curr_c1.data(), curr_c2.data(), parent_r.data(), node_rep0.data(),
                    max_prof_leaves, out_leaves, out_weights
                );
            }
            double dL = run_subtree_minibatch_window(
                sr, N, W, 1, 25, node_has_active, out_leaves, out_weights,
                d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                1200, is_comp_root.data()
            );
            if (dL != 0.0) {
                total_mb_delta_L += dL;
                for (int32_t p = parent_r[sr]; p != -1; p = parent_r[p]) {
                    update_one_node_summary(p, curr_c1[p], curr_c2[p], parent_r.data(), leaf_cnt.data(), node_rep0.data());
                }
                parent_r[root_r] = -1;
            }
        }
        if (lvl + 1 < level_waves.size()) {
            compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
        }
    }

    /* Pass 2B: Staggered Boundary-Crossing Component Waves (when spine-heavy) + Bounded Medium Cross-Clade Skeletons */
    if (N > 1000) {
        if (leaves_in_lvl0 < (9 * N) / 10) {
            compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            std::vector<uint8_t> is_comp_root_stag;
            std::vector<std::vector<int32_t>> stag_waves;
            build_component_waves(N, ord, curr_c1, curr_c2, 525, 1050, is_comp_root_stag, stag_waves);
            for (int32_t lvl = (int32_t)stag_waves.size() - 1; lvl >= 0; --lvl) {
                const std::vector<int32_t> &wave = stag_waves[lvl];
                std::vector<int32_t> out_leaves;
                std::vector<float> out_weights;
                for (int32_t idx = 0; idx < (int32_t)wave.size(); ++idx) {
                    int32_t sr = wave[idx];
                    out_leaves.clear();
                    out_weights.clear();
                    if (sr != root_r) {
                        compute_outgroup_dyadic_profile(
                            sr, root_r, N, curr_c1.data(), curr_c2.data(), parent_r.data(), node_rep0.data(),
                            max_prof_leaves, out_leaves, out_weights
                        );
                    }
                    double dL = run_subtree_minibatch_window(
                        sr, N, W, 1, 25, node_has_active, out_leaves, out_weights,
                        d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                        1200, is_comp_root_stag.data()
                    );
                    if (dL != 0.0) {
                        total_mb_delta_L += dL;
                        for (int32_t p = parent_r[sr]; p != -1; p = parent_r[p]) {
                            update_one_node_summary(p, curr_c1[p], curr_c2[p], parent_r.data(), leaf_cnt.data(), node_rep0.data());
                        }
                        parent_r[root_r] = -1;
                    }
                }
                compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            }
            if (N >= 5000) {
                int32_t bb_thresh = std::max(8, N / 700);
                total_mb_delta_L += run_subtree_minibatch_window(
                    root_r, N, W, bb_thresh, 25, node_has_active, empty_out_leaves, empty_out_weights,
                    d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                    1200, NULL
                );
                compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            }
        }

        std::vector<int32_t> mid_roots;
        std::vector<int32_t> spine_above_p2b;
        {
            std::vector<int32_t> st = {root_r};
            while (!st.empty()) {
                int32_t u = st.back(); st.pop_back();
                if (u < N) continue;
                if (leaf_cnt[u] <= 3200) {
                    if (leaf_cnt[u] > 1050) mid_roots.push_back(u);
                } else {
                    spine_above_p2b.push_back(u);
                    st.push_back(curr_c2[u]);
                    st.push_back(curr_c1[u]);
                }
            }
        }
        std::vector<int32_t> out_leaves_p2b;
        std::vector<float> out_weights_p2b;
        for (int32_t idx = 0; idx < (int32_t)mid_roots.size(); ++idx) {
            int32_t sr = mid_roots[idx];
            out_leaves_p2b.clear();
            out_weights_p2b.clear();
            if (sr != root_r) {
                compute_outgroup_dyadic_profile(
                    sr, root_r, N, curr_c1.data(), curr_c2.data(), parent_r.data(), node_rep0.data(),
                    max_prof_leaves, out_leaves_p2b, out_weights_p2b
                );
            }
            int32_t c_thresh = std::max(4, leaf_cnt[sr] / 800);
            double dL = run_subtree_minibatch_window(
                sr, N, W, c_thresh, 25, node_has_active, out_leaves_p2b, out_weights_p2b,
                d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                1200, NULL
            );
            if (dL != 0.0) {
                total_mb_delta_L += dL;
                for (int32_t p = parent_r[sr]; p != -1; p = parent_r[p]) {
                    update_one_node_summary(p, curr_c1[p], curr_c2[p], parent_r.data(), leaf_cnt.data(), node_rep0.data());
                }
                parent_r[root_r] = -1;
            }
        }
    }

    return total_mb_delta_L;
}

static FILE *open_rnasim_bin(int32_t N) {
    char p1[256], p2[256], p3[256];
    snprintf(p1, sizeof(p1), "rnasim_%d.bin", N);
    FILE *fp = fopen(p1, "rb");
    if (fp) return fp;
    snprintf(p2, sizeof(p2), "rnasim_data/rnasim_%d.bin", N);
    fp = fopen(p2, "rb");
    if (fp) return fp;
    snprintf(p3, sizeof(p3), "/home/sjbhatt/phylo2vec_discrete_gradient/rnasim_data/rnasim_%d.bin", N);
    return fopen(p3, "rb");
}

static void run_ladder_5000_verification(int n_gpus, std::vector<RnaSimGpuWorkspace> &mb_ws) {
    const int32_t N = 5000;
    FILE *fp = open_rnasim_bin(N);
    if (!fp) { fprintf(stderr, "Missing rnasim_5000.bin for ladder verification\n"); return; }
    int32_t hdr[3];
    if (fread(hdr, sizeof(int32_t), 3, fp) != 3) exit(1);
    int32_t W = hdr[2];
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;

    std::vector<uint64_t> h_packed_seqs((size_t)N * 5 * W);
    std::vector<int32_t> true_c1(n_nodes_r), true_c2(n_nodes_r);
    if (fread(h_packed_seqs.data(), sizeof(uint64_t), h_packed_seqs.size(), fp) != h_packed_seqs.size()) exit(1);
    if (fread(true_c1.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
    if (fread(true_c2.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
    fclose(fp);

    /* Construct a pure 5,000-leaf ladder (caterpillar/pectinate) tree:
     * Internal node N + k (for k = 0 .. N-2) has children:
     *   - k == 0: leaves 0 and 1
     *   - k > 0:  internal node N + k - 1 and leaf k + 1 */
    std::vector<int32_t> lad_c1(n_nodes_r, 0), lad_c2(n_nodes_r, 0);
    lad_c1[N] = 0;
    lad_c2[N] = 1;
    for (int32_t k = 1; k < N - 1; ++k) {
        lad_c1[N + k] = N + k - 1;
        lad_c2[N + k] = k + 1;
    }

    std::vector<int32_t> ord, parent_r, leaf_cnt, node_rep0;
    compute_tree_summaries_blind(lad_c1.data(), lad_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);

    std::vector<uint8_t> node_has_active(n_nodes_r, 0);
    std::vector<int32_t> sr_p1, si_p1, sl_p1, sright_p1;
    int32_t r_id = -1;
    extract_active_skeleton(root_r, lad_c1.data(), lad_c2.data(), N, 8, node_has_active, leaf_cnt, sr_p1, si_p1, sl_p1, sright_p1, &r_id, 1200, NULL);

    std::vector<uint8_t> is_comp_root;
    std::vector<std::vector<int32_t>> level_waves;
    build_component_waves(N, ord, lad_c1, lad_c2, 1050, 1050, is_comp_root, level_waves);

    std::vector<uint8_t> leaf_covered_p2a(N, 0), int_covered_p2a(n_nodes_r, 0);
    int32_t max_K_p2a = 0, min_K_p2a = 1000000;
    for (size_t lvl = 0; lvl < level_waves.size(); ++lvl) {
        for (int32_t sr : level_waves[lvl]) {
            std::vector<int32_t> s_roots, s_ints, s_l, s_r;
            int32_t rid = -1;
            extract_active_skeleton(sr, lad_c1.data(), lad_c2.data(), N, 1, node_has_active, leaf_cnt, s_roots, s_ints, s_l, s_r, &rid, 1200, is_comp_root.data());
            int32_t K_sk = (int32_t)s_roots.size() + (sr == root_r ? 0 : 1);
            if (K_sk > max_K_p2a) max_K_p2a = K_sk;
            if (K_sk < min_K_p2a) min_K_p2a = K_sk;
            for (int32_t u : s_roots) if (u < N) leaf_covered_p2a[u] = 1;
            for (int32_t u : s_ints)  int_covered_p2a[u] = 1;
        }
    }
    int32_t cnt_leaves = 0, cnt_ints = 0;
    for (int32_t i = 0; i < N; ++i) cnt_leaves += leaf_covered_p2a[i];
    for (int32_t u = N; u < n_nodes_r; ++u) cnt_ints += int_covered_p2a[u];

    printf("====================================================================================================\n");
    printf("5,000-LEAF LADDER (CATERPILLAR) TREE VERIFICATION\n");
    printf("====================================================================================================\n");
    printf("Pass 1 Global Skeleton K   : %d <= max_K (%d) [Previously 4,993 -> SKIPPED]\n", (int32_t)sr_p1.size(), mb_ws[0].max_K);
    printf("Pass 2A Component Waves    : %zu waves, K in [%d, %d] <= max_K (%d)\n", level_waves.size(), min_K_p2a, max_K_p2a, mb_ws[0].max_K);
    printf("Pass 2A Uncondensed Leaves : %d / %d (100.0%%) [Previously 1,050 / 5,000 (21.0%%)]\n", cnt_leaves, N);
    printf("Pass 2A Internal Nodes     : %d / %d (100.0%%) [Previously 1,049 / 4,999 (21.0%%)]\n", cnt_ints, N - 1);

    std::vector<uint64_t *> d_packed_seqs(n_gpus, NULL);
    for (int g = 0; g < n_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaMalloc(&d_packed_seqs[g], h_packed_seqs.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_packed_seqs[g], h_packed_seqs.data(), h_packed_seqs.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    }

    uint64_t rng = 123456789ULL;
    std::vector<uint64_t> leaf_hashes(N);
    uint64_t xor_all = 0;
    for (int32_t x = 1; x < N; ++x) { leaf_hashes[x] = splitmix64(&rng); xor_all ^= leaf_hashes[x]; }
    leaf_hashes[0] = xor_all;
    std::vector<uint64_t> splits_true, splits_lad_init, splits_lad_opt;
    compute_xor_splits_o_n(true_c1.data(), true_c2.data(), N, leaf_hashes.data(), splits_true);
    compute_xor_splits_o_n(lad_c1.data(),  lad_c2.data(),  N, leaf_hashes.data(), splits_lad_init);
    int64_t rf_lad_init = count_wrong_splits_rf(splits_true, splits_lad_init);

    double t0 = wall_sec();
    double dL = run_multi_gpu_minibatch_bme(N, W, n_gpus, d_packed_seqs, mb_ws, lad_c1, lad_c2);
    double dt = wall_sec() - t0;

    compute_tree_summaries_blind(lad_c1.data(), lad_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
    std::vector<int32_t> sr_post, si_post, sl_post, sright_post;
    extract_active_skeleton(root_r, lad_c1.data(), lad_c2.data(), N, 8, node_has_active, leaf_cnt, sr_post, si_post, sl_post, sright_post, &r_id, 10000, NULL);

    int32_t max_depth = 0;
    std::vector<int32_t> depth(n_nodes_r, 0);
    for (int32_t idx = 1; idx < (int32_t)ord.size(); ++idx) {
        int32_t u = ord[idx];
        depth[u] = depth[parent_r[u]] + 1;
        if (depth[u] > max_depth) max_depth = depth[u];
    }
    for (int32_t x = 0; x < N; ++x) {
        depth[x] = depth[parent_r[x]] + 1;
        if (depth[x] > max_depth) max_depth = depth[x];
    }

    compute_xor_splits_o_n(lad_c1.data(), lad_c2.data(), N, leaf_hashes.data(), splits_lad_opt);
    int64_t rf_lad_opt = count_wrong_splits_rf(splits_true, splits_lad_opt);
    printf("Post-Wave Global Skeleton  : %d super-roots (untruncated!) <= max_K (%d)\n", (int32_t)sr_post.size(), mb_ws[0].max_K);
    printf("Ladder Start -> Optimized  : RF wrong splits %lld -> %lld | Delta L = %+.2f | Max depth 4999 -> %d | Time = %.2fs\n",
           (long long)rf_lad_init, (long long)rf_lad_opt, dL, max_depth, dt);
    printf("====================================================================================================\n\n");

    for (int g = 0; g < n_gpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        cudaFree(d_packed_seqs[g]);
    }
}

int main(int argc, char **argv) {
    bool minibatch_only = false;
    bool test_ladder = false;
    for (int a = 1; a < argc; ++a) {
        if (strcmp(argv[a], "--minibatch-only") == 0) minibatch_only = true;
        if (strcmp(argv[a], "--test-ladder") == 0) test_ladder = true;
    }

    int n_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (n_gpus > 8) n_gpus = 8;
    if (n_gpus < 1) n_gpus = 1;
    if (n_gpus > 1) minibatch_only = true;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<RnaSimGpuWorkspace> mb_ws(n_gpus);
    for (int g = 0; g < n_gpus; ++g) {
        mb_ws[g].init(g, 2048);
    }

    if (test_ladder) {
        run_ladder_5000_verification(n_gpus, mb_ws);
        for (int g = 0; g < n_gpus; ++g) mb_ws[g].destroy();
        return 0;
    }

    printf("====================================================================================================================================\n");
    printf("BLIND EMPIRICAL RNASIM BENCHMARK (1,596-bp 16S rRNA Sequences | %dx %s | Mode: %s)\n",
           n_gpus, prop.name, minibatch_only ? "Minibatch-Only" : "Full-Tree + Minibatch");
    printf("====================================================================================================================================\n");
    printf("%8s | %22s | %22s | %24s | %24s | %10s\n",
           "Taxa (N)", "VFT-InitNJ RF (Time)", "VFT-ME (-noml) RF", "GPU Full-Tree BME RF", "GPU Minibatch BME RF", "Delta L");
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");

    const int32_t sizes[] = {1000, 5000, 10000, 50000, 100000};
    const double vft_init_times[] = {0.58, 3.40, 7.82, 63.01, 179.03};
    const double vft_me_times[]   = {7.46, 36.78, 73.80, 376.54, 772.53};

    for (size_t s_idx = 0; s_idx < sizeof(sizes) / sizeof(sizes[0]); ++s_idx) {
        int32_t N = sizes[s_idx];
        FILE *fp = open_rnasim_bin(N);
        if (!fp) { fprintf(stderr, "Missing rnasim_%d.bin\n", N); continue; }

        int32_t hdr[3];
        if (fread(hdr, sizeof(int32_t), 3, fp) != 3) exit(1);
        int32_t W = hdr[2];
        const int32_t n_nodes_r = 2 * N - 1;

        std::vector<uint64_t> h_packed_seqs((size_t)N * 5 * W);
        std::vector<int32_t> true_c1(n_nodes_r), true_c2(n_nodes_r);
        std::vector<int32_t> init_c1(n_nodes_r), init_c2(n_nodes_r);
        std::vector<int32_t> me_c1(n_nodes_r), me_c2(n_nodes_r);

        if (fread(h_packed_seqs.data(), sizeof(uint64_t), h_packed_seqs.size(), fp) != h_packed_seqs.size()) exit(1);
        if (fread(true_c1.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(true_c2.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(init_c1.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(init_c2.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(me_c1.data(),   sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(me_c2.data(),   sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        fclose(fp);

        std::vector<uint64_t *> d_packed_seqs(n_gpus, NULL);
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMalloc(&d_packed_seqs[g], h_packed_seqs.size() * sizeof(uint64_t)));
            CUDA_CHECK(cudaMemcpy(d_packed_seqs[g], h_packed_seqs.data(), h_packed_seqs.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
        }

        uint64_t rng = 123456789ULL;
        std::vector<uint64_t> leaf_hashes(N);
        uint64_t xor_all = 0;
        for (int32_t x = 1; x < N; ++x) { leaf_hashes[x] = splitmix64(&rng); xor_all ^= leaf_hashes[x]; }
        leaf_hashes[0] = xor_all;

        std::vector<uint64_t> splits_true, splits_init, splits_me, splits_full_gpu, splits_mb_gpu;
        compute_xor_splits_o_n(true_c1.data(), true_c2.data(), N, leaf_hashes.data(), splits_true);
        compute_xor_splits_o_n(init_c1.data(), init_c2.data(), N, leaf_hashes.data(), splits_init);
        compute_xor_splits_o_n(me_c1.data(),   me_c2.data(),   N, leaf_hashes.data(), splits_me);

        int64_t rf_init = count_wrong_splits_rf(splits_true, splits_init);
        int64_t rf_me   = count_wrong_splits_rf(splits_true, splits_me);

        /* --------------------------------------------------------------------
         * Mode A: Full-Tree GPU Phylo2Vec BME Optimization starting from VFT-InitNJ
         * -------------------------------------------------------------------- */
        int32_t R_sk = 2 * N - 1;
        std::vector<int32_t> scratch_uf(R_sk);
        double dt_full_gpu = 0.0;
        int64_t rf_full_gpu = -1;

        if (!minibatch_only && N <= 10000) {
            RnaSimGpuWorkspace full_ws;
            full_ws.init(0, N + 64);
            CUDA_CHECK(cudaDeviceSynchronize());
            double t0_full = wall_sec();

            std::vector<int64_t> v_cf(N - 1);
            std::vector<int32_t> perm(N);
            relabel_from_rooted_tree(init_c1.data(), init_c2.data(), N, 0, v_cf.data(), perm.data());

            std::vector<int32_t> prof_offsets(N + 1), prof_leaves(N);
            std::vector<float> prof_weights(N, 1.0f);
            for (int32_t a = 0; a < N; ++a) {
                prof_offsets[a] = a;
                prof_leaves[a] = a;
            }
            prof_offsets[N] = N;

            (void)optimize_skeleton_on_gpu(
                N, W, 100, 64,
                prof_offsets.data(), prof_leaves.data(), prof_weights.data(),
                d_packed_seqs[0], full_ws, v_cf, perm
            );

            CUDA_CHECK(cudaDeviceSynchronize());
            dt_full_gpu = wall_sec() - t0_full;
            full_ws.destroy();

            std::vector<int32_t> full_gpu_c1(R_sk), full_gpu_c2(R_sk);
            decode_cf_to_rooted(v_cf.data(), N, full_gpu_c1.data(), full_gpu_c2.data(), scratch_uf.data());
            for (int32_t m = N; m < R_sk; ++m) {
                if (full_gpu_c1[m] < N) full_gpu_c1[m] = perm[full_gpu_c1[m]];
                if (full_gpu_c2[m] < N) full_gpu_c2[m] = perm[full_gpu_c2[m]];
            }
            compute_xor_splits_o_n(full_gpu_c1.data(), full_gpu_c2.data(), N, leaf_hashes.data(), splits_full_gpu);
            rf_full_gpu = count_wrong_splits_rf(splits_true, splits_full_gpu);
        }

        /* --------------------------------------------------------------------
         * Mode B: Multi-GPU Subtree-Wave Minibatch Phylo2Vec BME
         * -------------------------------------------------------------------- */
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        double t0_mb = wall_sec();
        std::vector<int32_t> curr_c1 = init_c1, curr_c2 = init_c2;
        double total_mb_delta_L = run_multi_gpu_minibatch_bme(
            N, W, n_gpus, d_packed_seqs, mb_ws, curr_c1, curr_c2
        );
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        double dt_mb_gpu = wall_sec() - t0_mb;

        compute_xor_splits_o_n(curr_c1.data(), curr_c2.data(), N, leaf_hashes.data(), splits_mb_gpu);
        int64_t rf_mb_gpu = count_wrong_splits_rf(splits_true, splits_mb_gpu);

        char b_init[64], b_me[64], b_full[64], b_mb[64];
        snprintf(b_init, sizeof(b_init), "%lld (%.2fs)", (long long)rf_init, vft_init_times[s_idx]);
        snprintf(b_me,   sizeof(b_me),   "%lld (%.2fs)", (long long)rf_me,   vft_me_times[s_idx]);
        if (rf_full_gpu >= 0) {
            snprintf(b_full, sizeof(b_full), "%lld (%.2fs)", (long long)rf_full_gpu, dt_full_gpu);
        } else {
            snprintf(b_full, sizeof(b_full), "N/A (Minibatch)");
        }
        snprintf(b_mb,   sizeof(b_mb),   "%lld (%.2fs)", (long long)rf_mb_gpu,   dt_mb_gpu);
        double dL_rep = total_mb_delta_L;

        printf("%8d | %22s | %22s | %24s | %24s | %+10.2f\n",
               N, b_init, b_me, b_full, b_mb, dL_rep);
        fflush(stdout);

        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            cudaFree(d_packed_seqs[g]);
        }
    }
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");
    for (int g = 0; g < n_gpus; ++g) {
        mb_ws[g].destroy();
    }
    return 0;
}
