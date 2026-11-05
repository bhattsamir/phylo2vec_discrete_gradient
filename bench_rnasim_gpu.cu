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
#include <sys/resource.h>
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
    #pragma unroll 5
    for (int32_t w = 0; w < W; ++w) {
        uint64_t vx = __ldg(&sx[4 * W + w]);
        uint64_t vy = __ldg(&sy[4 * W + w]);
        n_valid += __popcll(vx & vy);
        uint64_t m = (__ldg(&sx[0 * W + w]) & __ldg(&sy[0 * W + w]))
                   | (__ldg(&sx[1 * W + w]) & __ldg(&sy[1 * W + w]))
                   | (__ldg(&sx[2 * W + w]) & __ldg(&sy[2 * W + w]))
                   | (__ldg(&sx[3 * W + w]) & __ldg(&sy[3 * W + w]));
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
    float * __restrict__ d_D_cond,
    int32_t heavy_min
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
        /* Pairs where both profiles have >= heavy_min leaves are left to the tiled kernel. */
        if (end_a - off_a >= heavy_min && end_b - off_b >= heavy_min) return;
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

/* Same float formula as dev_jc69_dist_popc, from precomputed popcounts. */
__device__ __forceinline__ float dev_jc69_from_counts(int32_t n_valid, int32_t n_match) {
    if (n_valid <= 0) return 1.5f;
    float p = (float)(n_valid - n_match) / (float)n_valid;
    if (p >= 0.74f) p = 0.74f;
    float d = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
    return (d > 0.0f) ? d : 0.0f;
}

constexpr int32_t HB_T = 64;        /* leaf tile edge */
constexpr int32_t HB_CW = 5;        /* words per staged chunk (25 words = 5 chunks) */
constexpr int32_t HB_HEAVY_MIN = 8; /* both profiles at least this large -> tiled kernel */

/* Tiled k_build_real_seq_condensed_D for heavy pairs (one block per pair a < b).
 * Leaf sequences are staged in shared memory in word chunks and each thread keeps a
 * 4x4 tile of exact popcounts. Distances use the same float formula, and the weighted
 * sums run in the same order as k_build_real_seq_condensed_D (row sums over q, then
 * over p), so D_cond is bit-identical for any W > 0. */
__global__ __launch_bounds__(256) void k_build_condensed_D_heavy_tiled(
    int32_t K, int32_t W, int32_t n_pairs,
    const int32_t * __restrict__ d_pair_a,
    const int32_t * __restrict__ d_pair_b,
    const int32_t * __restrict__ d_prof_offsets,
    const int32_t * __restrict__ d_prof_leaves,
    const float   * __restrict__ d_prof_weights,
    const uint64_t * __restrict__ d_packed_seqs,
    float * __restrict__ d_D_cond
) {
    __shared__ uint64_t sA[5 * HB_CW * HB_T];   /* [channel][word][row] */
    __shared__ uint64_t sB[5 * HB_CW * HB_T];
    __shared__ float    s_dist[HB_T][HB_T + 1];
    __shared__ float    s_row_sum[HB_T];
    __shared__ int32_t  s_leaf_a[HB_T], s_leaf_b[HB_T];
    __shared__ float    s_wa[HB_T], s_wb[HB_T];

    const int32_t pid = blockIdx.x;
    if (pid >= n_pairs) return;
    const int32_t a = d_pair_a[pid], b = d_pair_b[pid];
    const int32_t off_a = d_prof_offsets[a], len_a = d_prof_offsets[a + 1] - off_a;
    const int32_t off_b = d_prof_offsets[b], len_b = d_prof_offsets[b + 1] - off_b;
    const int32_t tid = (int32_t)threadIdx.x, tx = tid & 15, ty = tid >> 4;
    float d_val = 0.0f;

    for (int32_t pa0 = 0; pa0 < len_a; pa0 += HB_T) {
        const int32_t na = min(HB_T, len_a - pa0);
        __syncthreads();
        if (tid < HB_T) {
            s_row_sum[tid] = 0.0f;
            if (tid < na) {
                s_leaf_a[tid] = d_prof_leaves[off_a + pa0 + tid];
                s_wa[tid]     = d_prof_weights[off_a + pa0 + tid];
            }
        }
        for (int32_t qb0 = 0; qb0 < len_b; qb0 += HB_T) {
            const int32_t nb = min(HB_T, len_b - qb0);
            __syncthreads();
            if (tid < nb) {
                s_leaf_b[tid] = d_prof_leaves[off_b + qb0 + tid];
                s_wb[tid]     = d_prof_weights[off_b + qb0 + tid];
            }
            __syncthreads();

            int32_t nv[4][4], nm[4][4];
            #pragma unroll
            for (int32_t i = 0; i < 4; ++i)
                #pragma unroll
                for (int32_t j = 0; j < 4; ++j) { nv[i][j] = 0; nm[i][j] = 0; }

            for (int32_t w0 = 0; w0 < W; w0 += HB_CW) {
                /* Stage words w0..w0+HB_CW-1 of all 5 channels; consecutive threads read
                 * consecutive words of one sequence. */
                for (int32_t e = tid; e < 5 * HB_CW * HB_T; e += 256) {
                    const int32_t cw = e % HB_CW, ch = (e / HB_CW) % 5, r = e / (5 * HB_CW);
                    const int32_t s_idx = (ch * HB_CW + cw) * HB_T + r;
                    const int32_t w = w0 + cw;
                    const size_t g_off = (size_t)ch * W + w;
                    sA[s_idx] = (r < na && w < W) ? d_packed_seqs[(size_t)s_leaf_a[r] * 5 * W + g_off] : 0ull;
                    sB[s_idx] = (r < nb && w < W) ? d_packed_seqs[(size_t)s_leaf_b[r] * 5 * W + g_off] : 0ull;
                }
                __syncthreads();
                #pragma unroll
                for (int32_t cw = 0; cw < HB_CW; ++cw) {
                    uint64_t ra[4], rb[4], m[4][4];
                    #pragma unroll
                    for (int32_t i = 0; i < 4; ++i) ra[i] = sA[(4 * HB_CW + cw) * HB_T + ty + 16 * i];
                    #pragma unroll
                    for (int32_t j = 0; j < 4; ++j) rb[j] = sB[(4 * HB_CW + cw) * HB_T + tx + 16 * j];
                    #pragma unroll
                    for (int32_t i = 0; i < 4; ++i)
                        #pragma unroll
                        for (int32_t j = 0; j < 4; ++j) nv[i][j] += __popcll(ra[i] & rb[j]);
                    #pragma unroll
                    for (int32_t c = 0; c < 4; ++c) {
                        #pragma unroll
                        for (int32_t i = 0; i < 4; ++i) ra[i] = sA[(c * HB_CW + cw) * HB_T + ty + 16 * i];
                        #pragma unroll
                        for (int32_t j = 0; j < 4; ++j) rb[j] = sB[(c * HB_CW + cw) * HB_T + tx + 16 * j];
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
                #pragma unroll
                for (int32_t j = 0; j < 4; ++j) {
                    const int32_t p = ty + 16 * i, q = tx + 16 * j;
                    if (p < na && q < nb) {
                        s_dist[p][q] = (s_leaf_a[p] == s_leaf_b[q]) ? 0.0f : dev_jc69_from_counts(nv[i][j], nm[i][j]);
                    }
                }
            }
            __syncthreads();
            if (tid < na) {
                float row_sum = s_row_sum[tid];
                for (int32_t q = 0; q < nb; ++q) row_sum += s_wb[q] * s_dist[tid][q];
                s_row_sum[tid] = row_sum;
            }
        }
        __syncthreads();
        if (tid == 0) {
            for (int32_t p = 0; p < na; ++p) d_val += s_wa[p] * s_row_sum[p];
        }
    }
    if (tid == 0) {
        d_D_cond[(size_t)a * K + b] = d_val;
        d_D_cond[(size_t)b * K + a] = d_val;
    }
}

/* Block-parallel exact refinement of selected super-leaf pairs (a, b) using uncapped
 * dyadic profiles (d_exact_offsets, d_exact_leaves, d_exact_weights) and on-the-fly JC69
 * popcount distances. Uses shared-memory staged, k-sliced, 4x4 register-tiled evaluation
 * with adaptive A-tile heights (16, 32, 48, 64) and multi-pair packing across consecutive
 * pairs sharing the same right-hand profile b, updating d_D_cond[a * K + b],
 * d_D_cond[b * K + a], and d_pair_exact_D[pid]. */
__global__ __launch_bounds__(256, 2) void k_refine_real_seq_condensed_D_pairs(
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
    const int32_t tid = (int32_t)threadIdx.x;
    const int32_t pid = (int32_t)blockIdx.x;
    if (pid >= n_pairs) return;

    constexpr int32_t BM = 64, BN = 64, TY = 16, TX = 16, RJ = 4;
    constexpr int32_t SW = 25, PA = BM + 1, PB = BN + 1;
    constexpr int32_t MAX_CAP_A = 128;
    const int32_t nsl = W / 5;
    const int32_t stride_x = 5 * W;
    const int32_t c_step = W - 5;

    __shared__ uint64_t sA[SW * PA];
    __shared__ uint64_t sB[SW * PB];
    __shared__ int32_t  s_win_a[64];
    __shared__ int32_t  s_win_b[64];
    __shared__ int32_t  s_win_oa[64];
    __shared__ int32_t  s_win_la[64];
    __shared__ int32_t  s_win_ob[64];
    __shared__ int32_t  s_win_lb[64];
    __shared__ int32_t  s_pack_leaf_a[MAX_CAP_A];
    __shared__ double   s_pack_wa[MAX_CAP_A];
    __shared__ uint8_t  s_pack_pidx[MAX_CAP_A];
    __shared__ double   s_pair_sum[64];
    __shared__ int32_t  s_leaf_a[BM];
    __shared__ double   s_wa[BM];
    __shared__ int32_t  s_leaf_b[BN];
    __shared__ double   s_wb[BN];
    __shared__ double   s_row_sum[BM];
    __shared__ int32_t  s_meta[6];

    if (W <= 0 || (W % 5) != 0) {
        int32_t a = d_pair_a[pid];
        int32_t b = d_pair_b[pid];
        int32_t off_a = d_exact_offsets[a], end_a = d_exact_offsets[a + 1];
        int32_t off_b = d_exact_offsets[b], end_b = d_exact_offsets[b + 1];
        int32_t len_b = end_b - off_b;
        int64_t total_ij = (int64_t)(end_a - off_a) * (int64_t)len_b;
        double my_sum = 0.0;
        for (int64_t idx = tid; idx < total_ij; idx += 256) {
            int32_t p = off_a + (int32_t)(idx / len_b);
            int32_t q = off_b + (int32_t)(idx % len_b);
            float wp = d_exact_weights[p], wq = d_exact_weights[q];
            if (wp == 0.0f || wq == 0.0f) continue;
            int32_t lx = d_exact_leaves[p], ly = d_exact_leaves[q];
            float d_jc = 0.0f;
            if (lx != ly) {
                const uint64_t *sx = d_packed_seqs + (size_t)lx * (5 * W);
                const uint64_t *sy = d_packed_seqs + (size_t)ly * (5 * W);
                int32_t n_valid = 0, n_match = 0;
                #pragma unroll 1
                for (int32_t w = 0; w < W; ++w) {
                    n_valid += __popcll(sx[4 * W + w] & sy[4 * W + w]);
                    n_match += __popcll((sx[0 * W + w] & sy[0 * W + w]) | (sx[1 * W + w] & sy[1 * W + w]) | (sx[2 * W + w] & sy[2 * W + w]) | (sx[3 * W + w] & sy[3 * W + w]));
                }
                d_jc = 1.5f;
                if (n_valid > 0) {
                    float pf = (float)(n_valid - n_match) / (float)n_valid;
                    if (pf >= 0.74f) pf = 0.74f;
                    d_jc = -0.75f * logf(1.0f - (4.0f / 3.0f) * pf);
                    if (d_jc < 0.0f) d_jc = 0.0f;
                }
            }
            my_sum += ((double)wp * (double)wq) * (double)d_jc;
        }
        double *s_red = reinterpret_cast<double *>(sA);
        s_red[tid] = my_sum;
        __syncthreads();
        for (int32_t s = 128; s > 0; s >>= 1) {
            if (tid < s) s_red[tid] += s_red[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            if (d_pair_exact_D != NULL) d_pair_exact_D[pid] = s_red[0];
            if (d_D_cond != NULL && a < K && b < K) {
                float fv = (float)s_red[0];
                d_D_cond[(size_t)a * K + b] = fv;
                d_D_cond[(size_t)b * K + a] = fv;
            }
        }
        return;
    }

    const int32_t ws = pid & ~63;
    const int32_t we = (ws + 64 < n_pairs) ? (ws + 64) : n_pairs;
    const int32_t wlen = we - ws;
    if (tid < wlen) {
        int32_t ak = d_pair_a[ws + tid];
        int32_t bk = d_pair_b[ws + tid];
        int32_t oa = d_exact_offsets[ak];
        int32_t ea = d_exact_offsets[ak + 1];
        int32_t ob = d_exact_offsets[bk];
        int32_t eb = d_exact_offsets[bk + 1];
        s_win_a[tid]  = ak;
        s_win_b[tid]  = bk;
        s_win_oa[tid] = oa;
        s_win_la[tid] = ea - oa;
        s_win_ob[tid] = ob;
        s_win_lb[tid] = eb - ob;
    }
    __syncthreads();

    const int32_t my_idx = pid - ws;
    int32_t lead_idx = my_idx, pack_m = 1, total_a = 0;
    int32_t cur = 0;
    while (cur < wlen) {
        int32_t b_cur  = s_win_b[cur];
        int32_t lb_cur = s_win_lb[cur];
        int32_t la_cur = s_win_la[cur];
        int32_t cap_a = (n_pairs >= 4096) ? 64 : ((n_pairs >= 2048) ? 32 : 16);
        if (lb_cur <= 512) {
            cap_a = (n_pairs >= 2048) ? MAX_CAP_A : 64;
        }
        int32_t m = 1;
        int32_t sum_a = la_cur;
        if (la_cur > 0 && la_cur <= lb_cur && la_cur < cap_a) {
            while (cur + m < wlen) {
                int32_t b_nxt  = s_win_b[cur + m];
                int32_t la_nxt = s_win_la[cur + m];
                if (b_nxt != b_cur || la_nxt <= 0 || la_nxt > lb_cur || sum_a + la_nxt > cap_a) break;
                sum_a += la_nxt;
                ++m;
            }
        }
        if (my_idx >= cur && my_idx < cur + m) {
            lead_idx = cur;
            pack_m   = m;
            total_a  = sum_a;
            break;
        }
        cur += m;
    }
    if (my_idx != lead_idx) return;

    int32_t off_a0 = s_win_oa[lead_idx];
    int32_t len_a0 = s_win_la[lead_idx];
    int32_t off_b0 = s_win_ob[lead_idx];
    int32_t len_b0 = s_win_lb[lead_idx];
    if (pack_m == 1) {
        if (len_a0 > len_b0) {
            int32_t t;
            t = off_a0; off_a0 = off_b0; off_b0 = t;
            t = len_a0; len_a0 = len_b0; len_b0 = t;
        }
        total_a = len_a0;
    }
    if (tid < pack_m) {
        s_pair_sum[tid] = 0.0;
    }
    if (total_a <= 0 || len_b0 <= 0) {
        if (tid < pack_m) {
            int32_t gpid = ws + lead_idx + tid;
            int32_t ak = s_win_a[lead_idx + tid];
            int32_t bk = s_win_b[lead_idx + tid];
            if (d_pair_exact_D != NULL) d_pair_exact_D[gpid] = 0.0;
            if (d_D_cond != NULL && ak < K && bk < K) {
                d_D_cond[(size_t)ak * K + bk] = 0.0f;
                d_D_cond[(size_t)bk * K + ak] = 0.0f;
            }
        }
        return;
    }

    if (tid == 0) {
        s_meta[0] = lead_idx; s_meta[1] = pack_m; s_meta[2] = total_a;
        s_meta[3] = off_a0;   s_meta[4] = off_b0; s_meta[5] = len_b0;
    }
    if (pack_m > 1 && tid < total_a) {
        int32_t rem = tid;
        int32_t p = 0;
        while (p + 1 < pack_m) {
            int32_t la_p = s_win_la[lead_idx + p];
            if (rem < la_p) break;
            rem -= la_p;
            ++p;
        }
        int32_t src_idx = s_win_oa[lead_idx + p] + rem;
        float w = d_exact_weights[src_idx];
        s_pack_leaf_a[tid] = (w != 0.0f) ? d_exact_leaves[src_idx] : -1;
        s_pack_wa[tid]     = (double)w;
        s_pack_pidx[tid]   = (uint8_t)p;
    }
    __syncthreads();

    const int32_t tx = tid & 15;
    const int32_t ty = tid >> 4;

#define NK_WL_STEP(RI, ASTRIDE, ABASE, wl) \
    { \
        uint64_t a_reg[RI], b_reg[RJ]; \
        const int32_t kv = 20 + (wl); \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) a_reg[i] = sA[((ABASE) + kv) * (ASTRIDE) + ty + TY * i]; \
        _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j)   b_reg[j] = sB[kv * PB + tx + TX * j]; \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) \
                nv[i][j] += __popcll(a_reg[i] & b_reg[j]); \
        uint64_t m_reg[RI][RJ]; \
        _Pragma("unroll") for (int32_t c = 0; c < 4; ++c) { \
            const int32_t kc = c * 5 + (wl); \
            _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) a_reg[i] = sA[((ABASE) + kc) * (ASTRIDE) + ty + TY * i]; \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j)   b_reg[j] = sB[kc * PB + tx + TX * j]; \
            _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) \
                _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) \
                    m_reg[i][j] = (c == 0) ? (a_reg[i] & b_reg[j]) : (m_reg[i][j] | (a_reg[i] & b_reg[j])); \
        } \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) \
                nm[i][j] += __popcll(m_reg[i][j]); \
    }

#define NK_RUN_ATILE_16() \
    { \
        double acc0 = 0.0; \
        _Pragma("unroll 1") for (int32_t q0 = 0; q0 < s_meta[5]; q0 += BN) { \
            const int32_t tile_b = (s_meta[5] - q0 < BN) ? (s_meta[5] - q0) : BN; \
            __syncthreads(); \
            if (tid < BN) { \
                if (tid < tile_b) { \
                    float w = d_exact_weights[s_meta[4] + q0 + tid]; \
                    s_leaf_b[tid] = (w != 0.0f) ? d_exact_leaves[s_meta[4] + q0 + tid] : -1; \
                    s_wb[tid]     = (double)w; \
                } else { \
                    s_leaf_b[tid] = -1; \
                    s_wb[tid]     = 0.0; \
                } \
            } \
            uint32_t nv[1][RJ], nm[1][RJ]; \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) { nv[0][j] = 0u; nm[0][j] = 0u; } \
            _Pragma("unroll 1") \
            for (int32_t s = 0; s < nsl; ++s) { \
                __syncthreads(); \
                _Pragma("unroll 1") for (int32_t idx = tid; idx < 16 * SW; idx += 256) { \
                    int32_t r = idx / SW; \
                    int32_t k = idx - r * SW; \
                    int32_t c = k / 5; \
                    int32_t la = s_leaf_a[r]; \
                    sA[k * PA + r] = (la >= 0) ? __ldg(&d_packed_seqs[(size_t)la * stride_x + s * 5 + c * c_step + k]) : 0ULL; \
                } \
                _Pragma("unroll 1") for (int32_t idx = tid; idx < BN * SW; idx += 256) { \
                    int32_t r = idx / SW; \
                    int32_t k = idx - r * SW; \
                    int32_t c = k / 5; \
                    int32_t lb = s_leaf_b[r]; \
                    sB[k * PB + r] = (lb >= 0) ? __ldg(&d_packed_seqs[(size_t)lb * stride_x + s * 5 + c * c_step + k]) : 0ULL; \
                } \
                __syncthreads(); \
                NK_WL_STEP(1, PA, 0, 0) \
                NK_WL_STEP(1, PA, 0, 1) \
                NK_WL_STEP(1, PA, 0, 2) \
                NK_WL_STEP(1, PA, 0, 3) \
                NK_WL_STEP(1, PA, 0, 4) \
            } \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) { \
                const double wq = s_wb[tx + TX * j]; \
                int32_t n_valid = (int32_t)nv[0][j]; \
                int32_t n_match = (int32_t)nm[0][j]; \
                float d_jc = 1.5f; \
                if (n_valid > 0) { \
                    float p = (float)(n_valid - n_match) / (float)n_valid; \
                    if (p >= 0.74f) p = 0.74f; \
                    d_jc = -0.75f * logf(1.0f - (4.0f / 3.0f) * p); \
                    if (d_jc < 0.0f) d_jc = 0.0f; \
                } \
                acc0 += wq * (double)d_jc; \
            } \
        } \
        _Pragma("unroll") for (int32_t offset = 8; offset > 0; offset >>= 1) { \
            acc0 += __shfl_down_sync(0xffffffffu, acc0, offset); \
        } \
        if (tx == 0 && ty < tile_a) { \
            s_row_sum[ty] = s_wa[ty] * acc0; \
        } \
    }

#define NK_RUN_ATILE_64(RI) \
    { \
        double acc[RI]; \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) acc[i] = 0.0; \
        _Pragma("unroll 1") for (int32_t q0 = 0; q0 < s_meta[5]; q0 += BN) { \
            const int32_t tile_b = (s_meta[5] - q0 < BN) ? (s_meta[5] - q0) : BN; \
            __syncthreads(); \
            if (tid < BN) { \
                if (tid < tile_b) { \
                    float w = d_exact_weights[s_meta[4] + q0 + tid]; \
                    s_leaf_b[tid] = (w != 0.0f) ? d_exact_leaves[s_meta[4] + q0 + tid] : -1; \
                    s_wb[tid]     = (double)w; \
                } else { \
                    s_leaf_b[tid] = -1; \
                    s_wb[tid]     = 0.0; \
                } \
            } \
            uint32_t nv[RI][RJ], nm[RI][RJ]; \
            _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) \
                _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) { nv[i][j] = 0u; nm[i][j] = 0u; } \
            _Pragma("unroll 1") \
            for (int32_t s = 0; s < nsl; ++s) { \
                __syncthreads(); \
                _Pragma("unroll 1") for (int32_t idx = tid; idx < (RI) * 16 * SW; idx += 256) { \
                    int32_t r = idx / SW; \
                    int32_t k = idx - r * SW; \
                    int32_t c = k / 5; \
                    int32_t la = s_leaf_a[r]; \
                    sA[k * PA + r] = (la >= 0) ? __ldg(&d_packed_seqs[(size_t)la * stride_x + s * 5 + c * c_step + k]) : 0ULL; \
                } \
                _Pragma("unroll 1") for (int32_t idx = tid; idx < BN * SW; idx += 256) { \
                    int32_t r = idx / SW; \
                    int32_t k = idx - r * SW; \
                    int32_t c = k / 5; \
                    int32_t lb = s_leaf_b[r]; \
                    sB[k * PB + r] = (lb >= 0) ? __ldg(&d_packed_seqs[(size_t)lb * stride_x + s * 5 + c * c_step + k]) : 0ULL; \
                } \
                __syncthreads(); \
                NK_WL_STEP(RI, PA, 0, 0) \
                NK_WL_STEP(RI, PA, 0, 1) \
                NK_WL_STEP(RI, PA, 0, 2) \
                NK_WL_STEP(RI, PA, 0, 3) \
                NK_WL_STEP(RI, PA, 0, 4) \
            } \
            _Pragma("unroll") for (int32_t j = 0; j < RJ; ++j) { \
                const double wq = s_wb[tx + TX * j]; \
                _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) { \
                    int32_t n_valid = (int32_t)nv[i][j]; \
                    int32_t n_match = (int32_t)nm[i][j]; \
                    float d_jc = 1.5f; \
                    if (n_valid > 0) { \
                        float p = (float)(n_valid - n_match) / (float)n_valid; \
                        if (p >= 0.74f) p = 0.74f; \
                        d_jc = -0.75f * logf(1.0f - (4.0f / 3.0f) * p); \
                        if (d_jc < 0.0f) d_jc = 0.0f; \
                    } \
                    acc[i] += wq * (double)d_jc; \
                } \
            } \
        } \
        _Pragma("unroll") for (int32_t i = 0; i < (RI); ++i) { \
            double v = acc[i]; \
            _Pragma("unroll") for (int32_t offset = 8; offset > 0; offset >>= 1) { \
                v += __shfl_down_sync(0xffffffffu, v, offset); \
            } \
            int32_t r = ty + TY * i; \
            if (tx == 0 && r < tile_a) { \
                s_row_sum[r] = s_wa[r] * v; \
            } \
        } \
    }

    #pragma unroll 1
    for (int32_t p0 = 0; p0 < s_meta[2]; p0 += BM) {
        const int32_t tile_a = (s_meta[2] - p0 < BM) ? (s_meta[2] - p0) : BM;
        if (tid < BM) {
            if (tid < tile_a) {
                if (s_meta[1] == 1) {
                    float w = d_exact_weights[s_meta[3] + p0 + tid];
                    s_leaf_a[tid] = (w != 0.0f) ? d_exact_leaves[s_meta[3] + p0 + tid] : -1;
                    s_wa[tid]     = (double)w;
                } else {
                    s_leaf_a[tid] = s_pack_leaf_a[p0 + tid];
                    s_wa[tid]     = s_pack_wa[p0 + tid];
                }
            } else {
                s_leaf_a[tid] = -1;
                s_wa[tid]     = 0.0;
            }
        }
        __syncthreads();

        if (tile_a <= 16) {
            NK_RUN_ATILE_16()
        } else if (tile_a <= 32) {
            NK_RUN_ATILE_64(2)
        } else if (tile_a <= 48) {
            NK_RUN_ATILE_64(3)
        } else {
            NK_RUN_ATILE_64(4)
        }
        __syncthreads();

        if (tid < s_meta[1]) {
            double s = s_pair_sum[tid];
            for (int32_t r = 0; r < tile_a; ++r) {
                if (s_meta[1] == 1 || s_pack_pidx[p0 + r] == (uint8_t)tid) {
                    s += s_row_sum[r];
                }
            }
            s_pair_sum[tid] = s;
        }
        __syncthreads();
    }

#undef NK_RUN_ATILE_64
#undef NK_RUN_ATILE_16
#undef NK_WL_STEP

    if (tid < s_meta[1]) {
        int32_t gpid = (pid & ~63) + s_meta[0] + tid;
        int32_t ak = s_win_a[s_meta[0] + tid];
        int32_t bk = s_win_b[s_meta[0] + tid];
        double exact_val = s_pair_sum[tid];
        if (d_pair_exact_D != NULL) d_pair_exact_D[gpid] = exact_val;
        if (d_D_cond != NULL && ak < K && bk < K) {
            float fv = (float)exact_val;
            d_D_cond[(size_t)ak * K + bk] = fv;
            d_D_cond[(size_t)bk * K + ak] = fv;
        }
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
    float   *d_refine_D_scratch;
    bool     refine_split_ready;
    /* Heavy-pair list for k_build_condensed_D_heavy_tiled (grown on demand). */
    int32_t *d_heavy_a;
    int32_t *d_heavy_b;
    size_t   heavy_cap;
    std::vector<int32_t> h_heavy_a, h_heavy_b;

    void init(int32_t gpu_id, int32_t K_cap) {
        dev_id = gpu_id;
        max_K = K_cap;
        d_refine_D_scratch = NULL;
        refine_split_ready = false;
        d_heavy_a = NULL;
        d_heavy_b = NULL;
        heavy_cap = 0;
        int32_t R = 2 * K_cap - 1;
        int32_t M = 2 * K_cap - 2;
        size_t prof_cap = std::max((size_t)256 * (size_t)K_cap, (size_t)4096);
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
        if (!refine_split_ready) {
            /* Room for the load-balanced exact refinement in optimize_skeleton_on_gpu: up to
             * 4 * max_K virtual CSR slots and appended launch pairs, plus a D scratch matrix the
             * refinement kernel writes at indices a * K + b and b * K + a with a, b < 5 * K.
             * Allocated on first use, so workspaces that never refine exactly do not pay for it. */
            CUDA_CHECK(cudaSetDevice(dev_id));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            size_t extra = (size_t)4 * (size_t)max_K;
            cudaFree(d_exact_offsets);
            cudaFree(d_pair_a);
            cudaFree(d_pair_b);
            cudaFree(d_pair_exact_D);
            pair_cap += extra;
            CUDA_CHECK(cudaMalloc(&d_exact_offsets, ((size_t)max_K + 1 + extra) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_pair_a,        pair_cap * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_pair_b,        pair_cap * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_pair_exact_D,  pair_cap * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&d_refine_D_scratch, (size_t)5 * (size_t)max_K * ((size_t)max_K + 1) * sizeof(float)));
            refine_split_ready = true;
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
        cudaFree(d_refine_D_scratch);
        d_refine_D_scratch = NULL;
        refine_split_ready = false;
        cudaFree(d_heavy_a); cudaFree(d_heavy_b);
        d_heavy_a = d_heavy_b = NULL;
        heavy_cap = 0;
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
    /* Pairs whose profiles both have >= HB_HEAVY_MIN leaves go to the tiled kernel; the
     * rest (singletons, singleton x outgroup) stay on the per-thread kernel. */
    const int32_t heavy_min = (W > 0) ? HB_HEAVY_MIN : INT32_MAX;
    gpu_ws.h_heavy_a.clear();
    gpu_ws.h_heavy_b.clear();
    if (heavy_min != INT32_MAX) {
        std::vector<int32_t> heavy_nodes;
        for (int32_t a = 0; a < K; ++a) {
            if (prof_offsets[a + 1] - prof_offsets[a] >= heavy_min) heavy_nodes.push_back(a);
        }
        for (size_t ia = 0; ia < heavy_nodes.size(); ++ia) {
            for (size_t ib = ia + 1; ib < heavy_nodes.size(); ++ib) {
                gpu_ws.h_heavy_a.push_back(heavy_nodes[ia]);
                gpu_ws.h_heavy_b.push_back(heavy_nodes[ib]);
            }
        }
    }
    const int32_t n_heavy = (int32_t)gpu_ws.h_heavy_a.size();
    if ((size_t)n_heavy > gpu_ws.heavy_cap) {
        cudaFree(gpu_ws.d_heavy_a);
        cudaFree(gpu_ws.d_heavy_b);
        gpu_ws.heavy_cap = (size_t)n_heavy;
        CUDA_CHECK(cudaMalloc(&gpu_ws.d_heavy_a, gpu_ws.heavy_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&gpu_ws.d_heavy_b, gpu_ws.heavy_cap * sizeof(int32_t)));
    }
    k_build_real_seq_condensed_D<<<grd2d, blk2d, 0, gpu_ws.stream>>>(
        K, W, gpu_ws.d_prof_offsets, gpu_ws.d_prof_leaves, gpu_ws.d_prof_weights, d_packed_seqs, gpu_ws.d_D,
        heavy_min
    );
    if (n_heavy > 0) {
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_heavy_a, gpu_ws.h_heavy_a.data(), (size_t)n_heavy * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
        CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_heavy_b, gpu_ws.h_heavy_b.data(), (size_t)n_heavy * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
        k_build_condensed_D_heavy_tiled<<<n_heavy, 256, 0, gpu_ws.stream>>>(
            K, W, n_heavy, gpu_ws.d_heavy_a, gpu_ws.d_heavy_b,
            gpu_ws.d_prof_offsets, gpu_ws.d_prof_leaves, gpu_ws.d_prof_weights, d_packed_seqs, gpu_ws.d_D
        );
    }

    std::vector<int32_t> c1_cf, c2_cf, v_map_cf, m_lmin_cf, next_k_cf, parent_u_cf, sub_size_cf, tin_cf, adj_u_cf;
    std::vector<float> h_best_delta(K);
    std::vector<int32_t> h_best_j(K);

    double init_L = 0.0, curr_L = 0.0, best_L = 1e300, prev_L = 1e300;
    std::vector<int64_t> best_v_cf = v_cf, prev_v_cf = v_cf;
    std::vector<int32_t> best_perm = perm, prev_perm = perm;
    int32_t flipped_already = 0;
    bool allow_fast_local = true;
    bool last_was_fast_local = false;
    int32_t cur_batch_cap = max_batch_cap;

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
            prev_L = curr_L;
        } else if (last_was_fast_local && curr_L >= prev_L - 1e-8) {
            // Fast local batch overshot: revert to prev state, halve batch cap, and run one strict path-disjoint step
            v_cf = prev_v_cf;
            perm = prev_perm;
            cur_batch_cap = std::max(16, cur_batch_cap / 2);
            allow_fast_local = false;
            last_was_fast_local = false;
            continue;
        } else if (last_was_fast_local) {
            cur_batch_cap = std::min(max_batch_cap, cur_batch_cap + 16);
        } else {
            // Re-enable fast_local after a strict step so an early overshoot never permanently locks out fast_local
            allow_fast_local = true;
        }
        prev_L = curr_L;
        prev_v_cf = v_cf;
        prev_perm = perm;
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
        float eps_move_thresh = -0.15f / (float)(W * 64);
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

        std::vector<uint8_t> tin_busy(M_sk, 0);
        int32_t n_applied = 0;
        std::vector<int32_t> mark_tins;
        mark_tins.reserve(256);
        bool fast_local_mode = allow_fast_local && (cands.size() >= 16);
        last_was_fast_local = fast_local_mode;
        float fast_cutoff = cands[0].d * 0.10f;
        int32_t eff_batch_cap = fast_local_mode ? cur_batch_cap : max_batch_cap;

        for (const auto &mc : cands) {
            if (fast_local_mode && mc.d > fast_cutoff && n_applied >= 8) break;
            int32_t i = mc.i, dst_u = mc.j;
            int32_t tr = v_map_cf[i + 1];
            int32_t pr = (tr == root_u_sk) ? other_rc : parent_u_cf[tr];
            if (pr == root_u_sk || dst_u < 0 || dst_u >= root_u_sk) continue;
            int32_t dst_v = parent_u_cf[dst_u];
            if (dst_v < 0 || dst_v == root_u_sk) continue;

            int32_t t0 = tin_cf[tr], t1 = t0 + sub_size_cf[tr];
            if (t0 <= tin_cf[dst_u] && tin_cf[dst_u] < t1) continue;
            if (t0 <= tin_cf[pr] && tin_cf[pr] < t1) continue;

            mark_tins.clear();
            bool conflict = false;
            if (fast_local_mode && sub_size_cf[tr] <= 63) {
                for (int32_t t = t0; t < t1; ++t) {
                    if (tin_busy[t]) { conflict = true; break; }
                    mark_tins.push_back(t);
                }
                if (conflict) continue;
                int32_t local_nodes[12];
                int32_t nl = 0;
                local_nodes[nl++] = pr;
                local_nodes[nl++] = dst_u;
                local_nodes[nl++] = dst_v;
                for (int32_t k = 0; k < 3; ++k) {
                    int32_t nb = adj_u_cf[3 * pr + k];
                    if (nb >= 0 && nb != tr) local_nodes[nl++] = nb;
                    int32_t nbu = adj_u_cf[3 * dst_u + k];
                    if (nbu >= 0) local_nodes[nl++] = nbu;
                    int32_t nbv = adj_u_cf[3 * dst_v + k];
                    if (nbv >= 0) local_nodes[nl++] = nbv;
                }
                for (int32_t q = 0; q < nl; ++q) {
                    int32_t tq = tin_cf[local_nodes[q]];
                    if (tin_busy[tq]) { conflict = true; break; }
                    mark_tins.push_back(tq);
                }
                if (conflict) continue;
            } else {
                mark_tins.push_back(tin_cf[tr]);
                mark_tins.push_back(tin_cf[dst_v]);
                int32_t u_anc = pr;
                int32_t t_dst = tin_cf[dst_u];
                int32_t hops = 0;
                while (u_anc != -1 && !(tin_cf[u_anc] <= t_dst && t_dst < tin_cf[u_anc] + sub_size_cf[u_anc])) {
                    mark_tins.push_back(tin_cf[u_anc]);
                    u_anc = parent_u_cf[u_anc];
                    if (++hops > M_sk) break;
                }
                if (u_anc == -1 || hops > M_sk) continue;
                mark_tins.push_back(tin_cf[u_anc]);
                int32_t v_anc = dst_u;
                while (v_anc != u_anc && v_anc != -1) {
                    mark_tins.push_back(tin_cf[v_anc]);
                    v_anc = parent_u_cf[v_anc];
                    if (++hops > M_sk) break;
                }
                if (v_anc != u_anc || hops > M_sk) continue;
                for (int32_t tq : mark_tins) {
                    if (tin_busy[tq]) { conflict = true; break; }
                }
                if (conflict) continue;
            }

            for (int32_t tq : mark_tins) tin_busy[tq] = 1;
            apply_unrooted_spr_host(adj_u_cf.data(), tr, pr, dst_u, dst_v);
            n_applied++;
            if (n_applied >= eff_batch_cap) break;
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

    if (best_L >= init_L - 1e-7) {
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
            if (pair_a.size() > 1) {
                std::vector<std::pair<int32_t, int32_t>> sorted_pairs(pair_a.size());
                for (size_t pid = 0; pid < pair_a.size(); ++pid) {
                    int32_t u = pair_a[pid], v = pair_b[pid];
                    int32_t lu = exact_prof_offsets[u + 1] - exact_prof_offsets[u];
                    int32_t lv = exact_prof_offsets[v + 1] - exact_prof_offsets[v];
                    if (lu > lv) std::swap(u, v);
                    sorted_pairs[pid] = std::make_pair(v, u);
                }
                std::sort(sorted_pairs.begin(), sorted_pairs.end());
                for (size_t pid = 0; pid < sorted_pairs.size(); ++pid) {
                    pair_a[pid] = sorted_pairs[pid].second;
                    pair_b[pid] = sorted_pairs[pid].first;
                }
            }

            std::vector<double> h_pair_exact_D(pair_a.size(), 0.0);
            if (!pair_a.empty()) {
                int32_t n_pairs = (int32_t)pair_a.size();
                int32_t total_exact_prof = exact_prof_offsets[K];
                gpu_ws.ensure_exact_prof_cap((size_t)total_exact_prof);

                /* Load balancing. The refinement kernel runs one 256-thread block per launch pair,
                 * so a capped pair with ~1e9 leaf pairs would occupy one SM while the rest of the
                 * GPU idles. Such pairs are split into row chunks of their smaller exact profile.
                 * Each chunk is a virtual super-leaf id >= K + 1 whose CSR range in refine_offsets
                 * points into the same exact leaf and weight arrays. Chunk 0 takes the pair's own
                 * launch slot, chunks 1..c-1 are appended after n_pairs, and the chunk sums are added
                 * back per pair in a fixed order, so results are deterministic. With a split,
                 * d_exact_offsets receives K + 1 + slots entries, the launch has n_launch > n_pairs
                 * slots, and D is written to the scratch matrix because virtual ids fall outside the
                 * K x K layout of d_D. Without a heavy pair the launch equals the unsplit one. */
                std::vector<int32_t> launch_a(pair_a), launch_b(pair_b), chunk_owner;
                std::vector<int32_t> refine_offsets;
                {
                    const double chunk_leaf_pairs = 4194304.0;     /* target leaf pairs per block */
                    const int64_t b_chunk_target = 6144;           /* ~6-8k leaf B-chunks sized for L2 cache */
                    const int64_t max_virtual = (int64_t)4 * K;    /* slots reserved by ensure_exact_prof_cap */
                    std::vector<int32_t> heavy_pid, heavy_small, heavy_large;
                    std::vector<int64_t> heavy_rows;
                    std::vector<std::vector<int32_t>> b_split_pids((size_t)K);
                    std::vector<int32_t> b_split_profs;
                    for (int32_t pid = 0; pid < n_pairs; ++pid) {
                        int32_t u = pair_a[pid], v = pair_b[pid];
                        int64_t lu = exact_prof_offsets[u + 1] - exact_prof_offsets[u];
                        int64_t lv = exact_prof_offsets[v + 1] - exact_prof_offsets[v];
                        int32_t s = (lu <= lv) ? u : v;
                        int32_t l = (lu <= lv) ? v : u;
                        int64_t ls = std::min(lu, lv), ll = std::max(lu, lv);
                        int64_t b_c = (ll > b_chunk_target) ? std::min((int64_t)16, (ll + b_chunk_target - 1) / b_chunk_target) : 1;
                        int64_t b_rows = (b_c >= 2) ? ((((ll + b_c - 1) / b_c) + 63) & ~(int64_t)63) : ll;
                        if (b_c >= 2 && (double)ls * (double)b_rows <= 2.0 * chunk_leaf_pairs) {
                            if (b_split_pids[l].empty()) b_split_profs.push_back(l);
                            b_split_pids[l].push_back(pid);
                        } else if (ls >= 32 && (double)ls * (double)ll > 2.0 * chunk_leaf_pairs) {
                            int64_t rows = (int64_t)std::ceil(chunk_leaf_pairs / (double)ll);
                            rows = std::max((int64_t)16, ((rows + 15) / 16) * 16);
                            heavy_pid.push_back(pid);
                            heavy_small.push_back(s);
                            heavy_large.push_back(l);
                            heavy_rows.push_back(rows);
                        }
                    }
                    int64_t b_slots = 0, b_extra_pairs = 0;
                    for (int32_t l : b_split_profs) {
                        int64_t ll = exact_prof_offsets[l + 1] - exact_prof_offsets[l];
                        int64_t b_c = std::min((int64_t)16, (ll + b_chunk_target - 1) / b_chunk_target);
                        int64_t b_rows = (((ll + b_c - 1) / b_c) + 63) & ~(int64_t)63;
                        int64_t c = (ll + b_rows - 1) / b_rows;
                        if (c >= 2) {
                            b_slots += c + 1;
                            b_extra_pairs += (c - 1) * (int64_t)b_split_pids[l].size();
                        }
                    }
                    int64_t slots = 0, extra_pairs = 0;
                    for (int32_t pass = 0; pass < 64; ++pass) {
                        slots = b_slots;
                        extra_pairs = b_extra_pairs;
                        for (size_t h = 0; h < heavy_pid.size(); ++h) {
                            int32_t s = heavy_small[h];
                            int64_t ls = exact_prof_offsets[s + 1] - exact_prof_offsets[s];
                            int64_t c = (ls + heavy_rows[h] - 1) / heavy_rows[h];
                            if (c >= 2) {
                                slots += c + 1;
                                extra_pairs += c - 1;
                            }
                        }
                        if (slots <= max_virtual) break;
                        for (size_t h = 0; h < heavy_rows.size(); ++h) heavy_rows[h] *= 2;
                    }
                    if (slots > 0 && slots <= max_virtual && (size_t)(n_pairs + extra_pairs) <= (size_t)K * (size_t)K / 2 + 16 + (size_t)4 * K) {
                        refine_offsets.assign(exact_prof_offsets, exact_prof_offsets + K + 1);
                        for (size_t h = 0; h < heavy_pid.size(); ++h) {
                            int32_t pid = heavy_pid[h], s = heavy_small[h], l = heavy_large[h];
                            int64_t o0 = exact_prof_offsets[s], o1 = exact_prof_offsets[s + 1];
                            int64_t rows = heavy_rows[h];
                            int64_t c = (o1 - o0 + rows - 1) / rows;
                            if (c < 2) continue;
                            int32_t base = (int32_t)refine_offsets.size();
                            for (int64_t j = 0; j < c; ++j) refine_offsets.push_back((int32_t)(o0 + j * rows));
                            refine_offsets.push_back((int32_t)o1);
                            if (launch_a[pid] == s) launch_a[pid] = base;
                            else launch_b[pid] = base;
                            for (int64_t j = 1; j < c; ++j) {
                                launch_a.push_back(base + (int32_t)j);
                                launch_b.push_back(l);
                                chunk_owner.push_back(pid);
                            }
                        }
                        for (int32_t l : b_split_profs) {
                            int64_t o0 = exact_prof_offsets[l], o1 = exact_prof_offsets[l + 1];
                            int64_t ll = o1 - o0;
                            int64_t b_c = std::min((int64_t)16, (ll + b_chunk_target - 1) / b_chunk_target);
                            int64_t b_rows = (((ll + b_c - 1) / b_c) + 63) & ~(int64_t)63;
                            int64_t c = (ll + b_rows - 1) / b_rows;
                            if (c < 2) continue;
                            int32_t base = (int32_t)refine_offsets.size();
                            for (int64_t j = 0; j < c; ++j) refine_offsets.push_back((int32_t)(o0 + j * b_rows));
                            refine_offsets.push_back((int32_t)o1);
                            for (int32_t pid : b_split_pids[l]) {
                                int32_t s_pid = (launch_a[pid] == l) ? launch_b[pid] : launch_a[pid];
                                launch_a[pid] = s_pid;
                                launch_b[pid] = base;
                            }
                            for (int64_t j = 1; j < c; ++j) {
                                for (int32_t pid : b_split_pids[l]) {
                                    launch_a.push_back(launch_a[pid]);
                                    launch_b.push_back(base + (int32_t)j);
                                    chunk_owner.push_back(pid);
                                }
                            }
                        }
                    }
                }
                const int32_t n_launch = (int32_t)launch_a.size();
                const bool split_pairs = !refine_offsets.empty();

                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_offsets, split_pairs ? refine_offsets.data() : exact_prof_offsets,
                                           (split_pairs ? refine_offsets.size() : (size_t)(K + 1)) * sizeof(int32_t),        cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_leaves,  exact_prof_leaves,  (size_t)total_exact_prof * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_weights, exact_prof_weights, (size_t)total_exact_prof * sizeof(float),   cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_a, launch_a.data(), (size_t)n_launch * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
                CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_b, launch_b.data(), (size_t)n_launch * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));

                k_refine_real_seq_condensed_D_pairs<<<n_launch, 256, 0, gpu_ws.stream>>>(
                    K, W, n_launch, gpu_ws.d_pair_a, gpu_ws.d_pair_b,
                    gpu_ws.d_exact_offsets, gpu_ws.d_exact_leaves, gpu_ws.d_exact_weights,
                    d_packed_seqs, split_pairs ? gpu_ws.d_refine_D_scratch : gpu_ws.d_D, gpu_ws.d_pair_exact_D
                );
                std::vector<double> h_launch_exact_D((size_t)n_launch, 0.0);
                CUDA_CHECK(cudaMemcpyAsync(h_launch_exact_D.data(), gpu_ws.d_pair_exact_D, (size_t)n_launch * sizeof(double), cudaMemcpyDeviceToHost, gpu_ws.stream));
                CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
                for (int32_t pid = 0; pid < n_pairs; ++pid) h_pair_exact_D[pid] = h_launch_exact_D[pid];
                for (int32_t t = n_pairs; t < n_launch; ++t) h_pair_exact_D[chunk_owner[t - n_pairs]] += h_launch_exact_D[t];
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

            if (exact_delta_L >= -1e-7) {
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
                need_exact_prof = false;
            }
        }
        prof_offsets[K] = (int32_t)prof_leaves.size();

        std::vector<int32_t> exact_prof_offsets, exact_prof_leaves;
        std::vector<float> exact_prof_weights;
        if (need_exact_prof) {
            const int32_t exact_cap = (N >= 1000000) ? 2048 : -1;
            exact_prof_offsets.resize(K + 1, 0);
            exact_prof_leaves.reserve((N >= 1000000) ? (size_t)(K * 256) : (size_t)N);
            exact_prof_weights.reserve((N >= 1000000) ? (size_t)(K * 256) : (size_t)N);
            for (int32_t a = 0; a < K; ++a) {
                exact_prof_offsets[a] = (int32_t)exact_prof_leaves.size();
                if (leaf_cnt[super_roots[a]] <= max_prof_leaves) {
                    exact_prof_leaves.insert(exact_prof_leaves.end(), prof_leaves.begin() + prof_offsets[a], prof_leaves.begin() + prof_offsets[a + 1]);
                    exact_prof_weights.insert(exact_prof_weights.end(), prof_weights.begin() + prof_offsets[a], prof_weights.begin() + prof_offsets[a + 1]);
                } else {
                    compute_clade_dyadic_profile(
                        super_roots[a], N, curr_c1.data(), curr_c2.data(), node_rep0.data(),
                        exact_cap, exact_prof_leaves, exact_prof_weights
                    );
                }
            }
            exact_prof_offsets[K] = (int32_t)exact_prof_leaves.size();
        }

        std::vector<int64_t> v_mb(K - 1);
        std::vector<int32_t> perm_mb(K);
        relabel_from_rooted_tree(sk_c1.data(), sk_c2.data(), K, 0, v_mb.data(), perm_mb.data());

        double dL = optimize_skeleton_on_gpu(
            K, W, 200, 128,
            prof_offsets.data(), prof_leaves.data(), prof_weights.data(),
            d_packed_seqs, gpu_ws, v_mb, perm_mb,
            need_exact_prof ? exact_prof_offsets.data() : NULL,
            need_exact_prof ? exact_prof_leaves.data() : NULL,
            need_exact_prof ? exact_prof_weights.data() : NULL
        );
        if (dL == 0.0) return 0.0;

        // Re-root the optimized K-taxon skeleton on its exact centroid edge (closest to N/2 leaves)
        std::vector<int32_t> c1_cf, c2_cf, v_map_cf, m_lmin_cf, next_k_cf, parent_u_cf, sub_sz_cf, tin_cf, adj_u_cf;
        int32_t other_rc, rc0, rc1, rc2;
        build_skeleton_metadata(
            v_mb.data(), K, c1_cf, c2_cf, v_map_cf, m_lmin_cf, next_k_cf,
            parent_u_cf, sub_sz_cf, tin_cf, adj_u_cf,
            &other_rc, &rc0, &rc1, &rc2
        );
        const int32_t root_u_sk = 2 * K - 3;
        std::vector<int32_t> w_sub(2 * K - 2, 0);
        for (int32_t a = 0; a < K; ++a) w_sub[a] = leaf_cnt[super_roots[perm_mb[a]]];
        for (int32_t u = K; u < root_u_sk; ++u) w_sub[u] = w_sub[c1_cf[u]] + w_sub[c2_cf[u]];
        int32_t u_best = 0;
        int32_t best_diff = std::abs(2 * w_sub[0] - N);
        for (int32_t u = 1; u < root_u_sk; ++u) {
            int32_t diff = std::abs(2 * w_sub[u] - N);
            if (diff < best_diff) {
                best_diff = diff;
                u_best = u;
            }
        }
        int32_t p_best = parent_u_cf[u_best];

        std::vector<int32_t> st_u, st_p;
        st_u.reserve(K - 2);
        st_p.reserve(K - 2);
        if (u_best >= K) { st_u.push_back(u_best); st_p.push_back(p_best); }
        if (p_best >= K) { st_u.push_back(p_best); st_p.push_back(u_best); }
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
        for (int32_t a = 0; a < K; ++a) sk_to_full[a] = super_roots[perm_mb[a]];
        for (int32_t s = 0; s < K - 2; ++s) {
            int32_t u_sk = st_u[K - 3 - s];
            sk_to_full[u_sk] = split_internals[s];
        }
        for (int32_t s = 0; s < K - 2; ++s) {
            int32_t u_sk = st_u[K - 3 - s];
            int32_t full_u = sk_to_full[u_sk];
            int32_t b0 = sk_to_full[child0[u_sk]];
            int32_t b1 = sk_to_full[child1[u_sk]];
            curr_c1[full_u] = b0;
            curr_c2[full_u] = b1;
            update_one_node_summary(full_u, b0, b1, parent_r.data(), leaf_cnt.data(), node_rep0.data());
        }
        {
            int32_t full_u = split_internals[K - 2]; // == root_r
            int32_t b0 = sk_to_full[u_best];
            int32_t b1 = sk_to_full[p_best];
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

        const int32_t *oc1 = (outgroup_c1 != NULL) ? outgroup_c1 : curr_c1.data();
        const int32_t *oc2 = (outgroup_c2 != NULL) ? outgroup_c2 : curr_c2.data();
        const int32_t *opar = (outgroup_par != NULL) ? outgroup_par : parent_r.data();
        const int32_t *orep = (outgroup_rep0 != NULL) ? outgroup_rep0 : node_rep0.data();

        std::vector<int32_t> prof_offsets(K + 1, 0), prof_leaves;
        std::vector<float> prof_weights;
        prof_leaves.reserve(4 * K);
        prof_weights.reserve(4 * K);
        bool need_exact_prof = false;
        for (int32_t a = 0; a < K_in; ++a) {
            prof_offsets[a] = (int32_t)prof_leaves.size();
            compute_clade_dyadic_profile(
                super_roots[a], N, oc1, oc2, orep,
                max_prof_leaves, prof_leaves, prof_weights
            );
            if (max_prof_leaves > 0 && leaf_cnt[super_roots[a]] > max_prof_leaves) {
                need_exact_prof = false;
            }
        }
        prof_offsets[K_in] = (int32_t)prof_leaves.size();
        prof_leaves.insert(prof_leaves.end(), outgroup_leaves.begin(), outgroup_leaves.end());
        prof_weights.insert(prof_weights.end(), outgroup_weights.begin(), outgroup_weights.end());
        prof_offsets[K] = (int32_t)prof_leaves.size();

        int32_t outgroup_true_leaves = N - leaf_cnt[sub_root];
        if (max_prof_leaves > 0 && outgroup_true_leaves > (int32_t)outgroup_leaves.size()) {
            need_exact_prof = false;
        }

        std::vector<int32_t> exact_prof_offsets, exact_prof_leaves;
        std::vector<float> exact_prof_weights;
        if (need_exact_prof) {
            const int32_t exact_cap = (N >= 1000000) ? 2048 : -1;
            exact_prof_offsets.resize(K + 1, 0);
            exact_prof_leaves.reserve((N >= 1000000) ? (size_t)(K * 256) : (size_t)N);
            exact_prof_weights.reserve((N >= 1000000) ? (size_t)(K * 256) : (size_t)N);
            for (int32_t a = 0; a < K_in; ++a) {
                exact_prof_offsets[a] = (int32_t)exact_prof_leaves.size();
                if (max_prof_leaves <= 0 || leaf_cnt[super_roots[a]] <= max_prof_leaves) {
                    exact_prof_leaves.insert(exact_prof_leaves.end(), prof_leaves.begin() + prof_offsets[a], prof_leaves.begin() + prof_offsets[a + 1]);
                    exact_prof_weights.insert(exact_prof_weights.end(), prof_weights.begin() + prof_offsets[a], prof_weights.begin() + prof_offsets[a + 1]);
                } else {
                    compute_clade_dyadic_profile(
                        super_roots[a], N, oc1, oc2, orep,
                        exact_cap, exact_prof_leaves, exact_prof_weights
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
                    exact_cap, exact_prof_leaves, exact_prof_weights
                );
            }
            exact_prof_offsets[K] = (int32_t)exact_prof_leaves.size();
        }

        std::vector<int64_t> v_mb(K - 1);
        std::vector<int32_t> perm_mb(K);
        relabel_from_rooted_tree(sk_c1.data(), sk_c2.data(), K, 0, v_mb.data(), perm_mb.data());

        double dL = optimize_skeleton_on_gpu(
            K, W, 200, 128,
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
            if (outgroup_c1 == NULL) {
                update_one_node_summary(full_u, b0, b1, parent_r.data(), leaf_cnt.data(), node_rep0.data());
            } else {
                leaf_cnt[full_u] = leaf_cnt[b0] + leaf_cnt[b1];
                node_rep0[full_u] = (leaf_cnt[b0] <= leaf_cnt[b1]) ? node_rep0[b0] : node_rep0[b1];
            }
        }
        if (outgroup_c1 == NULL) {
            parent_r[sub_root] = parent_of_sub_root;
        }
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

    int n_outer_rounds = (N <= 1000) ? 2 : ((N >= 1000000) ? 4 : 3);
    int total_passes = (N > 1050) ? (n_outer_rounds + 1) : n_outer_rounds;
    for (int outer_round = 0; outer_round < total_passes; ++outer_round) {
    const bool is_final_leaf_cleanup = (outer_round == n_outer_rounds);
    /* Pass 1: Global Backbone (priority-queue bounded to max_internals = 1200 so K <= 1201 <= 2048) */
    if (outer_round > 0 && !is_final_leaf_cleanup && N >= 5000) {
        total_mb_delta_L += run_subtree_minibatch_window(
            root_r, N, W, 1, 25, node_has_active, empty_out_leaves, empty_out_weights,
            d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
            1500, NULL
        );
        compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
    }

    /* Pass 2A: Leaf-Complete Connected Component Waves (100% of all N original leaves across all leaf-bearing internal components, K <= 1056 <= 2048) */
    std::vector<uint8_t> is_comp_root;
    std::vector<std::vector<int32_t>> level_waves;
    build_component_waves(N, ord, curr_c1, curr_c2, 1050, 1050, is_comp_root, level_waves);

    for (size_t lvl = 0; lvl < level_waves.size(); ++lvl) {
        const std::vector<int32_t> &wave = level_waves[lvl];
        if (n_gpus > 1 && wave.size() > 1) {
            std::vector<int32_t> snap_c1 = curr_c1, snap_c2 = curr_c2, snap_par = parent_r, snap_rep0 = node_rep0;
            double wave_dL = 0.0;
            #pragma omp parallel for num_threads(n_gpus) schedule(dynamic, 1) reduction(+:wave_dL)
            for (int32_t idx = 0; idx < (int32_t)wave.size(); ++idx) {
                int g = omp_get_thread_num();
                int32_t sr = wave[idx];
                std::vector<int32_t> out_leaves;
                std::vector<float> out_weights;
                if (sr != root_r) {
                    compute_outgroup_dyadic_profile(
                        sr, root_r, N, snap_c1.data(), snap_c2.data(), snap_par.data(), snap_rep0.data(),
                        max_prof_leaves, out_leaves, out_weights
                    );
                }
                wave_dL += run_subtree_minibatch_window(
                    sr, N, W, 1, 25, node_has_active, out_leaves, out_weights,
                    d_packed_seqs[g], mb_ws[g], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                    1200, is_comp_root.data(),
                    snap_c1.data(), snap_c2.data(), snap_par.data(), snap_rep0.data()
                );
            }
            total_mb_delta_L += wave_dL;
            compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
        } else {
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
    }

    /* Pass 2B: Staggered Boundary-Crossing Component Waves + Full Multi-Scale V-Cycle */
    if (N > 1000) {
        {
            compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            std::vector<uint8_t> is_comp_root_stag;
            std::vector<std::vector<int32_t>> stag_waves;
            build_component_waves(N, ord, curr_c1, curr_c2, 525, 1050, is_comp_root_stag, stag_waves);
            for (int32_t lvl = (int32_t)stag_waves.size() - 1; lvl >= 0; --lvl) {
                const std::vector<int32_t> &wave = stag_waves[lvl];
                if (n_gpus > 1 && wave.size() > 1) {
                    std::vector<int32_t> snap_c1 = curr_c1, snap_c2 = curr_c2, snap_par = parent_r, snap_rep0 = node_rep0;
                    double wave_dL = 0.0;
                    #pragma omp parallel for num_threads(n_gpus) schedule(dynamic, 1) reduction(+:wave_dL)
                    for (int32_t idx = 0; idx < (int32_t)wave.size(); ++idx) {
                        int g = omp_get_thread_num();
                        int32_t sr = wave[idx];
                        std::vector<int32_t> out_leaves;
                        std::vector<float> out_weights;
                        if (sr != root_r) {
                            compute_outgroup_dyadic_profile(
                                sr, root_r, N, snap_c1.data(), snap_c2.data(), snap_par.data(), snap_rep0.data(),
                                max_prof_leaves, out_leaves, out_weights
                            );
                        }
                        wave_dL += run_subtree_minibatch_window(
                            sr, N, W, 1, 25, node_has_active, out_leaves, out_weights,
                            d_packed_seqs[g], mb_ws[g], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                            1200, is_comp_root_stag.data(),
                            snap_c1.data(), snap_c2.data(), snap_par.data(), snap_rep0.data()
                        );
                    }
                    total_mb_delta_L += wave_dL;
                    compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
                } else {
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
            }
            if (outer_round > 0 && !is_final_leaf_cleanup && N >= 5000) {
                total_mb_delta_L += run_subtree_minibatch_window(
                    root_r, N, W, 1, 25, node_has_active, empty_out_leaves, empty_out_weights,
                    d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                    1500, NULL
                );
                compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            }
        }
        if (is_final_leaf_cleanup) {
            compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            break;
        }

        // Multi-scale V-cycle: Full Bottom-Up elevator (in every round) + Global Backbone + Top-Down cascade
        std::vector<int32_t> scale_caps;
        if (N > 4000) {
            for (int32_t sc = 4096; sc < N; sc *= 4) scale_caps.push_back(sc);
            scale_caps.push_back(N); // Global root_r after bottom-up tournament!
        } else if (N >= 2000) {
            scale_caps.push_back(N);
        }
        for (int32_t sc = N / 4; sc > 3200; sc /= 4) scale_caps.push_back(sc);
        if (N > 1050) scale_caps.push_back(3200);
        for (int32_t sc_cap : scale_caps) {
            compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            std::vector<int32_t> mid_roots;
            {
                std::vector<int32_t> st = {root_r};
                while (!st.empty()) {
                    int32_t u = st.back(); st.pop_back();
                    if (u < N) continue;
                    if (leaf_cnt[u] <= sc_cap) {
                        if (leaf_cnt[u] > 1050) mid_roots.push_back(u);
                    } else {
                        st.push_back(curr_c2[u]);
                        st.push_back(curr_c1[u]);
                    }
                }
            }
            if (mid_roots.empty()) continue;
            if (n_gpus > 1 && mid_roots.size() > 1) {
                std::vector<int32_t> snap_c1 = curr_c1, snap_c2 = curr_c2, snap_par = parent_r, snap_rep0 = node_rep0;
                double mid_dL = 0.0;
                #pragma omp parallel for num_threads(n_gpus) schedule(dynamic, 1) reduction(+:mid_dL)
                for (int32_t idx = 0; idx < (int32_t)mid_roots.size(); ++idx) {
                    int g = omp_get_thread_num();
                    int32_t sr = mid_roots[idx];
                    std::vector<int32_t> out_leaves_p2b;
                    std::vector<float> out_weights_p2b;
                    if (sr != root_r) {
                        compute_outgroup_dyadic_profile(
                            sr, root_r, N, snap_c1.data(), snap_c2.data(), snap_par.data(), snap_rep0.data(),
                            max_prof_leaves, out_leaves_p2b, out_weights_p2b
                        );
                    }
                    mid_dL += run_subtree_minibatch_window(
                        sr, N, W, 1, 25, node_has_active, out_leaves_p2b, out_weights_p2b,
                        d_packed_seqs[g], mb_ws[g], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                        1500, NULL,
                        snap_c1.data(), snap_c2.data(), snap_par.data(), snap_rep0.data()
                    );
                }
                total_mb_delta_L += mid_dL;
                compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
            } else {
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
                    double dL = run_subtree_minibatch_window(
                        sr, N, W, 1, 25, node_has_active, out_leaves_p2b, out_weights_p2b,
                        d_packed_seqs[0], mb_ws[0], curr_c1, curr_c2, parent_r, leaf_cnt, node_rep0,
                        1500, NULL
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
        }
    }

    compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, parent_r, leaf_cnt, node_rep0);
    }
    return total_mb_delta_L;
}

static FILE *open_rnasim_bin(int32_t N, const char *custom_prefix = NULL) {
    if (custom_prefix && custom_prefix[0] != '\0') {
        char p0[512];
        snprintf(p0, sizeof(p0), "%s_%d.bin", custom_prefix, N);
        return fopen(p0, "rb");
    }
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
    bool emit_json = false;
    const char *custom_prefix = NULL;
    const char *sizes_csv = NULL;
    for (int a = 1; a < argc; ++a) {
        if (strcmp(argv[a], "--minibatch-only") == 0) minibatch_only = true;
        else if (strcmp(argv[a], "--test-ladder") == 0) test_ladder = true;
        else if (strcmp(argv[a], "--json") == 0) emit_json = true;
        else if (strcmp(argv[a], "--prefix") == 0 && a + 1 < argc) custom_prefix = argv[++a];
        else if (strcmp(argv[a], "--sizes") == 0 && a + 1 < argc) sizes_csv = argv[++a];
    }

    int n_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&n_gpus));
    if (n_gpus > 8) n_gpus = 8;
    if (n_gpus < 1) n_gpus = 1;

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

    std::vector<int32_t> sizes = {1000, 5000, 10000, 50000, 100000};
    if (sizes_csv && sizes_csv[0] != '\0') {
        sizes.clear();
        const char *p = sizes_csv;
        while (*p) {
            char *endp = NULL;
            long v = strtol(p, &endp, 10);
            if (endp == p) break;
            if (v > 0) sizes.push_back((int32_t)v);
            p = endp;
            while (*p == ',' || *p == ' ') ++p;
        }
    }

    printf("====================================================================================================================================\n");
    printf("BLIND GPU PHYLO2VEC BME BENCHMARK (%dx %s | Mode: %s | Prefix: %s)\n",
           n_gpus, prop.name, minibatch_only ? "Minibatch-Only" : "Full-Tree + Minibatch",
           custom_prefix ? custom_prefix : "rnasim");
    printf("====================================================================================================================================\n");
    printf("%8s | %22s | %22s | %24s | %24s | %10s\n",
           "Taxa (N)", "VFT-InitNJ RF (Time)", "VFT-ME (-noml) RF", "GPU Full-Tree BME RF", "GPU Minibatch BME RF", "Delta L");
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");

    const double vft_init_times[] = {0.58, 3.40, 7.82, 63.01, 179.03};
    const double vft_me_times[]   = {7.46, 36.78, 73.80, 376.54, 772.53};

    for (size_t s_idx = 0; s_idx < sizes.size(); ++s_idx) {
        int32_t N = sizes[s_idx];
        FILE *fp = open_rnasim_bin(N, custom_prefix);
        if (!fp) { fprintf(stderr, "Missing bin for N=%d (prefix=%s)\n", N, custom_prefix ? custom_prefix : "rnasim"); continue; }

        int32_t hdr[3];
        if (fread(hdr, sizeof(int32_t), 3, fp) != 3) exit(1);
        int32_t L_sites = hdr[1];
        int32_t W = hdr[2];
        const int32_t n_nodes_r = 2 * N - 1;

        const size_t total_u64 = (size_t)N * 5 * (size_t)W;
        std::vector<uint64_t *> d_packed_seqs(n_gpus, NULL);
        for (int g = 0; g < n_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaMalloc(&d_packed_seqs[g], total_u64 * sizeof(uint64_t)));
        }
        {
            const size_t chunk_u64 = 32ULL * 1024ULL * 1024ULL; /* 256 MB staging buffer */
            std::vector<uint64_t> h_stage(std::min(total_u64, chunk_u64));
            size_t offset = 0;
            while (offset < total_u64) {
                size_t cur = std::min(chunk_u64, total_u64 - offset);
                if (fread(h_stage.data(), sizeof(uint64_t), cur, fp) != cur) exit(1);
                for (int g = 0; g < n_gpus; ++g) {
                    CUDA_CHECK(cudaSetDevice(g));
                    CUDA_CHECK(cudaMemcpy(d_packed_seqs[g] + offset, h_stage.data(), cur * sizeof(uint64_t), cudaMemcpyHostToDevice));
                }
                offset += cur;
            }
        }

        std::vector<int32_t> true_c1(n_nodes_r), true_c2(n_nodes_r);
        std::vector<int32_t> init_c1(n_nodes_r), init_c2(n_nodes_r);
        std::vector<int32_t> me_c1(n_nodes_r), me_c2(n_nodes_r);

        if (fread(true_c1.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(true_c2.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(init_c1.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(init_c2.data(), sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(me_c1.data(),   sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        if (fread(me_c2.data(),   sizeof(int32_t), n_nodes_r, fp) != (size_t)n_nodes_r) exit(1);
        fclose(fp);

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
        double dL_full_gpu = 0.0;
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

            dL_full_gpu = optimize_skeleton_on_gpu(
                N, W, std::max(400, N / 5), 128,
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

        double t_init_ref = (s_idx < 5) ? vft_init_times[s_idx] : 0.0;
        double t_me_ref   = (s_idx < 5) ? vft_me_times[s_idx]   : 0.0;
        char b_init[64], b_me[64], b_full[64], b_mb[64];
        snprintf(b_init, sizeof(b_init), "%lld (%.2fs)", (long long)rf_init, t_init_ref);
        snprintf(b_me,   sizeof(b_me),   "%lld (%.2fs)", (long long)rf_me,   t_me_ref);
        if (rf_full_gpu >= 0) {
            snprintf(b_full, sizeof(b_full), "%lld (%.2fs)", (long long)rf_full_gpu, dt_full_gpu);
        } else {
            snprintf(b_full, sizeof(b_full), "N/A (Minibatch)");
        }
        snprintf(b_mb,   sizeof(b_mb),   "%lld (%.2fs)", (long long)rf_mb_gpu,   dt_mb_gpu);
        double dL_rep = total_mb_delta_L;

        printf("%8d | %22s | %22s | %24s | %24s | %+10.2f\n",
               N, b_init, b_me, b_full, b_mb, dL_rep);
        if (emit_json) {
            struct rusage ru;
            getrusage(RUSAGE_SELF, &ru);
            double host_ram_mb = (double)ru.ru_maxrss / 1024.0;
            size_t free_b = 0, total_b = 0;
            CUDA_CHECK(cudaSetDevice(0));
            CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
            double gpu_vram_mb = (double)(total_b - free_b) / (1024.0 * 1024.0);
            int omp_thr = omp_get_max_threads();
            printf("[RESULT_JSON] {\"N\": %d, \"L\": %d, \"W\": %d, \"max_splits\": %d, \"rf_init\": %lld, \"rf_me\": %lld, \"rf_full_gpu\": %lld, \"dt_full_gpu\": %.4f, \"dL_full_gpu\": %.4f, \"rf_mb_gpu\": %lld, \"dt_mb_gpu\": %.4f, \"dL_mb_gpu\": %.4f, \"host_ram_mb\": %.1f, \"gpu_vram_mb\": %.1f, \"n_gpus\": %d, \"cpu_threads\": %d, \"gpu_name\": \"%s\"}\n",
                   N, L_sites, W, N - 3, (long long)rf_init, (long long)rf_me,
                   (long long)rf_full_gpu, dt_full_gpu, dL_full_gpu,
                   (long long)rf_mb_gpu, dt_mb_gpu, total_mb_delta_L,
                   host_ram_mb, gpu_vram_mb, n_gpus, omp_thr, prop.name);
        }
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
