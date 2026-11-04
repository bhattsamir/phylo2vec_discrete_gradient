/*
 * qwen_tree_gpu.cu
 *
 * GPU Balanced Minimum Evolution (BME) Tree + 4-Point Quartet Violation Scanner +
 * Clade-Condensed Minibatch SPR Evolution for Qwen-14B Weight Hierarchies:
 *   - Qwen/Qwen2.5-14B (Base Ancestor)
 *   - Qwen/Qwen2.5-Coder-14B (Continual Pretraining + Code Specialist)
 *   - deepseek-ai/DeepSeek-R1-Distill-Qwen-14B (Reasoning Distillation Specialist)
 *
 * Taxa (N):
 *   - Per-layer MLP down_proj neurons: N = 13,824 (x 48 layers)
 *   - Vocabulary lm_head tokens:       N = 152,064
 *   - Full-model MLP down_proj:        N = 663,552 (all 48 layers combined)
 * Traits per taxon (D):
 *   - D = 5,120 residual-stream weights (stored in BF16, L2-normalized to unit sphere)
 *   - Pairwise distance: D(x, y) = ||w_hat_x - w_hat_y||_2^2 = 2 * (1 - <w_hat_x, w_hat_y>)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>

extern "C" {
void encode_rooted_to_v_c(
    const int32_t *child1_r,
    const int32_t *child2_r,
    int32_t n_leaves,
    int64_t *v_out
);
void relabel_cherry_first_c(
    const int64_t *v_in,
    int32_t n_leaves,
    int32_t flip_mode,
    int64_t *v_out,
    int32_t *perm_out
);
}

/* Exact O(M^3) Neighbor-Joining on an M x M distance matrix producing a rooted binary tree (c1_r, c2_r) */
static void build_nj_rooted_from_dist(
    const float *D_in, int32_t M,
    int32_t *c1_r, int32_t *c2_r
) {
    if (M == 2) {
        c1_r[2] = 0; c2_r[2] = 1;
        return;
    }
    std::vector<double> D((size_t)M * M);
    for (size_t i = 0; i < (size_t)M * M; ++i) D[i] = (double)D_in[i];
    std::vector<int32_t> active_node(M);
    for (int32_t i = 0; i < M; ++i) active_node[i] = i;
    std::vector<double> r_sum(M, 0.0);
    int32_t n_act = M;
    int32_t next_u = M;

    while (n_act > 2) {
        for (int32_t i = 0; i < n_act; ++i) {
            double s = 0.0;
            for (int32_t j = 0; j < n_act; ++j) s += D[(size_t)i * M + j];
            r_sum[i] = s;
        }
        double best_q = 1e300;
        int32_t best_i = 0, best_j = 1;
        double scale = (double)(n_act - 2);
        for (int32_t i = 0; i < n_act; ++i) {
            for (int32_t j = i + 1; j < n_act; ++j) {
                double q = scale * D[(size_t)i * M + j] - r_sum[i] - r_sum[j];
                if (q < best_q) {
                    best_q = q;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        int32_t u = next_u++;
        c1_r[u] = active_node[best_i];
        c2_r[u] = active_node[best_j];
        double d_ij = D[(size_t)best_i * M + best_j];
        for (int32_t k = 0; k < n_act; ++k) {
            if (k == best_i || k == best_j) continue;
            double d_uk = 0.5 * (D[(size_t)best_i * M + k] + D[(size_t)best_j * M + k] - d_ij);
            if (d_uk < 0.0) d_uk = 0.0;
            D[(size_t)best_i * M + k] = d_uk;
            D[(size_t)k * M + best_i] = d_uk;
        }
        active_node[best_i] = u;
        /* Swap best_j with n_act - 1 */
        int32_t last = n_act - 1;
        if (best_j != last) {
            active_node[best_j] = active_node[last];
            for (int32_t k = 0; k < n_act; ++k) {
                if (k == best_j || k == last) continue;
                D[(size_t)best_j * M + k] = D[(size_t)last * M + k];
                D[(size_t)k * M + best_j] = D[(size_t)k * M + last];
            }
            D[(size_t)best_j * M + best_j] = 0.0;
        }
        n_act--;
    }
    int32_t root_u = next_u++;
    c1_r[root_u] = active_node[0];
    c2_r[root_u] = active_node[1];
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static constexpr int32_t DIM = 5120;

static double wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ============================================================================
 * 1. O(N) Tree Encoding & Exact O(N log N) Robinson-Foulds Split Distance
 * ============================================================================ */
static void decode_cf_to_rooted(
    const int64_t *v_cf, int32_t N,
    int32_t *c1_r, int32_t *c2_r, int32_t *scratch_uf
) {
    for (int32_t x = 0; x < N; ++x) scratch_uf[x] = x;
    for (int32_t k = 0; k < N - 1; ++k) {
        int32_t i = N - 2 - k;
        int32_t m = N + k;
        int32_t x = (int32_t)v_cf[i];
        int32_t y = i + 1;
        c1_r[m] = scratch_uf[y];
        c2_r[m] = scratch_uf[x];
        scratch_uf[x] = m;
    }
}

static void encode_rooted_fast_o_n(
    const int32_t *child1_r, const int32_t *child2_r, int32_t N, int64_t *v_out
) {
    if (N <= 1) return;
    if (N == 2) { v_out[0] = 0; return; }
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    std::vector<int32_t> ord(N - 1);
    std::vector<int32_t> min_leaf(n_nodes_r);
    std::vector<int32_t> l_max_node(n_nodes_r, -1);
    std::vector<int32_t> c_stay_node(n_nodes_r, -1);
    std::vector<int32_t> node_of_lmax(N, -1);

    for (int32_t x = 0; x < N; ++x) min_leaf[x] = x;
    int32_t ord_len = 0;
    ord[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        int32_t b0 = child1_r[u], b1 = child2_r[u];
        if (b0 >= N) ord[ord_len++] = b0;
        if (b1 >= N) ord[ord_len++] = b1;
    }
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        if (min_leaf[c1] < min_leaf[c2]) {
            min_leaf[u] = min_leaf[c1];
            l_max_node[u] = min_leaf[c2];
            c_stay_node[u] = c1;
        } else {
            min_leaf[u] = min_leaf[c2];
            l_max_node[u] = min_leaf[c1];
            c_stay_node[u] = c2;
        }
        node_of_lmax[l_max_node[u]] = u;
    }
    bool all_cherry_first = true;
    v_out[0] = 0;
    for (int32_t i = 1; i < N - 1; ++i) {
        int32_t next_leaf = i + 1;
        int32_t u = node_of_lmax[next_leaf];
        int32_t w = c_stay_node[u];
        while (w >= N && l_max_node[w] > next_leaf) w = c_stay_node[w];
        if (w < N) v_out[i] = w;
        else { all_cherry_first = false; break; }
    }
    if (!all_cherry_first) {
        encode_rooted_to_v_c(child1_r, child2_r, N, v_out);
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
    std::sort(splits_out.begin(), splits_out.end());
}

static int64_t count_wrong_splits_rf(
    const std::vector<uint64_t> &splits_a,
    const std::vector<uint64_t> &splits_b
) {
    size_t i = 0, j = 0, common = 0;
    while (i < splits_a.size() && j < splits_b.size()) {
        if (splits_a[i] == splits_b[j]) { common++; i++; j++; }
        else if (splits_a[i] < splits_b[j]) i++;
        else j++;
    }
    return (int64_t)splits_a.size() - (int64_t)common;
}

static void compute_tree_summaries_blind(
    const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    std::vector<int32_t> &ord,
    std::vector<int32_t> &curr_parent,
    std::vector<int32_t> &leaf_cnt,
    std::vector<int32_t> &node_rep0
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    ord.resize(N - 1);
    curr_parent.assign(n_nodes_r, -1);
    leaf_cnt.assign(n_nodes_r, 1);
    node_rep0.resize(n_nodes_r);
    std::vector<int32_t> min_depth(n_nodes_r, 0);

    int32_t ord_len = 0;
    ord[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        curr_parent[b0] = u;
        curr_parent[b1] = u;
        if (b0 >= N) ord[ord_len++] = b0;
        if (b1 >= N) ord[ord_len++] = b1;
    }
    for (int32_t x = 0; x < N; ++x) node_rep0[x] = x;
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        leaf_cnt[u] = leaf_cnt[b0] + leaf_cnt[b1];
        if (min_depth[b0] <= min_depth[b1]) {
            min_depth[u] = 1 + min_depth[b0];
            node_rep0[u] = node_rep0[b0];
        } else {
            min_depth[u] = 1 + min_depth[b1];
            node_rep0[u] = node_rep0[b1];
        }
    }
}

/* ============================================================================
 * 2. GPU Weight Oracle & Exact Dyadic Clade Centroid Kernels (D = 5,120)
 * ============================================================================ */
__device__ __forceinline__ float bf16_bits_to_f32(uint16_t bits) {
    uint32_t u = ((uint32_t)bits) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ uint16_t f32_to_bf16_bits(float val) {
    return (uint16_t)(__float_as_uint(val) >> 16);
}

/* Normalize N rows of length D=5120 in BF16 in-place on GPU (1 warp = 32 threads per row) */
__global__ void k_l2_normalize_rows_bf16(int32_t N, uint16_t * __restrict__ d_W) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    uint4 *row_v4 = reinterpret_cast<uint4 *>(d_W + (size_t)warp_id * DIM);
    float sum_sq = 0.0f;
    for (int32_t idx = lane; idx < DIM / 8; idx += 32) {
        uint4 v = row_v4[idx];
        uint16_t b[8];
        memcpy(b, &v, 16);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float f = bf16_bits_to_f32(b[k]);
            sum_sq += f * f;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    float inv_norm = rsqrtf(fmaxf(sum_sq, 1e-30f));
    inv_norm = __shfl_sync(0xffffffff, inv_norm, 0);

    for (int32_t idx = lane; idx < DIM / 8; idx += 32) {
        uint4 v = row_v4[idx];
        uint16_t b[8];
        memcpy(b, &v, 16);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float f = bf16_bits_to_f32(b[k]) * inv_norm;
            b[k] = f32_to_bf16_bits(f);
        }
        memcpy(&v, b, 16);
        row_v4[idx] = v;
    }
}

/* Compute cosine similarity <w_base_i, w_ft_i> for each neuron i in [0, N) */
__global__ void k_neuron_self_cosine(
    int32_t N,
    const uint16_t * __restrict__ d_W_base,
    const uint16_t * __restrict__ d_W_ft,
    float * __restrict__ d_cos_out
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    const uint4 *r0 = reinterpret_cast<const uint4 *>(d_W_base + (size_t)warp_id * DIM);
    const uint4 *r1 = reinterpret_cast<const uint4 *>(d_W_ft   + (size_t)warp_id * DIM);
    float dot = 0.0f;
    for (int32_t idx = lane; idx < DIM / 8; idx += 32) {
        uint4 v0 = r0[idx], v1 = r1[idx];
        uint16_t b0[8], b1[8];
        memcpy(b0, &v0, 16);
        memcpy(b1, &v1, 16);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            dot += bf16_bits_to_f32(b0[k]) * bf16_bits_to_f32(b1[k]);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
    if (lane == 0) d_cos_out[warp_id] = dot;
}

/* Warp-cooperative 4-point Quartet Violation Scanner on GPU!
 * Each warp (32 threads) evaluates one internal node u in [N, root_r), loading the 4 representative
 * vectors (l_b0, l_b1, l_s, l_g) in coalesced 128-bit chunks and computing all 6 squared Euclidean
 * distances D(x, y) = 2 * (1 - <w_x, w_y>) simultaneously in registers! */
__global__ void k_detect_quartet_violations_bf16(
    int32_t N, int32_t root_r,
    const int32_t * __restrict__ c1_r,
    const int32_t * __restrict__ c2_r,
    const int32_t * __restrict__ curr_parent,
    const int32_t * __restrict__ node_rep0,
    const uint16_t * __restrict__ d_W,
    uint8_t * __restrict__ d_violated,
    float * __restrict__ d_viol_margin
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t u = N + warp_id;
    if (u >= root_r) return;

    int32_t p = curr_parent[u];
    if (p < N) {
        if (lane == 0) { d_violated[u] = 0; if (d_viol_margin) d_viol_margin[u] = 0.0f; }
        return;
    }

    int32_t b0 = c1_r[u], b1 = c2_r[u];
    int32_t s = (c1_r[p] == u) ? c2_r[p] : c1_r[p];
    int32_t g = -1;
    if (p != root_r) {
        int32_t gp = curr_parent[p];
        g = (c1_r[gp] == p) ? c2_r[gp] : c1_r[gp];
    } else {
        if (s >= N) {
            g = c2_r[s];
            s = c1_r[s];
        } else {
            if (lane == 0) { d_violated[u] = 0; if (d_viol_margin) d_viol_margin[u] = 0.0f; }
            return;
        }
    }

    int32_t l_b0 = node_rep0[b0];
    int32_t l_b1 = node_rep0[b1];
    int32_t l_s  = node_rep0[s];
    int32_t l_g  = node_rep0[g];

    const uint4 *r_b0 = reinterpret_cast<const uint4 *>(d_W + (size_t)l_b0 * DIM);
    const uint4 *r_b1 = reinterpret_cast<const uint4 *>(d_W + (size_t)l_b1 * DIM);
    const uint4 *r_s  = reinterpret_cast<const uint4 *>(d_W + (size_t)l_s  * DIM);
    const uint4 *r_g  = reinterpret_cast<const uint4 *>(d_W + (size_t)l_g  * DIM);

    float dot_b0_b1 = 0.0f, dot_s_g   = 0.0f;
    float dot_b0_s  = 0.0f, dot_b1_g  = 0.0f;
    float dot_b1_s  = 0.0f, dot_b0_g  = 0.0f;

    for (int32_t idx = lane; idx < DIM / 8; idx += 32) {
        uint4 v_b0 = r_b0[idx], v_b1 = r_b1[idx], v_s = r_s[idx], v_g = r_g[idx];
        uint16_t u_b0[8], u_b1[8], u_s[8], u_g[8];
        memcpy(u_b0, &v_b0, 16);
        memcpy(u_b1, &v_b1, 16);
        memcpy(u_s,  &v_s,  16);
        memcpy(u_g,  &v_g,  16);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float fb0 = bf16_bits_to_f32(u_b0[k]);
            float fb1 = bf16_bits_to_f32(u_b1[k]);
            float fs  = bf16_bits_to_f32(u_s[k]);
            float fg  = bf16_bits_to_f32(u_g[k]);
            dot_b0_b1 += fb0 * fb1;
            dot_s_g   += fs  * fg;
            dot_b0_s  += fb0 * fs;
            dot_b1_g  += fb1 * fg;
            dot_b1_s  += fb1 * fs;
            dot_b0_g  += fb0 * fg;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        dot_b0_b1 += __shfl_down_sync(0xffffffff, dot_b0_b1, offset);
        dot_s_g   += __shfl_down_sync(0xffffffff, dot_s_g,   offset);
        dot_b0_s  += __shfl_down_sync(0xffffffff, dot_b0_s,  offset);
        dot_b1_g  += __shfl_down_sync(0xffffffff, dot_b1_g,  offset);
        dot_b1_s  += __shfl_down_sync(0xffffffff, dot_b1_s,  offset);
        dot_b0_g  += __shfl_down_sync(0xffffffff, dot_b0_g,  offset);
    }

    if (lane == 0) {
        float d_b0_b1 = fmaxf(0.0f, 2.0f - 2.0f * dot_b0_b1);
        float d_s_g   = fmaxf(0.0f, 2.0f - 2.0f * dot_s_g);
        float d_b0_s  = fmaxf(0.0f, 2.0f - 2.0f * dot_b0_s);
        float d_b1_g  = fmaxf(0.0f, 2.0f - 2.0f * dot_b1_g);
        float d_b1_s  = fmaxf(0.0f, 2.0f - 2.0f * dot_b1_s);
        float d_b0_g  = fmaxf(0.0f, 2.0f - 2.0f * dot_b0_g);

        float sum_curr = d_b0_b1 + d_s_g;
        float sum_alt1 = d_b0_s + d_b1_g;
        float sum_alt2 = d_b1_s + d_b0_g;
        float best_alt = (sum_alt1 < sum_alt2) ? sum_alt1 : sum_alt2;
        float margin = sum_curr - best_alt;
        d_violated[u] = (margin > 1e-4f) ? 1 : 0;
        if (d_viol_margin) d_viol_margin[u] = (margin > 0.0f) ? margin : 0.0f;
    }
}

/* Build K x D exact dyadic centroid matrix d_centroids[a, 0..DIM-1] on GPU from CSR profile
 * (d_prof_offsets, d_prof_leaves, d_prof_weights):
 *   mu_a = sum_{p in prof(a)} w_p * w_hat_{leaf_p}
 * Each block handles one super-leaf a in [0, K), and threads iterate over d in [0, DIM). */
__global__ void k_build_superleaf_centroids(
    int32_t K,
    const int32_t * __restrict__ d_prof_offsets,
    const int32_t * __restrict__ d_prof_leaves,
    const float   * __restrict__ d_prof_weights,
    const uint16_t * __restrict__ d_W,
    float * __restrict__ d_centroids
) {
    int32_t a = blockIdx.x;
    if (a >= K) return;
    int32_t beg = d_prof_offsets[a];
    int32_t end = d_prof_offsets[a + 1];
    float *out_row = d_centroids + (size_t)a * DIM;

    for (int32_t d8 = threadIdx.x; d8 < DIM / 8; d8 += blockDim.x) {
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int32_t p = beg; p < end; ++p) {
            int32_t leaf = d_prof_leaves[p];
            float w = d_prof_weights[p];
            uint4 v = reinterpret_cast<const uint4 *>(d_W + (size_t)leaf * DIM)[d8];
            uint16_t b[8];
            memcpy(b, &v, 16);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                acc[k] += w * bf16_bits_to_f32(b[k]);
            }
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            out_row[d8 * 8 + k] = acc[k];
        }
    }
}

/* Build K x K condensed Squared Euclidean Distance matrix from super-leaf centroids:
 * By bilinearity of inner product and sum_{p in a} w_p = 1, ||w_hat_p||^2 = 1:
 *   D_cond[a, b] = sum_{p in a} sum_{q in b} w_p w_q ||w_hat_p - w_hat_q||^2
 *                = 2 - 2 * <mu_a, mu_b>
 * One warp (32 threads) computes one pair (a, b) with a < b. */
__global__ void k_build_condensed_D_from_centroids(
    int32_t K,
    const float * __restrict__ d_centroids,
    float * __restrict__ d_D_cond
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t total_pairs = K * K;
    if (warp_id >= total_pairs) return;
    int32_t a = warp_id / K;
    int32_t b = warp_id % K;
    if (a > b) return;
    if (a == b) {
        if (lane == 0) d_D_cond[(size_t)a * K + b] = 0.0f;
        return;
    }
    const float4 *ra = reinterpret_cast<const float4 *>(d_centroids + (size_t)a * DIM);
    const float4 *rb = reinterpret_cast<const float4 *>(d_centroids + (size_t)b * DIM);
    float dot = 0.0f;
    for (int32_t idx = lane; idx < DIM / 4; idx += 32) {
        float4 va = ra[idx], vb = rb[idx];
        dot += va.x * vb.x + va.y * vb.y + va.z * vb.z + va.w * vb.w;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
    if (lane == 0) {
        float d = fmaxf(0.0f, 2.0f - 2.0f * dot);
        d_D_cond[(size_t)a * K + b] = d;
        d_D_cond[(size_t)b * K + a] = d;
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
        int32_t pr = d_perm[row];
        int32_t pc = d_perm[col];
        subdist[(size_t)row * M + col] = (row == col) ? 0.0f : D_orig[(size_t)pr * N + pc];
    }
}

__global__ void k_bottom_up_leaves(
    float * __restrict__ subdist, const int32_t * __restrict__ c1_r,
    const int32_t * __restrict__ c2_r, int32_t N, int32_t M, int32_t root_u
) {
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= N) return;
    float *row_x = &subdist[(size_t)x * M];
    for (int32_t u = N; u < root_u; ++u) {
        row_x[u] = 0.5f * (row_x[c1_r[u]] + row_x[c2_r[u]]);
    }
}

__global__ void k_transpose_leaf_block(float * __restrict__ subdist, int32_t N, int32_t M, int32_t root_u) {
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t u_off = blockIdx.y * blockDim.y + threadIdx.y;
    int32_t u = N + u_off;
    if (x < N && u < root_u) {
        subdist[(size_t)u * M + x] = subdist[(size_t)x * M + u];
    }
}

__global__ void k_bottom_up_internals(
    float * __restrict__ subdist, const int32_t * __restrict__ c1_r,
    const int32_t * __restrict__ c2_r, int32_t N, int32_t M, int32_t root_u
) {
    int32_t u = N + blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= root_u) return;
    float *row_u = &subdist[(size_t)u * M];
    for (int32_t v = N; v < root_u; ++v) {
        row_u[v] = (u == v) ? 0.0f : 0.5f * (row_u[c1_r[v]] + row_u[c2_r[v]]);
    }
}

__global__ void k_top_down_and_root(
    float * __restrict__ subdist, const int32_t * __restrict__ c1_r,
    const int32_t * __restrict__ c2_r, const int32_t * __restrict__ tin_u,
    const int32_t * __restrict__ sub_sz_u, int32_t N, int32_t M, int32_t root_u,
    int32_t r_c0, int32_t r_c1, int32_t r_c2
) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= M) return;
    int32_t tv = tin_u[v];
    if (v != root_u) {
        int32_t c_in = -1, o1 = -1, o2 = -1;
        if (tin_u[r_c0] <= tv && tv < tin_u[r_c0] + sub_sz_u[r_c0]) {
            c_in = r_c0; o1 = r_c1; o2 = r_c2;
        } else if (tin_u[r_c1] <= tv && tv < tin_u[r_c1] + sub_sz_u[r_c1]) {
            c_in = r_c1; o1 = r_c0; o2 = r_c2;
        } else {
            c_in = r_c2; o1 = r_c0; o2 = r_c1;
        }
        float d_out = 0.5f * (subdist[(size_t)o1 * M + v] + subdist[(size_t)o2 * M + v]);
        subdist[(size_t)root_u * M + v] = d_out;
        int32_t curr = c_in;
        while (curr != v && curr >= N) {
            int32_t b0 = c1_r[curr], b1 = c2_r[curr];
            if (tin_u[b0] <= tv && tv < tin_u[b0] + sub_sz_u[b0]) {
                d_out = 0.5f * (d_out + subdist[(size_t)b1 * M + v]);
                subdist[(size_t)curr * M + v] = d_out;
                curr = b0;
            } else {
                d_out = 0.5f * (d_out + subdist[(size_t)b0 * M + v]);
                subdist[(size_t)curr * M + v] = d_out;
                curr = b1;
            }
        }
    }
    for (int32_t u = N; u < root_u; ++u) {
        if (u == v) {
            int32_t b0 = c1_r[u], b1 = c2_r[u];
            subdist[(size_t)u * M + u] = 0.5f * (subdist[(size_t)u * M + b0] + subdist[(size_t)u * M + b1]);
        } else if (!(tin_u[u] <= tv && tv < tin_u[u] + sub_sz_u[u])) {
            int32_t b0 = c1_r[u], b1 = c2_r[u];
            subdist[(size_t)u * M + v] = 0.5f * (subdist[(size_t)b0 * M + v] + subdist[(size_t)b1 * M + v]);
        }
    }
}

__global__ void k_compute_bme_length(
    const float * __restrict__ subdist, const int32_t * __restrict__ parent_u,
    int32_t N, int32_t M, double * __restrict__ d_curr_L
) {
    __shared__ double s_sum[256];
    int32_t tid = threadIdx.x;
    double my_sum = 0.0;
    for (int32_t x = tid; x < N; x += blockDim.x) {
        int32_t p = parent_u[x];
        my_sum += (double)subdist[(size_t)p * M + x];
    }
    int32_t root_u = M - 1;
    for (int32_t u = N + tid; u < root_u; u += blockDim.x) {
        int32_t p = parent_u[u];
        my_sum += 0.5 * ((double)subdist[(size_t)p * M + u] - (double)subdist[(size_t)u * M + u]);
    }
    s_sum[tid] = my_sum;
    __syncthreads();
    for (int32_t s = 128; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) *d_curr_L = s_sum[0];
}

__global__ void k_populate_nd_u(
    const float * __restrict__ subdist, const int32_t * __restrict__ adj_u,
    float * __restrict__ nd_u, int32_t N, int32_t M, int32_t root_u
) {
    int32_t u = N + blockIdx.x * blockDim.x + threadIdx.x;
    if (u > root_u) return;
    for (int32_t k = 0; k < 3; ++k) {
        int32_t nb = adj_u[3 * u + k];
        nd_u[3 * u + k] = (nb >= 0) ? subdist[(size_t)u * M + nb] : 0.0f;
    }
}

struct SprTask { int32_t u, c; float pow_val, dkc, dist; };
struct SprSharedQueue {
    int32_t u[256], c[256];
    float pow_val[256], dkc[256], dist[256];
    int32_t bounds[4];
};

__device__ __forceinline__ int32_t dev_find_nb_idx3(const int32_t * __restrict__ adj_u, int32_t u, int32_t target) {
    int32_t base = 3 * u;
    if (adj_u[base + 0] == target) return 0;
    if (adj_u[base + 1] == target) return 1;
    return 2;
}

__device__ __forceinline__ void dev_eval_cherry_or_enqueue(
    int32_t b_node, int32_t u_parent, int32_t sub_sz_b,
    float next_pow, float d_kc_b, float ndist_b,
    const int32_t * __restrict__ adj_u, const int32_t * __restrict__ parent_u,
    const float * __restrict__ nd_u, const float * __restrict__ row_root,
    const float * __restrict__ row_orig, float * __restrict__ edge_delta,
    SprSharedQueue * __restrict__ sq, int32_t * __restrict__ q_tail
) {
    if (sub_sz_b == 3) {
        int32_t p_idx = dev_find_nb_idx3(adj_u, b_node, u_parent);
        int32_t i0 = (p_idx == 0) ? 1 : 0;
        int32_t i1 = (p_idx == 2) ? 1 : 2;
        int32_t l0 = adj_u[3 * b_node + i0];
        int32_t l1 = adj_u[3 * b_node + i1];
        float r_l0 = row_root[l0], r_l1 = row_root[l1];
        float o_l0 = row_orig[l0], o_l1 = row_orig[l1];
        float c_l0 = nd_u[3 * b_node + i0];
        float c_l1 = nd_u[3 * b_node + i1];
        float pow_leaf = 0.5f * next_pow;
        float delta_base0 = 0.25f * ((r_l0 - o_l0) + (d_kc_b - c_l1));
        float delta_base1 = 0.25f * ((r_l1 - o_l1) + (d_kc_b - c_l0));
        edge_delta[(parent_u[l0] == b_node) ? l0 : b_node] = ndist_b + delta_base0 - pow_leaf * (r_l0 - o_l0);
        edge_delta[(parent_u[l1] == b_node) ? l1 : b_node] = ndist_b + delta_base1 - pow_leaf * (r_l1 - o_l1);
    } else if (*q_tail < 256) {
        int32_t pos = (*q_tail)++;
        sq->u[pos] = b_node; sq->c[pos] = u_parent;
        sq->pow_val[pos] = next_pow; sq->dkc[pos] = d_kc_b; sq->dist[pos] = ndist_b;
    }
}

__device__ __forceinline__ void dev_expand_subtree_stack(
    int32_t start_u, int32_t start_c, float start_pow, float start_dkc, float start_dist,
    int32_t n_leaves, int32_t n_nodes_u, const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u, const int32_t * __restrict__ sub_size,
    const float * __restrict__ nd_u, const float * __restrict__ row_root,
    const float * __restrict__ row_orig, float * __restrict__ edge_delta
) {
    SprTask stack[32];
    int32_t sp = 0;
    stack[sp++] = {start_u, start_c, start_pow, start_dkc, start_dist};
    while (sp > 0) {
        SprTask cur = stack[--sp];
        int32_t u = cur.u, C = cur.c;
        float pow_neg_len = cur.pow_val, d_kc = cur.dkc, curr_dist = cur.dist;
        while (u >= n_leaves) {
            int32_t c_idx = dev_find_nb_idx3(adj_u, u, C);
            int32_t i0 = (c_idx == 0) ? 1 : 0;
            int32_t i1 = (c_idx == 2) ? 1 : 2;
            int32_t b0 = adj_u[3 * u + i0], b1 = adj_u[3 * u + i1];
            float r_b0 = row_root[b0], r_b1 = row_root[b1];
            float o_b0 = row_orig[b0], o_b1 = row_orig[b1];
            float c_b0 = nd_u[3 * u + i0], c_b1 = nd_u[3 * u + i1];
            float next_pow = 0.5f * pow_neg_len;
            float delta_base0 = 0.25f * ((r_b0 - o_b0) + (d_kc - c_b1));
            float delta_base1 = 0.25f * ((r_b1 - o_b1) + (d_kc - c_b0));
            float ndist0 = curr_dist + delta_base0 - next_pow * (r_b0 - o_b0);
            float ndist1 = curr_dist + delta_base1 - next_pow * (r_b1 - o_b1);
            edge_delta[(parent_u[b0] == u) ? b0 : u] = ndist0;
            edge_delta[(parent_u[b1] == u) ? b1 : u] = ndist1;

            if (b0 < n_leaves && b1 < n_leaves) break;
            if (b0 >= n_leaves && b1 < n_leaves) {
                C = u; u = b0; pow_neg_len = next_pow; d_kc = 0.5f * (d_kc + r_b1); curr_dist = ndist0;
            } else if (b1 >= n_leaves && b0 < n_leaves) {
                C = u; u = b1; pow_neg_len = next_pow; d_kc = 0.5f * (d_kc + r_b0); curr_dist = ndist1;
            } else {
                int32_t s0 = sub_size[b0], s1 = sub_size[b1];
                if (s0 < s1) {
                    if (sp < 32) stack[sp++] = {b1, u, next_pow, 0.5f * (d_kc + r_b0), ndist1};
                    C = u; u = b0; pow_neg_len = next_pow; d_kc = 0.5f * (d_kc + r_b1); curr_dist = ndist0;
                } else {
                    if (sp < 32) stack[sp++] = {b0, u, next_pow, 0.5f * (d_kc + r_b1), ndist0};
                    C = u; u = b1; pow_neg_len = next_pow; d_kc = 0.5f * (d_kc + r_b0); curr_dist = ndist1;
                }
            }
        }
    }
}

__device__ __forceinline__ void dev_seed_half_tree_fifo(
    int32_t start_u, int32_t sib_root, int32_t parent, int32_t q_base,
    int32_t n_leaves, int32_t n_nodes_u, const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u, const int32_t * __restrict__ sub_size,
    const float * __restrict__ nd_u, const float * __restrict__ row_root,
    const float * __restrict__ row_orig, float * __restrict__ edge_delta,
    SprSharedQueue * __restrict__ sq, int32_t * __restrict__ out_head, int32_t * __restrict__ out_tail
) {
    int32_t q_head = q_base, q_tail = q_base;
    if (start_u >= n_leaves) {
        sq->u[q_tail] = start_u; sq->c[q_tail] = parent;
        sq->pow_val[q_tail] = 0.5f; sq->dkc[q_tail] = row_root[sib_root]; sq->dist[q_tail] = 0.0f;
        q_tail++;
    }
    while ((q_tail - q_head) > 0 && (q_tail - q_head) < 32 && q_tail < q_base + 96) {
        int32_t level_end = q_tail;
        while (q_head < level_end && q_tail < q_base + 96) {
            int32_t u = sq->u[q_head], C = sq->c[q_head];
            float pow_neg_len = sq->pow_val[q_head], d_kc = sq->dkc[q_head], curr_dist = sq->dist[q_head];
            q_head++;
            while (u >= n_leaves && q_tail < q_base + 96) {
                int32_t c_idx = dev_find_nb_idx3(adj_u, u, C);
                int32_t i0 = (c_idx == 0) ? 1 : 0;
                int32_t i1 = (c_idx == 2) ? 1 : 2;
                int32_t b0 = adj_u[3 * u + i0], b1 = adj_u[3 * u + i1];
                float r_b0 = row_root[b0], r_b1 = row_root[b1];
                float o_b0 = row_orig[b0], o_b1 = row_orig[b1];
                float c_b0 = nd_u[3 * u + i0], c_b1 = nd_u[3 * u + i1];
                float next_pow = 0.5f * pow_neg_len;
                float delta_base0 = 0.25f * ((r_b0 - o_b0) + (d_kc - c_b1));
                float delta_base1 = 0.25f * ((r_b1 - o_b1) + (d_kc - c_b0));
                float ndist0 = curr_dist + delta_base0 - next_pow * (r_b0 - o_b0);
                float ndist1 = curr_dist + delta_base1 - next_pow * (r_b1 - o_b1);
                edge_delta[(parent_u[b0] == u) ? b0 : u] = ndist0;
                edge_delta[(parent_u[b1] == u) ? b1 : u] = ndist1;

                int32_t s0 = (b0 >= n_leaves) ? sub_size[b0] : 1;
                int32_t s1 = (b1 >= n_leaves) ? sub_size[b1] : 1;
                if (s0 > 3 && s1 <= 3) {
                    if (s1 == 3) {
                        dev_eval_cherry_or_enqueue(b1, u, s1, next_pow, 0.5f * (d_kc + r_b0), ndist1,
                                                   adj_u, parent_u, nd_u, row_root, row_orig, edge_delta, sq, &q_tail);
                    }
                    C = u; u = b0; pow_neg_len = next_pow; d_kc = 0.5f * (d_kc + r_b1); curr_dist = ndist0;
                } else if (s1 > 3 && s0 <= 3) {
                    if (s0 == 3) {
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
    *out_head = q_head;
    *out_tail = q_tail;
}

__device__ __forceinline__ void dev_run_parallel_spr_dfs(
    int32_t tree_root, int32_t parent, int32_t other0, int32_t other1,
    int32_t n_leaves, int32_t n_nodes_u, const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u, const int32_t * __restrict__ sub_size,
    const float * __restrict__ nd_u, const float * __restrict__ row_root,
    const float * __restrict__ row_o0, const float * __restrict__ row_o1,
    float * __restrict__ edge_delta, SprSharedQueue * __restrict__ sq
) {
    if (threadIdx.x == 0) {
        edge_delta[(parent_u[tree_root] == parent) ? tree_root : parent] = 0.0f;
        edge_delta[(parent_u[other0]    == parent) ? other0    : parent] = 0.0f;
        edge_delta[(parent_u[other1]    == parent) ? other1    : parent] = 0.0f;
        dev_seed_half_tree_fifo(other0, other1, parent, 0, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                                nd_u, row_root, row_o1, edge_delta, sq, &sq->bounds[0], &sq->bounds[1]);
    } else if (threadIdx.x == 32) {
        dev_seed_half_tree_fifo(other1, other0, parent, 128, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                                nd_u, row_root, row_o0, edge_delta, sq, &sq->bounds[2], &sq->bounds[3]);
    }
    __syncthreads();

    int32_t h0 = sq->bounds[0], t0 = sq->bounds[1];
    int32_t h1 = sq->bounds[2], t1 = sq->bounds[3];
    int32_t n0 = t0 - h0, n1 = t1 - h1, total = n0 + n1;
    for (int32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int32_t q_idx = (idx < n0) ? (h0 + idx) : (h1 + (idx - n0));
        const float *r_orig = (idx < n0) ? row_o1 : row_o0;
        dev_expand_subtree_stack(sq->u[q_idx], sq->c[q_idx], sq->pow_val[q_idx], sq->dkc[q_idx], sq->dist[q_idx],
                                 n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u, row_root, r_orig, edge_delta);
    }
    __syncthreads();
}

__device__ __forceinline__ void dev_map_edge_delta_to_row(
    int32_t i, int32_t n_leaves, int32_t other_root_child,
    const int32_t * __restrict__ m_lmin, const int32_t * __restrict__ next_k_same_lmin,
    const float * __restrict__ edge_delta, float * __restrict__ out_row
) {
    for (int32_t j = threadIdx.x; j <= i; j += blockDim.x) out_row[j] = edge_delta[j];
    __syncthreads();
    int32_t n_active = n_leaves - 2 - i;
    for (int32_t k = threadIdx.x; k < n_active; k += blockDim.x) {
        int32_t lmin = m_lmin[k];
        if (lmin <= i && next_k_same_lmin[k] >= n_active) out_row[lmin] = edge_delta[n_leaves + k];
    }
    for (int32_t j = i + 1 + threadIdx.x; j <= 2 * i - 2; j += blockDim.x) {
        out_row[j] = edge_delta[(2 * n_leaves - 2) - 2 * i + j];
    }
    if (threadIdx.x == 0) {
        if (2 * i - 1 > i) out_row[2 * i - 1] = edge_delta[other_root_child];
        out_row[2 * i] = edge_delta[other_root_child];
    }
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
    float *row_tmp = &row_tmp_pool[(size_t)blockIdx.x * (2 * n_leaves)];

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
        if (threadIdx.x == 0) { best_delta_out[i] = 1e30f; best_j_out[i] = -1; }
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
    dev_map_edge_delta_to_row(i, n_leaves, other_root_child, m_lmin, next_k_same_lmin, edge_delta, row_tmp);
    __syncthreads();

    int32_t vi = (int32_t)v_curr[i];
    float b_val = 0.0f;
    int32_t b_j = vi;
    int32_t max_j = (i == 1) ? 1 : (2 * i - 2);
    for (int32_t j = threadIdx.x; j <= max_j; j += blockDim.x) {
        if (j == vi) continue;
        float val = row_tmp[j];
        if (val < b_val || (val == b_val && j < b_j)) { b_val = val; b_j = j; }
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
 * 4. GPU Workspace & Exact Dyadic Clade-Centroid Skeleton Optimizer
 * ============================================================================ */
struct QwenTreeGPUWorkspace {
    int32_t max_K;
    size_t prof_cap;
    float *d_centroids, *d_D_cond, *d_subdist, *d_nd_u, *d_row_tmp, *d_best_delta, *d_prof_weights;
    int32_t *d_prof_offsets, *d_prof_leaves;
    int32_t *d_perm, *d_c1_r, *d_c2_r, *d_m_lmin, *d_v_map, *d_next_k;
    int32_t *d_parent_u, *d_sub_size, *d_tin, *d_adj_u, *d_best_j;
    int64_t *d_v_curr;
    double *d_curr_L;

    void init(int32_t K_max, size_t max_leaves_cap) {
        max_K = K_max;
        int32_t M_max = 2 * K_max - 2;
        int32_t R_max = 2 * K_max - 1;
        prof_cap = std::max(max_leaves_cap, (size_t)256 * K_max);
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)K_max * DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_prof_offsets, (size_t)(K_max + 1) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_prof_leaves,  prof_cap * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_prof_weights, prof_cap * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_perm, (size_t)K_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_D_cond, (size_t)K_max * K_max * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_subdist, (size_t)M_max * M_max * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nd_u, (size_t)M_max * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_row_tmp, (size_t)K_max * (2 * K_max) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_best_delta, (size_t)K_max * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_best_j, (size_t)K_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_v_curr, (size_t)K_max * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_curr_L, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_c1_r, (size_t)R_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_c2_r, (size_t)R_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_m_lmin, (size_t)K_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_v_map, (size_t)K_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_next_k, (size_t)K_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_parent_u, (size_t)M_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sub_size, (size_t)M_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_tin, (size_t)M_max * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_adj_u, (size_t)M_max * 3 * sizeof(int32_t)));
        CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_top1, cudaFuncAttributeMaxDynamicSharedMemorySize, 90112));
    }

    void destroy() {
        CUDA_CHECK(cudaSetDevice(0));
        cudaFree(d_centroids);
        cudaFree(d_prof_offsets); cudaFree(d_prof_leaves); cudaFree(d_prof_weights);
        cudaFree(d_perm); cudaFree(d_D_cond);
        cudaFree(d_subdist); cudaFree(d_nd_u); cudaFree(d_row_tmp);
        cudaFree(d_best_delta); cudaFree(d_best_j); cudaFree(d_v_curr); cudaFree(d_curr_L);
        cudaFree(d_c1_r); cudaFree(d_c2_r); cudaFree(d_m_lmin);
        cudaFree(d_v_map); cudaFree(d_next_k); cudaFree(d_parent_u);
        cudaFree(d_sub_size); cudaFree(d_tin); cudaFree(d_adj_u);
    }
};

/* Compute exact uncapped dyadic leaf-weight profile w(x) = 2^{-depth_u(x)} for rooted clade u */
static void compute_clade_dyadic_profile_exact(
    int32_t u, int32_t N,
    const int32_t *c1_r, const int32_t *c2_r,
    std::vector<int32_t> &prof_leaves_out,
    std::vector<float> &prof_weights_out
) {
    if (u < N) {
        prof_leaves_out.push_back(u);
        prof_weights_out.push_back(1.0f);
        return;
    }
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
}

static double optimize_skeleton_on_gpu_bf16(
    int32_t K, int32_t max_iters,
    const int32_t *prof_offsets,
    const int32_t *prof_leaves,
    const float *prof_weights,
    const uint16_t *d_W,
    QwenTreeGPUWorkspace &gpu_ws,
    std::vector<int64_t> &v_cf,
    std::vector<int32_t> &perm
) {
    const int32_t M_sk = 2 * K - 2;
    const int32_t R_sk = 2 * K - 1;
    const int32_t root_u_sk = M_sk - 1;
    const std::vector<int64_t> init_v_cf = v_cf;
    const std::vector<int32_t> init_perm = perm;

    const int32_t total_prof = prof_offsets[K];
    CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_offsets, prof_offsets, (size_t)(K + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_leaves,  prof_leaves,  (size_t)total_prof * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_weights, prof_weights, (size_t)total_prof * sizeof(float), cudaMemcpyHostToDevice));

    /* Step 1: Build exact dyadic centroids for all K super-leaves on GPU */
    k_build_superleaf_centroids<<<K, 256>>>(
        K, gpu_ws.d_prof_offsets, gpu_ws.d_prof_leaves, gpu_ws.d_prof_weights,
        d_W, gpu_ws.d_centroids
    );
    /* Step 2: Build exact K x K condensed Squared Euclidean Distance matrix from centroids */
    int32_t total_pairs = K * K;
    k_build_condensed_D_from_centroids<<<(total_pairs + 7) / 8, 256>>>(
        K, gpu_ws.d_centroids, gpu_ws.d_D_cond
    );

    dim3 blk2d(16, 16);
    dim3 grd2d((K + 15) / 16, (K + 15) / 16);

    std::vector<int32_t> c1_cf(R_sk), c2_cf(R_sk), m_lmin_cf(K), v_map_cf(K);
    std::vector<int32_t> next_k_cf(K), last_seen(K), parent_u_cf(M_sk), sub_sz_cf(M_sk), tin_cf(M_sk), adj_u_cf(3 * M_sk);
    std::vector<int32_t> scratch_uf(R_sk);
    std::vector<float> h_best_delta(K);
    std::vector<int32_t> h_best_j(K);

    double init_L = 0.0, curr_L = 0.0, best_L = 1e300;
    std::vector<int64_t> best_v_cf = v_cf;
    std::vector<int32_t> best_perm = perm;
    int32_t flipped_already = 0;
    std::vector<int32_t> path_nodes;
    path_nodes.reserve(64);

    for (int32_t iter = 0; iter <= max_iters; ++iter) {
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
        for (int32_t j = 0; j < K; ++j) last_seen[j] = K;
        for (int32_t k = K - 3; k >= 0; --k) {
            int32_t lmin = m_lmin_cf[k];
            next_k_cf[k] = last_seen[lmin];
            last_seen[lmin] = k;
        }
        int32_t rc1 = c1_cf[R_sk - 1], rc2 = c2_cf[R_sk - 1];
        int32_t other_rc = (rc1 < rc2) ? rc1 : rc2;
        int32_t r_c0 = c1_cf[root_u_sk], r_c1 = c2_cf[root_u_sk], r_c2 = other_rc;

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

        CUDA_CHECK(cudaMemcpy(gpu_ws.d_perm, perm.data(), (size_t)K * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_v_curr, v_cf.data(), (size_t)(K - 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_c1_r, c1_cf.data(), (size_t)R_sk * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_c2_r, c2_cf.data(), (size_t)R_sk * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_m_lmin, m_lmin_cf.data(), (size_t)K * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_v_map, v_map_cf.data(), (size_t)K * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_next_k, next_k_cf.data(), (size_t)K * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_parent_u, parent_u_cf.data(), (size_t)M_sk * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_sub_size, sub_sz_cf.data(), (size_t)M_sk * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_tin, tin_cf.data(), (size_t)M_sk * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_adj_u, adj_u_cf.data(), (size_t)3 * M_sk * sizeof(int32_t), cudaMemcpyHostToDevice));

        k_copy_leaf_rows_perm<<<grd2d, blk2d>>>(gpu_ws.d_D_cond, gpu_ws.d_perm, gpu_ws.d_subdist, K, M_sk);
        k_bottom_up_leaves<<<(K + 255) / 256, 256>>>(gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, K, M_sk, root_u_sk);
        dim3 grd_tr((K + 15) / 16, (K - 2 + 15) / 16);
        k_transpose_leaf_block<<<grd_tr, blk2d>>>(gpu_ws.d_subdist, K, M_sk, root_u_sk);
        k_bottom_up_internals<<<(K - 2 + 255) / 256, 256>>>(gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, K, M_sk, root_u_sk);
        k_top_down_and_root<<<(M_sk + 255) / 256, 256>>>(
            gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, gpu_ws.d_tin, gpu_ws.d_sub_size,
            K, M_sk, root_u_sk, r_c0, r_c1, r_c2
        );
        k_compute_bme_length<<<1, 256>>>(gpu_ws.d_subdist, gpu_ws.d_parent_u, K, M_sk, gpu_ws.d_curr_L);
        CUDA_CHECK(cudaMemcpy(&curr_L, gpu_ws.d_curr_L, sizeof(double), cudaMemcpyDeviceToHost));
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

        k_populate_nd_u<<<(K - 1 + 255) / 256, 256>>>(gpu_ws.d_subdist, gpu_ws.d_adj_u, gpu_ws.d_nd_u, K, M_sk, root_u_sk);
        k_spr_rows_top1<<<K - 2, 128, (size_t)M_sk * sizeof(float)>>>(
            2, K, M_sk, other_rc, gpu_ws.d_v_curr, gpu_ws.d_v_map,
            gpu_ws.d_adj_u, gpu_ws.d_parent_u, gpu_ws.d_sub_size,
            gpu_ws.d_m_lmin, gpu_ws.d_next_k, gpu_ws.d_subdist,
            gpu_ws.d_nd_u, gpu_ws.d_row_tmp,
            gpu_ws.d_best_delta, gpu_ws.d_best_j
        );

        CUDA_CHECK(cudaMemcpy(h_best_delta.data(), gpu_ws.d_best_delta, (size_t)K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_best_j.data(), gpu_ws.d_best_j, (size_t)K * sizeof(int32_t), cudaMemcpyDeviceToHost));

        struct MoveCand { int32_t i, j; float d; };
        std::vector<MoveCand> cands;
        for (int32_t i = 1; i < K - 1; ++i) {
            if (h_best_delta[i] < -1e-4f) {
                cands.push_back({i, h_best_j[i], h_best_delta[i]});
            }
        }
        if (cands.empty()) {
            if (!flipped_already) {
                flipped_already = 1;
                std::vector<int64_t> v_tmp = v_cf;
                std::vector<int32_t> step_perm(K);
                relabel_cherry_first_c(v_tmp.data(), K, 1, v_cf.data(), step_perm.data());
                std::vector<int32_t> new_perm(K);
                for (int32_t a = 0; a < K; ++a) new_perm[a] = perm[step_perm[a]];
                perm = new_perm;
                continue;
            }
            break;
        }
        flipped_already = 0;
        std::sort(cands.begin(), cands.end(), [](const MoveCand &a, const MoveCand &b) {
            return a.d < b.d;
        });

        std::vector<uint8_t> node_busy(R_sk, 0);
        std::vector<uint8_t> leaf_busy(K, 0);
        std::vector<int64_t> v_cand = v_cf;
        int32_t n_applied = 0;
        for (const auto &mc : cands) {
            int32_t i = mc.i, j = mc.j;
            int32_t tr = v_map_cf[i + 1];
            int32_t pr = (tr == root_u_sk) ? other_rc : parent_u_cf[tr];
            int32_t dst = (j <= i) ? j : ((2 * K - 2) - 2 * i + j);
            if (dst < 0 || dst >= R_sk) continue;
            int32_t dst_u = (dst >= root_u_sk) ? other_rc : dst;

            int32_t lmin_old = (int32_t)v_cf[i];
            int32_t lmin_tr  = i + 1;
            int32_t lmin_new = (dst < K) ? dst : m_lmin_cf[dst - K];
            if (leaf_busy[lmin_old] || leaf_busy[lmin_tr] || leaf_busy[lmin_new]) continue;

            path_nodes.clear();
            path_nodes.push_back(tr);
            int32_t u_anc = pr;
            int32_t t_dst = tin_cf[dst_u];
            int32_t hops = 0;
            while (u_anc != -1 && !(tin_cf[u_anc] <= t_dst && t_dst < tin_cf[u_anc] + sub_sz_cf[u_anc])) {
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
            leaf_busy[lmin_old] = 1;
            leaf_busy[lmin_tr]  = 1;
            leaf_busy[lmin_new] = 1;
            v_cand[i] = j;
            n_applied++;
            if (n_applied >= 16) break;
        }
        if (n_applied == 0) {
            v_cand[cands[0].i] = cands[0].j;
        }

        std::vector<int32_t> step_perm(K);
        relabel_cherry_first_c(v_cand.data(), K, iter & 1, v_cf.data(), step_perm.data());
        std::vector<int32_t> new_perm(K);
        for (int32_t a = 0; a < K; ++a) new_perm[a] = perm[step_perm[a]];
        perm = new_perm;
    }

    if (best_L >= init_L - 1e-6) {
        v_cf = init_v_cf;
        perm = init_perm;
        return 0.0;
    }
    v_cf = best_v_cf;
    perm = best_perm;
    return best_L - init_L;
}

/* ============================================================================
 * 5. Active Skeleton Extraction & Blind Quartet Discovery
 * ============================================================================ */
struct BlindCladeDesc {
    int32_t rep_leaf;
    int32_t target_leaves;
    int32_t viol_count;
    float   sum_margin;
};

static int32_t resolve_clade_root_by_leaf(
    int32_t rep_leaf, int32_t target_leaves,
    const std::vector<int32_t> &curr_parent,
    const std::vector<int32_t> &leaf_cnt
) {
    int32_t u = rep_leaf;
    while (curr_parent[u] != -1 && leaf_cnt[curr_parent[u]] <= target_leaves) {
        u = curr_parent[u];
    }
    return u;
}

static void extract_active_skeleton(
    int32_t root_r,
    const int32_t *c1_r, const int32_t *c2_r, int32_t N,
    int32_t condense_thresh,
    const std::vector<uint8_t> &node_has_active,
    const std::vector<int32_t> &leaf_cnt,
    std::vector<int32_t> &super_roots,
    std::vector<int32_t> &split_internals,
    std::vector<int32_t> &skel_left,
    std::vector<int32_t> &skel_right,
    int32_t *out_skel_id
) {
    struct Frame { int32_t u; int32_t state; int32_t left_id; };
    std::vector<Frame> st;
    st.reserve(256);
    st.push_back({root_r, 0, -1});
    int32_t last_ret = -1;

    while (!st.empty()) {
        Frame &f = st.back();
        int32_t u = f.u;
        if (f.state == 0) {
            if (u < N || (!node_has_active[u] && leaf_cnt[u] <= condense_thresh)) {
                int32_t sid = (int32_t)super_roots.size();
                super_roots.push_back(u);
                last_ret = sid;
                st.pop_back();
            } else {
                f.state = 1;
                st.push_back({c1_r[u], 0, -1});
            }
        } else if (f.state == 1) {
            f.left_id = last_ret;
            f.state = 2;
            st.push_back({c2_r[u], 0, -1});
        } else {
            int32_t right_id = last_ret;
            int32_t my_idx = (int32_t)split_internals.size();
            split_internals.push_back(u);
            skel_left.push_back(f.left_id);
            skel_right.push_back(right_id);
            last_ret = -1 - my_idx;
            st.pop_back();
        }
    }
    *out_skel_id = -1 - last_ret;
}

static int32_t scan_quartet_violations_gpu(
    int32_t N, int32_t max_clade_leaves,
    const int32_t *c1_r, const int32_t *c2_r,
    const std::vector<int32_t> &ord,
    const std::vector<int32_t> &curr_parent,
    const std::vector<int32_t> &leaf_cnt,
    const std::vector<int32_t> &node_rep0,
    const uint16_t *d_W,
    std::vector<BlindCladeDesc> &corrupted_clades_out,
    double *out_total_margin = NULL
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;

    int32_t *d_c1 = NULL, *d_c2 = NULL, *d_par = NULL, *d_rep0 = NULL;
    uint8_t *d_viol = NULL;
    float *d_margin = NULL;
    CUDA_CHECK(cudaMalloc(&d_c1,     (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_c2,     (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_par,    (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_rep0,   (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_viol,   (size_t)n_nodes_r * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_margin, (size_t)n_nodes_r * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_c1,   c1_r,               (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c2,   c2_r,               (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_par,  curr_parent.data(), (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rep0, node_rep0.data(),   (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_viol,   0, (size_t)n_nodes_r * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_margin, 0, (size_t)n_nodes_r * sizeof(float)));

    /* 1 warp (32 threads) per internal node -> 8 internal nodes per 256-thread block */
    int32_t n_int = N - 1;
    k_detect_quartet_violations_bf16<<<(n_int + 7) / 8, 256>>>(
        N, root_r, d_c1, d_c2, d_par, d_rep0, d_W, d_viol, d_margin
    );

    std::vector<uint8_t> h_viol(n_nodes_r, 0);
    std::vector<float> h_margin(n_nodes_r, 0.0f);
    CUDA_CHECK(cudaMemcpy(h_viol.data(),   d_viol,   (size_t)n_nodes_r * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_margin.data(), d_margin, (size_t)n_nodes_r * sizeof(float),   cudaMemcpyDeviceToHost));
    cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_par); cudaFree(d_rep0); cudaFree(d_viol); cudaFree(d_margin);

    int32_t total_viol = 0;
    double sum_m = 0.0;
    for (int32_t u = N; u < root_r; ++u) {
        total_viol += (int32_t)h_viol[u];
        sum_m += (double)h_margin[u];
    }
    if (out_total_margin) *out_total_margin = sum_m;

    std::vector<int32_t> sub_viol(n_nodes_r, 0);
    std::vector<float> sub_margin(n_nodes_r, 0.0f);
    for (int32_t idx = (int32_t)ord.size() - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        sub_viol[u]   = (int32_t)h_viol[u] + sub_viol[b0] + sub_viol[b1];
        sub_margin[u] = h_margin[u] + sub_margin[b0] + sub_margin[b1];
    }

    corrupted_clades_out.clear();
    std::vector<int32_t> st = {root_r};
    while (!st.empty()) {
        int32_t u = st.back(); st.pop_back();
        if (u < N || sub_viol[u] == 0) continue;
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        bool has_clean_sister = (sub_viol[b0] == 0 && leaf_cnt[b0] >= 50) ||
                                (sub_viol[b1] == 0 && leaf_cnt[b1] >= 50);
        if (!has_clean_sister && leaf_cnt[u] <= max_clade_leaves && sub_viol[u] >= 2) {
            corrupted_clades_out.push_back({node_rep0[u], leaf_cnt[u], sub_viol[u], sub_margin[u]});
        } else {
            st.push_back(b1);
            st.push_back(b0);
        }
    }
    /* Sort corrupted clades by total violation margin descending so the most heavily rewired clades are optimized first */
    std::sort(corrupted_clades_out.begin(), corrupted_clades_out.end(),
              [](const BlindCladeDesc &a, const BlindCladeDesc &b) {
                  return a.sum_margin > b.sum_margin;
              });
    return total_viol;
}

/* ============================================================================
 * 6. Fast Hierarchical FastME + GPU BME Base Tree Builder
 * ============================================================================ */
/* Project C_sz leaves onto direction vector d_dir[0..DIM-1] on GPU (1 warp per leaf) */
__global__ void k_project_leaves_bf16(
    int32_t C_sz,
    const int32_t * __restrict__ d_leaves,
    const float * __restrict__ d_dir,
    const uint16_t * __restrict__ d_W,
    float * __restrict__ d_scores
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    if (warp_id >= C_sz) return;
    int32_t leaf = d_leaves[warp_id];
    const uint4 *r_w = reinterpret_cast<const uint4 *>(d_W + (size_t)leaf * DIM);
    float dot = 0.0f;
    for (int32_t idx = lane; idx < DIM / 8; idx += 32) {
        uint4 v = r_w[idx];
        uint16_t b[8];
        memcpy(b, &v, 16);
        int32_t base_d = idx * 8;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            dot += bf16_bits_to_f32(b[k]) * d_dir[base_d + k];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
    }
    if (lane == 0) d_scores[warp_id] = dot;
}

/* Recursively partitions `leaves` into clusters of size <= max_cluster_sz using
 * 2-pivot FastMap cosine projection, solves each leaf cluster with exact NJ + GPU BME,
 * and joins clusters via NJ + GPU BME! */
static int32_t build_initial_bme_subtree(
    const std::vector<int32_t> &leaves,
    int32_t N_total,
    const uint16_t *d_W,
    QwenTreeGPUWorkspace &gpu_ws,
    std::vector<int32_t> &c1_r,
    std::vector<int32_t> &c2_r,
    int32_t &next_internal_id,
    uint64_t *rng
) {
    int32_t M = (int32_t)leaves.size();
    if (M < 4) {
        int32_t cur = leaves[0];
        for (int32_t c = 1; c < M; ++c) {
            int32_t u = next_internal_id++;
            c1_r[u] = cur;
            c2_r[u] = leaves[c];
            cur = u;
        }
        return cur;
    }

    const int32_t max_fastme = 256;
    if (M <= max_fastme) {
        std::vector<int32_t> prof_ofs(M + 1);
        std::vector<float> prof_w(M, 1.0f);
        for (int32_t i = 0; i <= M; ++i) prof_ofs[i] = i;

        CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_offsets, prof_ofs.data(), (size_t)(M + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_leaves,  leaves.data(),   (size_t)M * sizeof(int32_t),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_weights, prof_w.data(),   (size_t)M * sizeof(float),         cudaMemcpyHostToDevice));

        k_build_superleaf_centroids<<<M, 256>>>(
            M, gpu_ws.d_prof_offsets, gpu_ws.d_prof_leaves, gpu_ws.d_prof_weights, d_W, gpu_ws.d_centroids
        );
        k_build_condensed_D_from_centroids<<<(M * M + 7) / 8, 256>>>(
            M, gpu_ws.d_centroids, gpu_ws.d_D_cond
        );
        std::vector<float> h_D((size_t)M * M);
        CUDA_CHECK(cudaMemcpy(h_D.data(), gpu_ws.d_D_cond, (size_t)M * M * sizeof(float), cudaMemcpyDeviceToHost));

        std::vector<int64_t> v_loc(M - 1), v_cf(M - 1);
        std::vector<int32_t> perm(M), loc_c1(2 * M - 1), loc_c2(2 * M - 1), uf(2 * M - 1);
        build_nj_rooted_from_dist(h_D.data(), M, loc_c1.data(), loc_c2.data());
        encode_rooted_to_v_c(loc_c1.data(), loc_c2.data(), M, v_loc.data());
        relabel_cherry_first_c(v_loc.data(), M, 0, v_cf.data(), perm.data());
        if (M >= 6) {
            optimize_skeleton_on_gpu_bf16(
                M, 20, prof_ofs.data(), leaves.data(), prof_w.data(),
                d_W, gpu_ws, v_cf, perm
            );
        }
        decode_cf_to_rooted(v_cf.data(), M, loc_c1.data(), loc_c2.data(), uf.data());

        std::vector<int32_t> map_node(2 * M - 1);
        for (int32_t a = 0; a < M; ++a) map_node[a] = leaves[perm[a]];
        for (int32_t s = 0; s < M - 1; ++s) {
            int32_t u = next_internal_id++;
            map_node[M + s] = u;
            c1_r[u] = map_node[loc_c1[M + s]];
            c2_r[u] = map_node[loc_c2[M + s]];
        }
        return map_node[2 * M - 2];
    }

    /* For M > max_fastme, split into K_sub (4 <= K_sub <= 64) balanced sub-clusters via recursive 2-pivot projection on GPU */
    int32_t target_sub = std::max(max_fastme, (M + 31) / 32);
    if (target_sub > (M + 3) / 4) target_sub = std::max(1, (M + 3) / 4);
    std::vector<std::vector<int32_t>> clusters = {leaves};

    while (true) {
        bool any_split = false;
        std::vector<std::vector<int32_t>> next_clusters;
        for (auto &cl : clusters) {
            if ((int32_t)cl.size() <= target_sub) {
                next_clusters.push_back(std::move(cl));
                continue;
            }
            any_split = true;
            int32_t C_sz = (int32_t)cl.size();
            int32_t p0 = cl[splitmix64(rng) % C_sz];
            int32_t p1 = cl[splitmix64(rng) % C_sz];
            if (p0 == p1) p1 = cl[(splitmix64(rng) + 1) % C_sz];

            uint16_t h_p0[DIM], h_p1[DIM];
            CUDA_CHECK(cudaMemcpy(h_p0, d_W + (size_t)p0 * DIM, DIM * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_p1, d_W + (size_t)p1 * DIM, DIM * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            float dir[DIM];
            for (int d = 0; d < DIM; ++d) {
                uint32_t u0 = ((uint32_t)h_p0[d]) << 16;
                uint32_t u1 = ((uint32_t)h_p1[d]) << 16;
                float f0, f1;
                memcpy(&f0, &u0, 4);
                memcpy(&f1, &u1, 4);
                dir[d] = f0 - f1;
            }
            CUDA_CHECK(cudaMemcpy(gpu_ws.d_centroids, dir, DIM * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_leaves, cl.data(), (size_t)C_sz * sizeof(int32_t), cudaMemcpyHostToDevice));
            k_project_leaves_bf16<<<(C_sz + 7) / 8, 256>>>(
                C_sz, gpu_ws.d_prof_leaves, gpu_ws.d_centroids, d_W, gpu_ws.d_prof_weights
            );
            std::vector<float> scores(C_sz);
            CUDA_CHECK(cudaMemcpy(scores.data(), gpu_ws.d_prof_weights, (size_t)C_sz * sizeof(float), cudaMemcpyDeviceToHost));

            std::vector<std::pair<float, int32_t>> proj(C_sz);
            for (int32_t i = 0; i < C_sz; ++i) proj[i] = {scores[i], cl[i]};

            int32_t mid = C_sz / 2;
            std::nth_element(proj.begin(), proj.begin() + mid, proj.end());
            std::vector<int32_t> left_cl(mid), right_cl(C_sz - mid);
            for (int32_t i = 0; i < mid; ++i) left_cl[i] = proj[i].second;
            for (int32_t i = mid; i < C_sz; ++i) right_cl[i - mid] = proj[i].second;
            next_clusters.push_back(std::move(left_cl));
            next_clusters.push_back(std::move(right_cl));
        }
        clusters = std::move(next_clusters);
        if (!any_split) break;
    }

    /* Recursively build each sub-cluster tree */
    int32_t K_c = (int32_t)clusters.size();
    std::vector<int32_t> sub_roots(K_c);
    for (int32_t c = 0; c < K_c; ++c) {
        sub_roots[c] = build_initial_bme_subtree(clusters[c], N_total, d_W, gpu_ws, c1_r, c2_r, next_internal_id, rng);
    }
    if (K_c < 4) {
        int32_t cur = sub_roots[0];
        for (int32_t c = 1; c < K_c; ++c) {
            int32_t u = next_internal_id++;
            c1_r[u] = cur;
            c2_r[u] = sub_roots[c];
            cur = u;
        }
        return cur;
    }

    /* Join the K_c sub-cluster roots using exact FastME + GPU BME on their dyadic profiles! */
    std::vector<int32_t> prof_offsets(K_c + 1, 0);
    std::vector<int32_t> prof_leaves;
    std::vector<float> prof_weights;
    for (int32_t a = 0; a < K_c; ++a) {
        prof_offsets[a] = (int32_t)prof_leaves.size();
        compute_clade_dyadic_profile_exact(sub_roots[a], N_total, c1_r.data(), c2_r.data(), prof_leaves, prof_weights);
    }
    prof_offsets[K_c] = (int32_t)prof_leaves.size();

    CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_offsets, prof_offsets.data(), (size_t)(K_c + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_leaves,  prof_leaves.data(),  prof_leaves.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ws.d_prof_weights, prof_weights.data(), prof_weights.size() * sizeof(float),  cudaMemcpyHostToDevice));
    k_build_superleaf_centroids<<<K_c, 256>>>(
        K_c, gpu_ws.d_prof_offsets, gpu_ws.d_prof_leaves, gpu_ws.d_prof_weights, d_W, gpu_ws.d_centroids
    );
    k_build_condensed_D_from_centroids<<<(K_c * K_c + 7) / 8, 256>>>(
        K_c, gpu_ws.d_centroids, gpu_ws.d_D_cond
    );
    std::vector<float> h_D((size_t)K_c * K_c);
    CUDA_CHECK(cudaMemcpy(h_D.data(), gpu_ws.d_D_cond, (size_t)K_c * K_c * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<int64_t> v_loc(K_c - 1), v_cf(K_c - 1);
    std::vector<int32_t> perm(K_c), loc_c1(2 * K_c - 1), loc_c2(2 * K_c - 1), uf(2 * K_c - 1);
    build_nj_rooted_from_dist(h_D.data(), K_c, loc_c1.data(), loc_c2.data());
    encode_rooted_to_v_c(loc_c1.data(), loc_c2.data(), K_c, v_loc.data());
    relabel_cherry_first_c(v_loc.data(), K_c, 0, v_cf.data(), perm.data());

    if (K_c >= 6) {
        optimize_skeleton_on_gpu_bf16(
            K_c, 40, prof_offsets.data(), prof_leaves.data(), prof_weights.data(),
            d_W, gpu_ws, v_cf, perm
        );
    }
    decode_cf_to_rooted(v_cf.data(), K_c, loc_c1.data(), loc_c2.data(), uf.data());

    std::vector<int32_t> map_node(2 * K_c - 1);
    for (int32_t a = 0; a < K_c; ++a) map_node[a] = sub_roots[perm[a]];
    for (int32_t s = 0; s < K_c - 1; ++s) {
        int32_t u = next_internal_id++;
        map_node[K_c + s] = u;
        c1_r[u] = map_node[loc_c1[K_c + s]];
        c2_r[u] = map_node[loc_c2[K_c + s]];
    }
    return map_node[2 * K_c - 2];
}

/* Refine tree (curr_c1, curr_c2) on weight matrix d_W using Clade-Condensed Minibatch BME */
static double refine_tree_minibatch_on_gpu(
    int32_t N, int32_t max_windows,
    const uint16_t *d_W,
    QwenTreeGPUWorkspace &gpu_ws,
    std::vector<int32_t> &curr_c1,
    std::vector<int32_t> &curr_c2,
    int32_t *out_init_viol = NULL,
    int32_t *out_final_viol = NULL,
    int32_t *out_windows_improved = NULL
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    std::vector<int32_t> ord, curr_parent, leaf_cnt, node_rep0;
    compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, curr_parent, leaf_cnt, node_rep0);

    std::vector<BlindCladeDesc> active_clades;
    int32_t init_viol = scan_quartet_violations_gpu(
        N, 1100, curr_c1.data(), curr_c2.data(),
        ord, curr_parent, leaf_cnt, node_rep0, d_W, active_clades
    );
    if (out_init_viol) *out_init_viol = init_viol;

    int32_t n_win = std::min((int32_t)active_clades.size(), max_windows);
    double total_delta_L = 0.0;
    int32_t wins_improved = 0;
    std::vector<uint8_t> node_has_active(n_nodes_r, 0);
    std::vector<int32_t> scratch_uf(n_nodes_r);

    for (int32_t w = 0; w < n_win; ++w) {
        const auto &cd = active_clades[w];
        int32_t u_curr = resolve_clade_root_by_leaf(cd.rep_leaf, cd.target_leaves, curr_parent, leaf_cnt);
        std::fill(node_has_active.begin(), node_has_active.end(), 0);
        std::vector<int32_t> st = {u_curr};
        while (!st.empty()) {
            int32_t u = st.back(); st.pop_back();
            node_has_active[u] = 1;
            if (u >= N) {
                st.push_back(curr_c1[u]);
                st.push_back(curr_c2[u]);
            }
        }
        int32_t p = curr_parent[u_curr];
        while (p != -1) {
            node_has_active[p] = 1;
            p = curr_parent[p];
        }

        int32_t condense_thresh = std::max(1200, N / 80);
        std::vector<int32_t> super_roots, split_internals, skel_left, skel_right;
        int32_t root_skel_id = -1;
        while (true) {
            super_roots.clear();
            split_internals.clear();
            skel_left.clear();
            skel_right.clear();
            extract_active_skeleton(
                root_r, curr_c1.data(), curr_c2.data(), N,
                condense_thresh, node_has_active, leaf_cnt,
                super_roots, split_internals, skel_left, skel_right, &root_skel_id
            );
            if ((int32_t)super_roots.size() <= gpu_ws.max_K || condense_thresh >= N) break;
            condense_thresh = (condense_thresh * 3) / 2;
        }

        int32_t K = (int32_t)super_roots.size();
        if (K > gpu_ws.max_K || K < 6) continue;

        std::vector<int32_t> sk_c1(2 * K - 1), sk_c2(2 * K - 1);
        for (int32_t s = 0; s < K - 1; ++s) {
            int32_t l_raw = skel_left[s];
            int32_t r_raw = skel_right[s];
            sk_c1[K + s] = (l_raw >= 0) ? l_raw : (K + (-1 - l_raw));
            sk_c2[K + s] = (r_raw >= 0) ? r_raw : (K + (-1 - r_raw));
        }

        std::vector<int32_t> prof_offsets(K + 1, 0);
        std::vector<int32_t> prof_leaves;
        std::vector<float> prof_weights;
        prof_leaves.reserve((size_t)N);
        prof_weights.reserve((size_t)N);
        for (int32_t a = 0; a < K; ++a) {
            prof_offsets[a] = (int32_t)prof_leaves.size();
            compute_clade_dyadic_profile_exact(
                super_roots[a], N, curr_c1.data(), curr_c2.data(), prof_leaves, prof_weights
            );
        }
        prof_offsets[K] = (int32_t)prof_leaves.size();

        std::vector<int64_t> v_sk(K - 1), v_cf(K - 1);
        std::vector<int32_t> perm(K);
        encode_rooted_to_v_c(sk_c1.data(), sk_c2.data(), K, v_sk.data());
        relabel_cherry_first_c(v_sk.data(), K, 0, v_cf.data(), perm.data());

        double dL = optimize_skeleton_on_gpu_bf16(
            K, 60, prof_offsets.data(), prof_leaves.data(), prof_weights.data(),
            d_W, gpu_ws, v_cf, perm
        );
        if (dL == 0.0) continue;
        total_delta_L += dL;
        wins_improved++;

        int32_t R_sk = 2 * K - 1;
        std::vector<int32_t> new_sk_c1(R_sk), new_sk_c2(R_sk);
        decode_cf_to_rooted(v_cf.data(), K, new_sk_c1.data(), new_sk_c2.data(), scratch_uf.data());

        std::vector<int32_t> sk_to_full(R_sk);
        for (int32_t a = 0; a < K; ++a) sk_to_full[a] = super_roots[perm[a]];
        for (int32_t s = 0; s < K - 1; ++s) sk_to_full[K + s] = split_internals[s];
        for (int32_t s = 0; s < K - 1; ++s) {
            int32_t full_u = sk_to_full[K + s];
            curr_c1[full_u] = sk_to_full[new_sk_c1[K + s]];
            curr_c2[full_u] = sk_to_full[new_sk_c2[K + s]];
        }
        compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, curr_parent, leaf_cnt, node_rep0);
    }

    if (out_final_viol) {
        std::vector<BlindCladeDesc> post_clades;
        *out_final_viol = scan_quartet_violations_gpu(
            N, 1100, curr_c1.data(), curr_c2.data(),
            ord, curr_parent, leaf_cnt, node_rep0, d_W, post_clades
        );
    }
    if (out_windows_improved) *out_windows_improved = wins_improved;
    return total_delta_L;
}

/* ============================================================================
 * 7. Main Driver: Run Per-Layer (48 x 13,824), Vocabulary (152,064), & Full MLP (663,552)
 * ============================================================================ */
static void load_bf16_slice_to_gpu(
    const char *path, size_t row_offset, int32_t N, uint16_t *d_W
) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open %s\n", path);
        exit(EXIT_FAILURE);
    }
    size_t byte_off = row_offset * (size_t)DIM * sizeof(uint16_t);
    size_t n_bytes  = (size_t)N * (size_t)DIM * sizeof(uint16_t);
    if (fseek(fp, (long)byte_off, SEEK_SET) != 0) {
        fprintf(stderr, "fseek failed on %s\n", path);
        exit(EXIT_FAILURE);
    }
    std::vector<uint16_t> buf((size_t)N * DIM);
    if (fread(buf.data(), 1, n_bytes, fp) != n_bytes) {
        fprintf(stderr, "fread failed on %s\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    CUDA_CHECK(cudaMemcpy(d_W, buf.data(), n_bytes, cudaMemcpyHostToDevice));
    /* L2-normalize each row in-place on GPU */
    int32_t threads = 256;
    int32_t warps_per_blk = threads / 32;
    int32_t blocks = (N + warps_per_blk - 1) / warps_per_blk;
    k_l2_normalize_rows_bf16<<<blocks, threads>>>(N, d_W);
    CUDA_CHECK(cudaDeviceSynchronize());
}


static void compute_raw_quartet_arrays_gpu(
    int32_t N,
    const int32_t *c1_r, const int32_t *c2_r,
    const std::vector<int32_t> &curr_parent,
    const std::vector<int32_t> &node_rep0,
    const uint16_t *d_W,
    std::vector<uint8_t> &h_viol,
    std::vector<float> &h_margin
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;
    int32_t *d_c1 = NULL, *d_c2 = NULL, *d_par = NULL, *d_rep0 = NULL;
    uint8_t *d_viol = NULL;
    float *d_margin = NULL;
    CUDA_CHECK(cudaMalloc(&d_c1,     (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_c2,     (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_par,    (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_rep0,   (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_viol,   (size_t)n_nodes_r * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_margin, (size_t)n_nodes_r * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_c1,   c1_r,               (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c2,   c2_r,               (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_par,  curr_parent.data(), (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rep0, node_rep0.data(),   (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_viol,   0, (size_t)n_nodes_r * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(d_margin, 0, (size_t)n_nodes_r * sizeof(float)));

    int32_t n_int = N - 1;
    k_detect_quartet_violations_bf16<<<(n_int + 7) / 8, 256>>>(
        N, root_r, d_c1, d_c2, d_par, d_rep0, d_W, d_viol, d_margin
    );
    h_viol.resize(n_nodes_r);
    h_margin.resize(n_nodes_r);
    CUDA_CHECK(cudaMemcpy(h_viol.data(),   d_viol,   (size_t)n_nodes_r * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_margin.data(), d_margin, (size_t)n_nodes_r * sizeof(float),   cudaMemcpyDeviceToHost));
    cudaFree(d_c1); cudaFree(d_c2); cudaFree(d_par); cudaFree(d_rep0); cudaFree(d_viol); cudaFree(d_margin);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    const char *base_mlp  = "/tmp/qwen_tree_exp/base_mlp_48L.bf16";
    const char *coder_mlp = "/tmp/qwen_tree_exp/coder_mlp_48L.bf16";
    const char *r1_mlp    = "/tmp/qwen_tree_exp/r1_mlp_48L.bf16";
    const char *out_bin   = "/tmp/qwen_tree_exp/clade_scores_48L.bin";

    const int32_t N_LAYERS = 48;
    const int32_t N = 13824;
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;

    CUDA_CHECK(cudaSetDevice(0));
    QwenTreeGPUWorkspace gpu_ws;
    gpu_ws.init(3200, 32768);

    uint16_t *d_W_base = NULL, *d_W_target = NULL;
    float *d_cos_buf = NULL;
    CUDA_CHECK(cudaMalloc(&d_W_base,   (size_t)N * DIM * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_W_target, (size_t)N * DIM * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_cos_buf,  (size_t)N * sizeof(float)));

    /* Output arrays across all 48 x 13824 neurons:
     * 1. clade_id[48 * N]          (int32)
     * 2. coder_clade_score[48 * N] (float)
     * 3. r1_clade_score[48 * N]    (float)
     * 4. coder_spr_frac[48 * N]    (float)
     * 5. r1_spr_frac[48 * N]       (float)
     * 6. coder_self_cos[48 * N]    (float)
     * 7. r1_self_cos[48 * N]       (float)
     */
    const size_t TOTAL = (size_t)N_LAYERS * N;
    std::vector<int32_t> all_clade_id(TOTAL, 0);
    std::vector<float> all_coder_score(TOTAL, 0.0f);
    std::vector<float> all_r1_score(TOTAL, 0.0f);
    std::vector<float> all_coder_spr(TOTAL, 0.0f);
    std::vector<float> all_r1_spr(TOTAL, 0.0f);
    std::vector<float> all_coder_cos(TOTAL, 0.0f);
    std::vector<float> all_r1_cos(TOTAL, 0.0f);

    printf("Building 48-Layer Qwen2.5-14B MLP Base Trees & Computing Clade Rewiring Scores...\n");
    fflush(stdout);

    for (int32_t l = 0; l < N_LAYERS; ++l) {
        double t0 = wall_sec();
        size_t row_off = (size_t)l * N;
        load_bf16_slice_to_gpu(base_mlp, row_off, N, d_W_base);

        std::vector<int32_t> base_c1(n_nodes_r), base_c2(n_nodes_r);
        std::vector<int32_t> all_leaves(N);
        std::iota(all_leaves.begin(), all_leaves.end(), 0);
        int32_t next_id = N;
        uint64_t rng = 20261103ULL + (uint64_t)N + (uint64_t)row_off;

        build_initial_bme_subtree(all_leaves, N, d_W_base, gpu_ws, base_c1, base_c2, next_id, &rng);
        int32_t base_v0 = 0, base_v1 = 0, base_wins = 0;
        refine_tree_minibatch_on_gpu(N, 20, d_W_base, gpu_ws, base_c1, base_c2, &base_v0, &base_v1, &base_wins);

        std::vector<int32_t> ord, par, lcnt, rep0;
        compute_tree_summaries_blind(base_c1.data(), base_c2.data(), N, ord, par, lcnt, rep0);

        /* Partition T_Base^(l) into monophyletic clades of size <= 192 */
        std::vector<int32_t> clade_roots;
        std::vector<int32_t> st = {root_r};
        while (!st.empty()) {
            int32_t u = st.back(); st.pop_back();
            if (u < N || lcnt[u] <= 192) {
                clade_roots.push_back(u);
            } else {
                st.push_back(base_c2[u]);
                st.push_back(base_c1[u]);
            }
        }
        for (int32_t cid = 0; cid < (int32_t)clade_roots.size(); ++cid) {
            std::vector<int32_t> cst = {clade_roots[cid]};
            while (!cst.empty()) {
                int32_t u = cst.back(); cst.pop_back();
                if (u < N) all_clade_id[row_off + u] = cid;
                else { cst.push_back(base_c1[u]); cst.push_back(base_c2[u]); }
            }
        }

        /* Compute Base quartet violation array */
        std::vector<uint8_t> v_base_arr;
        std::vector<float> m_base_arr;
        compute_raw_quartet_arrays_gpu(N, base_c1.data(), base_c2.data(), par, rep0, d_W_base, v_base_arr, m_base_arr);

        /* Compute XOR split hashes for T_Base^(l) */
        std::vector<uint64_t> leaf_hashes(N);
        uint64_t xor_all = 0;
        for (int32_t x = 1; x < N; ++x) {
            leaf_hashes[x] = splitmix64(&rng);
            xor_all ^= leaf_hashes[x];
        }
        leaf_hashes[0] = xor_all;
        std::vector<uint64_t> node_hash_base(n_nodes_r);
        for (int32_t x = 0; x < N; ++x) node_hash_base[x] = leaf_hashes[x];
        for (int32_t idx = (int32_t)ord.size() - 1; idx >= 0; --idx) {
            int32_t u = ord[idx];
            node_hash_base[u] = node_hash_base[base_c1[u]] ^ node_hash_base[base_c2[u]];
        }

        const char *t_paths[2] = {coder_mlp, r1_mlp};
        for (int m = 0; m < 2; ++m) {
            load_bf16_slice_to_gpu(t_paths[m], row_off, N, d_W_target);
            k_neuron_self_cosine<<<(N + 7) / 8, 256>>>(N, d_W_base, d_W_target, d_cos_buf);
            std::vector<float> h_cos(N);
            CUDA_CHECK(cudaMemcpy(h_cos.data(), d_cos_buf, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));

            std::vector<uint8_t> v_tar_arr;
            std::vector<float> m_tar_arr;
            compute_raw_quartet_arrays_gpu(N, base_c1.data(), base_c2.data(), par, rep0, d_W_target, v_tar_arr, m_tar_arr);

            /* Evolve T_Base^(l) -> T_Target^(l) via SPR */
            std::vector<int32_t> tar_c1 = base_c1, tar_c2 = base_c2;
            refine_tree_minibatch_on_gpu(N, 25, d_W_target, gpu_ws, tar_c1, tar_c2, NULL, NULL, NULL);

            std::vector<uint64_t> splits_tar;
            compute_xor_splits_o_n(tar_c1.data(), tar_c2.data(), N, leaf_hashes.data(), splits_tar);

            /* Check for each internal node u of T_Base^(l) whether its split was broken by SPR */
            std::vector<int32_t> split_broken(n_nodes_r, 0);
            int32_t rc1 = base_c1[root_r], rc2 = base_c2[root_r];
            int32_t skip_rc = (rc1 > rc2) ? rc1 : rc2;
            for (int32_t u = N; u < root_r; ++u) {
                if (u == skip_rc) continue;
                uint64_t h = node_hash_base[u];
                if (!std::binary_search(splits_tar.begin(), splits_tar.end(), h)) {
                    split_broken[u] = 1;
                }
            }

            /* Accumulate subtree net quartet margin and subtree broken splits */
            std::vector<float> sub_net_margin(n_nodes_r, 0.0f);
            std::vector<int32_t> sub_broken(n_nodes_r, 0);
            for (int32_t idx = (int32_t)ord.size() - 1; idx >= 0; --idx) {
                int32_t u = ord[idx];
                int32_t b0 = base_c1[u], b1 = base_c2[u];
                float net_m = m_tar_arr[u] - m_base_arr[u];
                sub_net_margin[u] = net_m + sub_net_margin[b0] + sub_net_margin[b1];
                sub_broken[u]     = split_broken[u] + sub_broken[b0] + sub_broken[b1];
            }

            /* Score each monophyletic clade cid */
            for (int32_t cid = 0; cid < (int32_t)clade_roots.size(); ++cid) {
                int32_t u = clade_roots[cid];
                float sz = (float)std::max(1, lcnt[u]);
                float net_viol_density = sub_net_margin[u] / sz;
                float spr_frac = (lcnt[u] > 1) ? ((float)sub_broken[u] / (float)(lcnt[u] - 1)) : 0.0f;

                /* Include ancestor clade context up to 1,024 leaves so macro-clade SPR moves also register */
                int32_t anc = par[u];
                float anc_viol = 0.0f, anc_spr = 0.0f;
                int32_t anc_cnt = 0;
                while (anc != -1 && lcnt[anc] <= 1200) {
                    anc_viol += sub_net_margin[anc] / (float)lcnt[anc];
                    anc_spr  += (float)sub_broken[anc] / (float)std::max(1, lcnt[anc] - 1);
                    anc_cnt++;
                    anc = par[anc];
                }
                if (anc_cnt > 0) {
                    net_viol_density = 0.6f * net_viol_density + 0.4f * (anc_viol / (float)anc_cnt);
                    spr_frac         = 0.6f * spr_frac         + 0.4f * (anc_spr / (float)anc_cnt);
                }

                float combined_score = net_viol_density + 0.5f * spr_frac;
                std::vector<int32_t> cst = {u};
                while (!cst.empty()) {
                    int32_t x = cst.back(); cst.pop_back();
                    if (x < N) {
                        if (m == 0) {
                            all_coder_score[row_off + x] = combined_score;
                            all_coder_spr[row_off + x]   = spr_frac;
                            all_coder_cos[row_off + x]   = h_cos[x];
                        } else {
                            all_r1_score[row_off + x] = combined_score;
                            all_r1_spr[row_off + x]   = spr_frac;
                            all_r1_cos[row_off + x]   = h_cos[x];
                        }
                    } else {
                        cst.push_back(base_c1[x]);
                        cst.push_back(base_c2[x]);
                    }
                }
            }
        }
        if ((l + 1) % 4 == 0 || l == N_LAYERS - 1) {
            printf("  Processed layer %02d/%02d (%.2fs)\n", l + 1, N_LAYERS, wall_sec() - t0);
            fflush(stdout);
        }
    }

    FILE *fp = fopen(out_bin, "wb");
    if (!fp) { fprintf(stderr, "Failed to open %s\n", out_bin); return 1; }
    fwrite(all_clade_id.data(),    sizeof(int32_t), TOTAL, fp);
    fwrite(all_coder_score.data(), sizeof(float),   TOTAL, fp);
    fwrite(all_r1_score.data(),    sizeof(float),   TOTAL, fp);
    fwrite(all_coder_spr.data(),   sizeof(float),   TOTAL, fp);
    fwrite(all_r1_spr.data(),      sizeof(float),   TOTAL, fp);
    fwrite(all_coder_cos.data(),   sizeof(float),   TOTAL, fp);
    fwrite(all_r1_cos.data(),      sizeof(float),   TOTAL, fp);
    fclose(fp);
    printf("Saved 48-layer clade scores to %s!\n", out_bin);

    cudaFree(d_W_base); cudaFree(d_W_target); cudaFree(d_cos_buf);
    gpu_ws.destroy();
    return 0;
}
