/*
 * bench_rnasim_gpu.cu
 *
 * Blind Empirical Evaluation on the Real RNASim Dataset (1,596-bp 16S rRNA
 * alignment simulated under biophysical secondary-structure selection,
 * Mirarab et al. 2015) comparing:
 *   1. VeryFastTree 4.0.5 Initial Top-Hit NJ (-nt -nome -noml -threads 64)
 *   2. VeryFastTree 4.0.5 Minimum Evolution (-nt -noml -threads 64: 40 rounds ME-NNI + 2 rounds range-10 ME-SPR)
 *   3. Official FastME / In-Memory C99 BME SPR (optimize_fastme_bme_c)
 *   4. Full-Tree GPU Phylo2Vec BME Discrete-Gradient Optimizer (exact JC69 sequence distances via GPU __popcll)
 *   5. Blind Clade-Condensed Minibatch GPU Phylo2Vec BME Optimizer (zero HLD, zero knowledge of true tree,
 *      distance-guided paired-clade windows directly on real 1,596-bp bit-packed RNA sequences)
 *
 * Compile:
 *   nvcc -O3 -arch=sm_86 bench_rnasim_gpu.cu phylo2vec_fastme.c -o bench_rnasim_gpu
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

extern "C" {
int32_t relabel_cherry_first_c(
    const int64_t *v,
    int32_t n_leaves,
    int32_t flip_mode,
    int64_t *v_cf_out,
    int32_t *perm_out
);
}

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
    /* Always skip max(rc1, rc2):
     * - If both rc1, rc2 >= N, rc1 and rc2 represent the SAME root split, so skipping
     *   max(rc1, rc2) retains exactly one copy of the root split.
     * - If one of (rc1, rc2) < N, the other is >= N and has N - 1 leaves (the complement
     *   of a single leaf, i.e. a trivial pendant split), so skipping max(rc1, rc2)
     *   excludes that trivial split.
     * Thus every rooting produces exactly N - 3 non-trivial internal splits. */
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

__global__ void k_build_full_jc69_D(
    int32_t N, int32_t W,
    const uint64_t * __restrict__ d_packed_seqs,
    float * __restrict__ d_D
) {
    int32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t a = blockIdx.y * blockDim.y + threadIdx.y;
    if (a >= N || b >= N) return;
    if (a == b) {
        d_D[(size_t)a * N + b] = 0.0f;
        return;
    }
    if (a < b) {
        float d = dev_jc69_dist_popc(a, b, d_packed_seqs, W);
        d_D[(size_t)a * N + b] = d;
        d_D[(size_t)b * N + a] = d;
    }
}

__global__ void k_build_real_seq_condensed_D(
    int32_t K, int32_t W,
    const int32_t * __restrict__ d_rep8,
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
        int32_t ra0 = d_rep8[8 * a + 0], ra7 = d_rep8[8 * a + 7];
        int32_t rb0 = d_rep8[8 * b + 0], rb7 = d_rep8[8 * b + 7];

        float d_val = 0.0f;
        if (ra0 == ra7 && rb0 == rb7) {
            d_val = dev_jc69_dist_popc(ra0, rb0, d_packed_seqs, W);
        } else if (ra0 == ra7) {
            #pragma unroll
            for (int m2 = 0; m2 < 8; ++m2) {
                d_val += dev_jc69_dist_popc(ra0, d_rep8[8 * b + m2], d_packed_seqs, W);
            }
            d_val *= 0.125f;
        } else if (rb0 == rb7) {
            #pragma unroll
            for (int m1 = 0; m1 < 8; ++m1) {
                d_val += dev_jc69_dist_popc(d_rep8[8 * a + m1], rb0, d_packed_seqs, W);
            }
            d_val *= 0.125f;
        } else {
            int32_t ra[8], rb[8];
            #pragma unroll
            for (int m = 0; m < 8; ++m) {
                ra[m] = d_rep8[8 * a + m];
                rb[m] = d_rep8[8 * b + m];
            }
            #pragma unroll
            for (int m1 = 0; m1 < 8; ++m1) {
                #pragma unroll
                for (int m2 = 0; m2 < 8; ++m2) {
                    d_val += dev_jc69_dist_popc(ra[m1], rb[m2], d_packed_seqs, W);
                }
            }
            d_val *= (1.0f / 64.0f);
        }
        d_D_cond[(size_t)a * K + b] = d_val;
        d_D_cond[(size_t)b * K + a] = d_val;
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
        float val = 0.5f * (subdist[(size_t)c1 * M + v] + subdist[(size_t)c2 * M + v]);
        subdist[(size_t)u * M + v] = val;
        subdist[(size_t)v * M + u] = val;
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

/* Exact Pauplin BME Tree Length L(T) = 0.5 * sum_{x=0}^{N-1} subdist[x, parent_u[x]] */
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
    if (other0 == -1) return;

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
    std::vector<int32_t> &leaf_cnt_out,
    std::vector<int32_t> &node_rep4_out
) {
    const int32_t n_nodes_r = 2 * N - 1;
    const int32_t root_r = n_nodes_r - 1;

    ord_out.resize(N - 1);
    int32_t ord_len = 0;
    ord_out[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord_out[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        if (b0 >= N) ord_out[ord_len++] = b0;
        if (b1 >= N) ord_out[ord_len++] = b1;
    }

    leaf_cnt_out.assign(n_nodes_r, 1);
    node_rep4_out.resize(8 * n_nodes_r);
    for (int32_t x = 0; x < N; ++x) {
        for (int32_t r = 0; r < 8; ++r) node_rep4_out[8 * x + r] = x;
    }
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord_out[idx];
        int32_t b0 = c1_r[u], b1 = c2_r[u];
        leaf_cnt_out[u] = leaf_cnt_out[b0] + leaf_cnt_out[b1];
        node_rep4_out[8 * u + 0] = node_rep4_out[8 * b0 + 0];
        node_rep4_out[8 * u + 1] = node_rep4_out[8 * b0 + 2];
        node_rep4_out[8 * u + 2] = node_rep4_out[8 * b0 + 4];
        node_rep4_out[8 * u + 3] = node_rep4_out[8 * b0 + 6];
        node_rep4_out[8 * u + 4] = node_rep4_out[8 * b1 + 0];
        node_rep4_out[8 * u + 5] = node_rep4_out[8 * b1 + 2];
        node_rep4_out[8 * u + 6] = node_rep4_out[8 * b1 + 4];
        node_rep4_out[8 * u + 7] = node_rep4_out[8 * b1 + 6];
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
    int32_t *out_root_skel_id
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
    *out_root_skel_id = -1 - last_ret;
}

/* ============================================================================
 * 5. GPU Workspace for Full-Tree and Minibatch Optimization
 * ============================================================================ */
struct RnaSimGpuWorkspace {
    int32_t max_K;
    int32_t *d_rep4;
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

    void init(int32_t K_cap) {
        max_K = K_cap;
        int32_t R = 2 * K_cap - 1;
        int32_t M = 2 * K_cap - 2;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&d_rep4,        (size_t)8 * K_cap * sizeof(int32_t)));
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

    void destroy() {
        CUDA_CHECK(cudaSetDevice(0));
        cudaFree(d_rep4); cudaFree(d_perm); cudaFree(d_D); cudaFree(d_subdist); cudaFree(d_nd_u); cudaFree(d_row_scratch);
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

/* Optimize a K-taxon skeleton on GPU using a single sequence-distance build + path-disjoint SPR batching */
static double optimize_skeleton_on_gpu(
    int32_t K, int32_t W, int32_t max_iters, int32_t max_batch_cap,
    const int32_t *orig_rep4, const uint64_t *d_packed_seqs,
    RnaSimGpuWorkspace &gpu_ws,
    std::vector<int64_t> &v_cf, std::vector<int32_t> &perm
) {
    const int32_t R_sk = 2 * K - 1;
    const int32_t M_sk = 2 * K - 2;
    const int32_t root_u_sk = M_sk - 1;

    /* Build K x K condensed sequence distance matrix ONCE in original skeleton leaf order */
    CUDA_CHECK(cudaMemcpy(gpu_ws.d_rep4, orig_rep4, (size_t)8 * K * sizeof(int32_t), cudaMemcpyHostToDevice));
    dim3 blk2d(16, 16);
    dim3 grd2d((K + 15) / 16, (K + 15) / 16);
    k_build_real_seq_condensed_D<<<grd2d, blk2d>>>(K, W, gpu_ws.d_rep4, d_packed_seqs, gpu_ws.d_D);

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

        CUDA_CHECK(cudaMemcpy(gpu_ws.d_perm,     perm.data(),        (size_t)K * sizeof(int32_t),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_v_curr,   v_cf.data(),        (size_t)(K - 1) * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_c1_r,     c1_cf.data(),       (size_t)R_sk * sizeof(int32_t),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_c2_r,     c2_cf.data(),       (size_t)R_sk * sizeof(int32_t),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_v_map,    v_map_cf.data(),    (size_t)K * sizeof(int32_t),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_m_lmin,   m_lmin_cf.data(),   (size_t)K * sizeof(int32_t),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_next_k,   next_k_cf.data(),   (size_t)K * sizeof(int32_t),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_parent_u, parent_u_cf.data(), (size_t)M_sk * sizeof(int32_t),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_sub_size, sub_size_cf.data(), (size_t)M_sk * sizeof(int32_t),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_tin,      tin_cf.data(),      (size_t)M_sk * sizeof(int32_t),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_ws.d_adj_u,    adj_u_cf.data(),    (size_t)3 * M_sk * sizeof(int32_t),cudaMemcpyHostToDevice));

        k_copy_leaf_rows_perm<<<grd2d, blk2d>>>(gpu_ws.d_D, gpu_ws.d_perm, gpu_ws.d_subdist, K, M_sk);
        k_bottom_up_leaves<<<(K + 255) / 256, 256>>>(gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, K, M_sk, root_u_sk);
        int32_t n_int = root_u_sk - K;
        if (n_int > 0) {
            dim3 grd_tr((K + 15) / 16, (n_int + 15) / 16);
            k_transpose_leaf_block<<<grd_tr, blk2d>>>(gpu_ws.d_subdist, K, M_sk, root_u_sk);
            k_bottom_up_internals<<<(n_int + 255) / 256, 256>>>(gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, K, M_sk, root_u_sk);
        }
        k_top_down_and_root<<<(M_sk + 255) / 256, 256>>>(
            gpu_ws.d_subdist, gpu_ws.d_c1_r, gpu_ws.d_c2_r, gpu_ws.d_tin, gpu_ws.d_sub_size,
            K, M_sk, root_u_sk, rc0, rc1, rc2
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
            gpu_ws.d_nd_u, gpu_ws.d_row_scratch,
            gpu_ws.d_best_delta, gpu_ws.d_best_j
        );
        CUDA_CHECK(cudaMemcpy(h_best_delta.data(), gpu_ws.d_best_delta, (size_t)K * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_best_j.data(),     gpu_ws.d_best_j,     (size_t)K * sizeof(int32_t), cudaMemcpyDeviceToHost));

        std::vector<MoveCand> cands;
        float eps_move_thresh = -1e-4f;
        for (int32_t i = 1; i < K - 1; ++i) {
            if (h_best_delta[i] < eps_move_thresh && h_best_j[i] >= 0 && h_best_j[i] < root_u_sk) {
                cands.push_back({i, h_best_j[i], h_best_delta[i]});
            }
        }
        if (cands.empty()) {
            if (flipped_already == 0) {
                /* Gauge 1: flip c_stay and c_move at every internal node */
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

            /* Ensure dst_u is not inside tr's subtree */
            if (tin_cf[tr] <= tin_cf[dst_u] && tin_cf[dst_u] < tin_cf[tr] + sub_size_cf[tr]) continue;
            if (node_busy[tr] || node_busy[pr] || node_busy[dst_u] || node_busy[dst_v]) continue;

            /* Trace tree path between pr and dst_u in parent_u_cf */
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
            path_nodes.push_back(u_anc); /* LCA */
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
    v_cf = best_v_cf;
    perm = best_perm;
    return best_L - init_L;
}

/* Helper to optimize one extracted skeleton on (curr_c1, curr_c2) and graft back */
static double run_one_minibatch_window(
    int32_t N, int32_t W, int32_t condense_thresh, int32_t max_iters,
    const std::vector<uint8_t> &node_has_active,
    const uint64_t *d_packed_seqs,
    RnaSimGpuWorkspace &gpu_ws,
    std::vector<int32_t> &curr_c1, std::vector<int32_t> &curr_c2,
    std::vector<int32_t> &ord, std::vector<int32_t> &leaf_cnt, std::vector<int32_t> &node_rep4
) {
    const int32_t root_r = 2 * N - 2;
    std::vector<int32_t> super_roots, split_internals, skel_left, skel_right;
    int32_t root_skel_id = -1;
    extract_active_skeleton(
        root_r, curr_c1.data(), curr_c2.data(), N,
        condense_thresh, node_has_active, leaf_cnt,
        super_roots, split_internals, skel_left, skel_right, &root_skel_id
    );

    int32_t K = (int32_t)super_roots.size();
    if (K < 6 || K > gpu_ws.max_K) return 0.0;

    std::vector<int32_t> sk_c1(2 * K - 1), sk_c2(2 * K - 1);
    for (int32_t s = 0; s < K - 1; ++s) {
        int32_t l_raw = skel_left[s], r_raw = skel_right[s];
        sk_c1[K + s] = (l_raw >= 0) ? l_raw : (K + (-1 - l_raw));
        sk_c2[K + s] = (r_raw >= 0) ? r_raw : (K + (-1 - r_raw));
    }

    std::vector<int32_t> orig_rep4(8 * K);
    for (int32_t a = 0; a < K; ++a) {
        int32_t ru = super_roots[a];
        for (int32_t r = 0; r < 8; ++r) orig_rep4[8 * a + r] = node_rep4[8 * ru + r];
    }

    std::vector<int64_t> v_mb(K - 1);
    std::vector<int32_t> perm_mb(K);
    relabel_from_rooted_tree(sk_c1.data(), sk_c2.data(), K, 0, v_mb.data(), perm_mb.data());

    double dL = optimize_skeleton_on_gpu(K, W, max_iters, 32, orig_rep4.data(), d_packed_seqs, gpu_ws, v_mb, perm_mb);

    int32_t R_k = 2 * K - 1;
    std::vector<int32_t> new_sk_c1(R_k), new_sk_c2(R_k), scratch_uf(R_k);
    decode_cf_to_rooted(v_mb.data(), K, new_sk_c1.data(), new_sk_c2.data(), scratch_uf.data());
    std::vector<int32_t> sk_to_full(R_k);
    for (int32_t a = 0; a < K; ++a) sk_to_full[a] = super_roots[perm_mb[a]];
    for (int32_t s = 0; s < K - 1; ++s) sk_to_full[K + s] = split_internals[s];
    for (int32_t s = 0; s < K - 1; ++s) {
        int32_t full_u = sk_to_full[K + s];
        curr_c1[full_u] = sk_to_full[new_sk_c1[K + s]];
        curr_c2[full_u] = sk_to_full[new_sk_c2[K + s]];
    }
    compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, leaf_cnt, node_rep4);
    return dL;
}


int main(void) {
    printf("====================================================================================================================================\n");
    printf("BLIND EMPIRICAL RNASIM BENCHMARK (Real 1,596-bp 16S rRNA Sequences: VeryFastTree 4.0.5 vs. GPU Phylo2Vec BME Optimizer)\n");
    printf("====================================================================================================================================\n");
    printf("%8s | %22s | %22s | %24s | %24s | %10s\n",
           "Taxa (N)", "VFT-InitNJ RF (Time)", "VFT-ME (-noml) RF", "GPU Full-Tree BME RF", "GPU Minibatch BME RF", "Delta L");
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");

    const int32_t sizes[] = {1000, 5000, 10000, 50000};
    const double vft_init_times[] = {0.58, 3.40, 7.82, 63.01};
    const double vft_me_times[]   = {7.46, 36.78, 73.80, 376.54};

    RnaSimGpuWorkspace gpu_ws;
    gpu_ws.init(10240);

    for (size_t s_idx = 0; s_idx < sizeof(sizes) / sizeof(sizes[0]); ++s_idx) {
        int32_t N = sizes[s_idx];
        char bin_path[256];
        snprintf(bin_path, sizeof(bin_path), "/home/sjbhatt/phylo2vec_discrete_gradient/rnasim_data/rnasim_%d.bin", N);
        FILE *fp = fopen(bin_path, "rb");
        if (!fp) { fprintf(stderr, "Missing %s\n", bin_path); continue; }

        int32_t hdr[3];
        if (fread(hdr, sizeof(int32_t), 3, fp) != 3) exit(1);
        int32_t W = hdr[2];
        const int32_t n_nodes_r = 2 * N - 1;
        const int32_t root_r = n_nodes_r - 1;

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

        uint64_t *d_packed_seqs = NULL;
        CUDA_CHECK(cudaMalloc(&d_packed_seqs, h_packed_seqs.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_packed_seqs, h_packed_seqs.data(), h_packed_seqs.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

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
        double total_full_delta_L = 0.0, dt_full_gpu = 0.0;
        int64_t rf_full_gpu = -1;

        if (N <= 10000) {

            CUDA_CHECK(cudaDeviceSynchronize());
            double t0_full = wall_sec();

            std::vector<int64_t> v_cf(N - 1);
            std::vector<int32_t> perm(N);
            relabel_from_rooted_tree(init_c1.data(), init_c2.data(), N, 0, v_cf.data(), perm.data());

            std::vector<int32_t> orig_rep4(8 * N);
            for (int32_t a = 0; a < N; ++a) {
                for (int32_t r = 0; r < 8; ++r) orig_rep4[8 * a + r] = a;
            }

            total_full_delta_L = optimize_skeleton_on_gpu(
                N, W, 100, 64, orig_rep4.data(), d_packed_seqs, gpu_ws, v_cf, perm
            );

            CUDA_CHECK(cudaDeviceSynchronize());
            dt_full_gpu = wall_sec() - t0_full;

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
         * Mode B: Blind Multigrid Clade-Condensed Minibatch GPU Phylo2Vec BME
         *         - Pass 1: Global Backbone Minibatch (optimizes top-level clades)
         *         - Pass 2: 100%-Covering Overlapping DFS Clade Windows
         * -------------------------------------------------------------------- */
        CUDA_CHECK(cudaDeviceSynchronize());
        double t0_mb = wall_sec();
        std::vector<int32_t> curr_c1 = init_c1, curr_c2 = init_c2;
        std::vector<int32_t> ord, leaf_cnt, node_rep4;
        compute_tree_summaries_blind(curr_c1.data(), curr_c2.data(), N, ord, leaf_cnt, node_rep4);

        double total_mb_delta_L = 0.0;
        std::vector<uint8_t> node_has_active(n_nodes_r, 0);

        /* Pass 1: Global Backbone Minibatch for N >= 5000 */
        if (N >= 5000) {
            int32_t bb_thresh = N / 700;
            if (bb_thresh < 8) bb_thresh = 8;
            double dL_p1 = run_one_minibatch_window(
                N, W, bb_thresh, 25, node_has_active, d_packed_seqs, gpu_ws,
                curr_c1, curr_c2, ord, leaf_cnt, node_rep4
            );
            total_mb_delta_L += dL_p1;
        }

        /* Pass 2: Collect all N leaves in current tree DFS order and sweep overlapping windows */
        std::vector<int32_t> dfs_leaves;
        dfs_leaves.reserve(N);
        {
            std::vector<int32_t> st = {root_r};
            while (!st.empty()) {
                int32_t u = st.back(); st.pop_back();
                if (u < N) {
                    dfs_leaves.push_back(u);
                } else {
                    st.push_back(curr_c2[u]);
                    st.push_back(curr_c1[u]);
                }
            }
        }

        int32_t win_size   = (N <= 1000) ? 550 : 950;
        int32_t win_stride = (N <= 1000) ? 450 : 600;
        int32_t condense_thresh = N / 250;
        if (condense_thresh < 8) condense_thresh = 8;

        for (int32_t round = 1; round <= 1; ++round) {
            dfs_leaves.clear();
            {
                std::vector<int32_t> st = {root_r};
                while (!st.empty()) {
                    int32_t u = st.back(); st.pop_back();
                    if (u < N) dfs_leaves.push_back(u);
                    else { st.push_back(curr_c2[u]); st.push_back(curr_c1[u]); }
                }
            }
            double round_dL = 0.0;
            for (int32_t start = 0; start < N; start += win_stride) {
                int32_t end = start + win_size;
                if (end > N) end = N;
                std::fill(node_has_active.begin(), node_has_active.end(), 0);
                for (int32_t idx = start; idx < end; ++idx) {
                    node_has_active[dfs_leaves[idx]] = 1;
                }
                for (int32_t idx_o = (int32_t)ord.size() - 1; idx_o >= 0; --idx_o) {
                    int32_t u = ord[idx_o];
                    node_has_active[u] = node_has_active[curr_c1[u]] | node_has_active[curr_c2[u]];
                }

                round_dL += run_one_minibatch_window(
                    N, W, condense_thresh, 25, node_has_active, d_packed_seqs, gpu_ws,
                    curr_c1, curr_c2, ord, leaf_cnt, node_rep4
                );
                if (end == N) break;
            }
            total_mb_delta_L += round_dL;

        }

        CUDA_CHECK(cudaDeviceSynchronize());
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
        double dL_rep = (rf_full_gpu >= 0) ? total_full_delta_L : total_mb_delta_L;

        printf("%8d | %22s | %22s | %24s | %24s | %+10.2f\n",
               N, b_init, b_me, b_full, b_mb, dL_rep);

        cudaFree(d_packed_seqs);
    }
    printf("------------------------------------------------------------------------------------------------------------------------------------\n");
    gpu_ws.destroy();
    return 0;
}
