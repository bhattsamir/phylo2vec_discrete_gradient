/*
 * ============================================================================
 * Phylo2Vec O(N^2) BME Discrete Gradient — CUDA Multi-GPU Implementation
 * ============================================================================
 *
 * Implements:
 *   1. Pass 1 (`k_bottom_up_leaf_cols` + 16x16 shared-memory `k_transpose_leaf_block`):
 *      Lockstep SIMT warp-coalesced bottom-up dynamic programming on leaf columns.
 *   2. Pass 2 (`k_bottom_up_int_cols`):
 *      Lockstep SIMT warp-coalesced bottom-up dynamic programming on internal columns.
 *   3. Pass 3 (`k_top_down_ancestor_walk` + `k_nd_u`):
 *      Stackless ancestor-walk top-down pass + diagonal zeroing + 3-way `nd_u` table.
 *   4. Pass 4 (`k_spr_rows_smem` with automatic fallback to `k_spr_rows_gmem`):
 *      - Uses `next_k_same_lmin[k] >= n_active` so each leaf destination column
 *        `0 <= j <= i` has at most ONE writer thread (zero write-write races,
 *        independent of warp/thread scheduling).
 *      - Guards root-branch column `2*i - 1` with `if (2 * i - 1 > i)` to prevent
 *        overlapping leaf column `j = 1` at `i = 1`.
 *      - Queries `cudaDevAttrMaxSharedMemoryPerBlockOptin` and opts into dynamic
 *        shared memory up to the device limit (227 KiB on Hopper sm_90), falling
 *        back automatically to `k_spr_rows_gmem` when `(2N-2)*sizeof(float)`
 *        exceeds per-block shared memory.
 *   5. Pass 4b (`k_spr_rows_top1`):
 *      Fused SPR row scan + warp/block argmin reduction that outputs only
 *      `(best_delta[i], best_j[i])` per row (8*N bytes total instead of 8*N^2 bytes),
 *      eliminating the O(N^2) full-gradient output buffer during optimization.
 *   6. Column-Tiled Low-Memory Pipeline (`k_tiled_leaf_cols` + `k_spr_rows_tiled`):
 *      Computes only `subdist_int_leaf[(N-1) x N]` in column tiles of width B_TILE
 *      and reconstructs the 3 active rows per SPR scan autonomously (`dev_compute_autonomous_row`).
 *
 * Compilation on NVIDIA GPU servers (sm_90 H100 / sm_89 L40S / sm_100 Blackwell):
 *   nvcc -O3 -arch=sm_90 -Xcompiler -fopenmp phylo2vec_cuda_8xh100.cu phylo2vec_fastme.c -o phylo2vec_cuda_8xh100
 *
 * Local verification on CPU/macOS (without nvcc, executes the exact kernel bodies
 * under a deterministic + reverse-thread-order SIMT block verifier):
 *   /usr/bin/clang -O3 -c phylo2vec_fastme.c -o phylo2vec_fastme.o && \
 *   /usr/bin/clang++ -O3 -std=c++17 -x c++ phylo2vec_cuda_8xh100.cu -x none phylo2vec_fastme.o -o phylo2vec_cuda_verify && ./phylo2vec_cuda_verify
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

extern "C" {
void calculate_discrete_gradient_full_c(
    const int64_t *v, const double *D, int32_t n_leaves, double *delta_full_out
);
}

static inline double now_ms_host(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

#ifdef __CUDACC__
#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
#define SIMT_PHASE_ACTIVE 1
#else
/*
 * Host SIMT Emulation for local verification when compiled with clang++/g++:
 * Simulates CUDA threadblocks with true barrier synchronization (`__syncthreads()`)
 * by running all threads `tx in [0 .. blockDim.x)` (in forward or reverse order)
 * through each barrier-delimited phase in lockstep.
 */
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __shared__
struct Dim3 { int x, y, z; };
static thread_local Dim3 blockIdx  = {0, 0, 0};
static thread_local Dim3 blockDim  = {1, 1, 1};
static thread_local Dim3 threadIdx = {0, 0, 0};
static thread_local int g_simt_target_phase = 0;
static thread_local int g_simt_curr_phase   = 0;
#define SIMT_PHASE_ACTIVE (g_simt_curr_phase == g_simt_target_phase)
#define __syncthreads() do { g_simt_curr_phase++; } while (0)
#endif

/* ============================================================================
 * STAGE 2 KERNELS: Lockstep Column-Parallel `subdist` + 16x16 Tiled Transpose
 *                  + Stackless Ancestor-Walk Top-Down Pass
 * ============================================================================ */

/* Pass 1: Bottom-up internal rows [N .. R-1] for leaf columns x in [0 .. N-1] */
__global__ void k_bottom_up_leaf_cols(
    const float * __restrict__ D,
    float * __restrict__ subdist,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    int32_t n_leaves,
    int32_t n_nodes_u
) {
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n_leaves) return;
    int32_t root_u = n_nodes_u - 1;

    for (int32_t i = 0; i < n_leaves; ++i) {
        subdist[(size_t)i * n_nodes_u + x] = D[(size_t)i * n_leaves + x];
    }
    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        subdist[(size_t)u * n_nodes_u + x] =
            0.5f * (subdist[(size_t)c1 * n_nodes_u + x] + subdist[(size_t)c2 * n_nodes_u + x]);
    }
}

/* Pass 1b: Coalesced 16x16 shared-memory tiled transpose of [N .. R-1, 0 .. N-1] -> [0 .. N-1, N .. R-1] */
__global__ void k_transpose_leaf_block(
    float * __restrict__ subdist,
    int32_t n_leaves,
    int32_t n_nodes_u
) {
    int32_t n_int = (n_nodes_u - 1) - n_leaves;
#ifdef __CUDACC__
    __shared__ float tile[16][17];
    int32_t x = blockIdx.x * 16 + threadIdx.x; /* leaf index in [0 .. N-1] */
    int32_t k = blockIdx.y * 16 + threadIdx.y; /* internal offset in [0 .. n_int-1] */
    if (x < n_leaves && k < n_int) {
        int32_t u = n_leaves + k;
        tile[threadIdx.y][threadIdx.x] = subdist[(size_t)u * n_nodes_u + x];
    }
    __syncthreads();
    int32_t k_out = blockIdx.y * 16 + threadIdx.x; /* internal offset */
    int32_t x_out = blockIdx.x * 16 + threadIdx.y; /* leaf row */
    if (x_out < n_leaves && k_out < n_int) {
        int32_t u_out = n_leaves + k_out;
        subdist[(size_t)x_out * n_nodes_u + u_out] = tile[threadIdx.x][threadIdx.y];
    }
#else
    int32_t x = blockIdx.x * 16 + threadIdx.x;
    int32_t k = blockIdx.y * 16 + threadIdx.y;
    if (x < n_leaves && k < n_int) {
        int32_t u = n_leaves + k;
        subdist[(size_t)x * n_nodes_u + u] = subdist[(size_t)u * n_nodes_u + x];
    }
#endif
}

/* Pass 2: Bottom-up internal rows [N .. R-1] for internal columns w in [N .. R-1] */
__global__ void k_bottom_up_int_cols(
    float * __restrict__ subdist,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    int32_t n_leaves,
    int32_t n_nodes_u
) {
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t root_u = n_nodes_u - 1;
    int32_t w = n_leaves + k;
    if (w >= root_u) return;

    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        subdist[(size_t)u * n_nodes_u + w] =
            0.5f * (subdist[(size_t)c1 * n_nodes_u + w] + subdist[(size_t)c2 * n_nodes_u + w]);
    }
}

/* Pass 3: Stackless Ancestor-Walk Top-Down Pass (O(sum_v depth(v))) + Symmetrization */
__global__ void k_top_down_ancestor_walk(
    float * __restrict__ subdist,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    const int32_t * __restrict__ tin,
    int32_t n_nodes_u,
    int32_t root_u_c0,
    int32_t root_u_c1,
    int32_t root_u_c2
) {
    int32_t v_node = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t root_u = n_nodes_u - 1;
    if (v_node >= root_u) {
        if (v_node == root_u) subdist[(size_t)root_u * n_nodes_u + root_u] = 0.0f;
        return;
    }

    int32_t tv = tin[v_node];
    int32_t curr;
    float r_val;
    if (tv < tin[root_u_c1]) {
        curr = root_u_c0;
        r_val = 0.5f * (subdist[(size_t)root_u_c1 * n_nodes_u + v_node] +
                        subdist[(size_t)root_u_c2 * n_nodes_u + v_node]);
    } else if (tv < tin[root_u_c2]) {
        curr = root_u_c1;
        r_val = 0.5f * (subdist[(size_t)root_u_c0 * n_nodes_u + v_node] +
                        subdist[(size_t)root_u_c2 * n_nodes_u + v_node]);
    } else {
        curr = root_u_c2;
        r_val = 0.5f * (subdist[(size_t)root_u_c0 * n_nodes_u + v_node] +
                        subdist[(size_t)root_u_c1 * n_nodes_u + v_node]);
    }
    subdist[(size_t)root_u * n_nodes_u + v_node] = r_val;
    subdist[(size_t)v_node * n_nodes_u + root_u] = r_val;

    int32_t g = root_u;
    while (curr != v_node) {
        int32_t c1 = child1_r[curr], c2 = child2_r[curr];
        int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
        int32_t sib       = (tv < tin[c2]) ? c2 : c1;
        float val = 0.5f * (subdist[(size_t)sib * n_nodes_u + v_node] +
                            subdist[(size_t)g * n_nodes_u + v_node]);
        subdist[(size_t)curr * n_nodes_u + v_node] = val;
        subdist[(size_t)v_node * n_nodes_u + curr] = val;
        g = curr;
        curr = next_curr;
    }
    subdist[(size_t)v_node * n_nodes_u + v_node] = 0.0f;
}

/* Pass 3b: Precompute 3-way neighbor distances `nd_u` */
__global__ void k_nd_u(
    const float * __restrict__ subdist,
    const int32_t * __restrict__ adj_u,
    float * __restrict__ nd_u,
    int32_t n_leaves,
    int32_t n_nodes_u
) {
    int32_t u = n_leaves + blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_nodes_u) return;
    int32_t n0 = adj_u[3 * u + 0];
    int32_t n1 = adj_u[3 * u + 1];
    int32_t n2 = adj_u[3 * u + 2];
    nd_u[3 * u + 0] = subdist[(size_t)n1 * n_nodes_u + n2];
    nd_u[3 * u + 1] = subdist[(size_t)n0 * n_nodes_u + n2];
    nd_u[3 * u + 2] = subdist[(size_t)n0 * n_nodes_u + n1];
}

/* ============================================================================
 * STAGE 3 DEVICE HELPER: Heavy-First Logarithmic-Stack & Parallel Subtree DFS
 * ============================================================================ */
struct SprSharedQueue {
    int32_t u[256];
    int32_t c[256];
    float   pow_val[256];
    float   dkc[256];
    float   dist[256];
    int32_t bounds[4]; /* [q_head0, q_tail0, q_head1, q_tail1] */
};

__device__ __forceinline__ void dev_expand_subtree_stack(
    int32_t u,
    int32_t C,
    float pow_neg_len,
    float d_kc,
    float curr_dist,
    int32_t n_leaves,
    int32_t n_nodes_u,
    const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size,
    const float * __restrict__ nd_u,
    const float * __restrict__ row_root,
    const float * __restrict__ row_orig,
    float * __restrict__ edge_delta
) {
    int32_t st_u[18];
    int32_t st_c[18];
    float   st_pow[18];
    float   st_dkc[18];
    float   st_dist[18];
    int32_t top = 0;

    for (;;) {
        int32_t a0 = adj_u[3 * u + 0];
        int32_t a1 = adj_u[3 * u + 1];

        int32_t ki = (a0 == C) ? 0 : ((a1 == C) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;

        int32_t b0 = adj_u[3 * u + idx0];
        int32_t b1 = adj_u[3 * u + idx1];
        float d_ab = nd_u[3 * u + ki];

        float r_b0 = row_root[b0];
        float r_b1 = row_root[b1];

        float d_ac0 = nd_u[3 * u + idx0] + (row_orig[b1] - r_b1) * pow_neg_len;
        float ndist0 = curr_dist + 0.25f * (d_ac0 + r_b0 - d_ab - d_kc);
        int32_t ce0 = (parent_u[u] == b0) ? u : b0;
        edge_delta[ce0] = ndist0;

        float d_ac1 = nd_u[3 * u + idx1] + (row_orig[b0] - r_b0) * pow_neg_len;
        float ndist1 = curr_dist + 0.25f * (d_ac1 + r_b1 - d_ab - d_kc);
        int32_t ce1 = (parent_u[u] == b1) ? u : b1;
        edge_delta[ce1] = ndist1;

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
            u = st_u[top];
            C = st_c[top];
            pow_neg_len = st_pow[top];
            d_kc = st_dkc[top];
            curr_dist = st_dist[top];
        }
    }
}

__device__ __forceinline__ void dev_eval_cherry_or_enqueue(
    int32_t b,
    int32_t u,
    int32_t sb,
    float pow_neg_len,
    float d_kc,
    float curr_dist,
    const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u,
    const float * __restrict__ nd_u,
    const float * __restrict__ row_root,
    const float * __restrict__ row_orig,
    float * __restrict__ edge_delta,
    SprSharedQueue * __restrict__ sq,
    int32_t * __restrict__ q_tail_ptr
) {
    if (sb == 3) {
        int32_t a0 = adj_u[3 * b + 0];
        int32_t a1 = adj_u[3 * b + 1];
        int32_t ki = (a0 == u) ? 0 : ((a1 == u) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;
        int32_t c0 = adj_u[3 * b + idx0];
        int32_t c1 = adj_u[3 * b + idx1];
        float d_ab = nd_u[3 * b + ki];
        float r_c0 = row_root[c0];
        float r_c1 = row_root[c1];
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
    int32_t start_node,
    int32_t orig_node,
    int32_t parent_node,
    int32_t q_base,
    int32_t n_leaves,
    int32_t n_nodes_u,
    const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size,
    const float * __restrict__ nd_u,
    const float * __restrict__ row_root,
    const float * __restrict__ row_orig,
    float * __restrict__ edge_delta,
    SprSharedQueue * __restrict__ sq,
    int32_t * __restrict__ out_head,
    int32_t * __restrict__ out_tail
) {
    int32_t q_tail = q_base;
    int32_t q_lim  = q_base + 128;
    int32_t tau    = (n_nodes_u > 512) ? (n_nodes_u / 64) : 16;

    if (start_node >= n_leaves) {
        int32_t n0 = adj_u[3 * start_node + 0];
        int32_t n1 = adj_u[3 * start_node + 1];
        int32_t k_in = (n0 == parent_node) ? 0 : ((n1 == parent_node) ? 1 : 2);
        int32_t k0 = (k_in == 0) ? 1 : 0;
        int32_t k1 = (k_in == 2) ? 1 : 2;

        int32_t B0 = adj_u[3 * start_node + k0];
        int32_t B1 = adj_u[3 * start_node + k1];
        float D_AB = nd_u[3 * start_node + k_in];
        float d_kc_init = row_root[orig_node];

        float new_dist0 = 0.25f * (row_orig[B1] + row_root[B0] - D_AB - d_kc_init);
        int32_t c_end0 = (parent_u[start_node] == B0) ? start_node : B0;
        edge_delta[c_end0] = new_dist0;

        float new_dist1 = 0.25f * (row_orig[B0] + row_root[B1] - D_AB - d_kc_init);
        int32_t c_end1 = (parent_u[start_node] == B1) ? start_node : B1;
        edge_delta[c_end1] = new_dist1;

        /* Heavy-threshold stack: split any branch with component size > tau */
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
            int32_t u = h_u[h_top];
            int32_t C = h_c[h_top];
            float pow_neg_len = h_pow[h_top];
            float d_kc = h_dkc[h_top];
            float curr_dist = h_dist[h_top];

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
                int32_t a0 = adj_u[3 * u + 0];
                int32_t a1 = adj_u[3 * u + 1];
                int32_t ki = (a0 == C) ? 0 : ((a1 == C) ? 1 : 2);
                int32_t idx0 = (ki == 0) ? 1 : 0;
                int32_t idx1 = (ki == 2) ? 1 : 2;

                int32_t b0 = adj_u[3 * u + idx0];
                int32_t b1 = adj_u[3 * u + idx1];
                float d_ab = nd_u[3 * u + ki];
                float r_b0 = row_root[b0];
                float r_b1 = row_root[b1];

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
    int32_t tree_root,
    int32_t parent,
    int32_t other0,
    int32_t other1,
    int32_t n_leaves,
    int32_t n_nodes_u,
    const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size,
    const float * __restrict__ nd_u,
    const float * __restrict__ row_root,
    const float * __restrict__ row_o0,
    const float * __restrict__ row_o1,
    float * __restrict__ edge_delta,
    SprSharedQueue * __restrict__ sq
) {
    if (SIMT_PHASE_ACTIVE) {
        if (threadIdx.x == 0) {
            edge_delta[(parent_u[tree_root] == parent) ? tree_root : parent] = 0.0f;
            edge_delta[(parent_u[other0]    == parent) ? other0    : parent] = 0.0f;
            edge_delta[(parent_u[other1]    == parent) ? other1    : parent] = 0.0f;
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
    }
    __syncthreads();

    if (SIMT_PHASE_ACTIVE) {
        int32_t h0 = sq->bounds[0], t0 = sq->bounds[1];
        int32_t h1 = sq->bounds[2], t1 = sq->bounds[3];
        int32_t n0 = t0 - h0;
        int32_t n1 = t1 - h1;
        int32_t total = n0 + n1;
        for (int32_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
            int32_t q_idx = (idx < n0) ? (h0 + idx) : (h1 + (idx - n0));
            const float *r_orig = (idx < n0) ? row_o1 : row_o0;
            dev_expand_subtree_stack(
                sq->u[q_idx], sq->c[q_idx], sq->pow_val[q_idx], sq->dkc[q_idx], sq->dist[q_idx],
                n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u, row_root, r_orig, edge_delta
            );
        }
    }
    __syncthreads();
}

/*
 * Race-Free Destination Mapping Helper:
 *   - Leaf columns `0 <= j <= i` are initialized to `edge_delta[j]`.
 *   - Among active merges `0 <= k < n_active` (`n_active = n_leaves - 2 - i`),
 *     merge `k` is the LAST (winning) merge for `lmin = m_lmin[k]` if and only if
 *     `next_k_same_lmin[k] >= n_active`.
 *   - Therefore, at most ONE thread `k` ever writes to `out_row[lmin]`, completely
 *     eliminating write-write races across repeated minimum leaves!
 *   - Root attachment `2*i - 1` is written only when `2*i - 1 > i` (i >= 2),
 *     preventing overlap with leaf column `j = 1` when `i = 1`.
 */
__device__ __forceinline__ void dev_map_edge_delta_to_row(
    int32_t i,
    int32_t n_leaves,
    int32_t other_root_child,
    const int32_t * __restrict__ m_lmin,
    const int32_t * __restrict__ next_k_same_lmin,
    const float * __restrict__ edge_delta,
    float * __restrict__ out_row
) {
    if (SIMT_PHASE_ACTIVE) {
        for (int32_t j = threadIdx.x; j <= i; j += blockDim.x) {
            out_row[j] = edge_delta[j];
        }
    }
    __syncthreads();

    if (SIMT_PHASE_ACTIVE) {
        int32_t n_active = n_leaves - 2 - i;
        for (int32_t k = threadIdx.x; k < n_active; k += blockDim.x) {
            int32_t lmin = m_lmin[k];
            if (lmin <= i && next_k_same_lmin[k] >= n_active) {
                out_row[lmin] = edge_delta[n_leaves + k];
            }
        }
        for (int32_t j = i + 1 + threadIdx.x; j <= 2 * i - 2; j += blockDim.x) {
            out_row[j] = edge_delta[(2 * n_leaves - 2) - 2 * i + j];
        }
        if (threadIdx.x == 0) {
            if (2 * i - 1 > i) {
                out_row[2 * i - 1] = edge_delta[other_root_child];
            }
            out_row[2 * i] = edge_delta[other_root_child];
        }
    }
}

/* ============================================================================
 * STAGE 3 KERNEL A: Shared-Memory `s_edge_delta` Kernel (for (2N-2)*4B <= max_smem)
 * ============================================================================ */
__global__ void k_spr_rows_smem(
    int32_t r_start,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t other_root_child,
    const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size,
    const int32_t * __restrict__ m_lmin,
    const int32_t * __restrict__ next_k_same_lmin,
    const float * __restrict__ subdist,
    const float * __restrict__ nd_u,
    float * __restrict__ delta_out_slice
#ifndef __CUDACC__
    , float * __restrict__ host_smem_buf
#endif
) {
#ifdef __CUDACC__
    extern __shared__ float s_edge_delta[];
    __shared__ SprSharedQueue s_q;
#else
    float *s_edge_delta = host_smem_buf;
    static SprSharedQueue s_q;
#endif
    int32_t v_index = r_start + blockIdx.x;
    int32_t root_u = n_nodes_u - 1;
    int32_t max_cols = 2 * (n_leaves - 1) - 1;

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
        s_edge_delta, &s_q
    );

    int32_t i = v_index - 1;
    float *out_row = &delta_out_slice[(size_t)(v_index - r_start) * max_cols];
    dev_map_edge_delta_to_row(i, n_leaves, other_root_child, m_lmin, next_k_same_lmin, s_edge_delta, out_row);
}

/* ============================================================================
 * STAGE 3 KERNEL B: Global-Memory Fallback (`k_spr_rows_gmem`, arbitrary N)
 * ============================================================================ */
__global__ void k_spr_rows_gmem(
    int32_t r_start,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t other_root_child,
    const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size,
    const int32_t * __restrict__ m_lmin,
    const int32_t * __restrict__ next_k_same_lmin,
    const float * __restrict__ subdist,
    const float * __restrict__ nd_u,
    float * __restrict__ edge_delta_pool,
    float * __restrict__ delta_out_slice
) {
#ifdef __CUDACC__
    __shared__ SprSharedQueue s_q;
#else
    static SprSharedQueue s_q;
#endif
    int32_t v_index = r_start + blockIdx.x;
    int32_t root_u = n_nodes_u - 1;
    int32_t max_cols = 2 * (n_leaves - 1) - 1;
    float *g_edge_delta = &edge_delta_pool[(size_t)blockIdx.x * n_nodes_u];

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
        g_edge_delta, &s_q
    );

    int32_t i = v_index - 1;
    float *out_row = &delta_out_slice[(size_t)(v_index - r_start) * max_cols];
    dev_map_edge_delta_to_row(i, n_leaves, other_root_child, m_lmin, next_k_same_lmin, g_edge_delta, out_row);
}

/* ============================================================================
 * STAGE 3 KERNEL C: Fused Top-1 Candidate Reduction (`k_spr_rows_top1`)
 *   - Reduces each row `i` on the fly across all 128 threads in shared memory
 *     to `(row_best_delta[i], row_best_j[i])` using O(N) total output memory!
 * ============================================================================ */
__global__ void k_spr_rows_top1(
    int32_t r_start,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t other_root_child,
    int32_t use_smem_edge,
    const int64_t * __restrict__ v_curr,
    const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size,
    const int32_t * __restrict__ m_lmin,
    const int32_t * __restrict__ next_k_same_lmin,
    const float * __restrict__ subdist,
    const float * __restrict__ nd_u,
    float * __restrict__ edge_delta_pool,
    float * __restrict__ row_tmp_pool,
    float * __restrict__ best_delta_out,
    int32_t * __restrict__ best_j_out
) {
#ifdef __CUDACC__
    extern __shared__ float s_edge_delta_top1_buf[];
    __shared__ SprSharedQueue s_q;
    __shared__ float s_best_val[128];
    __shared__ int32_t s_best_j[128];
#else
    float *s_edge_delta_top1_buf = NULL;
    static SprSharedQueue s_q;
    static float s_best_val[128];
    static int32_t s_best_j[128];
#endif
    int32_t v_index = r_start + blockIdx.x;
    int32_t i = v_index - 1;
    int32_t root_u = n_nodes_u - 1;
    float *edge_delta = use_smem_edge ? s_edge_delta_top1_buf : &edge_delta_pool[(size_t)blockIdx.x * n_nodes_u];
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
    if (other0 == -1) return;

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

    if (SIMT_PHASE_ACTIVE) {
        int32_t vi = (int32_t)v_curr[i];
        float b_val = 0.0f;
        int32_t b_j = vi;
        for (int32_t j = threadIdx.x; j <= 2 * i; j += blockDim.x) {
            if (j == vi) continue;
            float val = row_tmp[j];
            if (val < b_val || (val == b_val && j < b_j)) {
                b_val = val;
                b_j = j;
            }
        }
        s_best_val[threadIdx.x] = b_val;
        s_best_j[threadIdx.x]   = b_j;
    }
    __syncthreads();

    for (int32_t stride = 64; stride > 0; stride >>= 1) {
        if (SIMT_PHASE_ACTIVE && threadIdx.x < stride) {
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

    if (SIMT_PHASE_ACTIVE && threadIdx.x == 0) {
        best_delta_out[i] = s_best_val[0];
        best_j_out[i]     = s_best_j[0];
    }
}

/* ============================================================================
 * COLUMN-TILED LOW-MEMORY PIPELINE (`k_tiled_leaf_cols` + `k_spr_rows_tiled`)
 * ============================================================================ */
__global__ void k_tiled_leaf_cols(
    const float * __restrict__ D,
    float * __restrict__ tile_buf,
    float * __restrict__ subdist_int_leaf,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    const int32_t * __restrict__ tin,
    int32_t c_start,
    int32_t c_len,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t root_u_c0,
    int32_t root_u_c1,
    int32_t root_u_c2
) {
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= c_len) return;
    int32_t root_u = n_nodes_u - 1;
    int32_t v_node = c_start + x;

    for (int32_t i = 0; i < n_leaves; ++i) {
        tile_buf[(size_t)i * c_len + x] = D[(size_t)i * n_leaves + v_node];
    }
    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        tile_buf[(size_t)u * c_len + x] =
            0.5f * (tile_buf[(size_t)c1 * c_len + x] + tile_buf[(size_t)c2 * c_len + x]);
    }

    int32_t tv = tin[v_node];
    int32_t curr;
    if (tv < tin[root_u_c1]) {
        curr = root_u_c0;
        tile_buf[(size_t)root_u * c_len + x] =
            0.5f * (tile_buf[(size_t)root_u_c1 * c_len + x] + tile_buf[(size_t)root_u_c2 * c_len + x]);
    } else if (tv < tin[root_u_c2]) {
        curr = root_u_c1;
        tile_buf[(size_t)root_u * c_len + x] =
            0.5f * (tile_buf[(size_t)root_u_c0 * c_len + x] + tile_buf[(size_t)root_u_c2 * c_len + x]);
    } else {
        curr = root_u_c2;
        tile_buf[(size_t)root_u * c_len + x] =
            0.5f * (tile_buf[(size_t)root_u_c0 * c_len + x] + tile_buf[(size_t)root_u_c1 * c_len + x]);
    }
    int32_t g = root_u;
    while (curr != v_node) {
        int32_t c1 = child1_r[curr], c2 = child2_r[curr];
        int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
        int32_t sib       = (tv < tin[c2]) ? c2 : c1;
        tile_buf[(size_t)curr * c_len + x] =
            0.5f * (tile_buf[(size_t)sib * c_len + x] + tile_buf[(size_t)g * c_len + x]);
        g = curr;
        curr = next_curr;
    }

    for (int32_t u = n_leaves; u <= root_u; ++u) {
        subdist_int_leaf[(size_t)(u - n_leaves) * n_leaves + v_node] = tile_buf[(size_t)u * c_len + x];
    }
}

__device__ __forceinline__ void dev_finish_single_autonomous_row(
    int32_t w, int32_t n_leaves, int32_t n_nodes_u,
    int32_t root_u_c0, int32_t root_u_c1, int32_t root_u_c2,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    const int32_t * __restrict__ tin,
    float * __restrict__ row_w
) {
    int32_t root_u = n_nodes_u - 1;
    for (int32_t u = n_leaves; u < root_u; ++u) {
        row_w[u] = 0.5f * (row_w[child1_r[u]] + row_w[child2_r[u]]);
    }
    if (w == root_u) {
        row_w[root_u] = 0.0f;
    } else {
        int32_t tv = tin[w];
        int32_t curr;
        if (tv < tin[root_u_c1]) {
            curr = root_u_c0;
            row_w[root_u] = 0.5f * (row_w[root_u_c1] + row_w[root_u_c2]);
        } else if (tv < tin[root_u_c2]) {
            curr = root_u_c1;
            row_w[root_u] = 0.5f * (row_w[root_u_c0] + row_w[root_u_c2]);
        } else {
            curr = root_u_c2;
            row_w[root_u] = 0.5f * (row_w[root_u_c0] + row_w[root_u_c1]);
        }
        int32_t g = root_u;
        while (curr != w) {
            int32_t c1 = child1_r[curr], c2 = child2_r[curr];
            int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
            int32_t sib       = (tv < tin[c2]) ? c2 : c1;
            row_w[curr] = 0.5f * (row_w[sib] + row_w[g]);
            g = curr;
            curr = next_curr;
        }
        row_w[w] = 0.0f;
    }
}

__device__ __forceinline__ void dev_compute_3_autonomous_rows(
    int32_t w0, int32_t w1, int32_t w2,
    int32_t n_leaves, int32_t n_nodes_u,
    int32_t root_u_c0, int32_t root_u_c1, int32_t root_u_c2,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    const int32_t * __restrict__ tin,
    const float * __restrict__ D,
    const float * __restrict__ subdist_int_leaf,
    float * __restrict__ row0,
    float * __restrict__ row1,
    float * __restrict__ row2
) {
    const float *src0 = (w0 < n_leaves) ? &D[(size_t)w0 * n_leaves] : &subdist_int_leaf[(size_t)(w0 - n_leaves) * n_leaves];
    const float *src1 = (w1 < n_leaves) ? &D[(size_t)w1 * n_leaves] : &subdist_int_leaf[(size_t)(w1 - n_leaves) * n_leaves];
    const float *src2 = (w2 < n_leaves) ? &D[(size_t)w2 * n_leaves] : &subdist_int_leaf[(size_t)(w2 - n_leaves) * n_leaves];
    if (SIMT_PHASE_ACTIVE) {
        for (int32_t x = threadIdx.x; x < n_leaves; x += blockDim.x) {
            row0[x] = src0[x];
            row1[x] = src1[x];
            row2[x] = src2[x];
        }
    }
    __syncthreads();

    if (SIMT_PHASE_ACTIVE) {
        if (threadIdx.x == 0) {
            dev_finish_single_autonomous_row(w0, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2, child1_r, child2_r, tin, row0);
        } else if (threadIdx.x == 32) {
            dev_finish_single_autonomous_row(w1, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2, child1_r, child2_r, tin, row1);
        } else if (threadIdx.x == 64) {
            dev_finish_single_autonomous_row(w2, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2, child1_r, child2_r, tin, row2);
        }
    }
    __syncthreads();
}

__device__ __forceinline__ int32_t dev_get_auto_target_node(
    int32_t r_batch_start,
    int32_t k,
    int32_t root_u,
    int32_t other_root_child,
    const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ adj_u
) {
    int32_t b = k / 3;
    int32_t m = k - 3 * b;
    int32_t v_index = r_batch_start + b;
    int32_t tree_root = v_map[v_index];
    if (m == 0) return tree_root;
    int32_t parent = (tree_root == root_u) ? other_root_child : parent_u[tree_root];
    int32_t other0 = -1, other1 = -1;
    for (int32_t j = 0; j < 3; ++j) {
        int32_t nb_node = adj_u[3 * parent + j];
        if (nb_node != -1 && nb_node != tree_root) {
            if (other0 == -1) other0 = nb_node;
            else other1 = nb_node;
        }
    }
    return (m == 1) ? other0 : other1;
}

__global__ void k_load_auto_leaves_tiled(
    const float * __restrict__ D,
    const float * __restrict__ subdist_int_leaf,
    float * __restrict__ auto_col_buf,
    const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ adj_u,
    int32_t r_batch_start,
    int32_t K,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t other_root_child
) {
#ifdef __CUDACC__
    __shared__ float tile[16][17];
    int32_t root_u = n_nodes_u - 1;
    int32_t x_in = blockIdx.x * 16 + threadIdx.x;
    int32_t k_in = blockIdx.y * 16 + threadIdx.y;
    if (x_in < n_leaves && k_in < K) {
        int32_t w = dev_get_auto_target_node(r_batch_start, k_in, root_u, other_root_child, v_map, parent_u, adj_u);
        tile[threadIdx.y][threadIdx.x] = (w < n_leaves)
            ? D[(size_t)w * n_leaves + x_in]
            : subdist_int_leaf[(size_t)(w - n_leaves) * n_leaves + x_in];
    }
    __syncthreads();
    int32_t k_out = blockIdx.y * 16 + threadIdx.x;
    int32_t x_out = blockIdx.x * 16 + threadIdx.y;
    if (x_out < n_leaves && k_out < K) {
        auto_col_buf[(size_t)x_out * K + k_out] = tile[threadIdx.x][threadIdx.y];
    }
#endif
}

__global__ void k_reconstruct_auto_cols(
    float * __restrict__ auto_col_buf,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    const int32_t * __restrict__ tin,
    const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ adj_u,
    int32_t r_batch_start,
    int32_t K,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t root_u_c0,
    int32_t root_u_c1,
    int32_t root_u_c2,
    int32_t other_root_child
) {
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    int32_t root_u = n_nodes_u - 1;
    int32_t w = dev_get_auto_target_node(r_batch_start, k, root_u, other_root_child, v_map, parent_u, adj_u);

    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        auto_col_buf[(size_t)u * K + k] =
            0.5f * (auto_col_buf[(size_t)c1 * K + k] + auto_col_buf[(size_t)c2 * K + k]);
    }
    if (w == root_u) {
        auto_col_buf[(size_t)root_u * K + k] = 0.0f;
    } else {
        int32_t tv = tin[w];
        int32_t curr;
        if (tv < tin[root_u_c1]) {
            curr = root_u_c0;
            auto_col_buf[(size_t)root_u * K + k] =
                0.5f * (auto_col_buf[(size_t)root_u_c1 * K + k] + auto_col_buf[(size_t)root_u_c2 * K + k]);
        } else if (tv < tin[root_u_c2]) {
            curr = root_u_c1;
            auto_col_buf[(size_t)root_u * K + k] =
                0.5f * (auto_col_buf[(size_t)root_u_c0 * K + k] + auto_col_buf[(size_t)root_u_c2 * K + k]);
        } else {
            curr = root_u_c2;
            auto_col_buf[(size_t)root_u * K + k] =
                0.5f * (auto_col_buf[(size_t)root_u_c0 * K + k] + auto_col_buf[(size_t)root_u_c1 * K + k]);
        }
        int32_t g = root_u;
        while (curr != w) {
            int32_t c1 = child1_r[curr], c2 = child2_r[curr];
            int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
            int32_t sib       = (tv < tin[c2]) ? c2 : c1;
            auto_col_buf[(size_t)curr * K + k] =
                0.5f * (auto_col_buf[(size_t)sib * K + k] + auto_col_buf[(size_t)g * K + k]);
            g = curr;
            curr = next_curr;
        }
        auto_col_buf[(size_t)w * K + k] = 0.0f;
    }
}

__global__ void k_transpose_auto_cols(
    const float * __restrict__ auto_col_buf,
    float * __restrict__ auto_row_pool,
    int32_t K,
    int32_t n_nodes_u
) {
#ifdef __CUDACC__
    __shared__ float tile[16][17];
    int32_t k_in = blockIdx.x * 16 + threadIdx.x;
    int32_t u_in = blockIdx.y * 16 + threadIdx.y;
    if (k_in < K && u_in < n_nodes_u) {
        tile[threadIdx.y][threadIdx.x] = auto_col_buf[(size_t)u_in * K + k_in];
    }
    __syncthreads();
    int32_t u_out = blockIdx.y * 16 + threadIdx.x;
    int32_t k_out = blockIdx.x * 16 + threadIdx.y;
    if (k_out < K && u_out < n_nodes_u) {
        auto_row_pool[(size_t)k_out * n_nodes_u + u_out] = tile[threadIdx.x][threadIdx.y];
    }
#endif
}

__global__ void k_spr_rows_tiled(
    int32_t r_start,
    int32_t r_batch_offset,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t root_u_c0,
    int32_t root_u_c1,
    int32_t root_u_c2,
    int32_t other_root_child,
    int32_t precomputed_auto_rows,
    int32_t use_smem_edge,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    const int32_t * __restrict__ tin,
    const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size,
    const int32_t * __restrict__ m_lmin,
    const int32_t * __restrict__ next_k_same_lmin,
    const float * __restrict__ D,
    const float * __restrict__ subdist_int_leaf,
    const float * __restrict__ nd_u,
    float * __restrict__ auto_row_pool,
    float * __restrict__ edge_delta_pool,
    float * __restrict__ delta_out_slice
) {
#ifdef __CUDACC__
    extern __shared__ float s_edge_delta_tiled[];
    __shared__ SprSharedQueue s_q;
#else
    float *s_edge_delta_tiled = NULL;
    static SprSharedQueue s_q;
#endif
    int32_t b = blockIdx.x;
    int32_t v_index = r_start + r_batch_offset + b;
    int32_t root_u = n_nodes_u - 1;
    int32_t max_cols = 2 * (n_leaves - 1) - 1;

    float *row_root   = &auto_row_pool[(size_t)(3 * b + 0) * n_nodes_u];
    float *row_o0     = &auto_row_pool[(size_t)(3 * b + 1) * n_nodes_u];
    float *row_o1     = &auto_row_pool[(size_t)(3 * b + 2) * n_nodes_u];
    float *edge_delta = use_smem_edge ? s_edge_delta_tiled : &edge_delta_pool[(size_t)b * n_nodes_u];

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

    if (!precomputed_auto_rows) {
        dev_compute_3_autonomous_rows(
            tree_root, other0, other1, n_leaves, n_nodes_u,
            root_u_c0, root_u_c1, root_u_c2, child1_r, child2_r, tin,
            D, subdist_int_leaf, row_root, row_o0, row_o1
        );
    }

    dev_run_parallel_spr_dfs(
        tree_root, parent, other0, other1, n_leaves, n_nodes_u,
        adj_u, parent_u, sub_size, nd_u, row_root, row_o0, row_o1, edge_delta, &s_q
    );

    int32_t i = v_index - 1;
    float *out_row = &delta_out_slice[(size_t)(v_index - r_start) * max_cols];
    dev_map_edge_delta_to_row(i, n_leaves, other_root_child, m_lmin, next_k_same_lmin, edge_delta, out_row);
}

__global__ void k_load_nd_u_leaves_tiled(
    const float * __restrict__ D,
    const float * __restrict__ subdist_int_leaf,
    float * __restrict__ auto_col_buf,
    const int32_t * __restrict__ adj_u,
    int32_t u_batch_start,
    int32_t K,
    int32_t n_leaves
) {
#ifdef __CUDACC__
    __shared__ float tile[16][17];
    int32_t x_in = blockIdx.x * 16 + threadIdx.x;
    int32_t k_in = blockIdx.y * 16 + threadIdx.y;
    if (x_in < n_leaves && k_in < K) {
        int32_t b = k_in >> 1;
        int32_t m = k_in & 1;
        int32_t u = u_batch_start + b;
        int32_t w = adj_u[3 * u + m];
        tile[threadIdx.y][threadIdx.x] = (w < n_leaves)
            ? D[(size_t)w * n_leaves + x_in]
            : subdist_int_leaf[(size_t)(w - n_leaves) * n_leaves + x_in];
    }
    __syncthreads();
    int32_t k_out = blockIdx.y * 16 + threadIdx.x;
    int32_t x_out = blockIdx.x * 16 + threadIdx.y;
    if (x_out < n_leaves && k_out < K) {
        auto_col_buf[(size_t)x_out * K + k_out] = tile[threadIdx.x][threadIdx.y];
    }
#endif
}

__global__ void k_reconstruct_nd_u_cols(
    float * __restrict__ auto_col_buf,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    const int32_t * __restrict__ tin,
    const int32_t * __restrict__ adj_u,
    float * __restrict__ nd_u,
    int32_t u_batch_start,
    int32_t K,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t root_u_c0,
    int32_t root_u_c1,
    int32_t root_u_c2
) {
    int32_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K) {
        int32_t root_u = n_nodes_u - 1;
        int32_t b = k >> 1;
        int32_t m = k & 1;
        int32_t u_node = u_batch_start + b;
        int32_t w = adj_u[3 * u_node + m];

        for (int32_t u = n_leaves; u < root_u; ++u) {
            int32_t c1 = child1_r[u], c2 = child2_r[u];
            auto_col_buf[(size_t)u * K + k] =
                0.5f * (auto_col_buf[(size_t)c1 * K + k] + auto_col_buf[(size_t)c2 * K + k]);
        }
        if (w == root_u) {
            auto_col_buf[(size_t)root_u * K + k] = 0.0f;
        } else {
            int32_t tv = tin[w];
            int32_t curr;
            if (tv < tin[root_u_c1]) {
                curr = root_u_c0;
                auto_col_buf[(size_t)root_u * K + k] =
                    0.5f * (auto_col_buf[(size_t)root_u_c1 * K + k] + auto_col_buf[(size_t)root_u_c2 * K + k]);
            } else if (tv < tin[root_u_c2]) {
                curr = root_u_c1;
                auto_col_buf[(size_t)root_u * K + k] =
                    0.5f * (auto_col_buf[(size_t)root_u_c0 * K + k] + auto_col_buf[(size_t)root_u_c2 * K + k]);
            } else {
                curr = root_u_c2;
                auto_col_buf[(size_t)root_u * K + k] =
                    0.5f * (auto_col_buf[(size_t)root_u_c0 * K + k] + auto_col_buf[(size_t)root_u_c1 * K + k]);
            }
            int32_t g = root_u;
            while (curr != w) {
                int32_t c1 = child1_r[curr], c2 = child2_r[curr];
                int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
                int32_t sib       = (tv < tin[c2]) ? c2 : c1;
                auto_col_buf[(size_t)curr * K + k] =
                    0.5f * (auto_col_buf[(size_t)sib * K + k] + auto_col_buf[(size_t)g * K + k]);
                g = curr;
                curr = next_curr;
            }
            auto_col_buf[(size_t)w * K + k] = 0.0f;
        }
    }
    __syncthreads();
    if (k < K && (k & 1) == 0) {
        int32_t b = k >> 1;
        int32_t u_node = u_batch_start + b;
        int32_t n1 = adj_u[3 * u_node + 1];
        int32_t n2 = adj_u[3 * u_node + 2];
        nd_u[3 * u_node + 0] = auto_col_buf[(size_t)n2 * K + (k + 1)];
        nd_u[3 * u_node + 1] = auto_col_buf[(size_t)n2 * K + k];
        nd_u[3 * u_node + 2] = auto_col_buf[(size_t)n1 * K + k];
    }
}

__global__ void k_spr_rows_tiled_top1(
    int32_t r_start,
    int32_t r_batch_offset,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t other_root_child,
    int32_t use_smem_edge,
    const int64_t * __restrict__ v_curr,
    const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ adj_u,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ sub_size,
    const int32_t * __restrict__ m_lmin,
    const int32_t * __restrict__ next_k_same_lmin,
    const float * __restrict__ nd_u,
    float * __restrict__ auto_row_pool,
    float * __restrict__ edge_delta_pool,
    float * __restrict__ best_delta_out,
    int32_t * __restrict__ best_j_out
) {
#ifdef __CUDACC__
    extern __shared__ float s_edge_delta_top1[];
    __shared__ SprSharedQueue s_q;
    __shared__ float s_best_val[128];
    __shared__ int32_t s_best_j[128];
#else
    float *s_edge_delta_top1 = NULL;
    static SprSharedQueue s_q;
    static float s_best_val[128];
    static int32_t s_best_j[128];
#endif
    int32_t b = blockIdx.x;
    int32_t v_index = r_start + r_batch_offset + b;
    int32_t i = v_index - 1;
    int32_t root_u = n_nodes_u - 1;

    float *row_root   = &auto_row_pool[(size_t)(3 * b + 0) * n_nodes_u];
    float *row_o0     = &auto_row_pool[(size_t)(3 * b + 1) * n_nodes_u];
    float *row_o1     = &auto_row_pool[(size_t)(3 * b + 2) * n_nodes_u];
    float *edge_delta = use_smem_edge ? s_edge_delta_top1 : &edge_delta_pool[(size_t)b * n_nodes_u];

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
        adj_u, parent_u, sub_size, nd_u, row_root, row_o0, row_o1, edge_delta, &s_q
    );

    /* Reuse block b's 3*n_nodes_u slice in auto_row_pool as row_tmp */
    float *row_tmp = row_root;
    dev_map_edge_delta_to_row(i, n_leaves, other_root_child, m_lmin, next_k_same_lmin, edge_delta, row_tmp);
    __syncthreads();

    if (SIMT_PHASE_ACTIVE) {
        int32_t vi = (int32_t)v_curr[i];
        float b_val = 0.0f;
        int32_t b_j = vi;
        for (int32_t j = threadIdx.x; j <= 2 * i; j += blockDim.x) {
            if (j == vi) continue;
            float val = row_tmp[j];
            if (val < b_val || (val == b_val && j < b_j)) {
                b_val = val;
                b_j = j;
            }
        }
        s_best_val[threadIdx.x] = b_val;
        s_best_j[threadIdx.x]   = b_j;
    }
    __syncthreads();

    for (int32_t stride = 64; stride > 0; stride >>= 1) {
        if (SIMT_PHASE_ACTIVE && threadIdx.x < stride) {
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

    if (SIMT_PHASE_ACTIVE && threadIdx.x == 0) {
        best_delta_out[i] = s_best_val[0];
        best_j_out[i]     = s_best_j[0];
    }
}

__device__ __forceinline__ float dev_synth_D_entry(int32_t i, int32_t j, uint32_t seed) {
    if (i == j) return 0.0f;
    uint32_t a = (uint32_t)((i < j) ? i : j);
    uint32_t b = (uint32_t)((i < j) ? j : i);
    uint32_t h = seed ^ (a * 0x9E3779B9U) ^ (b * 0x85EBCA6BU);
    h ^= h >> 16; h *= 0x7FEB352DU; h ^= h >> 15; h *= 0x846CA68BU; h ^= h >> 16;
    return 0.1f + 2.0f * ((float)(h & 0xFFFFFFU) / 16777216.0f);
}

__global__ void k_init_symmetric_D(float * __restrict__ D, int32_t N, uint32_t seed) {
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        D[(size_t)i * N + j] = dev_synth_D_entry(i, j, seed);
    }
}

__global__ void k_init_D_shard(
    float * __restrict__ D_shard, int32_t N, int32_t c_shard_start, int32_t W_shard, uint32_t seed
) {
    int32_t x_loc = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t i     = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && x_loc < W_shard) {
        D_shard[(size_t)i * W_shard + x_loc] = dev_synth_D_entry(i, c_shard_start + x_loc, seed);
    }
}

__global__ void k_tiled_leaf_cols_sharded(
    const float * __restrict__ D_shard,
    uint32_t seed_fallback,
    float * __restrict__ tile_buf,
    float * __restrict__ subdist_il_shard,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    const int32_t * __restrict__ tin,
    int32_t c_shard_start,
    int32_t W_shard,
    int32_t t_local_start,
    int32_t c_len,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t root_u_c0,
    int32_t root_u_c1,
    int32_t root_u_c2
) {
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= c_len) return;
    int32_t root_u = n_nodes_u - 1;
    int32_t x_loc  = t_local_start + x;
    int32_t v_node = c_shard_start + x_loc;

    if (D_shard != NULL) {
        for (int32_t i = 0; i < n_leaves; ++i) {
            tile_buf[(size_t)i * c_len + x] = D_shard[(size_t)i * W_shard + x_loc];
        }
    } else {
        for (int32_t i = 0; i < n_leaves; ++i) {
            tile_buf[(size_t)i * c_len + x] = dev_synth_D_entry(i, v_node, seed_fallback);
        }
    }
    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        tile_buf[(size_t)u * c_len + x] =
            0.5f * (tile_buf[(size_t)c1 * c_len + x] + tile_buf[(size_t)c2 * c_len + x]);
    }

    int32_t tv = tin[v_node];
    int32_t curr;
    if (tv < tin[root_u_c1]) {
        curr = root_u_c0;
        tile_buf[(size_t)root_u * c_len + x] =
            0.5f * (tile_buf[(size_t)root_u_c1 * c_len + x] + tile_buf[(size_t)root_u_c2 * c_len + x]);
    } else if (tv < tin[root_u_c2]) {
        curr = root_u_c1;
        tile_buf[(size_t)root_u * c_len + x] =
            0.5f * (tile_buf[(size_t)root_u_c0 * c_len + x] + tile_buf[(size_t)root_u_c2 * c_len + x]);
    } else {
        curr = root_u_c2;
        tile_buf[(size_t)root_u * c_len + x] =
            0.5f * (tile_buf[(size_t)root_u_c0 * c_len + x] + tile_buf[(size_t)root_u_c1 * c_len + x]);
    }
    int32_t g = root_u;
    while (curr != v_node) {
        int32_t c1 = child1_r[curr], c2 = child2_r[curr];
        int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
        int32_t sib       = (tv < tin[c2]) ? c2 : c1;
        tile_buf[(size_t)curr * c_len + x] =
            0.5f * (tile_buf[(size_t)sib * c_len + x] + tile_buf[(size_t)g * c_len + x]);
        g = curr;
        curr = next_curr;
    }

    for (int32_t u = n_leaves; u <= root_u; ++u) {
        subdist_il_shard[(size_t)(u - n_leaves) * W_shard + x_loc] = tile_buf[(size_t)u * c_len + x];
    }
}

__global__ void k_load_nd_u_leaves_sharded(
    const float * __restrict__ D_shard,
    uint32_t seed_fallback,
    const float * __restrict__ subdist_il_shard,
    float * __restrict__ stage_col_buf,
    const int32_t * __restrict__ adj_u,
    int32_t c_shard_start,
    int32_t W_shard,
    int32_t u_batch_start,
    int32_t K,
    int32_t n_leaves
) {
#ifdef __CUDACC__
    __shared__ float tile[16][17];
    int32_t x_loc = blockIdx.x * 16 + threadIdx.x;
    int32_t k_in  = blockIdx.y * 16 + threadIdx.y;
    if (x_loc < W_shard && k_in < K) {
        int32_t b = k_in >> 1;
        int32_t m = k_in & 1;
        int32_t u = u_batch_start + b;
        int32_t w = adj_u[3 * u + m];
        if (w < n_leaves) {
            tile[threadIdx.y][threadIdx.x] = (D_shard != NULL)
                ? D_shard[(size_t)w * W_shard + x_loc]
                : dev_synth_D_entry(w, c_shard_start + x_loc, seed_fallback);
        } else {
            tile[threadIdx.y][threadIdx.x] = subdist_il_shard[(size_t)(w - n_leaves) * W_shard + x_loc];
        }
    }
    __syncthreads();
    int32_t k_out = blockIdx.y * 16 + threadIdx.x;
    int32_t x_out = blockIdx.x * 16 + threadIdx.y;
    if (x_out < W_shard && k_out < K) {
        stage_col_buf[(size_t)x_out * K + k_out] = tile[threadIdx.x][threadIdx.y];
    }
#endif
}

__global__ void k_load_auto_leaves_sharded(
    const float * __restrict__ D_shard,
    uint32_t seed_fallback,
    const float * __restrict__ subdist_il_shard,
    float * __restrict__ stage_col_buf,
    const int32_t * __restrict__ v_map,
    const int32_t * __restrict__ parent_u,
    const int32_t * __restrict__ adj_u,
    int32_t c_shard_start,
    int32_t W_shard,
    int32_t v_idx_start,
    int32_t K,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t other_root_child
) {
#ifdef __CUDACC__
    __shared__ float tile[16][17];
    int32_t x_loc = blockIdx.x * 16 + threadIdx.x;
    int32_t k_in  = blockIdx.y * 16 + threadIdx.y;
    if (x_loc < W_shard && k_in < K) {
        int32_t b = k_in / 3;
        int32_t m = k_in - 3 * b;
        int32_t v_index = v_idx_start + b;
        int32_t root_u = n_nodes_u - 1;
        int32_t tr = v_map[v_index];
        int32_t par = (tr == root_u) ? other_root_child : parent_u[tr];
        int32_t o0 = -1, o1 = -1;
        for (int32_t j = 0; j < 3; ++j) {
            int32_t nb = adj_u[3 * par + j];
            if (nb != -1 && nb != tr) {
                if (o0 == -1) o0 = nb;
                else o1 = nb;
            }
        }
        int32_t w = (m == 0) ? tr : ((m == 1) ? o0 : o1);
        if (w < n_leaves) {
            tile[threadIdx.y][threadIdx.x] = (D_shard != NULL)
                ? D_shard[(size_t)w * W_shard + x_loc]
                : dev_synth_D_entry(w, c_shard_start + x_loc, seed_fallback);
        } else {
            tile[threadIdx.y][threadIdx.x] = subdist_il_shard[(size_t)(w - n_leaves) * W_shard + x_loc];
        }
    }
    __syncthreads();
    int32_t k_out = blockIdx.y * 16 + threadIdx.x;
    int32_t x_out = blockIdx.x * 16 + threadIdx.y;
    if (x_out < W_shard && k_out < K) {
        stage_col_buf[(size_t)x_out * K + k_out] = tile[threadIdx.x][threadIdx.y];
    }
#endif
}

/* ============================================================================
 * CLADE-CONDENSED MINIBATCH KERNELS FOR 1M - 10M+ TAXA (Theorem 5.5)
 * ============================================================================ */
__host__ __device__ static inline void dev_taxon_feature_vec(
    int32_t x, uint32_t seed, float out_phi[8], float *out_norm2
) {
    uint32_t h = (uint32_t)x * 0x9E3779B9U ^ seed;
    float n2 = 0.0f;
    for (int d = 0; d < 8; ++d) {
        h ^= h >> 16;
        h *= 0x85EBCA6BU;
        h ^= h >> 13;
        h *= 0xC2B2AE35U;
        h ^= h >> 16;
        float val = 0.1f + (float)(h & 0xFFFFU) * (1.0f / 65536.0f);
        out_phi[d] = val;
        n2 += val * val;
    }
    *out_norm2 = n2;
}

__global__ void k_condense_super_profiles(
    const int32_t * __restrict__ leaf_to_super,
    const float * __restrict__ omega_leaf,
    float * __restrict__ phi_bar,
    float * __restrict__ mu2_bar,
    int32_t n_total,
    uint32_t seed
) {
#ifdef __CUDACC__
    int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < n_total) {
        int32_t z = leaf_to_super[x];
        float w = omega_leaf[x];
        float phi[8], n2 = 0.0f;
        dev_taxon_feature_vec(x, seed, phi, &n2);
        float *dst = &phi_bar[(size_t)z * 8];
        for (int d = 0; d < 8; ++d) {
            atomicAdd(&dst[d], w * phi[d]);
        }
        atomicAdd(&mu2_bar[z], w * n2);
    }
#endif
}

__global__ void k_build_condensed_D(
    const float * __restrict__ phi_bar,
    const float * __restrict__ mu2_bar,
    float * __restrict__ D_eff,
    int32_t n_eff
) {
#ifdef __CUDACC__
    int32_t z2 = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t z1 = blockIdx.y * blockDim.y + threadIdx.y;
    if (z1 < n_eff && z2 < n_eff) {
        if (z1 == z2) {
            D_eff[(size_t)z1 * n_eff + z2] = 0.0f;
        } else {
            const float *p1 = &phi_bar[(size_t)z1 * 8];
            const float *p2 = &phi_bar[(size_t)z2 * 8];
            float dot = 0.0f;
            for (int d = 0; d < 8; ++d) dot += p1[d] * p2[d];
            float dist = mu2_bar[z1] + mu2_bar[z2] - 2.0f * dot;
            D_eff[(size_t)z1 * n_eff + z2] = (dist > 0.0f) ? dist : 0.0f;
        }
    }
#endif
}

/* ============================================================================
 * HOST TREE SETUP (Stage 1: O(N) Tree Decoding + `next_k_same_lmin` Table)
 * ============================================================================ */
struct HostTreeTopology {
    int32_t n_leaves;
    int32_t n_pairs;
    int32_t n_nodes_r;
    int32_t n_nodes_u;
    int32_t root_u;
    int32_t other_root_child;
    int32_t root_u_c0, root_u_c1, root_u_c2;
    int32_t *child1_r;
    int32_t *child2_r;
    int32_t *v_map;
    int32_t *m_lmin;
    int32_t *next_k_same_lmin;
    int32_t *parent_u;
    int32_t *sub_size;
    int32_t *tin;
    int32_t *adj_u;
};

static void build_host_tree_topology(const int64_t *v, int32_t n_leaves, HostTreeTopology *T) {
    T->n_leaves = n_leaves;
    T->n_pairs = n_leaves - 1;
    T->n_nodes_r = 2 * n_leaves - 1;
    T->n_nodes_u = 2 * n_leaves - 2;
    T->root_u = T->n_nodes_u - 1;

    int32_t n_pairs = T->n_pairs;
    int32_t n_nodes_r = T->n_nodes_r;
    int32_t n_nodes_u = T->n_nodes_u;
    int32_t root_u = T->root_u;

    T->child1_r = (int32_t *)malloc((size_t)n_nodes_r * sizeof(int32_t));
    T->child2_r = (int32_t *)malloc((size_t)n_nodes_r * sizeof(int32_t));
    T->v_map    = (int32_t *)calloc((size_t)n_leaves, sizeof(int32_t));
    T->m_lmin   = (int32_t *)malloc((size_t)n_pairs * sizeof(int32_t));
    T->next_k_same_lmin = (int32_t *)malloc((size_t)n_pairs * sizeof(int32_t));
    T->parent_u = (int32_t *)malloc((size_t)n_nodes_u * sizeof(int32_t));
    T->sub_size = (int32_t *)malloc((size_t)n_nodes_u * sizeof(int32_t));
    T->tin      = (int32_t *)malloc((size_t)n_nodes_u * sizeof(int32_t));
    T->adj_u    = (int32_t *)malloc((size_t)(3 * n_nodes_u) * sizeof(int32_t));

    int32_t *parents_uf = (int32_t *)malloc((size_t)n_leaves * sizeof(int32_t));
    int32_t *last_seen = (int32_t *)malloc((size_t)n_leaves * sizeof(int32_t));

    /* Branchless O(N) cherry-first component-root tree builder */
    for (int32_t i = 0; i < n_leaves; ++i) parents_uf[i] = i;
    for (int32_t k = 0; k < n_pairs; ++k) {
        int32_t i = n_pairs - 1 - k;
        int32_t x = (int32_t)v[i];
        int32_t y = i + 1;
        int32_t c1 = parents_uf[y];
        int32_t c2 = parents_uf[x];
        int32_t m = n_leaves + k;
        T->child1_r[m] = c1;
        T->child2_r[m] = c2;
        T->v_map[y] = c1;
        T->m_lmin[k] = x;
        parents_uf[x] = m;
    }

    for (int32_t j = 0; j < n_leaves; ++j) last_seen[j] = n_leaves;
    for (int32_t k = n_pairs - 1; k >= 0; --k) {
        int32_t lmin = T->m_lmin[k];
        T->next_k_same_lmin[k] = last_seen[lmin];
        last_seen[lmin] = k;
    }

    int32_t rc1 = T->child1_r[n_nodes_r - 1], rc2 = T->child2_r[n_nodes_r - 1];
    T->other_root_child = (rc1 < rc2) ? rc1 : rc2;
    T->root_u_c0 = T->child1_r[root_u];
    T->root_u_c1 = T->child2_r[root_u];
    T->root_u_c2 = T->other_root_child;

    for (int32_t u = 0; u < n_leaves; ++u) T->sub_size[u] = 1;
    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = T->child1_r[u], c2 = T->child2_r[u];
        T->parent_u[c1] = u; T->parent_u[c2] = u;
        T->sub_size[u] = 1 + T->sub_size[c1] + T->sub_size[c2];
        T->adj_u[3 * u + 0] = c1; T->adj_u[3 * u + 1] = c2;
    }
    T->parent_u[T->root_u_c0] = root_u;
    T->parent_u[T->root_u_c1] = root_u;
    T->parent_u[T->root_u_c2] = root_u;
    T->parent_u[root_u] = -1;

    for (int32_t u = 0; u < n_leaves; ++u) {
        T->adj_u[3 * u + 0] = T->parent_u[u];
        T->adj_u[3 * u + 1] = -1;
        T->adj_u[3 * u + 2] = -1;
    }
    for (int32_t u = n_leaves; u < root_u; ++u) T->adj_u[3 * u + 2] = T->parent_u[u];
    T->adj_u[3 * root_u + 0] = T->root_u_c0;
    T->adj_u[3 * root_u + 1] = T->root_u_c1;
    T->adj_u[3 * root_u + 2] = T->root_u_c2;

    T->tin[root_u] = 0;
    T->tin[T->root_u_c0] = 1;
    T->tin[T->root_u_c1] = 1 + T->sub_size[T->root_u_c0];
    T->tin[T->root_u_c2] = 1 + T->sub_size[T->root_u_c0] + T->sub_size[T->root_u_c1];
    for (int32_t u = root_u - 1; u >= n_leaves; --u) {
        int32_t c1 = T->child1_r[u], c2 = T->child2_r[u];
        T->tin[c1] = T->tin[u] + 1;
        T->tin[c2] = T->tin[u] + 1 + T->sub_size[c1];
    }

    free(parents_uf);
    free(last_seen);
}

static void free_host_tree_topology(HostTreeTopology *T) {
    free(T->child1_r); free(T->child2_r); free(T->v_map);
    free(T->m_lmin); free(T->next_k_same_lmin);
    free(T->parent_u); free(T->sub_size); free(T->tin); free(T->adj_u);
}

/* ============================================================================
 * COMPLETE EXECUTION & VALIDATION HARNESS (CUDA Runtime OR Host SIMT Verifier)
 * ============================================================================ */
static int run_and_validate_cuda_pipeline(
    const int64_t *v, const double *D_f64, int32_t N,
    double *subdist_ms_out, double *spr_smem_ms_out, double *spr_tiled_ms_out,
    double *max_err_smem_out, double *max_err_gmem_out, double *max_err_tiled_out,
    int reverse_thread_order
) {
    HostTreeTopology T;
    build_host_tree_topology(v, N, &T);

    int32_t n_pairs = T.n_pairs;
    int32_t n_nodes_r = T.n_nodes_r;
    int32_t n_nodes_u = T.n_nodes_u;
    int32_t max_cols = 2 * n_pairs - 1;
    int32_t n_rows_spr = N - 2;

    float *h_D = (float *)malloc((size_t)N * N * sizeof(float));
    for (size_t idx = 0; idx < (size_t)N * N; ++idx) h_D[idx] = (float)D_f64[idx];

    double *delta_ref = (double *)calloc((size_t)n_pairs * max_cols, sizeof(double));
    calculate_discrete_gradient_full_c(v, D_f64, N, delta_ref);

    float *h_delta_smem  = (float *)calloc((size_t)n_pairs * max_cols, sizeof(float));
    float *h_delta_gmem  = (float *)calloc((size_t)n_pairs * max_cols, sizeof(float));
    float *h_delta_tiled = (float *)calloc((size_t)n_pairs * max_cols, sizeof(float));
    float *h_best_delta  = (float *)calloc((size_t)n_pairs, sizeof(float));
    int32_t *h_best_j    = (int32_t *)calloc((size_t)n_pairs, sizeof(int32_t));

#ifdef __CUDACC__
    (void)reverse_thread_order;
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));
    int max_smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    size_t req_smem = (size_t)n_nodes_u * sizeof(float);
    int use_smem_edge = (req_smem + 8192 <= (size_t)max_smem_optin) ? 1 : 0;
    if (use_smem_edge) {
        CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req_smem));
        CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_top1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req_smem));
        CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_tiled, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req_smem));
    }

    int32_t *d_child1_r, *d_child2_r, *d_v_map, *d_m_lmin, *d_next_k, *d_parent_u, *d_sub_size, *d_tin, *d_adj_u, *d_best_j;
    int64_t *d_v;
    float *d_D, *d_subdist, *d_nd_u, *d_nd_u_tiled, *d_delta_smem, *d_delta_gmem, *d_delta_tiled, *d_edge_pool, *d_row_tmp, *d_best_delta;
    float *d_subdist_int_leaf, *d_tile_buf, *d_auto_col_buf, *d_auto_pool;

    const int32_t B_TILE = (N < 2048) ? N : 2048;
    const int32_t B_ROW_TILE = (n_rows_spr < 1024) ? n_rows_spr : 1024;
    CUDA_CHECK(cudaMalloc(&d_child1_r, (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_child2_r, (size_t)n_nodes_r * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_v_map,    (size_t)N * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_m_lmin,   (size_t)n_pairs * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_k,   (size_t)n_pairs * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_parent_u, (size_t)n_nodes_u * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_sub_size, (size_t)n_nodes_u * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_tin,      (size_t)n_nodes_u * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_adj_u,    (size_t)(3 * n_nodes_u) * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_v,        (size_t)n_pairs * sizeof(int64_t)));

    CUDA_CHECK(cudaMalloc(&d_D,           (size_t)N * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_subdist,     (size_t)n_nodes_u * n_nodes_u * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nd_u,        (size_t)(3 * n_nodes_u) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_nd_u_tiled,  (size_t)(3 * n_nodes_u) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta_smem,  (size_t)n_pairs * max_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta_gmem,  (size_t)n_pairs * max_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta_tiled, (size_t)n_pairs * max_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edge_pool,   (size_t)n_rows_spr * n_nodes_u * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_tmp,     (size_t)n_rows_spr * (2 * N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_delta,  (size_t)n_pairs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_j,      (size_t)n_pairs * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_subdist_int_leaf, (size_t)(N - 1) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tile_buf,         (size_t)n_nodes_u * B_TILE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_auto_col_buf,     (size_t)3 * B_ROW_TILE * n_nodes_u * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_auto_pool,        (size_t)3 * B_ROW_TILE * n_nodes_u * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_child1_r, T.child1_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_child2_r, T.child2_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v_map,    T.v_map,    (size_t)N * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m_lmin,   T.m_lmin,   (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_next_k,   T.next_k_same_lmin, (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_parent_u, T.parent_u, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sub_size, T.sub_size, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tin,      T.tin,      (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_u,    T.adj_u,    (size_t)(3 * n_nodes_u) * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v,        v,          (size_t)n_pairs * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D,        h_D,        (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_delta_smem,  0, (size_t)n_pairs * max_cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_delta_gmem,  0, (size_t)n_pairs * max_cols * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_delta_tiled, 0, (size_t)n_pairs * max_cols * sizeof(float)));

    cudaEvent_t ev_sub0, ev_sub1, ev_smem0, ev_smem1, ev_tiled0, ev_tiled1;
    CUDA_CHECK(cudaEventCreate(&ev_sub0));
    CUDA_CHECK(cudaEventCreate(&ev_sub1));
    CUDA_CHECK(cudaEventCreate(&ev_smem0));
    CUDA_CHECK(cudaEventCreate(&ev_smem1));
    CUDA_CHECK(cudaEventCreate(&ev_tiled0));
    CUDA_CHECK(cudaEventCreate(&ev_tiled1));

    int32_t n_int = (n_nodes_u - 1) - N;
    int n_passes = (subdist_ms_out != NULL) ? 3 : 1;
    float best_ms_sub = 1e30f, best_ms_smem = 1e30f, best_ms_tiled = 1e30f;

    for (int pass = 0; pass < n_passes; ++pass) {
        CUDA_CHECK(cudaEventRecord(ev_sub0));
        k_bottom_up_leaf_cols<<<(N + 255) / 256, 256>>>(d_D, d_subdist, d_child1_r, d_child2_r, N, n_nodes_u);
        CUDA_CHECK(cudaGetLastError());
        dim3 tr_grid((N + 15) / 16, (n_int + 15) / 16);
        dim3 tr_block(16, 16);
        k_transpose_leaf_block<<<tr_grid, tr_block>>>(d_subdist, N, n_nodes_u);
        CUDA_CHECK(cudaGetLastError());
        if (n_int > 0) {
            k_bottom_up_int_cols<<<(n_int + 255) / 256, 256>>>(d_subdist, d_child1_r, d_child2_r, N, n_nodes_u);
            CUDA_CHECK(cudaGetLastError());
        }
        k_top_down_ancestor_walk<<<(n_nodes_u + 255) / 256, 256>>>(
            d_subdist, d_child1_r, d_child2_r, d_tin, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2
        );
        CUDA_CHECK(cudaGetLastError());
        k_nd_u<<<(N - 1 + 255) / 256, 256>>>(d_subdist, d_adj_u, d_nd_u, N, n_nodes_u);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_sub1));

        CUDA_CHECK(cudaEventRecord(ev_smem0));
        if (use_smem_edge) {
            k_spr_rows_smem<<<n_rows_spr, 128, req_smem>>>(
                2, N, n_nodes_u, T.other_root_child, d_v_map, d_adj_u, d_parent_u,
                d_sub_size, d_m_lmin, d_next_k, d_subdist, d_nd_u, d_delta_smem + max_cols
            );
        } else {
            k_spr_rows_gmem<<<n_rows_spr, 128>>>(
                2, N, n_nodes_u, T.other_root_child, d_v_map, d_adj_u, d_parent_u,
                d_sub_size, d_m_lmin, d_next_k, d_subdist, d_nd_u, d_edge_pool, d_delta_smem + max_cols
            );
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_smem1));

        if (pass == 0) {
            k_spr_rows_gmem<<<n_rows_spr, 128>>>(
                2, N, n_nodes_u, T.other_root_child, d_v_map, d_adj_u, d_parent_u,
                d_sub_size, d_m_lmin, d_next_k, d_subdist, d_nd_u, d_edge_pool, d_delta_gmem + max_cols
            );
            CUDA_CHECK(cudaGetLastError());

            k_spr_rows_top1<<<n_rows_spr, 128, use_smem_edge ? req_smem : 0>>>(
                2, N, n_nodes_u, T.other_root_child, use_smem_edge, d_v, d_v_map, d_adj_u, d_parent_u,
                d_sub_size, d_m_lmin, d_next_k, d_subdist, d_nd_u, d_edge_pool, d_row_tmp, d_best_delta, d_best_j
            );
            CUDA_CHECK(cudaGetLastError());
        }

        /* Self-contained Column-Tiled execution (computes d_subdist_int_leaf and d_nd_u_tiled without d_subdist) */
        CUDA_CHECK(cudaEventRecord(ev_tiled0));
        for (int32_t c0 = 0; c0 < N; c0 += B_TILE) {
            int32_t c_len = (c0 + B_TILE <= N) ? B_TILE : (N - c0);
            k_tiled_leaf_cols<<<(c_len + 63) / 64, 64>>>(
                d_D, d_tile_buf, d_subdist_int_leaf, d_child1_r, d_child2_r, d_tin,
                c0, c_len, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2
            );
            CUDA_CHECK(cudaGetLastError());
        }
        for (int32_t u_off = 0; u_off < n_rows_spr; u_off += B_ROW_TILE) {
            int32_t u_len = (u_off + B_ROW_TILE <= n_rows_spr) ? B_ROW_TILE : (n_rows_spr - u_off);
            int32_t K_nd = 2 * u_len;
            dim3 ld_nd_grid((N + 15) / 16, (K_nd + 15) / 16);
            dim3 ld_nd_block(16, 16);
            k_load_nd_u_leaves_tiled<<<ld_nd_grid, ld_nd_block>>>(
                d_D, d_subdist_int_leaf, d_auto_col_buf, d_adj_u, N + u_off, K_nd, N
            );
            CUDA_CHECK(cudaGetLastError());
            k_reconstruct_nd_u_cols<<<(K_nd + 63) / 64, 64>>>(
                d_auto_col_buf, d_child1_r, d_child2_r, d_tin, d_adj_u, d_nd_u_tiled,
                N + u_off, K_nd, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2
            );
            CUDA_CHECK(cudaGetLastError());
        }
        for (int32_t r_off = 0; r_off < n_rows_spr; r_off += B_ROW_TILE) {
            int32_t b_len = (r_off + B_ROW_TILE <= n_rows_spr) ? B_ROW_TILE : (n_rows_spr - r_off);
            int32_t K = 3 * b_len;
            dim3 ld_grid((N + 15) / 16, (K + 15) / 16);
            dim3 ld_block(16, 16);
            k_load_auto_leaves_tiled<<<ld_grid, ld_block>>>(
                d_D, d_subdist_int_leaf, d_auto_col_buf, d_v_map, d_parent_u, d_adj_u,
                2 + r_off, K, N, n_nodes_u, T.other_root_child
            );
            CUDA_CHECK(cudaGetLastError());
            k_reconstruct_auto_cols<<<(K + 63) / 64, 64>>>(
                d_auto_col_buf, d_child1_r, d_child2_r, d_tin, d_v_map, d_parent_u, d_adj_u,
                2 + r_off, K, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2, T.other_root_child
            );
            CUDA_CHECK(cudaGetLastError());
            dim3 tr_auto_grid((K + 15) / 16, (n_nodes_u + 15) / 16);
            dim3 tr_auto_block(16, 16);
            k_transpose_auto_cols<<<tr_auto_grid, tr_auto_block>>>(
                d_auto_col_buf, d_auto_pool, K, n_nodes_u
            );
            CUDA_CHECK(cudaGetLastError());
            k_spr_rows_tiled<<<b_len, 128, use_smem_edge ? req_smem : 0>>>(
                2, r_off, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2, T.other_root_child,
                1, use_smem_edge,
                d_child1_r, d_child2_r, d_tin, d_v_map, d_adj_u, d_parent_u, d_sub_size,
                d_m_lmin, d_next_k, d_D, d_subdist_int_leaf, d_nd_u_tiled, d_auto_pool, d_edge_pool, d_delta_tiled + max_cols
            );
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(ev_tiled1));
        CUDA_CHECK(cudaEventSynchronize(ev_tiled1));

        float ms_sub = 0.0f, ms_smem = 0.0f, ms_tiled = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_sub, ev_sub0, ev_sub1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_smem, ev_smem0, ev_smem1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, ev_tiled0, ev_tiled1));
        if (ms_sub < best_ms_sub) best_ms_sub = ms_sub;
        if (ms_smem < best_ms_smem) best_ms_smem = ms_smem;
        if (ms_tiled < best_ms_tiled) best_ms_tiled = ms_tiled;
    }

    if (subdist_ms_out) *subdist_ms_out = (double)best_ms_sub;
    if (spr_smem_ms_out) *spr_smem_ms_out = (double)best_ms_smem;
    if (spr_tiled_ms_out) *spr_tiled_ms_out = (double)best_ms_tiled;

    CUDA_CHECK(cudaMemcpy(h_delta_smem,  d_delta_smem,  (size_t)n_pairs * max_cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta_gmem,  d_delta_gmem,  (size_t)n_pairs * max_cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta_tiled, d_delta_tiled, (size_t)n_pairs * max_cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_delta,  d_best_delta,  (size_t)n_pairs * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_j,      d_best_j,      (size_t)n_pairs * sizeof(int32_t), cudaMemcpyDeviceToHost));

    cudaFree(d_subdist_int_leaf); cudaFree(d_tile_buf); cudaFree(d_auto_col_buf); cudaFree(d_auto_pool);
    cudaFree(d_child1_r); cudaFree(d_child2_r); cudaFree(d_v_map); cudaFree(d_m_lmin); cudaFree(d_next_k);
    cudaFree(d_parent_u); cudaFree(d_sub_size); cudaFree(d_tin); cudaFree(d_adj_u); cudaFree(d_v);
    cudaFree(d_D); cudaFree(d_subdist); cudaFree(d_nd_u); cudaFree(d_nd_u_tiled); cudaFree(d_delta_smem); cudaFree(d_delta_gmem);
    cudaFree(d_delta_tiled); cudaFree(d_edge_pool); cudaFree(d_row_tmp); cudaFree(d_best_delta); cudaFree(d_best_j);
    cudaEventDestroy(ev_sub0); cudaEventDestroy(ev_sub1);
    cudaEventDestroy(ev_smem0); cudaEventDestroy(ev_smem1);
    cudaEventDestroy(ev_tiled0); cudaEventDestroy(ev_tiled1);
#else
    /* Local Host SIMT Emulation: executes the exact kernel bodies and tests both forward and reverse thread order */
    struct timespec t0, t1, t2, t3;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    float *subdist = (float *)calloc((size_t)n_nodes_u * n_nodes_u, sizeof(float));
    float *nd_u    = (float *)calloc((size_t)(3 * n_nodes_u), sizeof(float));

    blockDim = {256, 1, 1};
    for (int bx = 0; bx < (N + 255) / 256; ++bx) {
        blockIdx = {bx, 0, 0};
        for (int tx = 0; tx < 256; ++tx) {
            threadIdx = {tx, 0, 0};
            k_bottom_up_leaf_cols(h_D, subdist, T.child1_r, T.child2_r, N, n_nodes_u);
        }
    }
    int32_t n_int = (n_nodes_u - 1) - N;
    blockDim = {16, 16, 1};
    for (int by = 0; by < (n_int + 15) / 16; ++by) {
        for (int bx = 0; bx < (N + 15) / 16; ++bx) {
            blockIdx = {bx, by, 0};
            for (int ty = 0; ty < 16; ++ty) {
                for (int tx = 0; tx < 16; ++tx) {
                    threadIdx = {tx, ty, 0};
                    k_transpose_leaf_block(subdist, N, n_nodes_u);
                }
            }
        }
    }
    blockDim = {256, 1, 1};
    for (int bx = 0; bx < (n_int + 255) / 256; ++bx) {
        blockIdx = {bx, 0, 0};
        for (int tx = 0; tx < 256; ++tx) {
            threadIdx = {tx, 0, 0};
            k_bottom_up_int_cols(subdist, T.child1_r, T.child2_r, N, n_nodes_u);
        }
    }
    for (int bx = 0; bx < (n_nodes_u + 255) / 256; ++bx) {
        blockIdx = {bx, 0, 0};
        for (int tx = 0; tx < 256; ++tx) {
            threadIdx = {tx, 0, 0};
            k_top_down_ancestor_walk(subdist, T.child1_r, T.child2_r, T.tin, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2);
        }
    }
    for (int bx = 0; bx < (N - 1 + 255) / 256; ++bx) {
        blockIdx = {bx, 0, 0};
        for (int tx = 0; tx < 256; ++tx) {
            threadIdx = {tx, 0, 0};
            k_nd_u(subdist, T.adj_u, nd_u, N, n_nodes_u);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    float *smem_buf  = (float *)calloc((size_t)n_nodes_u, sizeof(float));
    float *edge_pool = (float *)calloc((size_t)n_rows_spr * n_nodes_u, sizeof(float));
    float *row_tmp   = (float *)calloc((size_t)n_rows_spr * (2 * N), sizeof(float));

    uint32_t fuzz_rng = 0xCAFEBABEU ^ (uint32_t)N ^ (uint32_t)(reverse_thread_order * 7919);
    int perm[128];
    auto build_thread_perm = [&](int mode) {
        for (int s = 0; s < 128; ++s) {
            perm[s] = (mode == 1) ? (127 - s) : s;
        }
        if (mode >= 2) {
            for (int s = 127; s > 0; --s) {
                fuzz_rng ^= fuzz_rng << 13; fuzz_rng ^= fuzz_rng >> 17; fuzz_rng ^= fuzz_rng << 5;
                int r = (int)(fuzz_rng % (uint32_t)(s + 1));
                int tmp = perm[s]; perm[s] = perm[r]; perm[r] = tmp;
            }
        }
    };

    blockDim = {128, 1, 1};
    for (int bx = 0; bx < n_rows_spr; ++bx) {
        blockIdx = {bx, 0, 0};
        for (int phase = 0; phase < 4; ++phase) {
            g_simt_target_phase = phase;
            build_thread_perm(reverse_thread_order);
            for (int step = 0; step < 128; ++step) {
                int tx = perm[step];
                threadIdx = {tx, 0, 0};
                g_simt_curr_phase = 0;
                k_spr_rows_smem(
                    2, N, n_nodes_u, T.other_root_child, T.v_map, T.adj_u, T.parent_u,
                    T.sub_size, T.m_lmin, T.next_k_same_lmin, subdist, nd_u,
                    h_delta_smem + max_cols, smem_buf
                );
            }
        }
        for (int phase = 0; phase < 4; ++phase) {
            g_simt_target_phase = phase;
            build_thread_perm(reverse_thread_order);
            for (int step = 0; step < 128; ++step) {
                int tx = perm[step];
                threadIdx = {tx, 0, 0};
                g_simt_curr_phase = 0;
                k_spr_rows_gmem(
                    2, N, n_nodes_u, T.other_root_child, T.v_map, T.adj_u, T.parent_u,
                    T.sub_size, T.m_lmin, T.next_k_same_lmin, subdist, nd_u,
                    edge_pool, h_delta_gmem + max_cols
                );
            }
        }
        for (int phase = 0; phase < 13; ++phase) {
            g_simt_target_phase = phase;
            build_thread_perm(reverse_thread_order);
            for (int step = 0; step < 128; ++step) {
                int tx = perm[step];
                threadIdx = {tx, 0, 0};
                g_simt_curr_phase = 0;
                k_spr_rows_top1(
                    2, N, n_nodes_u, T.other_root_child, 0, v, T.v_map, T.adj_u, T.parent_u,
                    T.sub_size, T.m_lmin, T.next_k_same_lmin, subdist, nd_u,
                    edge_pool, row_tmp, h_best_delta, h_best_j
                );
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);

    /* Column-tiled host SIMT execution */
    const int32_t B_TILE = 128;
    float *subdist_int_leaf = (float *)calloc((size_t)(N - 1) * N, sizeof(float));
    float *tile_buf         = (float *)calloc((size_t)n_nodes_u * B_TILE, sizeof(float));
    float *auto_pool        = (float *)calloc((size_t)n_rows_spr * 3 * n_nodes_u, sizeof(float));

    for (int32_t c0 = 0; c0 < N; c0 += B_TILE) {
        int32_t c_len = (c0 + B_TILE <= N) ? B_TILE : (N - c0);
        blockDim = {128, 1, 1};
        blockIdx = {0, 0, 0};
        for (int tx = 0; tx < c_len; ++tx) {
            threadIdx = {tx, 0, 0};
            k_tiled_leaf_cols(
                h_D, tile_buf, subdist_int_leaf, T.child1_r, T.child2_r, T.tin,
                c0, c_len, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2
            );
        }
    }
    for (int bx = 0; bx < n_rows_spr; ++bx) {
        blockIdx = {bx, 0, 0};
        for (int phase = 0; phase < 6; ++phase) {
            g_simt_target_phase = phase;
            build_thread_perm(reverse_thread_order);
            for (int step = 0; step < 128; ++step) {
                int tx = perm[step];
                threadIdx = {tx, 0, 0};
                g_simt_curr_phase = 0;
                k_spr_rows_tiled(
                    2, 0, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2, T.other_root_child,
                    0, 0,
                    T.child1_r, T.child2_r, T.tin, T.v_map, T.adj_u, T.parent_u, T.sub_size,
                    T.m_lmin, T.next_k_same_lmin, h_D, subdist_int_leaf, nd_u,
                    auto_pool, edge_pool, h_delta_tiled + max_cols
                );
            }
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t3);

    auto diff_ms = [](struct timespec a, struct timespec b) {
        return (double)(b.tv_sec - a.tv_sec) * 1000.0 + (double)(b.tv_nsec - a.tv_nsec) * 1e-6;
    };
    if (subdist_ms_out) *subdist_ms_out = diff_ms(t0, t1);
    if (spr_smem_ms_out) *spr_smem_ms_out = diff_ms(t1, t2);
    if (spr_tiled_ms_out) *spr_tiled_ms_out = diff_ms(t2, t3);

    free(subdist); free(nd_u); free(smem_buf); free(edge_pool); free(row_tmp);
    free(subdist_int_leaf); free(tile_buf); free(auto_pool);
#endif

    double max_err_smem = 0.0, max_err_gmem = 0.0, max_err_tiled = 0.0;
    int ok = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        /* Current coordinate delta must be strictly 0 and finite */
        float d_vi_s = h_delta_smem[(size_t)i * max_cols + v[i]];
        float d_vi_g = h_delta_gmem[(size_t)i * max_cols + v[i]];
        float d_vi_t = h_delta_tiled[(size_t)i * max_cols + v[i]];
        if (!isfinite(d_vi_s) || !isfinite(d_vi_g) || !isfinite(d_vi_t) ||
            fabsf(d_vi_s) > 1e-6f || fabsf(d_vi_g) > 1e-6f || fabsf(d_vi_t) > 1e-6f) {
            ok = 0;
        }
        float row_min = 0.0f;
        for (int32_t j = 0; j <= 2 * i; ++j) {
            double ref = delta_ref[(size_t)i * max_cols + j];
            double es = fabs((double)h_delta_smem[(size_t)i * max_cols + j] - ref);
            double eg = fabs((double)h_delta_gmem[(size_t)i * max_cols + j] - ref);
            double et = fabs((double)h_delta_tiled[(size_t)i * max_cols + j] - ref);
            if (!isfinite(es) || !isfinite(eg) || !isfinite(et)) ok = 0;
            if (es > max_err_smem) max_err_smem = es;
            if (eg > max_err_gmem) max_err_gmem = eg;
            if (et > max_err_tiled) max_err_tiled = et;
            if (j != (int32_t)v[i] && h_delta_gmem[(size_t)i * max_cols + j] < row_min) {
                row_min = h_delta_gmem[(size_t)i * max_cols + j];
            }
        }
        float best_d = h_best_delta[i];
        int32_t best_j = h_best_j[i];
        if (!isfinite(best_d) || !isfinite(row_min) || fabsf(best_d - row_min) > 1e-6f) {
            ok = 0;
        }
        if (best_j < 0 || best_j > 2 * i) {
            ok = 0;
        } else {
            float chosen_val = h_delta_gmem[(size_t)i * max_cols + best_j];
            if (!isfinite(chosen_val) || fabsf(chosen_val - row_min) > 1e-6f) {
                ok = 0;
            }
        }
    }

    if (max_err_smem_out) *max_err_smem_out = max_err_smem;
    if (max_err_gmem_out) *max_err_gmem_out = max_err_gmem;
    if (max_err_tiled_out) *max_err_tiled_out = max_err_tiled;

    free_host_tree_topology(&T);
    free(h_D); free(delta_ref);
    free(h_delta_smem); free(h_delta_gmem); free(h_delta_tiled);
    free(h_best_delta); free(h_best_j);
    return ok;
}

#ifdef __CUDACC__
static int run_full_subdist_top1_multi_gpu(
    int32_t N, const HostTreeTopology *T, const int64_t *v, int num_gpus_use,
    double *ms_sub_out, double *ms_spr_out, double *vram_mb_out,
    float *best_delta_out, int32_t *best_i_out, int32_t *best_j_out
) {
    int32_t n_pairs = T->n_pairs;
    int32_t n_nodes_r = T->n_nodes_r;
    int32_t n_nodes_u = T->n_nodes_u;
    int32_t n_rows_spr = N - 2;
    int32_t n_int = (n_nodes_u - 1) - N;
    const int32_t B_ROW_TILE = (n_rows_spr < 1024) ? n_rows_spr : 1024;

    size_t req_smem = (size_t)n_nodes_u * sizeof(float);
    int max_smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    int use_smem_edge = (req_smem + 8192 <= (size_t)max_smem_optin) ? 1 : 0;

    int32_t *d_child1_r[8], *d_child2_r[8], *d_v_map[8], *d_m_lmin[8], *d_next_k[8];
    int32_t *d_parent_u[8], *d_sub_size[8], *d_tin[8], *d_adj_u[8], *d_best_j[8];
    int64_t *d_v[8];
    float *d_D[8], *d_subdist[8], *d_nd_u[8], *d_edge_pool[8], *d_row_tmp[8], *d_best_delta[8];
    cudaEvent_t ev_sub0[8], ev_sub1[8], ev_spr0[8], ev_spr1[8];
    int32_t r_start_g[8], r_len_g[8];

    size_t vram_per_gpu = ((size_t)N * N + (size_t)n_nodes_u * n_nodes_u + (size_t)3 * n_nodes_u +
                           (size_t)B_ROW_TILE * (2 * N) + (size_t)(use_smem_edge ? 0 : B_ROW_TILE) * n_nodes_u) * sizeof(float);
    if (vram_mb_out) *vram_mb_out = (double)vram_per_gpu / (1024.0 * 1024.0);

    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        if (vram_per_gpu + (512ULL << 20) > free_mem) {
            return 0; /* Insufficient free VRAM for un-tiled (2N-2)^2 subdist matrix */
        }
    }

    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        if (use_smem_edge) {
            CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_top1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req_smem));
        }
        int32_t r0 = (int32_t)(((int64_t)n_rows_spr * g) / num_gpus_use);
        int32_t r1 = (int32_t)(((int64_t)n_rows_spr * (g + 1)) / num_gpus_use);
        r_start_g[g] = r0;
        r_len_g[g]   = r1 - r0;

        CUDA_CHECK(cudaMalloc(&d_child1_r[g], (size_t)n_nodes_r * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_child2_r[g], (size_t)n_nodes_r * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_v_map[g],    (size_t)N * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_m_lmin[g],   (size_t)n_pairs * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_next_k[g],   (size_t)n_pairs * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_parent_u[g], (size_t)n_nodes_u * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sub_size[g], (size_t)n_nodes_u * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_tin[g],      (size_t)n_nodes_u * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_adj_u[g],    (size_t)(3 * n_nodes_u) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_v[g],        (size_t)n_pairs * sizeof(int64_t)));

        CUDA_CHECK(cudaMalloc(&d_D[g],          (size_t)N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_subdist[g],    (size_t)n_nodes_u * n_nodes_u * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nd_u[g],       (size_t)(3 * n_nodes_u) * sizeof(float)));
        d_edge_pool[g] = NULL;
        if (!use_smem_edge) {
            CUDA_CHECK(cudaMalloc(&d_edge_pool[g], (size_t)B_ROW_TILE * n_nodes_u * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&d_row_tmp[g],    (size_t)B_ROW_TILE * (2 * N) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_best_delta[g], (size_t)n_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_best_j[g],     (size_t)n_pairs * sizeof(int32_t)));

        CUDA_CHECK(cudaMemcpy(d_child1_r[g], T->child1_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_child2_r[g], T->child2_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_map[g],    T->v_map,    (size_t)N * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m_lmin[g],   T->m_lmin,   (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_next_k[g],   T->next_k_same_lmin, (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_parent_u[g], T->parent_u, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sub_size[g], T->sub_size, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tin[g],      T->tin,      (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_adj_u[g],    T->adj_u,    (size_t)(3 * n_nodes_u) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v[g],        v,           (size_t)n_pairs * sizeof(int64_t), cudaMemcpyHostToDevice));

        dim3 init_grid((N + 15) / 16, (N + 15) / 16);
        dim3 init_block(16, 16);
        k_init_symmetric_D<<<init_grid, init_block>>>(d_D[g], N, 1337U + (uint32_t)N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventCreate(&ev_sub0[g]));
        CUDA_CHECK(cudaEventCreate(&ev_sub1[g]));
        CUDA_CHECK(cudaEventCreate(&ev_spr0[g]));
        CUDA_CHECK(cudaEventCreate(&ev_spr1[g]));
    }
    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* Launch Stage 2 (subdist) and Stage 3 (sharded SPR top-1) concurrently across all active GPUs */
    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventRecord(ev_sub0[g]));
        k_bottom_up_leaf_cols<<<(N + 255) / 256, 256>>>(d_D[g], d_subdist[g], d_child1_r[g], d_child2_r[g], N, n_nodes_u);
        dim3 tr_grid((N + 15) / 16, (n_int + 15) / 16);
        dim3 tr_block(16, 16);
        k_transpose_leaf_block<<<tr_grid, tr_block>>>(d_subdist[g], N, n_nodes_u);
        if (n_int > 0) {
            k_bottom_up_int_cols<<<(n_int + 255) / 256, 256>>>(d_subdist[g], d_child1_r[g], d_child2_r[g], N, n_nodes_u);
        }
        k_top_down_ancestor_walk<<<(n_nodes_u + 255) / 256, 256>>>(
            d_subdist[g], d_child1_r[g], d_child2_r[g], d_tin[g], n_nodes_u, T->root_u_c0, T->root_u_c1, T->root_u_c2
        );
        k_nd_u<<<(N - 1 + 255) / 256, 256>>>(d_subdist[g], d_adj_u[g], d_nd_u[g], N, n_nodes_u);
        CUDA_CHECK(cudaEventRecord(ev_sub1[g]));

        CUDA_CHECK(cudaEventRecord(ev_spr0[g]));
        for (int32_t r_off = r_start_g[g]; r_off < r_start_g[g] + r_len_g[g]; r_off += B_ROW_TILE) {
            int32_t b_len = (r_off + B_ROW_TILE <= r_start_g[g] + r_len_g[g]) ? B_ROW_TILE : (r_start_g[g] + r_len_g[g] - r_off);
            k_spr_rows_top1<<<b_len, 128, use_smem_edge ? req_smem : 0>>>(
                2 + r_off, N, n_nodes_u, T->other_root_child, use_smem_edge, d_v[g], d_v_map[g], d_adj_u[g], d_parent_u[g],
                d_sub_size[g], d_m_lmin[g], d_next_k[g], d_subdist[g], d_nd_u[g], d_edge_pool[g], d_row_tmp[g], d_best_delta[g], d_best_j[g]
            );
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(ev_spr1[g]));
    }

    float max_ms_sub = 0.0f, max_ms_spr = 0.0f;
    float *h_best_delta = (float *)calloc((size_t)n_pairs, sizeof(float));
    int32_t *h_best_j   = (int32_t *)calloc((size_t)n_pairs, sizeof(int32_t));

    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventSynchronize(ev_spr1[g]));
        float ms_s = 0.0f, ms_r = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_s, ev_sub0[g], ev_sub1[g]));
        CUDA_CHECK(cudaEventElapsedTime(&ms_r, ev_spr0[g], ev_spr1[g]));
        if (ms_s > max_ms_sub) max_ms_sub = ms_s;
        if (ms_r > max_ms_spr) max_ms_spr = ms_r;

        if (r_len_g[g] > 0) {
            CUDA_CHECK(cudaMemcpy(h_best_delta + 1 + r_start_g[g], d_best_delta[g] + 1 + r_start_g[g],
                                  (size_t)r_len_g[g] * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_best_j + 1 + r_start_g[g], d_best_j[g] + 1 + r_start_g[g],
                                  (size_t)r_len_g[g] * sizeof(int32_t), cudaMemcpyDeviceToHost));
        }

        cudaFree(d_child1_r[g]); cudaFree(d_child2_r[g]); cudaFree(d_v_map[g]); cudaFree(d_m_lmin[g]); cudaFree(d_next_k[g]);
        cudaFree(d_parent_u[g]); cudaFree(d_sub_size[g]); cudaFree(d_tin[g]); cudaFree(d_adj_u[g]); cudaFree(d_v[g]);
        cudaFree(d_D[g]); cudaFree(d_subdist[g]); cudaFree(d_nd_u[g]);
        if (d_edge_pool[g]) cudaFree(d_edge_pool[g]);
        cudaFree(d_row_tmp[g]); cudaFree(d_best_delta[g]); cudaFree(d_best_j[g]);
        cudaEventDestroy(ev_sub0[g]); cudaEventDestroy(ev_sub1[g]);
        cudaEventDestroy(ev_spr0[g]); cudaEventDestroy(ev_spr1[g]);
    }

    float g_best_val = 0.0f;
    int32_t g_best_i = 1, g_best_j = 0;
    for (int32_t i = 1; i < n_pairs; ++i) {
        if (h_best_delta[i] < g_best_val) {
            g_best_val = h_best_delta[i];
            g_best_i   = i;
            g_best_j   = h_best_j[i];
        }
    }
    free(h_best_delta);
    free(h_best_j);

    if (ms_sub_out) *ms_sub_out = (double)max_ms_sub;
    if (ms_spr_out) *ms_spr_out = (double)max_ms_spr;
    if (best_delta_out) *best_delta_out = g_best_val;
    if (best_i_out) *best_i_out = g_best_i;
    if (best_j_out) *best_j_out = g_best_j;
    return 1;
}

static int run_tiled_top1_multi_gpu(
    int32_t N, const HostTreeTopology *T, const int64_t *v, int num_gpus_use,
    double *ms_tiled_out, double *vram_mb_out,
    float *best_delta_out, int32_t *best_i_out, int32_t *best_j_out
) {
    int32_t n_pairs = T->n_pairs;
    int32_t n_nodes_r = T->n_nodes_r;
    int32_t n_nodes_u = T->n_nodes_u;
    int32_t n_rows_spr = N - 2;
    const int32_t B_TILE = (N < 1024) ? N : 1024;
    const int32_t B_ROW_TILE = (n_rows_spr < 512) ? n_rows_spr : 512;

    size_t req_smem = (size_t)n_nodes_u * sizeof(float);
    int max_smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    int use_smem_edge = (req_smem + 8192 <= (size_t)max_smem_optin) ? 1 : 0;

    int32_t *d_child1_r[8], *d_child2_r[8], *d_v_map[8], *d_m_lmin[8], *d_next_k[8];
    int32_t *d_parent_u[8], *d_sub_size[8], *d_tin[8], *d_adj_u[8], *d_best_j[8];
    int64_t *d_v[8];
    float *d_D[8], *d_subdist_int_leaf[8], *d_tile_buf[8], *d_auto_col_buf[8], *d_auto_pool[8];
    float *d_nd_u[8], *d_edge_pool[8], *d_best_delta[8];
    cudaEvent_t ev0[8], ev1[8];
    int32_t r_start_g[8], r_len_g[8];

    size_t vram_per_gpu = ((size_t)N * N + (size_t)(N - 1) * N + (size_t)n_nodes_u * B_TILE +
                           (size_t)6 * B_ROW_TILE * n_nodes_u + (size_t)3 * n_nodes_u +
                           (size_t)(use_smem_edge ? 0 : B_ROW_TILE) * n_nodes_u) * sizeof(float);
    if (vram_mb_out) *vram_mb_out = (double)vram_per_gpu / (1024.0 * 1024.0);

    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        if (vram_per_gpu + (512ULL << 20) > free_mem) {
            return 0;
        }
    }

    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        if (use_smem_edge) {
            CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_tiled_top1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req_smem));
        }
        int32_t r0 = (int32_t)(((int64_t)n_rows_spr * g) / num_gpus_use);
        int32_t r1 = (int32_t)(((int64_t)n_rows_spr * (g + 1)) / num_gpus_use);
        r_start_g[g] = r0;
        r_len_g[g]   = r1 - r0;

        CUDA_CHECK(cudaMalloc(&d_child1_r[g], (size_t)n_nodes_r * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_child2_r[g], (size_t)n_nodes_r * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_v_map[g],    (size_t)N * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_m_lmin[g],   (size_t)n_pairs * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_next_k[g],   (size_t)n_pairs * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_parent_u[g], (size_t)n_nodes_u * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sub_size[g], (size_t)n_nodes_u * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_tin[g],      (size_t)n_nodes_u * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_adj_u[g],    (size_t)(3 * n_nodes_u) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_v[g],        (size_t)n_pairs * sizeof(int64_t)));

        CUDA_CHECK(cudaMalloc(&d_D[g],                (size_t)N * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_subdist_int_leaf[g], (size_t)(N - 1) * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tile_buf[g],         (size_t)n_nodes_u * B_TILE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_auto_col_buf[g],     (size_t)3 * B_ROW_TILE * n_nodes_u * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_auto_pool[g],        (size_t)3 * B_ROW_TILE * n_nodes_u * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nd_u[g],             (size_t)(3 * n_nodes_u) * sizeof(float)));
        d_edge_pool[g] = NULL;
        if (!use_smem_edge) {
            CUDA_CHECK(cudaMalloc(&d_edge_pool[g], (size_t)B_ROW_TILE * n_nodes_u * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&d_best_delta[g],       (size_t)n_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_best_j[g],           (size_t)n_pairs * sizeof(int32_t)));

        CUDA_CHECK(cudaMemcpy(d_child1_r[g], T->child1_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_child2_r[g], T->child2_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_map[g],    T->v_map,    (size_t)N * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m_lmin[g],   T->m_lmin,   (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_next_k[g],   T->next_k_same_lmin, (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_parent_u[g], T->parent_u, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sub_size[g], T->sub_size, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tin[g],      T->tin,      (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_adj_u[g],    T->adj_u,    (size_t)(3 * n_nodes_u) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v[g],        v,           (size_t)n_pairs * sizeof(int64_t), cudaMemcpyHostToDevice));

        dim3 init_grid((N + 15) / 16, (N + 15) / 16);
        dim3 init_block(16, 16);
        k_init_symmetric_D<<<init_grid, init_block>>>(d_D[g], N, 1337U + (uint32_t)N);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventCreate(&ev0[g]));
        CUDA_CHECK(cudaEventCreate(&ev1[g]));
    }
    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventRecord(ev0[g]));
        for (int32_t c0 = 0; c0 < N; c0 += B_TILE) {
            int32_t c_len = (c0 + B_TILE <= N) ? B_TILE : (N - c0);
            k_tiled_leaf_cols<<<(c_len + 63) / 64, 64>>>(
                d_D[g], d_tile_buf[g], d_subdist_int_leaf[g], d_child1_r[g], d_child2_r[g], d_tin[g],
                c0, c_len, N, n_nodes_u, T->root_u_c0, T->root_u_c1, T->root_u_c2
            );
            CUDA_CHECK(cudaGetLastError());
        }
        /* Shard nd_u reconstruction across GPUs */
        for (int32_t u_off = r_start_g[g]; u_off < r_start_g[g] + r_len_g[g]; u_off += B_ROW_TILE) {
            int32_t u_len = (u_off + B_ROW_TILE <= r_start_g[g] + r_len_g[g]) ? B_ROW_TILE : (r_start_g[g] + r_len_g[g] - u_off);
            int32_t K_nd = 2 * u_len;
            dim3 ld_nd_grid((N + 15) / 16, (K_nd + 15) / 16);
            dim3 ld_nd_block(16, 16);
            k_load_nd_u_leaves_tiled<<<ld_nd_grid, ld_nd_block>>>(
                d_D[g], d_subdist_int_leaf[g], d_auto_col_buf[g], d_adj_u[g], N + u_off, K_nd, N
            );
            CUDA_CHECK(cudaGetLastError());
            k_reconstruct_nd_u_cols<<<(K_nd + 63) / 64, 64>>>(
                d_auto_col_buf[g], d_child1_r[g], d_child2_r[g], d_tin[g], d_adj_u[g], d_nd_u[g],
                N + u_off, K_nd, N, n_nodes_u, T->root_u_c0, T->root_u_c1, T->root_u_c2
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (num_gpus_use > 1) {
        for (int g = 0; g < num_gpus_use; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        for (int src = 0; src < num_gpus_use; ++src) {
            for (int dst = 0; dst < num_gpus_use; ++dst) {
                if (src != dst && r_len_g[src] > 0) {
                    size_t off = (size_t)3 * (N + r_start_g[src]);
                    size_t cnt = (size_t)3 * r_len_g[src] * sizeof(float);
                    CUDA_CHECK(cudaMemcpyPeerAsync(d_nd_u[dst] + off, dst, d_nd_u[src] + off, src, cnt));
                }
            }
        }
        for (int g = 0; g < num_gpus_use; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        for (int32_t r_off = r_start_g[g]; r_off < r_start_g[g] + r_len_g[g]; r_off += B_ROW_TILE) {
            int32_t b_len = (r_off + B_ROW_TILE <= r_start_g[g] + r_len_g[g]) ? B_ROW_TILE : (r_start_g[g] + r_len_g[g] - r_off);
            int32_t K = 3 * b_len;
            dim3 ld_grid((N + 15) / 16, (K + 15) / 16);
            dim3 ld_block(16, 16);
            k_load_auto_leaves_tiled<<<ld_grid, ld_block>>>(
                d_D[g], d_subdist_int_leaf[g], d_auto_col_buf[g], d_v_map[g], d_parent_u[g], d_adj_u[g],
                2 + r_off, K, N, n_nodes_u, T->other_root_child
            );
            CUDA_CHECK(cudaGetLastError());
            k_reconstruct_auto_cols<<<(K + 63) / 64, 64>>>(
                d_auto_col_buf[g], d_child1_r[g], d_child2_r[g], d_tin[g], d_v_map[g], d_parent_u[g], d_adj_u[g],
                2 + r_off, K, N, n_nodes_u, T->root_u_c0, T->root_u_c1, T->root_u_c2, T->other_root_child
            );
            CUDA_CHECK(cudaGetLastError());
            dim3 tr_auto_grid((K + 15) / 16, (n_nodes_u + 15) / 16);
            dim3 tr_auto_block(16, 16);
            k_transpose_auto_cols<<<tr_auto_grid, tr_auto_block>>>(
                d_auto_col_buf[g], d_auto_pool[g], K, n_nodes_u
            );
            CUDA_CHECK(cudaGetLastError());
            k_spr_rows_tiled_top1<<<b_len, 128, use_smem_edge ? req_smem : 0>>>(
                2, r_off, N, n_nodes_u, T->other_root_child, use_smem_edge, d_v[g], d_v_map[g], d_adj_u[g],
                d_parent_u[g], d_sub_size[g], d_m_lmin[g], d_next_k[g], d_nd_u[g], d_auto_pool[g],
                d_edge_pool[g], d_best_delta[g], d_best_j[g]
            );
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(ev1[g]));
    }

    float max_ms = 0.0f;
    float *h_best_delta = (float *)calloc((size_t)n_pairs, sizeof(float));
    int32_t *h_best_j   = (int32_t *)calloc((size_t)n_pairs, sizeof(int32_t));

    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventSynchronize(ev1[g]));
        float ms_g = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_g, ev0[g], ev1[g]));
        if (ms_g > max_ms) max_ms = ms_g;

        if (r_len_g[g] > 0) {
            CUDA_CHECK(cudaMemcpy(h_best_delta + 1 + r_start_g[g], d_best_delta[g] + 1 + r_start_g[g],
                                  (size_t)r_len_g[g] * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_best_j + 1 + r_start_g[g], d_best_j[g] + 1 + r_start_g[g],
                                  (size_t)r_len_g[g] * sizeof(int32_t), cudaMemcpyDeviceToHost));
        }

        cudaFree(d_child1_r[g]); cudaFree(d_child2_r[g]); cudaFree(d_v_map[g]); cudaFree(d_m_lmin[g]); cudaFree(d_next_k[g]);
        cudaFree(d_parent_u[g]); cudaFree(d_sub_size[g]); cudaFree(d_tin[g]); cudaFree(d_adj_u[g]); cudaFree(d_v[g]);
        cudaFree(d_D[g]); cudaFree(d_subdist_int_leaf[g]); cudaFree(d_tile_buf[g]);
        cudaFree(d_auto_col_buf[g]); cudaFree(d_auto_pool[g]); cudaFree(d_nd_u[g]);
        if (d_edge_pool[g]) cudaFree(d_edge_pool[g]);
        cudaFree(d_best_delta[g]); cudaFree(d_best_j[g]);
        cudaEventDestroy(ev0[g]); cudaEventDestroy(ev1[g]);
    }

    float g_best_val = 0.0f;
    int32_t g_best_i = 1, g_best_j = 0;
    for (int32_t i = 1; i < n_pairs; ++i) {
        if (h_best_delta[i] < g_best_val) {
            g_best_val = h_best_delta[i];
            g_best_i   = i;
            g_best_j   = h_best_j[i];
        }
    }
    free(h_best_delta);
    free(h_best_j);

    if (ms_tiled_out) *ms_tiled_out = (double)max_ms;
    if (best_delta_out) *best_delta_out = g_best_val;
    if (best_i_out) *best_i_out = g_best_i;
    if (best_j_out) *best_j_out = g_best_j;
    return 1;
}

static int run_sharded_multi_gpu_top1(
    int32_t N, const HostTreeTopology *T, const int64_t *v, int num_gpus_use,
    int32_t max_batches_per_gpu,
    double *ms_stage2_out, double *ms_stage3_full_out, double *vram_per_gpu_mb_out, int *has_d_shard_out,
    float *best_delta_out, int32_t *best_i_out, int32_t *best_j_out
) {
    int32_t n_pairs = T->n_pairs;
    int32_t n_nodes_r = T->n_nodes_r;
    int32_t n_nodes_u = T->n_nodes_u;
    int32_t n_rows_spr = N - 2;
    const int32_t B_TILE_MAX = (N >= 250000) ? 512 : 1024;
    const int32_t B_ROW_MAX  = (N >= 250000) ? 128 : 256;
    const int32_t B_TILE = (N < B_TILE_MAX) ? N : B_TILE_MAX;
    const int32_t B_ROW_TILE = (n_rows_spr < B_ROW_MAX) ? n_rows_spr : B_ROW_MAX;
    uint32_t seed = 1337U + (uint32_t)N;

    size_t req_smem = (size_t)n_nodes_u * sizeof(float);
    int max_smem_optin = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
    int use_smem_edge = (req_smem + 8192 <= (size_t)max_smem_optin) ? 1 : 0;

    int32_t c_start_g[8], W_shard_g[8], r_start_g[8], r_len_g[8];
    int32_t W_max = 0, r_len_max = 0;
    for (int g = 0; g < num_gpus_use; ++g) {
        int32_t c0 = (int32_t)(((int64_t)N * g) / num_gpus_use);
        int32_t c1 = (int32_t)(((int64_t)N * (g + 1)) / num_gpus_use);
        c_start_g[g] = c0;
        W_shard_g[g] = c1 - c0;
        if (W_shard_g[g] > W_max) W_max = W_shard_g[g];

        int32_t r0 = (int32_t)(((int64_t)n_rows_spr * g) / num_gpus_use);
        int32_t r1 = (int32_t)(((int64_t)n_rows_spr * (g + 1)) / num_gpus_use);
        r_start_g[g] = r0;
        r_len_g[g]   = r1 - r0;
        if (r_len_g[g] > r_len_max) r_len_max = r_len_g[g];
    }

    size_t base_floats = (size_t)(N - 1) * W_max + (size_t)n_nodes_u * B_TILE +
                         (size_t)6 * B_ROW_TILE * n_nodes_u + (size_t)3 * n_nodes_u +
                         (size_t)(use_smem_edge ? 0 : B_ROW_TILE) * n_nodes_u;
    size_t d_shard_floats = (size_t)N * W_max;
    size_t vram_with_d    = (base_floats + d_shard_floats) * sizeof(float);
    size_t vram_without_d = base_floats * sizeof(float);

    int alloc_d_shard = 1;
    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        size_t free_mem = 0, total_mem = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        if (vram_without_d + (256ULL << 20) > free_mem) {
            if (vram_per_gpu_mb_out) *vram_per_gpu_mb_out = (double)vram_without_d / (1024.0 * 1024.0);
            return 0;
        }
        if (vram_with_d + (512ULL << 20) > free_mem) {
            alloc_d_shard = 0;
        }
    }
    size_t vram_used = alloc_d_shard ? vram_with_d : vram_without_d;
    if (vram_per_gpu_mb_out) *vram_per_gpu_mb_out = (double)vram_used / (1024.0 * 1024.0);
    if (has_d_shard_out) *has_d_shard_out = alloc_d_shard;

    int32_t *d_child1_r[8], *d_child2_r[8], *d_v_map[8], *d_m_lmin[8], *d_next_k[8];
    int32_t *d_parent_u[8], *d_sub_size[8], *d_tin[8], *d_adj_u[8], *d_best_j[8];
    int64_t *d_v[8];
    float *d_D_shard[8], *d_subdist_il_shard[8], *d_tile_buf[8], *d_auto_col_buf[8], *d_auto_pool[8];
    float *d_nd_u[8], *d_edge_pool[8], *d_best_delta[8];
    cudaEvent_t ev_s2_0[8], ev_s2_1[8], ev_s3_0[8], ev_s3_1[8];

    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        if (use_smem_edge) {
            CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_tiled_top1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req_smem));
        }
        CUDA_CHECK(cudaMalloc(&d_child1_r[g], (size_t)n_nodes_r * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_child2_r[g], (size_t)n_nodes_r * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_v_map[g],    (size_t)N * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_m_lmin[g],   (size_t)n_pairs * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_next_k[g],   (size_t)n_pairs * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_parent_u[g], (size_t)n_nodes_u * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_sub_size[g], (size_t)n_nodes_u * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_tin[g],      (size_t)n_nodes_u * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_adj_u[g],    (size_t)(3 * n_nodes_u) * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_v[g],        (size_t)n_pairs * sizeof(int64_t)));

        d_D_shard[g] = NULL;
        if (alloc_d_shard) {
            CUDA_CHECK(cudaMalloc(&d_D_shard[g], (size_t)N * W_shard_g[g] * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&d_subdist_il_shard[g], (size_t)(N - 1) * W_shard_g[g] * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tile_buf[g],         (size_t)n_nodes_u * B_TILE * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_auto_col_buf[g],     (size_t)3 * B_ROW_TILE * n_nodes_u * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_auto_pool[g],        (size_t)3 * B_ROW_TILE * n_nodes_u * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_nd_u[g],             (size_t)(3 * n_nodes_u) * sizeof(float)));
        d_edge_pool[g] = NULL;
        if (!use_smem_edge) {
            CUDA_CHECK(cudaMalloc(&d_edge_pool[g], (size_t)B_ROW_TILE * n_nodes_u * sizeof(float)));
        }
        CUDA_CHECK(cudaMalloc(&d_best_delta[g],       (size_t)n_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_best_j[g],           (size_t)n_pairs * sizeof(int32_t)));

        CUDA_CHECK(cudaMemcpy(d_child1_r[g], T->child1_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_child2_r[g], T->child2_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_map[g],    T->v_map,    (size_t)N * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_m_lmin[g],   T->m_lmin,   (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_next_k[g],   T->next_k_same_lmin, (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_parent_u[g], T->parent_u, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sub_size[g], T->sub_size, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_tin[g],      T->tin,      (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_adj_u[g],    T->adj_u,    (size_t)(3 * n_nodes_u) * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v[g],        v,           (size_t)n_pairs * sizeof(int64_t), cudaMemcpyHostToDevice));

        if (alloc_d_shard && W_shard_g[g] > 0) {
            dim3 init_grid((W_shard_g[g] + 15) / 16, (N + 15) / 16);
            dim3 init_block(16, 16);
            k_init_D_shard<<<init_grid, init_block>>>(d_D_shard[g], N, c_start_g[g], W_shard_g[g], seed);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaEventCreate(&ev_s2_0[g]));
        CUDA_CHECK(cudaEventCreate(&ev_s2_1[g]));
        CUDA_CHECK(cudaEventCreate(&ev_s3_0[g]));
        CUDA_CHECK(cudaEventCreate(&ev_s3_1[g]));
    }
    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* Stage 2a: Sharded leaf-column tiling (each GPU computes W_shard_g[g] = N/G columns) */
    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventRecord(ev_s2_0[g]));
        for (int32_t t_loc = 0; t_loc < W_shard_g[g]; t_loc += B_TILE) {
            int32_t c_len = (t_loc + B_TILE <= W_shard_g[g]) ? B_TILE : (W_shard_g[g] - t_loc);
            k_tiled_leaf_cols_sharded<<<(c_len + 63) / 64, 64>>>(
                d_D_shard[g], seed, d_tile_buf[g], d_subdist_il_shard[g], d_child1_r[g], d_child2_r[g], d_tin[g],
                c_start_g[g], W_shard_g[g], t_loc, c_len, N, n_nodes_u, T->root_u_c0, T->root_u_c1, T->root_u_c2
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }

    /* Stage 2b: Sharded nd_u reconstruction across GPUs */
    int32_t num_total_batches = (r_len_max + B_ROW_TILE - 1) / B_ROW_TILE;
    size_t stage_slot_floats = (size_t)W_max * (3 * B_ROW_TILE);
    for (int32_t b_step = 0; b_step < num_total_batches; ++b_step) {
        for (int src = 0; src < num_gpus_use; ++src) {
            CUDA_CHECK(cudaSetDevice(src));
            for (int dst = 0; dst < num_gpus_use; ++dst) {
                int32_t u_off = r_start_g[dst] + b_step * B_ROW_TILE;
                int32_t u_end = r_start_g[dst] + r_len_g[dst];
                if (u_off >= u_end) continue;
                int32_t u_len = (u_off + B_ROW_TILE <= u_end) ? B_ROW_TILE : (u_end - u_off);
                int32_t K_nd  = 2 * u_len;
                float *out_buf = (src == dst)
                    ? (d_auto_col_buf[dst] + (size_t)c_start_g[src] * K_nd)
                    : (d_tile_buf[src] + (size_t)dst * stage_slot_floats);
                dim3 ld_grid((W_shard_g[src] + 15) / 16, (K_nd + 15) / 16);
                dim3 ld_block(16, 16);
                k_load_nd_u_leaves_sharded<<<ld_grid, ld_block>>>(
                    d_D_shard[src], seed, d_subdist_il_shard[src], out_buf, d_adj_u[src],
                    c_start_g[src], W_shard_g[src], N + u_off, K_nd, N
                );
                CUDA_CHECK(cudaGetLastError());
            }
        }
        if (num_gpus_use > 1) {
            for (int g = 0; g < num_gpus_use; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (int src = 0; src < num_gpus_use; ++src) {
                for (int dst = 0; dst < num_gpus_use; ++dst) {
                    if (src == dst) continue;
                    int32_t u_off = r_start_g[dst] + b_step * B_ROW_TILE;
                    int32_t u_end = r_start_g[dst] + r_len_g[dst];
                    if (u_off >= u_end) continue;
                    int32_t u_len = (u_off + B_ROW_TILE <= u_end) ? B_ROW_TILE : (u_end - u_off);
                    int32_t K_nd  = 2 * u_len;
                    float *src_ptr = d_tile_buf[src] + (size_t)dst * stage_slot_floats;
                    float *dst_ptr = d_auto_col_buf[dst] + (size_t)c_start_g[src] * K_nd;
                    CUDA_CHECK(cudaMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, (size_t)W_shard_g[src] * K_nd * sizeof(float)));
                }
            }
            for (int g = 0; g < num_gpus_use; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
        for (int dst = 0; dst < num_gpus_use; ++dst) {
            int32_t u_off = r_start_g[dst] + b_step * B_ROW_TILE;
            int32_t u_end = r_start_g[dst] + r_len_g[dst];
            if (u_off >= u_end) continue;
            int32_t u_len = (u_off + B_ROW_TILE <= u_end) ? B_ROW_TILE : (u_end - u_off);
            int32_t K_nd  = 2 * u_len;
            CUDA_CHECK(cudaSetDevice(dst));
            k_reconstruct_nd_u_cols<<<(K_nd + 63) / 64, 64>>>(
                d_auto_col_buf[dst], d_child1_r[dst], d_child2_r[dst], d_tin[dst], d_adj_u[dst], d_nd_u[dst],
                N + u_off, K_nd, N, n_nodes_u, T->root_u_c0, T->root_u_c1, T->root_u_c2
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (num_gpus_use > 1) {
        for (int g = 0; g < num_gpus_use; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        for (int src = 0; src < num_gpus_use; ++src) {
            for (int dst = 0; dst < num_gpus_use; ++dst) {
                if (src != dst && r_len_g[src] > 0) {
                    size_t off = (size_t)3 * (N + r_start_g[src]);
                    size_t cnt = (size_t)3 * r_len_g[src] * sizeof(float);
                    CUDA_CHECK(cudaMemcpyPeerAsync(d_nd_u[dst] + off, dst, d_nd_u[src] + off, src, cnt));
                }
            }
        }
        for (int g = 0; g < num_gpus_use; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventRecord(ev_s2_1[g]));
        CUDA_CHECK(cudaEventRecord(ev_s3_0[g]));
    }

    /* Stage 3: Sharded autonomous row reconstruction + Top-1 SPR reduction */
    int32_t run_batches = (max_batches_per_gpu > 0 && max_batches_per_gpu < num_total_batches)
        ? max_batches_per_gpu : num_total_batches;

    for (int32_t b_step = 0; b_step < run_batches; ++b_step) {
        for (int src = 0; src < num_gpus_use; ++src) {
            CUDA_CHECK(cudaSetDevice(src));
            for (int dst = 0; dst < num_gpus_use; ++dst) {
                int32_t r_off = r_start_g[dst] + b_step * B_ROW_TILE;
                int32_t r_end = r_start_g[dst] + r_len_g[dst];
                if (r_off >= r_end) continue;
                int32_t b_len = (r_off + B_ROW_TILE <= r_end) ? B_ROW_TILE : (r_end - r_off);
                int32_t K     = 3 * b_len;
                float *out_buf = (src == dst)
                    ? (d_auto_col_buf[dst] + (size_t)c_start_g[src] * K)
                    : (d_tile_buf[src] + (size_t)dst * stage_slot_floats);
                dim3 ld_grid((W_shard_g[src] + 15) / 16, (K + 15) / 16);
                dim3 ld_block(16, 16);
                k_load_auto_leaves_sharded<<<ld_grid, ld_block>>>(
                    d_D_shard[src], seed, d_subdist_il_shard[src], out_buf, d_v_map[src], d_parent_u[src], d_adj_u[src],
                    c_start_g[src], W_shard_g[src], 2 + r_off, K, N, n_nodes_u, T->other_root_child
                );
                CUDA_CHECK(cudaGetLastError());
            }
        }
        if (num_gpus_use > 1) {
            for (int g = 0; g < num_gpus_use; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            for (int src = 0; src < num_gpus_use; ++src) {
                for (int dst = 0; dst < num_gpus_use; ++dst) {
                    if (src == dst) continue;
                    int32_t r_off = r_start_g[dst] + b_step * B_ROW_TILE;
                    int32_t r_end = r_start_g[dst] + r_len_g[dst];
                    if (r_off >= r_end) continue;
                    int32_t b_len = (r_off + B_ROW_TILE <= r_end) ? B_ROW_TILE : (r_end - r_off);
                    int32_t K     = 3 * b_len;
                    float *src_ptr = d_tile_buf[src] + (size_t)dst * stage_slot_floats;
                    float *dst_ptr = d_auto_col_buf[dst] + (size_t)c_start_g[src] * K;
                    CUDA_CHECK(cudaMemcpyPeerAsync(dst_ptr, dst, src_ptr, src, (size_t)W_shard_g[src] * K * sizeof(float)));
                }
            }
            for (int g = 0; g < num_gpus_use; ++g) {
                CUDA_CHECK(cudaSetDevice(g));
                CUDA_CHECK(cudaDeviceSynchronize());
            }
        }
        for (int dst = 0; dst < num_gpus_use; ++dst) {
            int32_t r_off = r_start_g[dst] + b_step * B_ROW_TILE;
            int32_t r_end = r_start_g[dst] + r_len_g[dst];
            if (r_off >= r_end) continue;
            int32_t b_len = (r_off + B_ROW_TILE <= r_end) ? B_ROW_TILE : (r_end - r_off);
            int32_t K     = 3 * b_len;
            CUDA_CHECK(cudaSetDevice(dst));
            k_reconstruct_auto_cols<<<(K + 63) / 64, 64>>>(
                d_auto_col_buf[dst], d_child1_r[dst], d_child2_r[dst], d_tin[dst], d_v_map[dst], d_parent_u[dst], d_adj_u[dst],
                2 + r_off, K, N, n_nodes_u, T->root_u_c0, T->root_u_c1, T->root_u_c2, T->other_root_child
            );
            CUDA_CHECK(cudaGetLastError());
            dim3 tr_auto_grid((K + 15) / 16, (n_nodes_u + 15) / 16);
            dim3 tr_auto_block(16, 16);
            k_transpose_auto_cols<<<tr_auto_grid, tr_auto_block>>>(
                d_auto_col_buf[dst], d_auto_pool[dst], K, n_nodes_u
            );
            CUDA_CHECK(cudaGetLastError());
            k_spr_rows_tiled_top1<<<b_len, 128, use_smem_edge ? req_smem : 0>>>(
                2, r_off, N, n_nodes_u, T->other_root_child, use_smem_edge, d_v[dst], d_v_map[dst], d_adj_u[dst],
                d_parent_u[dst], d_sub_size[dst], d_m_lmin[dst], d_next_k[dst], d_nd_u[dst], d_auto_pool[dst],
                d_edge_pool[dst], d_best_delta[dst], d_best_j[dst]
            );
            CUDA_CHECK(cudaGetLastError());
        }
    }
    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventRecord(ev_s3_1[g]));
    }

    float max_s2_ms = 0.0f, max_s3_ms = 0.0f;
    float *h_best_delta = (float *)calloc((size_t)n_pairs, sizeof(float));
    int32_t *h_best_j   = (int32_t *)calloc((size_t)n_pairs, sizeof(int32_t));

    for (int g = 0; g < num_gpus_use; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaEventSynchronize(ev_s3_1[g]));
        float s2_g = 0.0f, s3_g = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&s2_g, ev_s2_0[g], ev_s2_1[g]));
        CUDA_CHECK(cudaEventElapsedTime(&s3_g, ev_s3_0[g], ev_s3_1[g]));
        if (s2_g > max_s2_ms) max_s2_ms = s2_g;
        if (s3_g > max_s3_ms) max_s3_ms = s3_g;

        int32_t eval_rows = run_batches * B_ROW_TILE;
        if (eval_rows > r_len_g[g]) eval_rows = r_len_g[g];
        if (eval_rows > 0) {
            CUDA_CHECK(cudaMemcpy(h_best_delta + 1 + r_start_g[g], d_best_delta[g] + 1 + r_start_g[g],
                                  (size_t)eval_rows * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_best_j + 1 + r_start_g[g], d_best_j[g] + 1 + r_start_g[g],
                                  (size_t)eval_rows * sizeof(int32_t), cudaMemcpyDeviceToHost));
        }

        cudaFree(d_child1_r[g]); cudaFree(d_child2_r[g]); cudaFree(d_v_map[g]); cudaFree(d_m_lmin[g]); cudaFree(d_next_k[g]);
        cudaFree(d_parent_u[g]); cudaFree(d_sub_size[g]); cudaFree(d_tin[g]); cudaFree(d_adj_u[g]); cudaFree(d_v[g]);
        if (d_D_shard[g]) cudaFree(d_D_shard[g]);
        cudaFree(d_subdist_il_shard[g]); cudaFree(d_tile_buf[g]);
        cudaFree(d_auto_col_buf[g]); cudaFree(d_auto_pool[g]); cudaFree(d_nd_u[g]);
        if (d_edge_pool[g]) cudaFree(d_edge_pool[g]);
        cudaFree(d_best_delta[g]); cudaFree(d_best_j[g]);
        cudaEventDestroy(ev_s2_0[g]); cudaEventDestroy(ev_s2_1[g]);
        cudaEventDestroy(ev_s3_0[g]); cudaEventDestroy(ev_s3_1[g]);
    }

    float g_best_val = 0.0f;
    int32_t g_best_i = 1, g_best_j = 0;
    for (int32_t i = 1; i < n_pairs; ++i) {
        if (h_best_delta[i] < g_best_val) {
            g_best_val = h_best_delta[i];
            g_best_i   = i;
            g_best_j   = h_best_j[i];
        }
    }
    free(h_best_delta);
    free(h_best_j);

    double scale = (run_batches > 0) ? ((double)num_total_batches / (double)run_batches) : 1.0;
    if (ms_stage2_out) *ms_stage2_out = (double)max_s2_ms;
    if (ms_stage3_full_out) *ms_stage3_full_out = (double)max_s3_ms * scale;
    if (best_delta_out) *best_delta_out = g_best_val;
    if (best_i_out) *best_i_out = g_best_i;
    if (best_j_out) *best_j_out = g_best_j;
    return 1;
}

static void enable_all_peer_access(int num_gpus) {
    for (int g = 0; g < num_gpus && g < 8; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        for (int p = 0; p < num_gpus && p < 8; ++p) {
            if (g == p) continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, g, p));
            if (can_access) {
                cudaError_t p_err = cudaDeviceEnablePeerAccess(p, 0);
                if (p_err != cudaSuccess && p_err != cudaErrorPeerAccessAlreadyEnabled) {
                    CUDA_CHECK(p_err);
                } else {
                    cudaGetLastError();
                }
            }
        }
    }
    CUDA_CHECK(cudaSetDevice(0));
}

static int run_large_n_scaling_sweep(int num_gpus_detected) {
    int multi_gpus = (num_gpus_detected > 8) ? 8 : ((num_gpus_detected >= 1) ? num_gpus_detected : 1);
    printf("\n========================================================================================================================\n");
    printf("LARGE-N & MULTI-GPU SCALING SWEEP (Top-1 Reduction & Column-Tiled Pipeline, 1x vs %dx GPU)\n", multi_gpus);
    printf("========================================================================================================================\n");
    printf("   N   | 1xGPU Full (ms) [sub + spr] | %dxGPU Full (ms) | Full VRAM | 1xGPU Tiled (ms) | %dxGPU Tiled (ms) | Tiled VRAM | Top-1 Match\n",
           multi_gpus, multi_gpus);
    printf("------------------------------------------------------------------------------------------------------------------------\n");

    int32_t large_sizes[] = {1000, 2000, 4000, 10000, 20000, 40000};
    int n_large = 6;
    int sweep_ok = 1;

    for (int s = 0; s < n_large; ++s) {
        int32_t N = large_sizes[s];
        int64_t *v = (int64_t *)malloc((size_t)(N - 1) * sizeof(int64_t));
        uint32_t rng = 424242U ^ (uint32_t)N;
        v[0] = 0;
        for (int32_t i = 1; i < N - 1; ++i) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            v[i] = (int64_t)(rng % (uint32_t)(i + 1));
        }

        HostTreeTopology T;
        build_host_tree_topology(v, N, &T);

        double ms_sub_1 = 0, ms_spr_1 = 0, vram_full = 0;
        double ms_sub_2 = 0, ms_spr_2 = 0;
        float bd_full_1 = 0, bd_full_2 = 0;
        int32_t bi_full_1 = 0, bj_full_1 = 0, bi_full_2 = 0, bj_full_2 = 0;

        int ok_full_1 = run_full_subdist_top1_multi_gpu(N, &T, v, 1, &ms_sub_1, &ms_spr_1, &vram_full, &bd_full_1, &bi_full_1, &bj_full_1);
        int ok_full_2 = ok_full_1 ? run_full_subdist_top1_multi_gpu(N, &T, v, multi_gpus, &ms_sub_2, &ms_spr_2, NULL, &bd_full_2, &bi_full_2, &bj_full_2) : 0;

        double ms_tiled_1 = 0, ms_tiled_2 = 0, ms_t2_s2 = 0, ms_t2_s3 = 0, vram_tiled = 0;
        float bd_tiled_1 = 0, bd_tiled_2 = 0;
        int32_t bi_tiled_1 = 0, bj_tiled_1 = 0, bi_tiled_2 = 0, bj_tiled_2 = 0;

        run_tiled_top1_multi_gpu(N, &T, v, 1, &ms_tiled_1, &vram_tiled, &bd_tiled_1, &bi_tiled_1, &bj_tiled_1);
        run_sharded_multi_gpu_top1(N, &T, v, multi_gpus, 0, &ms_t2_s2, &ms_t2_s3, NULL, NULL, &bd_tiled_2, &bi_tiled_2, &bj_tiled_2);
        ms_tiled_2 = ms_t2_s2 + ms_t2_s3;

        double diff_12_tiled = fabs((double)bd_tiled_1 - (double)bd_tiled_2);
        if (ok_full_1 && ok_full_2) {
            double diff_12_full = fabs((double)bd_full_1 - (double)bd_full_2);
            double diff_ft      = fabs((double)bd_full_1 - (double)bd_tiled_1);
            int row_ok = (bd_full_1 < 0.0f && diff_12_full < 1e-6 && diff_12_tiled < 1e-6 && diff_ft < 1e-4);
            if (!row_ok) sweep_ok = 0;
            printf(" %5d | %8.2f [%6.2f + %6.2f] |    %9.2f    | %6.0f MB |     %9.2f    |     %9.2f    |  %6.0f MB | %s (%.2e)\n",
                   N, ms_sub_1 + ms_spr_1, ms_sub_1, ms_spr_1, ms_sub_2 + ms_spr_2, vram_full,
                   ms_tiled_1, ms_tiled_2, vram_tiled, row_ok ? "PASS" : "FAIL", diff_ft);
        } else {
            int row_ok = (bd_tiled_1 < 0.0f && diff_12_tiled < 1e-6 && bi_tiled_1 == bi_tiled_2 && bj_tiled_1 == bj_tiled_2);
            if (!row_ok) sweep_ok = 0;
            printf(" %5d |            OOM (>free VRAM) |   OOM (>free)   | %6.0f MB |     %9.2f    |     %9.2f    |  %6.0f MB | %s (1v2=%.1e)\n",
                   N, vram_full, ms_tiled_1, ms_tiled_2, vram_tiled, row_ok ? "PASS" : "FAIL", diff_12_tiled);
        }
        fflush(stdout);

        free_host_tree_topology(&T);
        free(v);
    }
    printf("------------------------------------------------------------------------------------------------------------------------\n");

    /* Extreme-Scale Distributed Column-Sharded Sweep across all detected GPUs */
    printf("\n========================================================================================================================\n");
    printf("EXTREME-SCALE DISTRIBUTED COLUMN-SHARDED SWEEP (%dx GPU, Sharded subdist_int_leaf [4N^2/G bytes/GPU])\n", multi_gpus);
    printf("========================================================================================================================\n");
    printf("    N   | Moves (2N^2) | Stage2 100%% (ms) | Stage3 Sweep (ms) | Total Sweep (s) | VRAM / GPU | Total VRAM | Mode & Status\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");

    int32_t extreme_sizes[] = {40000, 80000, 120000, 180000, 250000, 350000, 380000, 400000};
    int n_extreme = 8;
    for (int s = 0; s < n_extreme; ++s) {
        int32_t N = extreme_sizes[s];
        int64_t *v = (int64_t *)malloc((size_t)(N - 1) * sizeof(int64_t));
        uint32_t rng = 424242U ^ (uint32_t)N;
        v[0] = 0;
        for (int32_t i = 1; i < N - 1; ++i) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            v[i] = (int64_t)(rng % (uint32_t)(i + 1));
        }

        HostTreeTopology T;
        build_host_tree_topology(v, N, &T);

        /* Run 100% of rows for N <= 40,000; for N >= 80,000 run 100% of Stage 2 + 4 batches/GPU of Stage 3 */
        int32_t max_batches = (N <= 40000) ? 0 : 4;
        double ms_s2 = 0, ms_s3 = 0, vram_per_gpu = 0;
        int has_d_shard = 0;
        float bd_sh = 0;
        int32_t bi_sh = 0, bj_sh = 0;

        int ok_sh = run_sharded_multi_gpu_top1(
            N, &T, v, multi_gpus, max_batches, &ms_s2, &ms_s3, &vram_per_gpu, &has_d_shard, &bd_sh, &bi_sh, &bj_sh
        );
        double moves = 2.0 * (double)N * (double)N;
        if (ok_sh) {
            int row_ok = (bd_sh < 0.0f);
            if (!row_ok) sweep_ok = 0;
            printf(" %6d |   %9.2e  |     %10.2f   |      %10.2f   |     %9.3f   |  %6.0f MB |  %6.1f GB | %s [%s]\n",
                   N, moves, ms_s2, ms_s3, (ms_s2 + ms_s3) / 1000.0,
                   vram_per_gpu, (vram_per_gpu * multi_gpus) / 1024.0,
                   has_d_shard ? (max_batches == 0 ? "Full-Shard" : "Shard+Calib") : "StreamD+Calib",
                   row_ok ? "PASS" : "FAIL");
        } else {
            printf(" %6d |   %9.2e  |              OOM (>free VRAM on %dx GPU; needed %.0f MB/GPU)\n",
                   N, moves, multi_gpus, vram_per_gpu);
            free_host_tree_topology(&T);
            free(v);
            break;
        }
        fflush(stdout);

        free_host_tree_topology(&T);
        free(v);
    }
    printf("------------------------------------------------------------------------------------------------------------------------\n");

    /* Ultra-Scale Clade-Condensed Minibatch Sweep (1M, 5M, 10M Taxa, Theorem 5.5) */
    printf("\n========================================================================================================================\n");
    printf("ULTRA-SCALE CLADE-CONDENSED MINIBATCH SWEEP (%dx GPU, N = 1,000,000 .. 10,000,000 Taxa, Active N_eff = 20,000)\n", multi_gpus);
    printf("========================================================================================================================\n");
    printf("    N_total | Active N_eff | Condense (ms) | Batch Grad (ms) | Batches/Epoch | Full Epoch (s) | VRAM / GPU | Status\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");

    int32_t mb_totals[] = {1000000, 5000000, 10000000};
    const int32_t N_eff = 20000;
    const int32_t B_act = (3 * N_eff) / 4; /* 15,000 active unpacked leaves + 5,000 condensed super-leaves */
    const int32_t K_sup = N_eff - B_act;

    int64_t *v_eff = (int64_t *)malloc((size_t)(N_eff - 1) * sizeof(int64_t));
    uint32_t mb_rng = 777777U;
    v_eff[0] = 0;
    for (int32_t i = 1; i < N_eff - 1; ++i) {
        mb_rng ^= mb_rng << 13; mb_rng ^= mb_rng >> 17; mb_rng ^= mb_rng << 5;
        v_eff[i] = (int64_t)(mb_rng % (uint32_t)(i + 1));
    }
    HostTreeTopology T_eff;
    build_host_tree_topology(v_eff, N_eff, &T_eff);

    for (int m = 0; m < 3; ++m) {
        int32_t N_tot = mb_totals[m];
        uint32_t seed = 9001U + (uint32_t)m;
        int32_t *h_leaf_to_super = (int32_t *)malloc((size_t)N_tot * sizeof(int32_t));
        float   *h_omega_leaf    = (float *)malloc((size_t)N_tot * sizeof(float));

        /* Assign first B_act leaves as individual active leaves (omega = 1.0) and partition
         * the remaining (N_tot - B_act) leaves into K_sup condensed binary clades with exact
         * dyadic weights omega_x = 2^{-d_{S_z}(r_z, x)} summing to 1.0 per super-leaf. */
        for (int32_t x = 0; x < B_act; ++x) {
            h_leaf_to_super[x] = x;
            h_omega_leaf[x]    = 1.0f;
        }
        int32_t rem_leaves = N_tot - B_act;
        int32_t cur_x = B_act;
        for (int32_t s = 0; s < K_sup; ++s) {
            int32_t z = B_act + s;
            int32_t x0 = B_act + (int32_t)(((int64_t)rem_leaves * s) / K_sup);
            int32_t x1 = B_act + (int32_t)(((int64_t)rem_leaves * (s + 1)) / K_sup);
            int32_t m_sz = x1 - x0;
            /* Balanced binary subtree of m_sz leaves: split m_sz into powers of 2 so sum(omega) == 1.0f */
            int32_t p2 = 1;
            while ((p2 << 1) <= m_sz) p2 <<= 1;
            int32_t n_deep = 2 * (m_sz - p2);
            float w_shallow = 1.0f / (float)p2;
            float w_deep    = 0.5f * w_shallow;
            for (int32_t k = 0; k < m_sz; ++k) {
                h_leaf_to_super[cur_x] = z;
                h_omega_leaf[cur_x]    = (k < n_deep) ? w_deep : w_shallow;
                cur_x++;
            }
        }

        /* Run GPU Clade-Condensation + 100% Full Multi-GPU Stage 2 & Stage 3 on N_eff = 20,000 */
        int32_t n_pairs = T_eff.n_pairs;
        int32_t n_nodes_r = T_eff.n_nodes_r;
        int32_t n_nodes_u = T_eff.n_nodes_u;
        int32_t n_rows_spr = N_eff - 2;
        int32_t n_int = (n_nodes_u - 1) - N_eff;
        const int32_t B_ROW_TILE = 1024;

        size_t req_smem = (size_t)n_nodes_u * sizeof(float);
        int max_smem_optin = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0));
        int use_smem_edge = (req_smem + 8192 <= (size_t)max_smem_optin) ? 1 : 0;

        int32_t *d_leaf_to_super[8], *d_child1_r[8], *d_child2_r[8], *d_v_map[8], *d_m_lmin[8], *d_next_k[8];
        int32_t *d_parent_u[8], *d_sub_size[8], *d_tin[8], *d_adj_u[8], *d_best_j[8];
        int64_t *d_v[8];
        float *d_omega_leaf[8], *d_phi_bar[8], *d_mu2_bar[8];
        float *d_D[8], *d_subdist[8], *d_nd_u[8], *d_edge_pool[8], *d_row_tmp[8], *d_best_delta[8];
        cudaEvent_t ev_c0[8], ev_c1[8], ev_g0[8], ev_g1[8];
        int32_t r_start_g[8], r_len_g[8];

        size_t vram_per_gpu = ((size_t)2 * N_tot + (size_t)N_eff * N_eff + (size_t)n_nodes_u * n_nodes_u +
                               (size_t)3 * n_nodes_u + (size_t)B_ROW_TILE * (2 * N_eff) +
                               (size_t)(use_smem_edge ? 0 : B_ROW_TILE) * n_nodes_u) * sizeof(float);

        for (int g = 0; g < multi_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            if (use_smem_edge) {
                CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_top1, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req_smem));
            }
            int32_t r0 = (int32_t)(((int64_t)n_rows_spr * g) / multi_gpus);
            int32_t r1 = (int32_t)(((int64_t)n_rows_spr * (g + 1)) / multi_gpus);
            r_start_g[g] = r0;
            r_len_g[g]   = r1 - r0;

            CUDA_CHECK(cudaMalloc(&d_leaf_to_super[g], (size_t)N_tot * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_omega_leaf[g],    (size_t)N_tot * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_phi_bar[g],       (size_t)N_eff * 8 * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_mu2_bar[g],       (size_t)N_eff * sizeof(float)));

            CUDA_CHECK(cudaMalloc(&d_child1_r[g], (size_t)n_nodes_r * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_child2_r[g], (size_t)n_nodes_r * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_v_map[g],    (size_t)N_eff * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_m_lmin[g],   (size_t)n_pairs * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_next_k[g],   (size_t)n_pairs * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_parent_u[g], (size_t)n_nodes_u * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_sub_size[g], (size_t)n_nodes_u * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_tin[g],      (size_t)n_nodes_u * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_adj_u[g],    (size_t)(3 * n_nodes_u) * sizeof(int32_t)));
            CUDA_CHECK(cudaMalloc(&d_v[g],        (size_t)n_pairs * sizeof(int64_t)));

            CUDA_CHECK(cudaMalloc(&d_D[g],          (size_t)N_eff * N_eff * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_subdist[g],    (size_t)n_nodes_u * n_nodes_u * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_nd_u[g],       (size_t)(3 * n_nodes_u) * sizeof(float)));
            d_edge_pool[g] = NULL;
            if (!use_smem_edge) {
                CUDA_CHECK(cudaMalloc(&d_edge_pool[g], (size_t)B_ROW_TILE * n_nodes_u * sizeof(float)));
            }
            CUDA_CHECK(cudaMalloc(&d_row_tmp[g],    (size_t)B_ROW_TILE * (2 * N_eff) * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_best_delta[g], (size_t)n_pairs * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_best_j[g],     (size_t)n_pairs * sizeof(int32_t)));

            CUDA_CHECK(cudaMemcpy(d_leaf_to_super[g], h_leaf_to_super, (size_t)N_tot * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_omega_leaf[g],    h_omega_leaf,    (size_t)N_tot * sizeof(float),   cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_phi_bar[g], 0, (size_t)N_eff * 8 * sizeof(float)));
            CUDA_CHECK(cudaMemset(d_mu2_bar[g], 0, (size_t)N_eff * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_child1_r[g], T_eff.child1_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_child2_r[g], T_eff.child2_r, (size_t)n_nodes_r * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_v_map[g],    T_eff.v_map,    (size_t)N_eff * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_m_lmin[g],   T_eff.m_lmin,   (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_next_k[g],   T_eff.next_k_same_lmin, (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_parent_u[g], T_eff.parent_u, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_sub_size[g], T_eff.sub_size, (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_tin[g],      T_eff.tin,      (size_t)n_nodes_u * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_adj_u[g],    T_eff.adj_u,    (size_t)(3 * n_nodes_u) * sizeof(int32_t), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_v[g],        v_eff,          (size_t)n_pairs * sizeof(int64_t), cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaEventCreate(&ev_c0[g]));
            CUDA_CHECK(cudaEventCreate(&ev_c1[g]));
            CUDA_CHECK(cudaEventCreate(&ev_g0[g]));
            CUDA_CHECK(cudaEventCreate(&ev_g1[g]));
        }
        for (int g = 0; g < multi_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        for (int g = 0; g < multi_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaEventRecord(ev_c0[g]));
            k_condense_super_profiles<<<(N_tot + 255) / 256, 256>>>(
                d_leaf_to_super[g], d_omega_leaf[g], d_phi_bar[g], d_mu2_bar[g], N_tot, seed
            );
            dim3 bd_grid((N_eff + 15) / 16, (N_eff + 15) / 16);
            dim3 bd_block(16, 16);
            k_build_condensed_D<<<bd_grid, bd_block>>>(d_phi_bar[g], d_mu2_bar[g], d_D[g], N_eff);
            CUDA_CHECK(cudaEventRecord(ev_c1[g]));

            CUDA_CHECK(cudaEventRecord(ev_g0[g]));
            k_bottom_up_leaf_cols<<<(N_eff + 255) / 256, 256>>>(d_D[g], d_subdist[g], d_child1_r[g], d_child2_r[g], N_eff, n_nodes_u);
            dim3 tr_grid((N_eff + 15) / 16, (n_int + 15) / 16);
            dim3 tr_block(16, 16);
            k_transpose_leaf_block<<<tr_grid, tr_block>>>(d_subdist[g], N_eff, n_nodes_u);
            if (n_int > 0) {
                k_bottom_up_int_cols<<<(n_int + 255) / 256, 256>>>(d_subdist[g], d_child1_r[g], d_child2_r[g], N_eff, n_nodes_u);
            }
            k_top_down_ancestor_walk<<<(n_nodes_u + 255) / 256, 256>>>(
                d_subdist[g], d_child1_r[g], d_child2_r[g], d_tin[g], n_nodes_u, T_eff.root_u_c0, T_eff.root_u_c1, T_eff.root_u_c2
            );
            k_nd_u<<<(N_eff - 1 + 255) / 256, 256>>>(d_subdist[g], d_adj_u[g], d_nd_u[g], N_eff, n_nodes_u);

            for (int32_t r_off = r_start_g[g]; r_off < r_start_g[g] + r_len_g[g]; r_off += B_ROW_TILE) {
                int32_t b_len = (r_off + B_ROW_TILE <= r_start_g[g] + r_len_g[g]) ? B_ROW_TILE : (r_start_g[g] + r_len_g[g] - r_off);
                k_spr_rows_top1<<<b_len, 128, use_smem_edge ? req_smem : 0>>>(
                    2 + r_off, N_eff, n_nodes_u, T_eff.other_root_child, use_smem_edge, d_v[g], d_v_map[g], d_adj_u[g], d_parent_u[g],
                    d_sub_size[g], d_m_lmin[g], d_next_k[g], d_subdist[g], d_nd_u[g], d_edge_pool[g], d_row_tmp[g], d_best_delta[g], d_best_j[g]
                );
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaEventRecord(ev_g1[g]));
        }

        /* Verify exact condensation identity on super-leaf pair (B_act, B_act + 1) against brute-force double sum */
        float gpu_d_pair = 0.0f;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaEventSynchronize(ev_g1[0]));
        CUDA_CHECK(cudaMemcpy(&gpu_d_pair, &d_D[0][(size_t)B_act * N_eff + (B_act + 1)], sizeof(float), cudaMemcpyDeviceToHost));

        int32_t s0_start = B_act;
        int32_t s0_end   = B_act + (int32_t)(((int64_t)rem_leaves * 1) / K_sup);
        int32_t s1_start = s0_end;
        int32_t s1_end   = B_act + (int32_t)(((int64_t)rem_leaves * 2) / K_sup);
        double exact_pair_sum = 0.0;
        for (int32_t x = s0_start; x < s0_end; ++x) {
            float px[8], nx2 = 0.0f;
            dev_taxon_feature_vec(x, seed, px, &nx2);
            for (int32_t y = s1_start; y < s1_end; ++y) {
                float py[8], ny2 = 0.0f;
                dev_taxon_feature_vec(y, seed, py, &ny2);
                double dxy = 0.0;
                for (int d = 0; d < 8; ++d) {
                    double diff = (double)px[d] - (double)py[d];
                    dxy += diff * diff;
                }
                exact_pair_sum += (double)h_omega_leaf[x] * (double)h_omega_leaf[y] * dxy;
            }
        }
        double rel_err_cond = fabs((double)gpu_d_pair - exact_pair_sum) / (exact_pair_sum + 1e-12);

        float max_ms_cond = 0.0f, max_ms_grad = 0.0f;
        float best_val = 0.0f;
        float *h_best_delta = (float *)calloc((size_t)n_pairs, sizeof(float));
        for (int g = 0; g < multi_gpus; ++g) {
            CUDA_CHECK(cudaSetDevice(g));
            CUDA_CHECK(cudaEventSynchronize(ev_g1[g]));
            float mc = 0.0f, mg = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&mc, ev_c0[g], ev_c1[g]));
            CUDA_CHECK(cudaEventElapsedTime(&mg, ev_g0[g], ev_g1[g]));
            if (mc > max_ms_cond) max_ms_cond = mc;
            if (mg > max_ms_grad) max_ms_grad = mg;
            if (r_len_g[g] > 0) {
                CUDA_CHECK(cudaMemcpy(h_best_delta + 1 + r_start_g[g], d_best_delta[g] + 1 + r_start_g[g],
                                      (size_t)r_len_g[g] * sizeof(float), cudaMemcpyDeviceToHost));
            }
            cudaFree(d_leaf_to_super[g]); cudaFree(d_omega_leaf[g]); cudaFree(d_phi_bar[g]); cudaFree(d_mu2_bar[g]);
            cudaFree(d_child1_r[g]); cudaFree(d_child2_r[g]); cudaFree(d_v_map[g]); cudaFree(d_m_lmin[g]); cudaFree(d_next_k[g]);
            cudaFree(d_parent_u[g]); cudaFree(d_sub_size[g]); cudaFree(d_tin[g]); cudaFree(d_adj_u[g]); cudaFree(d_v[g]);
            cudaFree(d_D[g]); cudaFree(d_subdist[g]); cudaFree(d_nd_u[g]);
            if (d_edge_pool[g]) cudaFree(d_edge_pool[g]);
            cudaFree(d_row_tmp[g]); cudaFree(d_best_delta[g]); cudaFree(d_best_j[g]);
            cudaEventDestroy(ev_c0[g]); cudaEventDestroy(ev_c1[g]);
            cudaEventDestroy(ev_g0[g]); cudaEventDestroy(ev_g1[g]);
        }
        for (int32_t i = 1; i < n_pairs; ++i) {
            if (h_best_delta[i] < best_val) best_val = h_best_delta[i];
        }
        free(h_best_delta);
        free(h_leaf_to_super);
        free(h_omega_leaf);

        int32_t batches_per_epoch = (N_tot + B_act - 1) / B_act;
        double full_epoch_s = (double)batches_per_epoch * ((double)(max_ms_cond + max_ms_grad) / 1000.0);
        int mb_ok = (best_val < 0.0f && rel_err_cond < 1e-4);
        if (!mb_ok) sweep_ok = 0;

        printf(" %10d |       %6d |     %8.2f  |       %8.2f  |          %4d |      %8.2f  |   %5.0f MB | %s (err=%.1e)\n",
               N_tot, N_eff, max_ms_cond, max_ms_grad, batches_per_epoch, full_epoch_s,
               (double)vram_per_gpu / (1024.0 * 1024.0), mb_ok ? "PASS" : "FAIL", rel_err_cond);
        fflush(stdout);
    }
    printf("------------------------------------------------------------------------------------------------------------------------\n");
    free_host_tree_topology(&T_eff);
    free(v_eff);
    return sweep_ok;
}
#endif

int main(void) {
    printf("====================================================================================================\n");
#ifdef __CUDACC__
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    enable_all_peer_access(num_gpus);
    printf("PHYLO2VEC CUDA DISCRETE GRADIENT BENCHMARK & VERIFICATION (%d CUDA GPU(s) detected)\n", num_gpus);
    for (int g = 0; g < num_gpus; ++g) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, g));
        printf("  GPU %d: %s (%.1f GB Memory, %d SMs, %zu KB Shared Mem/SM)\n",
               g, prop.name, (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               prop.multiProcessorCount, prop.sharedMemPerMultiprocessor / 1024);
    }
#else
    printf("PHYLO2VEC CUDA KERNEL VERIFICATION SUITE (Host SIMT Forward, Reverse & Randomized Thread-Order Verifier)\n");
#endif
    printf("====================================================================================================\n\n");

    int all_ok = 1;

    /* 1. Verify both N=6 repeated-minimum-leaf and root-overlap counterexamples under forward, reverse, AND randomized thread order */
    {
        const int32_t N = 6;
        int64_t v1[5] = {0, 0, 0, 0, 0};
        int64_t v2[5] = {0, 1, 1, 1, 1};
        double D1[36], D2[36];
        for (int a = 0; a < N; ++a) {
            for (int b = 0; b < N; ++b) {
                D1[a * N + b] = (a == b) ? 0.0 : 1.0;
                D2[a * N + b] = (a == b) ? 0.0 : 1.0;
            }
        }
        D1[0 * N + 2] = D1[2 * N + 0] = 10.0;
        D2[1 * N + 2] = D2[2 * N + 1] = 10.0;

        const char *mode_names[3] = {"FORWARD", "REVERSE", "RANDOMIZED FUZZ"};
        for (int rev = 0; rev <= 2; ++rev) {
            double es1 = 0, eg1 = 0, et1 = 0, es2 = 0, eg2 = 0, et2 = 0;
            int ok1 = run_and_validate_cuda_pipeline(v1, D1, N, NULL, NULL, NULL, &es1, &eg1, &et1, rev);
            int ok2 = run_and_validate_cuda_pipeline(v2, D2, N, NULL, NULL, NULL, &es2, &eg2, &et2, rev);
            if (!ok1 || !ok2 || es1 > 1e-6 || eg1 > 1e-6 || et1 > 1e-6 || es2 > 1e-6 || eg2 > 1e-6 || et2 > 1e-6) {
                all_ok = 0;
            }
            printf("[%s] Counterexample Verification (%s thread scheduling):\n",
                   (ok1 && ok2) ? "PASS" : "FAIL", mode_names[rev]);
            printf("       Case 1 (v=[0,0,0,0,0], D[0,2]=10): smem_err=%.3e, gmem_err=%.3e, tiled_err=%.3e\n", es1, eg1, et1);
            printf("       Case 2 (v=[0,1,1,1,1], D[1,2]=10): smem_err=%.3e, gmem_err=%.3e, tiled_err=%.3e\n", es2, eg2, et2);
        }

        /* Randomized SIMT thread-order fuzzing across 10 random trees + extremal caterpillars at N=16 and N=32 */
        uint32_t f_rng = 0x9E3779B9U;
        int fuzz_ok = 1;
        double max_fuzz_err = 0.0;
        for (int trial = 0; trial < 10; ++trial) {
            int32_t Nf = (trial < 5) ? 16 : 32;
            int64_t vf[31];
            double *Df = (double *)malloc((size_t)Nf * Nf * sizeof(double));
            vf[0] = 0;
            for (int32_t i = 1; i < Nf - 1; ++i) {
                if (trial % 5 == 0) vf[i] = 0;
                else if (trial % 5 == 1) vf[i] = i;
                else {
                    f_rng ^= f_rng << 13; f_rng ^= f_rng >> 17; f_rng ^= f_rng << 5;
                    vf[i] = (int64_t)(f_rng % (uint32_t)(i + 1));
                }
            }
            for (int32_t i = 0; i < Nf; ++i) {
                Df[i * Nf + i] = 0.0;
                for (int32_t j = i + 1; j < Nf; ++j) {
                    f_rng ^= f_rng << 13; f_rng ^= f_rng >> 17; f_rng ^= f_rng << 5;
                    double d = 0.05 + 5.0 * ((double)(f_rng & 0xFFFF) / 65536.0);
                    Df[i * Nf + j] = Df[j * Nf + i] = d;
                }
            }
            double es = 0, eg = 0, et = 0;
            int ok_t = run_and_validate_cuda_pipeline(vf, Df, Nf, NULL, NULL, NULL, &es, &eg, &et, 2 + trial);
            if (!ok_t || es > 2e-5 || eg > 2e-5 || et > 2e-5) fuzz_ok = 0;
            if (es > max_fuzz_err) max_fuzz_err = es;
            if (eg > max_fuzz_err) max_fuzz_err = eg;
            if (et > max_fuzz_err) max_fuzz_err = et;
            free(Df);
        }
        if (!fuzz_ok) all_ok = 0;
        printf("[%s] Randomized SIMT Thread-Order Fuzzing (10 trials, N=16,32, Fisher-Yates per barrier): max_err=%.3e\n\n",
               fuzz_ok ? "PASS" : "FAIL", max_fuzz_err);
    }

    /* 2. Benchmark & Validate across N */
    printf("----------------------------------------------------------------------------------------------------------------\n");
    printf("   N   | CPU Ref (ms) | GPU sub (ms) | SPR smem (ms) | Tiled (ms) |  smem Diff  |  gmem Diff  |  Tiled Diff | Status\n");
    printf("----------------------------------------------------------------------------------------------------------------\n");
#ifdef __CUDACC__
    int32_t sizes[] = {100, 250, 500, 1000, 2000, 4000};
    int n_sizes = 6;
#else
    int32_t sizes[] = {50, 100, 250, 500};
    int n_sizes = 4;
#endif
    uint32_t rng = 1337U;
    for (int s = 0; s < n_sizes; ++s) {
        int32_t N = sizes[s];
        int64_t *v = (int64_t *)malloc((size_t)(N - 1) * sizeof(int64_t));
        double *D = (double *)malloc((size_t)N * N * sizeof(double));
        v[0] = 0;
        for (int32_t i = 1; i < N - 1; ++i) {
            rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
            v[i] = (int64_t)(rng % (uint32_t)(i + 1));
        }
        for (int32_t i = 0; i < N; ++i) {
            D[(size_t)i * N + i] = 0.0;
            for (int32_t j = i + 1; j < N; ++j) {
                rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                double d = 0.1 + 2.0 * ((double)(rng & 0xFFFFFF) / (double)0x1000000);
                D[(size_t)i * N + j] = d;
                D[(size_t)j * N + i] = d;
            }
        }
        double t0_cpu = now_ms_host();
        double *tmp_ref = (double *)calloc((size_t)(N - 1) * (2 * (N - 1) - 1), sizeof(double));
        calculate_discrete_gradient_full_c(v, D, N, tmp_ref);
        double ms_cpu = now_ms_host() - t0_cpu;
        free(tmp_ref);

        double ms_sub = 0, ms_smem = 0, ms_tiled = 0;
        double es = 0, eg = 0, et = 0;
        int ok = run_and_validate_cuda_pipeline(v, D, N, &ms_sub, &ms_smem, &ms_tiled, &es, &eg, &et, 0);
        if (!ok || es > 5e-5 || eg > 5e-5 || et > 5e-5) all_ok = 0;
        printf(" %5d |  %10.3f  |  %10.3f  |   %9.3f   | %9.3f  |  %.3e  |  %.3e  |  %.3e  | %s\n",
               N, ms_cpu, ms_sub, ms_smem, ms_tiled, es, eg, et, ok ? "PASS" : "FAIL");
        free(v);
        free(D);
    }
    printf("----------------------------------------------------------------------------------------------------------------\n");

#ifdef __CUDACC__
    if (!run_large_n_scaling_sweep(num_gpus)) {
        all_ok = 0;
    }
#endif

    if (!all_ok) {
        fprintf(stderr, "CUDA pipeline validation FAILED.\n");
        return 1;
    }
    printf("\nAll CUDA kernels (smem, gmem fallback, top-1 candidate reduction, and column-tiled) verified (exit code 0).\n");
    return 0;
}
