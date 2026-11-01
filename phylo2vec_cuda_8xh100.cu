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
 * STAGE 3 DEVICE HELPER: Heavy-First Logarithmic-Stack Half-Tree DFS
 * ============================================================================ */
__device__ __forceinline__ void dev_run_half_tree_dfs(
    int32_t start_node,
    int32_t orig_node,
    int32_t parent_node,
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
    if (start_node < n_leaves) return;

    int32_t n0 = adj_u[3 * start_node + 0];
    int32_t n1 = adj_u[3 * start_node + 1];
    int32_t n2 = adj_u[3 * start_node + 2];

    int32_t k_in = (n0 == parent_node) ? 0 : ((n1 == parent_node) ? 1 : 2);
    int32_t k0 = (k_in == 0) ? 1 : 0;
    int32_t k1 = (k_in == 2) ? 1 : 2;

    int32_t B0 = adj_u[3 * start_node + k0];
    int32_t B1 = adj_u[3 * start_node + k1];
    float D_AB = nd_u[3 * start_node + k_in];
    float d_kc_init = row_root[orig_node];

    int32_t st_u[32];
    int32_t st_c[32];
    float   st_pow[32];
    float   st_dkc[32];
    float   st_dist[32];
    int32_t top = 0;

    float new_dist0 = 0.25f * (row_orig[B1] + row_root[B0] - D_AB - d_kc_init);
    int32_t c_end0 = (parent_u[start_node] == B0) ? start_node : B0;
    edge_delta[c_end0] = new_dist0;

    float new_dist1 = 0.25f * (row_orig[B0] + row_root[B1] - D_AB - d_kc_init);
    int32_t c_end1 = (parent_u[start_node] == B1) ? start_node : B1;
    edge_delta[c_end1] = new_dist1;

    if (B0 >= n_leaves && B1 >= n_leaves) {
        int32_t s0 = (parent_u[B0] == start_node) ? sub_size[B0] : (n_nodes_u - sub_size[start_node]);
        int32_t s1 = (parent_u[B1] == start_node) ? sub_size[B1] : (n_nodes_u - sub_size[start_node]);
        if (s0 >= s1) {
            st_u[top] = B0; st_c[top] = start_node; st_pow[top] = 0.25f;
            st_dkc[top] = 0.5f * (d_kc_init + row_root[B1]); st_dist[top] = new_dist0; top++;
            st_u[top] = B1; st_c[top] = start_node; st_pow[top] = 0.25f;
            st_dkc[top] = 0.5f * (d_kc_init + row_root[B0]); st_dist[top] = new_dist1; top++;
        } else {
            st_u[top] = B1; st_c[top] = start_node; st_pow[top] = 0.25f;
            st_dkc[top] = 0.5f * (d_kc_init + row_root[B0]); st_dist[top] = new_dist1; top++;
            st_u[top] = B0; st_c[top] = start_node; st_pow[top] = 0.25f;
            st_dkc[top] = 0.5f * (d_kc_init + row_root[B1]); st_dist[top] = new_dist0; top++;
        }
    } else if (B0 >= n_leaves) {
        st_u[top] = B0; st_c[top] = start_node; st_pow[top] = 0.25f;
        st_dkc[top] = 0.5f * (d_kc_init + row_root[B1]); st_dist[top] = new_dist0; top++;
    } else if (B1 >= n_leaves) {
        st_u[top] = B1; st_c[top] = start_node; st_pow[top] = 0.25f;
        st_dkc[top] = 0.5f * (d_kc_init + row_root[B0]); st_dist[top] = new_dist1; top++;
    }

    while (top > 0) {
        --top;
        int32_t u = st_u[top];
        int32_t C = st_c[top];
        float pow_neg_len = st_pow[top];
        float d_kc = st_dkc[top];
        float curr_dist = st_dist[top];

        int32_t a0 = adj_u[3 * u + 0];
        int32_t a1 = adj_u[3 * u + 1];
        int32_t a2 = adj_u[3 * u + 2];

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
                st_u[top] = b1; st_c[top] = u; st_pow[top] = next_pow;
                st_dkc[top] = 0.5f * (d_kc + r_b0); st_dist[top] = ndist1; top++;
            } else {
                st_u[top] = b1; st_c[top] = u; st_pow[top] = next_pow;
                st_dkc[top] = 0.5f * (d_kc + r_b0); st_dist[top] = ndist1; top++;
                st_u[top] = b0; st_c[top] = u; st_pow[top] = next_pow;
                st_dkc[top] = 0.5f * (d_kc + r_b1); st_dist[top] = ndist0; top++;
            }
        } else if (b0 >= n_leaves) {
            st_u[top] = b0; st_c[top] = u; st_pow[top] = next_pow;
            st_dkc[top] = 0.5f * (d_kc + r_b1); st_dist[top] = ndist0; top++;
        } else if (b1 >= n_leaves) {
            st_u[top] = b1; st_c[top] = u; st_pow[top] = next_pow;
            st_dkc[top] = 0.5f * (d_kc + r_b0); st_dist[top] = ndist1; top++;
        }
    }
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
#else
    float *s_edge_delta = host_smem_buf;
#endif
    int32_t v_index = r_start + blockIdx.x;
    int32_t root_u = n_nodes_u - 1;
    int32_t max_cols = 2 * (n_leaves - 1) - 1;

    if (SIMT_PHASE_ACTIVE) {
        for (int32_t u = threadIdx.x; u < root_u; u += blockDim.x) {
            s_edge_delta[u] = 0.0f;
        }
    }
    __syncthreads();

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

    if (SIMT_PHASE_ACTIVE) {
        const float *row_root = &subdist[(size_t)tree_root * n_nodes_u];
        if (threadIdx.x == 0) {
            dev_run_half_tree_dfs(
                other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                nd_u, row_root, &subdist[(size_t)other1 * n_nodes_u], s_edge_delta
            );
        } else if (threadIdx.x == 1) {
            dev_run_half_tree_dfs(
                other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                nd_u, row_root, &subdist[(size_t)other0 * n_nodes_u], s_edge_delta
            );
        }
    }
    __syncthreads();

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
    int32_t v_index = r_start + blockIdx.x;
    int32_t root_u = n_nodes_u - 1;
    int32_t max_cols = 2 * (n_leaves - 1) - 1;
    float *g_edge_delta = &edge_delta_pool[(size_t)blockIdx.x * n_nodes_u];

    if (SIMT_PHASE_ACTIVE) {
        for (int32_t u = threadIdx.x; u < root_u; u += blockDim.x) {
            g_edge_delta[u] = 0.0f;
        }
    }
    __syncthreads();

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

    if (SIMT_PHASE_ACTIVE) {
        const float *row_root = &subdist[(size_t)tree_root * n_nodes_u];
        if (threadIdx.x == 0) {
            dev_run_half_tree_dfs(
                other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                nd_u, row_root, &subdist[(size_t)other1 * n_nodes_u], g_edge_delta
            );
        } else if (threadIdx.x == 1) {
            dev_run_half_tree_dfs(
                other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                nd_u, row_root, &subdist[(size_t)other0 * n_nodes_u], g_edge_delta
            );
        }
    }
    __syncthreads();

    int32_t i = v_index - 1;
    float *out_row = &delta_out_slice[(size_t)(v_index - r_start) * max_cols];
    dev_map_edge_delta_to_row(i, n_leaves, other_root_child, m_lmin, next_k_same_lmin, g_edge_delta, out_row);
}

/* ============================================================================
 * STAGE 3 KERNEL C: Fused Top-1 Candidate Reduction (`k_spr_rows_top1`)
 *   - Reduces each row `i` on the fly to `(row_best_delta[i], row_best_j[i])`
 *     using O(N) total output memory instead of O(N^2) `delta_full` memory!
 * ============================================================================ */
__global__ void k_spr_rows_top1(
    int32_t r_start,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t other_root_child,
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
    int32_t v_index = r_start + blockIdx.x;
    int32_t i = v_index - 1;
    int32_t root_u = n_nodes_u - 1;
    float *g_edge_delta = &edge_delta_pool[(size_t)blockIdx.x * n_nodes_u];
    float *row_tmp = &row_tmp_pool[(size_t)blockIdx.x * (2 * n_leaves)];

    if (SIMT_PHASE_ACTIVE) {
        for (int32_t u = threadIdx.x; u < root_u; u += blockDim.x) {
            g_edge_delta[u] = 0.0f;
        }
    }
    __syncthreads();

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

    if (SIMT_PHASE_ACTIVE) {
        const float *row_root = &subdist[(size_t)tree_root * n_nodes_u];
        if (threadIdx.x == 0) {
            dev_run_half_tree_dfs(
                other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                nd_u, row_root, &subdist[(size_t)other1 * n_nodes_u], g_edge_delta
            );
        } else if (threadIdx.x == 1) {
            dev_run_half_tree_dfs(
                other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                nd_u, row_root, &subdist[(size_t)other0 * n_nodes_u], g_edge_delta
            );
        }
    }
    __syncthreads();

    dev_map_edge_delta_to_row(i, n_leaves, other_root_child, m_lmin, next_k_same_lmin, g_edge_delta, row_tmp);
    __syncthreads();

    if (SIMT_PHASE_ACTIVE && threadIdx.x == 0) {
        int32_t vi = (int32_t)v_curr[i];
        float b_val = 0.0f;
        int32_t b_j = vi;
        for (int32_t j = 0; j <= 2 * i; ++j) {
            if (j == vi) continue;
            float val = row_tmp[j];
            if (val < b_val) {
                b_val = val;
                b_j = j;
            }
        }
        best_delta_out[i] = b_val;
        best_j_out[i] = b_j;
    }
}

/* ============================================================================
 * COLUMN-TILED LOW-MEMORY PIPELINE (`k_tiled_leaf_cols` + `k_spr_rows_tiled`)
 * ============================================================================ */
__global__ void k_tiled_leaf_cols(
    const float * __restrict__ D_tile,
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
        tile_buf[(size_t)i * c_len + x] = D_tile[(size_t)i * c_len + x];
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

__device__ __forceinline__ void dev_compute_autonomous_row(
    int32_t w, int32_t n_leaves, int32_t n_nodes_u,
    int32_t root_u_c0, int32_t root_u_c1, int32_t root_u_c2,
    const int32_t * __restrict__ child1_r,
    const int32_t * __restrict__ child2_r,
    const int32_t * __restrict__ tin,
    const float * __restrict__ D,
    const float * __restrict__ subdist_int_leaf,
    float * __restrict__ row_w
) {
    int32_t root_u = n_nodes_u - 1;
    const float *src = (w < n_leaves)
        ? &D[(size_t)w * n_leaves]
        : &subdist_int_leaf[(size_t)(w - n_leaves) * n_leaves];
    if (SIMT_PHASE_ACTIVE) {
        for (int32_t x = threadIdx.x; x < n_leaves; x += blockDim.x) {
            row_w[x] = src[x];
        }
    }
    __syncthreads();

    if (SIMT_PHASE_ACTIVE && threadIdx.x == 0) {
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
    __syncthreads();
}

__global__ void k_spr_rows_tiled(
    int32_t r_start,
    int32_t n_leaves,
    int32_t n_nodes_u,
    int32_t root_u_c0,
    int32_t root_u_c1,
    int32_t root_u_c2,
    int32_t other_root_child,
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
    float * __restrict__ auto_row_pool, /* 4 * n_nodes_u floats per block: row_root, row_o0, row_o1, edge_delta */
    float * __restrict__ delta_out_slice
) {
    int32_t v_index = r_start + blockIdx.x;
    int32_t root_u = n_nodes_u - 1;
    int32_t max_cols = 2 * (n_leaves - 1) - 1;

    float *row_root   = &auto_row_pool[(size_t)blockIdx.x * 4 * n_nodes_u + 0 * n_nodes_u];
    float *row_o0     = &auto_row_pool[(size_t)blockIdx.x * 4 * n_nodes_u + 1 * n_nodes_u];
    float *row_o1     = &auto_row_pool[(size_t)blockIdx.x * 4 * n_nodes_u + 2 * n_nodes_u];
    float *edge_delta = &auto_row_pool[(size_t)blockIdx.x * 4 * n_nodes_u + 3 * n_nodes_u];

    if (SIMT_PHASE_ACTIVE) {
        for (int32_t u = threadIdx.x; u < root_u; u += blockDim.x) {
            edge_delta[u] = 0.0f;
        }
    }
    __syncthreads();

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

    dev_compute_autonomous_row(tree_root, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2,
                               child1_r, child2_r, tin, D, subdist_int_leaf, row_root);
    dev_compute_autonomous_row(other0, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2,
                               child1_r, child2_r, tin, D, subdist_int_leaf, row_o0);
    dev_compute_autonomous_row(other1, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2,
                               child1_r, child2_r, tin, D, subdist_int_leaf, row_o1);

    if (SIMT_PHASE_ACTIVE) {
        if (threadIdx.x == 0) {
            dev_run_half_tree_dfs(
                other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                nd_u, row_root, row_o1, edge_delta
            );
        } else if (threadIdx.x == 1) {
            dev_run_half_tree_dfs(
                other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size,
                nd_u, row_root, row_o0, edge_delta
            );
        }
    }
    __syncthreads();

    int32_t i = v_index - 1;
    float *out_row = &delta_out_slice[(size_t)(v_index - r_start) * max_cols];
    dev_map_edge_delta_to_row(i, n_leaves, other_root_child, m_lmin, next_k_same_lmin, edge_delta, out_row);
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

    int32_t *pairs = (int32_t *)malloc((size_t)(2 * n_pairs) * sizeof(int32_t));
    int32_t *parents_uf = (int32_t *)malloc((size_t)n_nodes_r * sizeof(int32_t));
    int32_t *min_leaf = (int32_t *)malloc((size_t)n_nodes_r * sizeof(int32_t));
    int32_t *last_seen = (int32_t *)malloc((size_t)n_leaves * sizeof(int32_t));

    for (int32_t k = 0; k < n_pairs; ++k) {
        int32_t i = n_pairs - 1 - k;
        pairs[2 * k]     = (int32_t)v[i];
        pairs[2 * k + 1] = i + 1;
    }
    for (int32_t i = 0; i < n_nodes_r; ++i) {
        parents_uf[i] = i;
        min_leaf[i] = i;
    }
    for (int32_t i = 0; i < n_pairs; ++i) {
        int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
        while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
        while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
        int32_t m = n_leaves + i;
        T->child1_r[m] = p2;
        T->child2_r[m] = p1;
        parents_uf[p1] = m;
        parents_uf[p2] = m;
        int32_t ml1 = min_leaf[p2], ml2 = min_leaf[p1];
        if (ml1 < ml2) {
            min_leaf[m] = ml1; T->v_map[ml2] = p1; T->m_lmin[i] = ml1;
        } else {
            min_leaf[m] = ml2; T->v_map[ml1] = p2; T->m_lmin[i] = ml2;
        }
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

    free(pairs);
    free(parents_uf);
    free(min_leaf);
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
    if (req_smem <= (size_t)max_smem_optin) {
        CUDA_CHECK(cudaFuncSetAttribute(k_spr_rows_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)req_smem));
    }

    int32_t *d_child1_r, *d_child2_r, *d_v_map, *d_m_lmin, *d_next_k, *d_parent_u, *d_sub_size, *d_tin, *d_adj_u, *d_best_j;
    int64_t *d_v;
    float *d_D, *d_subdist, *d_nd_u, *d_delta_smem, *d_delta_gmem, *d_delta_tiled, *d_edge_pool, *d_row_tmp, *d_best_delta;
    float *d_subdist_int_leaf, *d_tile_buf, *d_D_tile, *d_auto_pool;

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
    CUDA_CHECK(cudaMalloc(&d_delta_smem,  (size_t)n_pairs * max_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta_gmem,  (size_t)n_pairs * max_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_delta_tiled, (size_t)n_pairs * max_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_edge_pool,   (size_t)n_rows_spr * n_nodes_u * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_row_tmp,     (size_t)n_rows_spr * (2 * N) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_delta,  (size_t)n_pairs * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_best_j,      (size_t)n_pairs * sizeof(int32_t)));

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

    cudaEvent_t ev0, ev1, ev2, ev3;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventCreate(&ev2));
    CUDA_CHECK(cudaEventCreate(&ev3));

    int32_t n_int = (n_nodes_u - 1) - N;
    CUDA_CHECK(cudaEventRecord(ev0));
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
    CUDA_CHECK(cudaEventRecord(ev1));

    if (req_smem <= (size_t)max_smem_optin) {
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
    CUDA_CHECK(cudaEventRecord(ev2));

    k_spr_rows_gmem<<<n_rows_spr, 128>>>(
        2, N, n_nodes_u, T.other_root_child, d_v_map, d_adj_u, d_parent_u,
        d_sub_size, d_m_lmin, d_next_k, d_subdist, d_nd_u, d_edge_pool, d_delta_gmem + max_cols
    );
    CUDA_CHECK(cudaGetLastError());

    k_spr_rows_top1<<<n_rows_spr, 128>>>(
        2, N, n_nodes_u, T.other_root_child, d_v, d_v_map, d_adj_u, d_parent_u,
        d_sub_size, d_m_lmin, d_next_k, d_subdist, d_nd_u, d_edge_pool, d_row_tmp, d_best_delta, d_best_j
    );
    CUDA_CHECK(cudaGetLastError());

    /* Column-Tiled execution */
    const int32_t B_TILE = 128;
    CUDA_CHECK(cudaMalloc(&d_subdist_int_leaf, (size_t)(N - 1) * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tile_buf,         (size_t)n_nodes_u * B_TILE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D_tile,           (size_t)N * B_TILE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_auto_pool,        (size_t)n_rows_spr * 4 * n_nodes_u * sizeof(float)));
    float *h_D_tile = (float *)malloc((size_t)N * B_TILE * sizeof(float));

    CUDA_CHECK(cudaEventRecord(ev2));
    for (int32_t c0 = 0; c0 < N; c0 += B_TILE) {
        int32_t c_len = (c0 + B_TILE <= N) ? B_TILE : (N - c0);
        for (int32_t i = 0; i < N; ++i) {
            memcpy(&h_D_tile[(size_t)i * c_len], &h_D[(size_t)i * N + c0], (size_t)c_len * sizeof(float));
        }
        CUDA_CHECK(cudaMemcpy(d_D_tile, h_D_tile, (size_t)N * c_len * sizeof(float), cudaMemcpyHostToDevice));
        k_tiled_leaf_cols<<<(c_len + 127) / 128, 128>>>(
            d_D_tile, d_tile_buf, d_subdist_int_leaf, d_child1_r, d_child2_r, d_tin,
            c0, c_len, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2
        );
        CUDA_CHECK(cudaGetLastError());
    }
    k_spr_rows_tiled<<<n_rows_spr, 128>>>(
        2, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2, T.other_root_child,
        d_child1_r, d_child2_r, d_tin, d_v_map, d_adj_u, d_parent_u, d_sub_size,
        d_m_lmin, d_next_k, d_D, d_subdist_int_leaf, d_nd_u, d_auto_pool, d_delta_tiled + max_cols
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev3));
    CUDA_CHECK(cudaEventSynchronize(ev3));

    float ms_sub = 0.0f, ms_smem = 0.0f, ms_tiled = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_sub, ev0, ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms_smem, ev1, ev2));
    CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, ev2, ev3));
    if (subdist_ms_out) *subdist_ms_out = (double)ms_sub;
    if (spr_smem_ms_out) *spr_smem_ms_out = (double)ms_smem;
    if (spr_tiled_ms_out) *spr_tiled_ms_out = (double)ms_tiled;

    CUDA_CHECK(cudaMemcpy(h_delta_smem,  d_delta_smem,  (size_t)n_pairs * max_cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta_gmem,  d_delta_gmem,  (size_t)n_pairs * max_cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta_tiled, d_delta_tiled, (size_t)n_pairs * max_cols * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_delta,  d_best_delta,  (size_t)n_pairs * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_best_j,      d_best_j,      (size_t)n_pairs * sizeof(int32_t), cudaMemcpyDeviceToHost));

    free(h_D_tile);
    cudaFree(d_subdist_int_leaf); cudaFree(d_tile_buf); cudaFree(d_D_tile); cudaFree(d_auto_pool);
    cudaFree(d_child1_r); cudaFree(d_child2_r); cudaFree(d_v_map); cudaFree(d_m_lmin); cudaFree(d_next_k);
    cudaFree(d_parent_u); cudaFree(d_sub_size); cudaFree(d_tin); cudaFree(d_adj_u); cudaFree(d_v);
    cudaFree(d_D); cudaFree(d_subdist); cudaFree(d_nd_u); cudaFree(d_delta_smem); cudaFree(d_delta_gmem);
    cudaFree(d_delta_tiled); cudaFree(d_edge_pool); cudaFree(d_row_tmp); cudaFree(d_best_delta); cudaFree(d_best_j);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2); cudaEventDestroy(ev3);
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

    blockDim = {128, 1, 1};
    for (int bx = 0; bx < n_rows_spr; ++bx) {
        blockIdx = {bx, 0, 0};
        for (int phase = 0; phase < 4; ++phase) {
            g_simt_target_phase = phase;
            for (int step = 0; step < 128; ++step) {
                int tx = reverse_thread_order ? (127 - step) : step;
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
            for (int step = 0; step < 128; ++step) {
                int tx = reverse_thread_order ? (127 - step) : step;
                threadIdx = {tx, 0, 0};
                g_simt_curr_phase = 0;
                k_spr_rows_gmem(
                    2, N, n_nodes_u, T.other_root_child, T.v_map, T.adj_u, T.parent_u,
                    T.sub_size, T.m_lmin, T.next_k_same_lmin, subdist, nd_u,
                    edge_pool, h_delta_gmem + max_cols
                );
            }
        }
        for (int phase = 0; phase < 5; ++phase) {
            g_simt_target_phase = phase;
            for (int tx = 0; tx < 128; ++tx) {
                threadIdx = {tx, 0, 0};
                g_simt_curr_phase = 0;
                k_spr_rows_top1(
                    2, N, n_nodes_u, T.other_root_child, v, T.v_map, T.adj_u, T.parent_u,
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
    float *D_tile           = (float *)calloc((size_t)N * B_TILE, sizeof(float));
    float *auto_pool        = (float *)calloc((size_t)n_rows_spr * 4 * n_nodes_u, sizeof(float));

    for (int32_t c0 = 0; c0 < N; c0 += B_TILE) {
        int32_t c_len = (c0 + B_TILE <= N) ? B_TILE : (N - c0);
        for (int32_t i = 0; i < N; ++i) {
            memcpy(&D_tile[(size_t)i * c_len], &h_D[(size_t)i * N + c0], (size_t)c_len * sizeof(float));
        }
        blockDim = {128, 1, 1};
        blockIdx = {0, 0, 0};
        for (int tx = 0; tx < c_len; ++tx) {
            threadIdx = {tx, 0, 0};
            k_tiled_leaf_cols(
                D_tile, tile_buf, subdist_int_leaf, T.child1_r, T.child2_r, T.tin,
                c0, c_len, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2
            );
        }
    }
    for (int bx = 0; bx < n_rows_spr; ++bx) {
        blockIdx = {bx, 0, 0};
        for (int phase = 0; phase < 10; ++phase) {
            g_simt_target_phase = phase;
            for (int step = 0; step < 128; ++step) {
                int tx = reverse_thread_order ? (127 - step) : step;
                threadIdx = {tx, 0, 0};
                g_simt_curr_phase = 0;
                k_spr_rows_tiled(
                    2, N, n_nodes_u, T.root_u_c0, T.root_u_c1, T.root_u_c2, T.other_root_child,
                    T.child1_r, T.child2_r, T.tin, T.v_map, T.adj_u, T.parent_u, T.sub_size,
                    T.m_lmin, T.next_k_same_lmin, h_D, subdist_int_leaf, nd_u,
                    auto_pool, h_delta_tiled + max_cols
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
    free(subdist_int_leaf); free(tile_buf); free(D_tile); free(auto_pool);
#endif

    double max_err_smem = 0.0, max_err_gmem = 0.0, max_err_tiled = 0.0;
    int ok = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        /* Current coordinate delta must be strictly 0 */
        if (fabsf(h_delta_smem[(size_t)i * max_cols + v[i]]) > 1e-6f ||
            fabsf(h_delta_gmem[(size_t)i * max_cols + v[i]]) > 1e-6f ||
            fabsf(h_delta_tiled[(size_t)i * max_cols + v[i]]) > 1e-6f) {
            ok = 0;
        }
        float row_min = 0.0f;
        for (int32_t j = 0; j <= 2 * i; ++j) {
            double ref = delta_ref[(size_t)i * max_cols + j];
            double es = fabs((double)h_delta_smem[(size_t)i * max_cols + j] - ref);
            double eg = fabs((double)h_delta_gmem[(size_t)i * max_cols + j] - ref);
            double et = fabs((double)h_delta_tiled[(size_t)i * max_cols + j] - ref);
            if (isnan(es) || isnan(eg) || isnan(et)) ok = 0;
            if (es > max_err_smem) max_err_smem = es;
            if (eg > max_err_gmem) max_err_gmem = eg;
            if (et > max_err_tiled) max_err_tiled = et;
            if (j != (int32_t)v[i] && h_delta_gmem[(size_t)i * max_cols + j] < row_min) {
                row_min = h_delta_gmem[(size_t)i * max_cols + j];
            }
        }
        if (fabsf(h_best_delta[i] - row_min) > 1e-6f) ok = 0;
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

int main(void) {
    printf("====================================================================================================\n");
#ifdef __CUDACC__
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    printf("PHYLO2VEC CUDA DISCRETE GRADIENT BENCHMARK & VERIFICATION (%d CUDA GPU(s) detected)\n", num_gpus);
    for (int g = 0; g < num_gpus; ++g) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, g));
        printf("  GPU %d: %s (%.1f GB Memory, %d SMs, %zu KB Shared Mem/SM)\n",
               g, prop.name, (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               prop.multiProcessorCount, prop.sharedMemPerMultiprocessor / 1024);
    }
#else
    printf("PHYLO2VEC CUDA KERNEL VERIFICATION SUITE (Host SIMT Forward & Reverse Thread-Order Verifier)\n");
#endif
    printf("====================================================================================================\n\n");

    int all_ok = 1;

    /* 1. Verify both N=6 repeated-minimum-leaf and root-overlap counterexamples under forward AND reverse thread order */
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

        for (int rev = 0; rev <= 1; ++rev) {
            double es1 = 0, eg1 = 0, et1 = 0, es2 = 0, eg2 = 0, et2 = 0;
            int ok1 = run_and_validate_cuda_pipeline(v1, D1, N, NULL, NULL, NULL, &es1, &eg1, &et1, rev);
            int ok2 = run_and_validate_cuda_pipeline(v2, D2, N, NULL, NULL, NULL, &es2, &eg2, &et2, rev);
            if (!ok1 || !ok2 || es1 > 1e-6 || eg1 > 1e-6 || et1 > 1e-6 || es2 > 1e-6 || eg2 > 1e-6 || et2 > 1e-6) {
                all_ok = 0;
            }
            printf("[%s] Counterexample Verification (%s thread scheduling):\n",
                   (ok1 && ok2) ? "PASS" : "FAIL", rev ? "REVERSE" : "FORWARD");
            printf("       Case 1 (v=[0,0,0,0,0], D[0,2]=10): smem_err=%.3e, gmem_err=%.3e, tiled_err=%.3e\n", es1, eg1, et1);
            printf("       Case 2 (v=[0,1,1,1,1], D[1,2]=10): smem_err=%.3e, gmem_err=%.3e, tiled_err=%.3e\n", es2, eg2, et2);
        }
        printf("\n");
    }

    /* 2. Benchmark & Validate across N = 50, 100, 250, 500 */
    printf("----------------------------------------------------------------------------------------------------\n");
    printf("   N   | subdist (ms) | SPR smem (ms) | Tiled (ms) |  smem Diff  |  gmem Diff  |  Tiled Diff | Status\n");
    printf("----------------------------------------------------------------------------------------------------\n");
    int32_t sizes[4] = {50, 100, 250, 500};
    uint32_t rng = 1337U;
    for (int s = 0; s < 4; ++s) {
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
        double ms_sub = 0, ms_smem = 0, ms_tiled = 0;
        double es = 0, eg = 0, et = 0;
        int ok = run_and_validate_cuda_pipeline(v, D, N, &ms_sub, &ms_smem, &ms_tiled, &es, &eg, &et, 0);
        if (!ok || es > 5e-5 || eg > 5e-5 || et > 5e-5) all_ok = 0;
        printf(" %5d |  %9.3f   |   %9.3f   | %9.3f  |  %.3e  |  %.3e  |  %.3e  | %s\n",
               N, ms_sub, ms_smem, ms_tiled, es, eg, et, ok ? "PASS" : "FAIL");
        free(v);
        free(D);
    }
    printf("----------------------------------------------------------------------------------------------------\n");

    if (!all_ok) {
        fprintf(stderr, "CUDA pipeline validation FAILED.\n");
        return 1;
    }
    printf("\nAll CUDA kernels (smem, gmem fallback, top-1 candidate reduction, and column-tiled) verified (exit code 0).\n");
    return 0;
}
