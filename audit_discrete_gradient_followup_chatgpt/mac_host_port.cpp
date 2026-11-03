#define restrict __restrict__
#include <cstddef>
using dispatch_queue_t=int;
const int QOS_CLASS_USER_INTERACTIVE=0;
static int dispatch_get_global_queue(int,int){return 0;}
template<class F> void dispatch_apply(size_t n,int,F f){for(size_t i=0;i<n;i++)f(i);}
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/*
 * bench_parallel_gpu.m
 * --------------------
 * Comprehensive Large-N Benchmark (N = 100 .. 4,000) of the Phylo2Vec Discrete Gradient:
 *   1. 1-Core CPU (FP64, C99 baseline)
 *   2. 10-Core CPU (FP64, Column-Parallel Lockstep `subdist` + O(N log N) Ancestor Top-Down + Heavy-First DFS)
 *   3. 10-Core CPU (FP32, SIMD-vectorized single-precision)
 *   4. Apple M5 Metal GPU (`subdist` Matrix DP in Metal + Zero-Copy Hybrid GPU+CPU Unified Memory Pipeline)
 *   5. Full Apple M5 Metal GPU (End-to-End GPU: `subdist` + Heavy-First Register-Stack GPU SPR Scan)
 *
 * Compile & run:
 *   /usr/bin/clang -O3 -march=native -fobjc-arc -framework Metal -framework Foundation \
 *       bench_parallel_gpu.m phylo2vec_fastme.c -o bench_parallel_gpu && ./bench_parallel_gpu
 */

extern "C" void calculate_discrete_gradient_full_c(
    const int64_t *v, const double *D, int32_t n_leaves, double *delta_full_out
);

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static uint32_t rng_state = 133742U;
static inline uint32_t xorshift32(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}
static inline double rand_uniform(void) {
    return (double)(xorshift32() & 0xFFFFFF) / (double)0x1000000;
}

/* ============================================================================
 * Heavy-First 32-Slot Stack Tree Traversal (FP64 & FP32)
 * Guaranteed max stack depth <= floor(log2(2N)) + 2 <= 18 for ANY binary tree!
 * Fits in 1 KB of L1 cache / registers instead of a 320 KB BFS queue.
 * ============================================================================ */
static inline void run_half_tree_dfs_f64(
    int32_t start_node, int32_t orig_node, int32_t parent_node, int32_t n_leaves, int32_t n_nodes_u,
    const int32_t * restrict adj_u, const int32_t * restrict parent_u, const int32_t * restrict sub_size,
    const double * restrict nd_u, const double * restrict row_root, const double * restrict row_orig,
    double * restrict edge_delta
) {
    if (start_node < n_leaves) return;
    int32_t s_node[32], s_prev[32];
    double s_pow[32], s_dkc[32], s_dist[32];
    int32_t top = 0;

    int32_t n0 = adj_u[3 * start_node + 0];
    int32_t n1 = adj_u[3 * start_node + 1];
    int32_t n2 = adj_u[3 * start_node + 2];
    int32_t k_in = (n0 == parent_node) ? 0 : ((n1 == parent_node) ? 1 : 2);
    int32_t k0 = (k_in == 0) ? 1 : 0;
    int32_t k1 = (k_in == 2) ? 1 : 2;
    int32_t B0 = adj_u[3 * start_node + k0];
    int32_t B1 = adj_u[3 * start_node + k1];
    double D_AB = nd_u[3 * start_node + k_in];
    double d_kc_init = row_root[orig_node];

    {
        double D_AC = row_orig[B1], D_kB = row_root[B0];
        double new_dist = 0.25 * (D_AC + D_kB - D_AB - d_kc_init);
        int32_t child_end = (parent_u[start_node] == B0) ? start_node : B0;
        edge_delta[child_end] = new_dist;
        if (B0 >= n_leaves) {
            s_node[top] = B0; s_prev[top] = start_node;
            s_pow[top] = 0.25; s_dkc[top] = 0.5 * (d_kc_init + row_root[B1]);
            s_dist[top] = new_dist; top++;
        }
    }
    {
        double D_AC = row_orig[B0], D_kB = row_root[B1];
        double new_dist = 0.25 * (D_AC + D_kB - D_AB - d_kc_init);
        int32_t child_end = (parent_u[start_node] == B1) ? start_node : B1;
        edge_delta[child_end] = new_dist;
        if (B1 >= n_leaves) {
            s_node[top] = B1; s_prev[top] = start_node;
            s_pow[top] = 0.25; s_dkc[top] = 0.5 * (d_kc_init + row_root[B0]);
            s_dist[top] = new_dist; top++;
        }
    }

    while (top > 0) {
        --top;
        int32_t u = s_node[top], C = s_prev[top];
        double pow_neg_len = s_pow[top], d_kc = s_dkc[top], curr_dist = s_dist[top];

        const int32_t *adj = &adj_u[3 * u];
        const double *nd = &nd_u[3 * u];
        int32_t ki = (adj[0] == C) ? 0 : ((adj[1] == C) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;
        int32_t b0 = adj[idx0], b1 = adj[idx1];
        double d_ab = nd[ki], r_b0 = row_root[b0], r_b1 = row_root[b1];

        double d_ac0 = nd[idx0] + (row_orig[b1] - r_b1) * pow_neg_len;
        double ndist0 = curr_dist + 0.25 * (d_ac0 + r_b0 - d_ab - d_kc);
        int32_t c_end0 = (parent_u[u] == b0) ? u : b0;
        edge_delta[c_end0] = ndist0;

        double d_ac1 = nd[idx1] + (row_orig[b0] - r_b0) * pow_neg_len;
        double ndist1 = curr_dist + 0.25 * (d_ac1 + r_b1 - d_ab - d_kc);
        int32_t c_end1 = (parent_u[u] == b1) ? u : b1;
        edge_delta[c_end1] = ndist1;

        double next_pow = pow_neg_len * 0.5;
        if (b0 >= n_leaves && b1 >= n_leaves) {
            int32_t sz0 = (parent_u[b0] == u) ? sub_size[b0] : (n_nodes_u - sub_size[u]);
            int32_t sz1 = (parent_u[b1] == u) ? sub_size[b1] : (n_nodes_u - sub_size[u]);
            if (sz0 >= sz1) {
                s_node[top] = b0; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5 * (d_kc + r_b1); s_dist[top] = ndist0; top++;
                s_node[top] = b1; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5 * (d_kc + r_b0); s_dist[top] = ndist1; top++;
            } else {
                s_node[top] = b1; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5 * (d_kc + r_b0); s_dist[top] = ndist1; top++;
                s_node[top] = b0; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5 * (d_kc + r_b1); s_dist[top] = ndist0; top++;
            }
        } else if (b0 >= n_leaves) {
            s_node[top] = b0; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5 * (d_kc + r_b1); s_dist[top] = ndist0; top++;
        } else if (b1 >= n_leaves) {
            s_node[top] = b1; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5 * (d_kc + r_b0); s_dist[top] = ndist1; top++;
        }
    }
}

static inline void run_half_tree_dfs_f32(
    int32_t start_node, int32_t orig_node, int32_t parent_node, int32_t n_leaves, int32_t n_nodes_u,
    const int32_t * restrict adj_u, const int32_t * restrict parent_u, const int32_t * restrict sub_size,
    const float * restrict nd_u, const float * restrict row_root, const float * restrict row_orig,
    float * restrict edge_delta
) {
    if (start_node < n_leaves) return;
    int32_t s_node[32], s_prev[32];
    float s_pow[32], s_dkc[32], s_dist[32];
    int32_t top = 0;

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

    {
        float D_AC = row_orig[B1], D_kB = row_root[B0];
        float new_dist = 0.25f * (D_AC + D_kB - D_AB - d_kc_init);
        int32_t child_end = (parent_u[start_node] == B0) ? start_node : B0;
        edge_delta[child_end] = new_dist;
        if (B0 >= n_leaves) {
            s_node[top] = B0; s_prev[top] = start_node;
            s_pow[top] = 0.25f; s_dkc[top] = 0.5f * (d_kc_init + row_root[B1]);
            s_dist[top] = new_dist; top++;
        }
    }
    {
        float D_AC = row_orig[B0], D_kB = row_root[B1];
        float new_dist = 0.25f * (D_AC + D_kB - D_AB - d_kc_init);
        int32_t child_end = (parent_u[start_node] == B1) ? start_node : B1;
        edge_delta[child_end] = new_dist;
        if (B1 >= n_leaves) {
            s_node[top] = B1; s_prev[top] = start_node;
            s_pow[top] = 0.25f; s_dkc[top] = 0.5f * (d_kc_init + row_root[B0]);
            s_dist[top] = new_dist; top++;
        }
    }

    while (top > 0) {
        --top;
        int32_t u = s_node[top], C = s_prev[top];
        float pow_neg_len = s_pow[top], d_kc = s_dkc[top], curr_dist = s_dist[top];

        const int32_t *adj = &adj_u[3 * u];
        const float *nd = &nd_u[3 * u];
        int32_t ki = (adj[0] == C) ? 0 : ((adj[1] == C) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;
        int32_t b0 = adj[idx0], b1 = adj[idx1];
        float d_ab = nd[ki], r_b0 = row_root[b0], r_b1 = row_root[b1];

        float d_ac0 = nd[idx0] + (row_orig[b1] - r_b1) * pow_neg_len;
        float ndist0 = curr_dist + 0.25f * (d_ac0 + r_b0 - d_ab - d_kc);
        int32_t c_end0 = (parent_u[u] == b0) ? u : b0;
        edge_delta[c_end0] = ndist0;

        float d_ac1 = nd[idx1] + (row_orig[b0] - r_b0) * pow_neg_len;
        float ndist1 = curr_dist + 0.25f * (d_ac1 + r_b1 - d_ab - d_kc);
        int32_t c_end1 = (parent_u[u] == b1) ? u : b1;
        edge_delta[c_end1] = ndist1;

        float next_pow = pow_neg_len * 0.5f;
        if (b0 >= n_leaves && b1 >= n_leaves) {
            int32_t sz0 = (parent_u[b0] == u) ? sub_size[b0] : (n_nodes_u - sub_size[u]);
            int32_t sz1 = (parent_u[b1] == u) ? sub_size[b1] : (n_nodes_u - sub_size[u]);
            if (sz0 >= sz1) {
                s_node[top] = b0; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5f * (d_kc + r_b1); s_dist[top] = ndist0; top++;
                s_node[top] = b1; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5f * (d_kc + r_b0); s_dist[top] = ndist1; top++;
            } else {
                s_node[top] = b1; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5f * (d_kc + r_b0); s_dist[top] = ndist1; top++;
                s_node[top] = b0; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5f * (d_kc + r_b1); s_dist[top] = ndist0; top++;
            }
        } else if (b0 >= n_leaves) {
            s_node[top] = b0; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5f * (d_kc + r_b1); s_dist[top] = ndist0; top++;
        } else if (b1 >= n_leaves) {
            s_node[top] = b1; s_prev[top] = u; s_pow[top] = next_pow; s_dkc[top] = 0.5f * (d_kc + r_b0); s_dist[top] = ndist1; top++;
        }
    }
}

#define N_CPU_THREADS 10

/* Build tree topology & destinations (shared helper) */
static void build_tree_structures(
    const int64_t * restrict v,
    int32_t n_leaves,
    int32_t * restrict iwork,
    int32_t *out_other_root_child,
    int32_t *out_r0, int32_t *out_r1, int32_t *out_r2
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;

    int32_t *pairs = iwork;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *min_leaf = parents_uf + n_nodes_r;
    int32_t *v_map = min_leaf + n_nodes_r;
    int32_t *m_lmin = v_map + n_leaves;
    int32_t *parent_u = m_lmin + n_leaves;
    int32_t *sub_size = parent_u + n_nodes_u;
    int32_t *tin = sub_size + n_nodes_u;
    int32_t *dfs_order = tin + n_nodes_u;
    int32_t *adj_u = dfs_order + n_nodes_u;

    for (int32_t x = 0; x < n_leaves; ++x) {
        parents_uf[x] = x;
        min_leaf[x] = x;
    }
    for (int32_t k = 0; k < n_pairs; ++k) {
        int32_t i = n_pairs - 1 - k;
        int32_t m = n_leaves + k;
        int32_t x = (int32_t)v[i];
        int32_t y = i + 1;
        int32_t c1 = parents_uf[y];
        int32_t c2 = parents_uf[x];
        child1_r[m] = c1;
        child2_r[m] = c2;
        min_leaf[m] = x;
        m_lmin[k] = x;
        v_map[y] = c1;
        parents_uf[x] = m;
    }

    int32_t rc1 = child1_r[n_nodes_r - 1], rc2 = child2_r[n_nodes_r - 1];
    int32_t other_root_child = (rc1 < rc2) ? rc1 : rc2;
    int32_t root_u_c0 = child1_r[root_u];
    int32_t root_u_c1 = child2_r[root_u];
    int32_t root_u_c2 = other_root_child;

    for (int32_t u = 0; u < n_leaves; ++u) sub_size[u] = 1;
    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        parent_u[c1] = u; parent_u[c2] = u;
        sub_size[u] = 1 + sub_size[c1] + sub_size[c2];
        adj_u[3 * u + 0] = c1; adj_u[3 * u + 1] = c2;
    }
    parent_u[root_u_c0] = root_u;
    parent_u[root_u_c1] = root_u;
    parent_u[root_u_c2] = root_u;
    parent_u[root_u] = -1;

    for (int32_t u = 0; u < n_leaves; ++u) {
        adj_u[3 * u + 0] = parent_u[u];
        adj_u[3 * u + 1] = -1;
        adj_u[3 * u + 2] = -1;
    }
    for (int32_t u = n_leaves; u < root_u; ++u) adj_u[3 * u + 2] = parent_u[u];
    adj_u[3 * root_u + 0] = root_u_c0;
    adj_u[3 * root_u + 1] = root_u_c1;
    adj_u[3 * root_u + 2] = root_u_c2;

    tin[root_u] = 0;
    tin[root_u_c0] = 1;
    tin[root_u_c1] = 1 + sub_size[root_u_c0];
    tin[root_u_c2] = 1 + sub_size[root_u_c0] + sub_size[root_u_c1];
    for (int32_t u = root_u - 1; u >= n_leaves; --u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        tin[c1] = tin[u] + 1;
        tin[c2] = tin[u] + 1 + sub_size[c1];
    }
    for (int32_t u = 0; u < n_nodes_u; ++u) dfs_order[tin[u]] = u;

    *out_other_root_child = other_root_child;
    *out_r0 = root_u_c0;
    *out_r1 = root_u_c1;
    *out_r2 = root_u_c2;
}

/* ============================================================================
 * 10-Core CPU (FP64): Column-Parallel `subdist` + O(N log N) Top-Down + DFS SPR
 * ============================================================================ */
static void calculate_discrete_gradient_multicore_f64(
    const int64_t * restrict v,
    const double * restrict D,
    int32_t n_leaves,
    double * restrict delta_full_out,
    int32_t * restrict iwork,
    double * restrict dwork,
    int32_t * restrict thr_iwork,
    double * restrict thr_dwork
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;
    const int32_t max_cols = 2 * n_pairs - 1;
    memset(delta_full_out, 0, (size_t)n_pairs * max_cols * sizeof(double));

    int32_t other_root_child, root_u_c0, root_u_c1, root_u_c2;
    build_tree_structures(v, n_leaves, iwork, &other_root_child, &root_u_c0, &root_u_c1, &root_u_c2);

    int32_t *pairs = iwork;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *min_leaf = parents_uf + n_nodes_r;
    int32_t *v_map = min_leaf + n_nodes_r;
    int32_t *m_lmin = v_map + n_leaves;
    int32_t *parent_u = m_lmin + n_leaves;
    int32_t *sub_size = parent_u + n_nodes_u;
    int32_t *tin = sub_size + n_nodes_u;
    int32_t *dfs_order = tin + n_nodes_u;
    int32_t *adj_u = dfs_order + n_nodes_u;

    double *subdist = dwork;
    double *nd_u = subdist + (size_t)n_nodes_u * n_nodes_u;
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);

    // Pass 1: Column-parallel bottom-up on leaf columns [0 .. N-1]
    dispatch_apply(N_CPU_THREADS, q, [&](size_t tid) {
        int32_t c_start = (int32_t)((tid * n_leaves) / N_CPU_THREADS);
        int32_t c_end   = (int32_t)(((tid + 1) * n_leaves) / N_CPU_THREADS);
        int32_t c_len   = c_end - c_start;
        if (c_len <= 0) return;

        for (int32_t i = 0; i < n_leaves; ++i) {
            memcpy(&subdist[(size_t)i * n_nodes_u + c_start], &D[(size_t)i * n_leaves + c_start], (size_t)c_len * sizeof(double));
        }
        for (int32_t u = n_leaves; u < root_u; ++u) {
            const double * restrict r1 = &subdist[(size_t)child1_r[u] * n_nodes_u + c_start];
            const double * restrict r2 = &subdist[(size_t)child2_r[u] * n_nodes_u + c_start];
            double * restrict ru = &subdist[(size_t)u * n_nodes_u + c_start];
            for (int32_t x = 0; x < c_len; ++x) ru[x] = 0.5 * (r1[x] + r2[x]);
        }
    });

    // Transpose [N .. root_u-1, 0 .. N-1] -> [0 .. N-1, N .. root_u-1] in 64x64 cache blocks
    dispatch_apply(N_CPU_THREADS, q, [&](size_t tid) {
        int32_t x_start = (int32_t)((tid * n_leaves) / N_CPU_THREADS);
        int32_t x_end   = (int32_t)(((tid + 1) * n_leaves) / N_CPU_THREADS);
        for (int32_t u0 = n_leaves; u0 < root_u; u0 += 64) {
            int32_t u1 = (u0 + 64 < root_u) ? (u0 + 64) : root_u;
            for (int32_t x = x_start; x < x_end; ++x) {
                double * restrict row_x = &subdist[(size_t)x * n_nodes_u];
                for (int32_t u = u0; u < u1; ++u) row_x[u] = subdist[(size_t)u * n_nodes_u + x];
            }
        }
    });

    // Pass 2: Column-parallel bottom-up on internal columns [N .. root_u-1]
    int32_t n_int = root_u - n_leaves;
    dispatch_apply(N_CPU_THREADS, q, [&](size_t tid) {
        int32_t w_start = n_leaves + (int32_t)((tid * n_int) / N_CPU_THREADS);
        int32_t w_end   = n_leaves + (int32_t)(((tid + 1) * n_int) / N_CPU_THREADS);
        int32_t w_len   = w_end - w_start;
        if (w_len <= 0) return;

        for (int32_t u = n_leaves; u < root_u; ++u) {
            const double * restrict r1 = &subdist[(size_t)child1_r[u] * n_nodes_u + w_start];
            const double * restrict r2 = &subdist[(size_t)child2_r[u] * n_nodes_u + w_start];
            double * restrict ru = &subdist[(size_t)u * n_nodes_u + w_start];
            for (int32_t w = 0; w < w_len; ++w) ru[w] = 0.5 * (r1[w] + r2[w]);
        }
    });

    // Pass 3: O(N log N) Parallel Ancestor-Walk Top-Down + Symmetrization
    int32_t tin_c1_root = tin[root_u_c1];
    int32_t tin_c2_root = tin[root_u_c2];
    dispatch_apply(N_CPU_THREADS, q, [&](size_t tid) {
        int32_t v_start = (int32_t)((tid * root_u) / N_CPU_THREADS);
        int32_t v_end   = (int32_t)(((tid + 1) * root_u) / N_CPU_THREADS);
        for (int32_t v_node = v_start; v_node < v_end; ++v_node) {
            int32_t tv = tin[v_node];
            int32_t curr;
            double val_r;
            if (tv < tin_c1_root) {
                curr = root_u_c0;
                val_r = 0.5 * (subdist[(size_t)root_u_c1 * n_nodes_u + v_node] + subdist[(size_t)root_u_c2 * n_nodes_u + v_node]);
            } else if (tv < tin_c2_root) {
                curr = root_u_c1;
                val_r = 0.5 * (subdist[(size_t)root_u_c0 * n_nodes_u + v_node] + subdist[(size_t)root_u_c2 * n_nodes_u + v_node]);
            } else {
                curr = root_u_c2;
                val_r = 0.5 * (subdist[(size_t)root_u_c0 * n_nodes_u + v_node] + subdist[(size_t)root_u_c1 * n_nodes_u + v_node]);
            }
            subdist[(size_t)root_u * n_nodes_u + v_node] = val_r;
            subdist[(size_t)v_node * n_nodes_u + root_u] = val_r;

            int32_t g = root_u;
            while (curr != v_node) {
                int32_t c1 = child1_r[curr];
                int32_t c2 = child2_r[curr];
                int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
                int32_t sib       = (tv < tin[c2]) ? c2 : c1;
                double val = 0.5 * (subdist[(size_t)sib * n_nodes_u + v_node] + subdist[(size_t)g * n_nodes_u + v_node]);
                subdist[(size_t)curr * n_nodes_u + v_node] = val;
                subdist[(size_t)v_node * n_nodes_u + curr] = val;
                g = curr;
                curr = next_curr;
            }
            subdist[(size_t)v_node * n_nodes_u + v_node] = 0.0;
        }
    });
    subdist[(size_t)root_u * n_nodes_u + root_u] = 0.0;

    for (int32_t u = n_leaves; u <= root_u; ++u) {
        int32_t n0 = adj_u[3 * u + 0], n1 = adj_u[3 * u + 1], n2 = adj_u[3 * u + 2];
        nd_u[3 * u + 0] = subdist[(size_t)n1 * n_nodes_u + n2];
        nd_u[3 * u + 1] = subdist[(size_t)n0 * n_nodes_u + n2];
        nd_u[3 * u + 2] = subdist[(size_t)n0 * n_nodes_u + n1];
    }

    // Stage 3: Parallel Row Scan across 10 CPU cores using L1-Resident 32-Slot Heavy-First Stack + O(1) incremental dest_row
    int32_t n_rows = n_leaves - 2;
    dispatch_apply(N_CPU_THREADS, q, [&](size_t tid) {
        int32_t r_start = 2 + (int32_t)((tid * n_rows) / N_CPU_THREADS);
        int32_t r_end   = 2 + (int32_t)(((tid + 1) * n_rows) / N_CPU_THREADS);
        if (r_start >= r_end) return;

        double *edge_delta = &thr_dwork[tid * n_nodes_u];
        int32_t *dest_row = &thr_iwork[tid * n_leaves];

        int32_t i_init = r_end - 2;
        for (int32_t j = 0; j < n_leaves; ++j) dest_row[j] = j;
        for (int32_t k = 0; k < (n_leaves - 2 - i_init); ++k) {
            dest_row[m_lmin[k]] = n_leaves + k;
        }

        for (int32_t v_index = r_end - 1; v_index >= r_start; --v_index) {
            int32_t tree_root = v_map[v_index];
            int32_t parent = (tree_root == root_u) ? other_root_child : parent_u[tree_root];
            const double * restrict row_root = &subdist[(size_t)tree_root * n_nodes_u];

            int32_t other0 = -1, other1 = -1;
            for (int32_t k = 0; k < 3; ++k) {
                int32_t nb_node = adj_u[3 * parent + k];
                if (nb_node != -1 && nb_node != tree_root) {
                    if (other0 == -1) other0 = nb_node;
                    else other1 = nb_node;
                }
            }
            if (other0 != -1) {
                edge_delta[(parent_u[tree_root] == parent) ? tree_root : parent] = 0.0;
                edge_delta[(parent_u[other0]    == parent) ? other0    : parent] = 0.0;
                edge_delta[(parent_u[other1]    == parent) ? other1    : parent] = 0.0;

                run_half_tree_dfs_f64(
                    other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                    row_root, &subdist[(size_t)other1 * n_nodes_u], edge_delta
                );
                run_half_tree_dfs_f64(
                    other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                    row_root, &subdist[(size_t)other0 * n_nodes_u], edge_delta
                );

                int32_t i = v_index - 1;
                double * restrict out_row = &delta_full_out[(size_t)i * max_cols];
                for (int32_t j = 0; j <= i; ++j) out_row[j] = edge_delta[dest_row[j]];
                for (int32_t j = i + 1; j <= 2 * i - 2; ++j) {
                    out_row[j] = edge_delta[(2 * n_leaves - 2) - 2 * i + j];
                }
                if (2 * i - 1 > i) {
                    out_row[2 * i - 1] = edge_delta[other_root_child];
                }
                out_row[2 * i]     = edge_delta[other_root_child];
            }

            int32_t k_next = n_leaves - 1 - v_index;
            if (k_next >= 0 && k_next < n_pairs - 1) {
                dest_row[m_lmin[k_next]] = n_leaves + k_next;
            }
        }
    });
}

/* ============================================================================
 * Autonomous Row / Column Helper (Complete Column-Autonomy Theorem):
 * Computes the ENTIRE row `row_w[0 .. 2N-3]` of `subdist` for any node `w`
 * in O(N) time purely inside L1 cache using ONLY `D` (if w < N) or
 * `subdist_int_leaf` (if w >= N) — ZERO full `subdist` matrix needed!
 * ============================================================================ */
static inline void compute_autonomous_row_f64(
    int32_t w, int32_t n_leaves, int32_t n_nodes_u,
    int32_t root_u_c0, int32_t root_u_c1, int32_t root_u_c2,
    const int32_t * restrict child1_r, const int32_t * restrict child2_r,
    const int32_t * restrict tin,
    const double * restrict D, const double * restrict subdist_int_leaf,
    double * restrict row_w
) {
    const int32_t root_u = n_nodes_u - 1;
    if (w < n_leaves) {
        memcpy(row_w, &D[(size_t)w * n_leaves], (size_t)n_leaves * sizeof(double));
    } else {
        memcpy(row_w, &subdist_int_leaf[(size_t)(w - n_leaves) * n_leaves], (size_t)n_leaves * sizeof(double));
    }
    for (int32_t u = n_leaves; u < root_u; ++u) {
        row_w[u] = 0.5 * (row_w[child1_r[u]] + row_w[child2_r[u]]);
    }
    if (w == root_u) {
        row_w[root_u] = 0.0;
        return;
    }
    int32_t tv = tin[w];
    int32_t curr;
    if (tv < tin[root_u_c1]) {
        curr = root_u_c0;
        row_w[root_u] = 0.5 * (row_w[root_u_c1] + row_w[root_u_c2]);
    } else if (tv < tin[root_u_c2]) {
        curr = root_u_c1;
        row_w[root_u] = 0.5 * (row_w[root_u_c0] + row_w[root_u_c2]);
    } else {
        curr = root_u_c2;
        row_w[root_u] = 0.5 * (row_w[root_u_c0] + row_w[root_u_c1]);
    }
    int32_t g = root_u;
    while (curr != w) {
        int32_t c1 = child1_r[curr], c2 = child2_r[curr];
        int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
        int32_t sib       = (tv < tin[c2]) ? c2 : c1;
        row_w[curr] = 0.5 * (row_w[sib] + row_w[g]);
        g = curr;
        curr = next_curr;
    }
    row_w[w] = 0.0;
}

/* ============================================================================
 * Column-Tiled Low-Memory 10-Core CPU Implementation:
 *   - Tiles the leaf columns in L2-resident blocks of width B_TILE = 128.
 *   - NEVER allocates the (2N-2) x (2N-2) `subdist` matrix!
 *   - Generates `row_root`, `row_other0`, `row_other1` on the fly in L1 cache!
 * ============================================================================ */
#define B_TILE 128

static void calculate_discrete_gradient_tiled_f64(
    const int64_t * restrict v,
    const double * restrict D,
    int32_t n_leaves,
    double * restrict delta_full_out,
    int32_t * restrict iwork,
    double * restrict subdist_int_leaf, /* shape: (N - 1) x N — 4x smaller than subdist! */
    int32_t * restrict thr_iwork,
    double * restrict thr_tile_work     /* per-thread L1/L2 tile scratch */
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;
    const int32_t max_cols = 2 * n_pairs - 1;
    memset(delta_full_out, 0, (size_t)n_pairs * max_cols * sizeof(double));

    int32_t other_root_child, root_u_c0, root_u_c1, root_u_c2;
    build_tree_structures(v, n_leaves, iwork, &other_root_child, &root_u_c0, &root_u_c1, &root_u_c2);

    int32_t *pairs = iwork;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *min_leaf = parents_uf + n_nodes_r;
    int32_t *v_map = min_leaf + n_nodes_r;
    int32_t *m_lmin = v_map + n_leaves;
    int32_t *parent_u = m_lmin + n_leaves;
    int32_t *sub_size = parent_u + n_nodes_u;
    int32_t *tin = sub_size + n_nodes_u;
    int32_t *dfs_order = tin + n_nodes_u;
    int32_t *adj_u = dfs_order + n_nodes_u;

    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    int32_t n_tiles = (n_leaves + B_TILE - 1) / B_TILE;
    int32_t tin_c1_root = tin[root_u_c1];
    int32_t tin_c2_root = tin[root_u_c2];

    // Step 1: Process leaf columns in L2-resident tiles of width B_TILE = 128
    dispatch_apply(N_CPU_THREADS, q, [&](size_t tid) {
        int32_t t_start = (int32_t)((tid * n_tiles) / N_CPU_THREADS);
        int32_t t_end   = (int32_t)(((tid + 1) * n_tiles) / N_CPU_THREADS);
        double *tile_buf = &thr_tile_work[tid * (size_t)n_nodes_u * B_TILE];

        for (int32_t t = t_start; t < t_end; ++t) {
            int32_t c_start = t * B_TILE;
            int32_t c_end   = (c_start + B_TILE < n_leaves) ? (c_start + B_TILE) : n_leaves;
            int32_t c_len   = c_end - c_start;

            for (int32_t i = 0; i < n_leaves; ++i) {
                memcpy(&tile_buf[(size_t)i * B_TILE], &D[(size_t)i * n_leaves + c_start], (size_t)c_len * sizeof(double));
            }
            for (int32_t u = n_leaves; u < root_u; ++u) {
                const double * restrict r1 = &tile_buf[(size_t)child1_r[u] * B_TILE];
                const double * restrict r2 = &tile_buf[(size_t)child2_r[u] * B_TILE];
                double * restrict ru = &tile_buf[(size_t)u * B_TILE];
                for (int32_t x = 0; x < c_len; ++x) ru[x] = 0.5 * (r1[x] + r2[x]);
            }
            // Top-down ancestor walk on the tile's leaf columns
            for (int32_t x = 0; x < c_len; ++x) {
                int32_t v_node = c_start + x;
                int32_t tv = tin[v_node];
                int32_t curr;
                if (tv < tin_c1_root) {
                    curr = root_u_c0;
                    tile_buf[(size_t)root_u * B_TILE + x] = 0.5 * (tile_buf[(size_t)root_u_c1 * B_TILE + x] + tile_buf[(size_t)root_u_c2 * B_TILE + x]);
                } else if (tv < tin_c2_root) {
                    curr = root_u_c1;
                    tile_buf[(size_t)root_u * B_TILE + x] = 0.5 * (tile_buf[(size_t)root_u_c0 * B_TILE + x] + tile_buf[(size_t)root_u_c2 * B_TILE + x]);
                } else {
                    curr = root_u_c2;
                    tile_buf[(size_t)root_u * B_TILE + x] = 0.5 * (tile_buf[(size_t)root_u_c0 * B_TILE + x] + tile_buf[(size_t)root_u_c1 * B_TILE + x]);
                }
                int32_t g = root_u;
                while (curr != v_node) {
                    int32_t c1 = child1_r[curr], c2 = child2_r[curr];
                    int32_t next_curr = (tv < tin[c2]) ? c1 : c2;
                    int32_t sib       = (tv < tin[c2]) ? c2 : c1;
                    tile_buf[(size_t)curr * B_TILE + x] = 0.5 * (tile_buf[(size_t)sib * B_TILE + x] + tile_buf[(size_t)g * B_TILE + x]);
                    g = curr;
                    curr = next_curr;
                }
            }
            // Store internal rows [N .. root_u] into subdist_int_leaf
            for (int32_t u = n_leaves; u <= root_u; ++u) {
                memcpy(&subdist_int_leaf[(size_t)(u - n_leaves) * n_leaves + c_start],
                       &tile_buf[(size_t)u * B_TILE], (size_t)c_len * sizeof(double));
            }
        }
    });

    // Step 2: Populate `nd_u` in parallel using autonomous L1 rows
    double *nd_u = &subdist_int_leaf[(size_t)(n_nodes_u - n_leaves) * n_leaves];
    int32_t n_int_all = root_u - n_leaves + 1;
    dispatch_apply(N_CPU_THREADS, q, [&](size_t tid) {
        int32_t u_start = n_leaves + (int32_t)((tid * n_int_all) / N_CPU_THREADS);
        int32_t u_end   = n_leaves + (int32_t)(((tid + 1) * n_int_all) / N_CPU_THREADS);
        double *row_tmp = &thr_tile_work[tid * (size_t)n_nodes_u * B_TILE];
        for (int32_t u = u_start; u < u_end; ++u) {
            int32_t n0 = adj_u[3 * u + 0], n1 = adj_u[3 * u + 1], n2 = adj_u[3 * u + 2];
            compute_autonomous_row_f64(n0, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2,
                                       child1_r, child2_r, tin, D, subdist_int_leaf, row_tmp);
            nd_u[3 * u + 2] = row_tmp[n1];
            nd_u[3 * u + 1] = row_tmp[n2];
            compute_autonomous_row_f64(n1, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2,
                                       child1_r, child2_r, tin, D, subdist_int_leaf, row_tmp);
            nd_u[3 * u + 0] = row_tmp[n2];
        }
    });

    // Step 3: Stage 3 SPR Scan using 3 L1-resident autonomous rows (`row_root`, `row_other0`, `row_other1`)
    int32_t n_rows = n_leaves - 2;
    dispatch_apply(N_CPU_THREADS, q, [&](size_t tid) {
        int32_t r_start = 2 + (int32_t)((tid * n_rows) / N_CPU_THREADS);
        int32_t r_end   = 2 + (int32_t)(((tid + 1) * n_rows) / N_CPU_THREADS);
        if (r_start >= r_end) return;

        double *t_base     = &thr_tile_work[tid * (size_t)n_nodes_u * B_TILE];
        double *edge_delta = t_base;
        double *row_root   = t_base + n_nodes_u;
        double *row_other0 = t_base + 2 * n_nodes_u;
        double *row_other1 = t_base + 3 * n_nodes_u;
        int32_t *dest_row  = &thr_iwork[tid * n_leaves];

        int32_t i_init = r_end - 2;
        for (int32_t j = 0; j < n_leaves; ++j) dest_row[j] = j;
        for (int32_t k = 0; k < (n_leaves - 2 - i_init); ++k) {
            dest_row[m_lmin[k]] = n_leaves + k;
        }

        for (int32_t v_index = r_end - 1; v_index >= r_start; --v_index) {
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
            if (other0 != -1) {
                compute_autonomous_row_f64(tree_root, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2,
                                           child1_r, child2_r, tin, D, subdist_int_leaf, row_root);
                compute_autonomous_row_f64(other0, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2,
                                           child1_r, child2_r, tin, D, subdist_int_leaf, row_other0);
                compute_autonomous_row_f64(other1, n_leaves, n_nodes_u, root_u_c0, root_u_c1, root_u_c2,
                                           child1_r, child2_r, tin, D, subdist_int_leaf, row_other1);

                edge_delta[(parent_u[tree_root] == parent) ? tree_root : parent] = 0.0;
                edge_delta[(parent_u[other0]    == parent) ? other0    : parent] = 0.0;
                edge_delta[(parent_u[other1]    == parent) ? other1    : parent] = 0.0;

                run_half_tree_dfs_f64(
                    other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                    row_root, row_other1, edge_delta
                );
                run_half_tree_dfs_f64(
                    other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                    row_root, row_other0, edge_delta
                );

                int32_t i = v_index - 1;
                double * restrict out_row = &delta_full_out[(size_t)i * max_cols];
                for (int32_t j = 0; j <= i; ++j) out_row[j] = edge_delta[dest_row[j]];
                for (int32_t j = i + 1; j <= 2 * i - 2; ++j) {
                    out_row[j] = edge_delta[(2 * n_leaves - 2) - 2 * i + j];
                }
                if (2 * i - 1 > i) {
                    out_row[2 * i - 1] = edge_delta[other_root_child];
                }
                out_row[2 * i]     = edge_delta[other_root_child];
            }

            int32_t k_next = n_leaves - 1 - v_index;
            if (k_next >= 0 && k_next < n_pairs - 1) {
                dest_row[m_lmin[k_next]] = n_leaves + k_next;
            }
        }
    });
}

/* ============================================================================
 * Metal GPU Shaders:
 *   - Stage 2: Lockstep Column-Parallel `subdist` + 16x16 Threadgroup-Tiled
 *     Transpose + O(sum_v depth(v)) Stackless Ancestor-Walk Top-Down Pass
 * ============================================================================ */