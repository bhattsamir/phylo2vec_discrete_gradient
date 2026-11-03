#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __cplusplus
#define restrict __restrict__
extern "C" {
#endif

/*
 * Ultra-optimized Matrix-Based O(N^2) Phylo2Vec BME Discrete Gradient & Tree Optimizer in C99 (v3).
 *
 * Exports:
 *   - calculate_discrete_gradient_c      : Lower-triangular j < i discrete gradient (N-1 x N-2)
 *   - calculate_discrete_gradient_full_c : Full 0 <= j <= 2i discrete gradient (N-1 x (2N-3))
 *   - relabel_cherry_first_c             : O(N) cherry-first taxon relabeling (v_pi[i] <= i)
 *   - encode_unrooted_to_v_c             : Exact O(N) inverse bijection from unrooted tree to v
 *   - optimize_phylo2vec_bme_c           : End-to-end Phylo2Vec BME tree optimizer
 *   - optimize_fastme_bme_c              : Full unrooted FastME SPR hill-climber in C99
 *
 * Compile with:
 *   /usr/bin/clang -O3 -march=native -ffast-math -shared -fPIC phylo2vec_fastme.c -o libphylo2vec_fastme.dylib
 */

static int32_t *g_iwork = NULL;
static double *g_dwork = NULL;
static int32_t g_cap_leaves = 0;

static int ensure_workspace(int32_t n_leaves) {
    if (n_leaves < 4 || n_leaves > 100000) return -1;
    if (n_leaves <= g_cap_leaves && g_iwork != NULL && g_dwork != NULL) return 0;
    free(g_iwork);
    free(g_dwork);
    g_iwork = NULL;
    g_dwork = NULL;
    g_cap_leaves = 0;

    size_t nl = (size_t)n_leaves;
    size_t n_pairs = nl - 1;
    size_t n_nodes_r = 2 * nl - 1;
    size_t n_nodes_u = 2 * nl - 2;

    size_t i_count =
        2 * n_pairs +
        5 * n_nodes_r +
        nl +
        n_pairs * nl +
        n_nodes_u * 4 +
        n_nodes_u * 3 +
        n_nodes_u * 2 +
        2 * n_pairs + 4 * n_nodes_r + 3 * n_nodes_u +
        4 * n_nodes_r + 12 * nl;
    size_t d_count =
        n_nodes_u * n_nodes_u +
        n_nodes_u * 3 +
        n_nodes_u * 4 +
        nl * nl +
        n_nodes_r * nl + n_nodes_r +
        4 * nl;

    g_iwork = (int32_t *)malloc(i_count * sizeof(int32_t));
    g_dwork = (double *)malloc(d_count * sizeof(double));
    if (!g_iwork || !g_dwork) {
        free(g_iwork);
        free(g_dwork);
        g_iwork = NULL;
        g_dwork = NULL;
        return -1;
    }
    g_cap_leaves = n_leaves;
    return 0;
}

static inline double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

/* ============================================================================
 * Half-Tree Non-Recursive BFS & Heavy-First Register-Stack DFS
 * for O(N^2) Pauplin / FastME SPR Edge Deltas
 * ============================================================================ */
typedef struct {
    int32_t u;
    int32_t C;
    double pow_neg_len;
    double d_kc;
    double curr_dist;
} HeavyStackFrame;

static inline void run_half_tree_dfs(
    int32_t start_node,
    int32_t orig_node,
    int32_t parent_node,
    int32_t n_leaves,
    int32_t n_nodes_u,
    const int32_t * restrict adj_u,
    const int32_t * restrict parent_u,
    const int32_t * restrict sub_size,
    const double * restrict nd_u,
    const double * restrict row_root,
    const double * restrict row_orig,
    double * restrict edge_delta
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
    double D_AB = nd_u[3 * start_node + k_in];
    double d_kc_init = row_root[orig_node];

    HeavyStackFrame stack[32];
    int32_t sp = 0;

    double new_dist0 = 0.25 * (row_orig[B1] + row_root[B0] - D_AB - d_kc_init);
    int32_t c_end0 = (parent_u[start_node] == B0) ? start_node : B0;
    edge_delta[c_end0] = new_dist0;

    double new_dist1 = 0.25 * (row_orig[B0] + row_root[B1] - D_AB - d_kc_init);
    int32_t c_end1 = (parent_u[start_node] == B1) ? start_node : B1;
    edge_delta[c_end1] = new_dist1;

    int32_t push0 = (B0 >= n_leaves);
    int32_t push1 = (B1 >= n_leaves);
    if (push0 && push1) {
        int32_t s0 = (parent_u[B0] == start_node) ? sub_size[B0] : (n_nodes_u - sub_size[start_node]);
        int32_t s1 = (parent_u[B1] == start_node) ? sub_size[B1] : (n_nodes_u - sub_size[start_node]);
        if (s0 >= s1) {
            stack[sp++] = (HeavyStackFrame){B0, start_node, 0.25, 0.5 * (d_kc_init + row_root[B1]), new_dist0};
            stack[sp++] = (HeavyStackFrame){B1, start_node, 0.25, 0.5 * (d_kc_init + row_root[B0]), new_dist1};
        } else {
            stack[sp++] = (HeavyStackFrame){B1, start_node, 0.25, 0.5 * (d_kc_init + row_root[B0]), new_dist1};
            stack[sp++] = (HeavyStackFrame){B0, start_node, 0.25, 0.5 * (d_kc_init + row_root[B1]), new_dist0};
        }
    } else if (push0) {
        stack[sp++] = (HeavyStackFrame){B0, start_node, 0.25, 0.5 * (d_kc_init + row_root[B1]), new_dist0};
    } else if (push1) {
        stack[sp++] = (HeavyStackFrame){B1, start_node, 0.25, 0.5 * (d_kc_init + row_root[B0]), new_dist1};
    }

    while (sp > 0) {
        HeavyStackFrame f = stack[--sp];
        int32_t u = f.u;
        int32_t C = f.C;
        double pow_neg_len = f.pow_neg_len;
        double d_kc = f.d_kc;
        double curr_dist = f.curr_dist;

        const int32_t *adj = &adj_u[3 * u];
        const double *nd = &nd_u[3 * u];

        int32_t ki = (adj[0] == C) ? 0 : ((adj[1] == C) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;

        int32_t b0 = adj[idx0];
        int32_t b1 = adj[idx1];
        double d_ab = nd[ki];

        double r_b0 = row_root[b0];
        double r_b1 = row_root[b1];

        double d_ac0 = nd[idx0] + (row_orig[b1] - r_b1) * pow_neg_len;
        double ndist0 = curr_dist + 0.25 * (d_ac0 + r_b0 - d_ab - d_kc);
        int32_t ce0 = (parent_u[u] == b0) ? u : b0;
        edge_delta[ce0] = ndist0;

        double d_ac1 = nd[idx1] + (row_orig[b0] - r_b0) * pow_neg_len;
        double ndist1 = curr_dist + 0.25 * (d_ac1 + r_b1 - d_ab - d_kc);
        int32_t ce1 = (parent_u[u] == b1) ? u : b1;
        edge_delta[ce1] = ndist1;

        int32_t p0 = (b0 >= n_leaves);
        int32_t p1 = (b1 >= n_leaves);
        if (p0 && p1) {
            int32_t s0 = (parent_u[b0] == u) ? sub_size[b0] : (n_nodes_u - sub_size[u]);
            int32_t s1 = (parent_u[b1] == u) ? sub_size[b1] : (n_nodes_u - sub_size[u]);
            if (s0 >= s1) {
                stack[sp++] = (HeavyStackFrame){b0, u, pow_neg_len * 0.5, 0.5 * (d_kc + r_b1), ndist0};
                stack[sp++] = (HeavyStackFrame){b1, u, pow_neg_len * 0.5, 0.5 * (d_kc + r_b0), ndist1};
            } else {
                stack[sp++] = (HeavyStackFrame){b1, u, pow_neg_len * 0.5, 0.5 * (d_kc + r_b0), ndist1};
                stack[sp++] = (HeavyStackFrame){b0, u, pow_neg_len * 0.5, 0.5 * (d_kc + r_b1), ndist0};
            }
        } else if (p0) {
            stack[sp++] = (HeavyStackFrame){b0, u, pow_neg_len * 0.5, 0.5 * (d_kc + r_b1), ndist0};
        } else if (p1) {
            stack[sp++] = (HeavyStackFrame){b1, u, pow_neg_len * 0.5, 0.5 * (d_kc + r_b0), ndist1};
        }
    }
}

static inline void run_half_tree_bfs(
    int32_t start_node,
    int32_t orig_node,
    int32_t parent_node,
    int32_t n_leaves,
    const int32_t * restrict adj_u,
    const int32_t * restrict parent_u,
    const double * restrict nd_u,
    const double * restrict row_root,
    const double * restrict row_orig,
    double * restrict edge_delta,
    int32_t * restrict q_node,
    int32_t * restrict q_prev,
    double * restrict q_pow,
    double * restrict q_dkc,
    double * restrict q_dist
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
    double D_AB = nd_u[3 * start_node + k_in];
    double d_kc_init = row_root[orig_node];

    int32_t head = 0, tail = 0;

    // Branch 0 at start_node: B = B0, A = B1, C = orig_node
    {
        double D_AC = row_orig[B1];
        double D_kB = row_root[B0];
        double new_dist = 0.25 * (D_AC + D_kB - D_AB - d_kc_init);
        int32_t child_end = (parent_u[start_node] == B0) ? start_node : B0;
        edge_delta[child_end] = new_dist;
        if (B0 >= n_leaves) {
            q_node[tail] = B0;
            q_prev[tail] = start_node;
            q_pow[tail] = 0.25;
            q_dkc[tail] = 0.5 * (d_kc_init + row_root[B1]);
            q_dist[tail] = new_dist;
            tail++;
        }
    }
    // Branch 1 at start_node: B = B1, A = B0, C = orig_node
    {
        double D_AC = row_orig[B0];
        double D_kB = row_root[B1];
        double new_dist = 0.25 * (D_AC + D_kB - D_AB - d_kc_init);
        int32_t child_end = (parent_u[start_node] == B1) ? start_node : B1;
        edge_delta[child_end] = new_dist;
        if (B1 >= n_leaves) {
            q_node[tail] = B1;
            q_prev[tail] = start_node;
            q_pow[tail] = 0.25;
            q_dkc[tail] = 0.5 * (d_kc_init + row_root[B0]);
            q_dist[tail] = new_dist;
            tail++;
        }
    }

    while (head < tail) {
        int32_t u = q_node[head];
        int32_t C = q_prev[head];
        double pow_neg_len = q_pow[head];
        double d_kc = q_dkc[head];
        double curr_dist = q_dist[head];
        head++;

        const int32_t *adj = &adj_u[3 * u];
        const double *nd = &nd_u[3 * u];

        int32_t ki = (adj[0] == C) ? 0 : ((adj[1] == C) ? 1 : 2);
        int32_t idx0 = (ki == 0) ? 1 : 0;
        int32_t idx1 = (ki == 2) ? 1 : 2;

        int32_t b0 = adj[idx0];
        int32_t b1 = adj[idx1];
        double d_ab = nd[ki];

        double r_b0 = row_root[b0];
        double r_b1 = row_root[b1];

        {
            double d_ac = nd[idx0] + (row_orig[b1] - r_b1) * pow_neg_len;
            double ndist0 = curr_dist + 0.25 * (d_ac + r_b0 - d_ab - d_kc);
            int32_t c_end0 = (parent_u[u] == b0) ? u : b0;
            edge_delta[c_end0] = ndist0;
            if (b0 >= n_leaves) {
                q_node[tail] = b0;
                q_prev[tail] = u;
                q_pow[tail] = pow_neg_len * 0.5;
                q_dkc[tail] = 0.5 * (d_kc + r_b1);
                q_dist[tail] = ndist0;
                tail++;
            }
        }
        {
            double d_ac = nd[idx1] + (row_orig[b0] - r_b0) * pow_neg_len;
            double ndist1 = curr_dist + 0.25 * (d_ac + r_b1 - d_ab - d_kc);
            int32_t c_end1 = (parent_u[u] == b1) ? u : b1;
            edge_delta[c_end1] = ndist1;
            if (b1 >= n_leaves) {
                q_node[tail] = b1;
                q_prev[tail] = u;
                q_pow[tail] = pow_neg_len * 0.5;
                q_dkc[tail] = 0.5 * (d_kc + r_b0);
                q_dist[tail] = ndist1;
                tail++;
            }
        }
    }
}

static inline void decode_v_to_pairs_fast(
    const int64_t * restrict v,
    int32_t n_pairs,
    int32_t * restrict pairs
) {
    int32_t head = n_pairs - 1;
    pairs[2 * head] = 0;
    pairs[2 * head + 1] = 1;
    int32_t curr_len = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        int32_t next_leaf = i + 1;
        int32_t vi = (int32_t)v[i];
        if (vi <= i) {
            --head;
            pairs[2 * head] = vi;
            pairs[2 * head + 1] = next_leaf;
        } else {
            int32_t idx = vi - curr_len;
            int32_t c0 = pairs[2 * (head + idx - 1)];
            memmove(&pairs[2 * (head - 1)], &pairs[2 * head], (size_t)(2 * idx) * sizeof(int32_t));
            --head;
            pairs[2 * (head + idx)] = c0;
            pairs[2 * (head + idx) + 1] = next_leaf;
        }
        curr_len++;
    }
}

/* ============================================================================
 * 1. Lower-Triangular (j < i) Discrete Gradient (backward-compatible)
 * ============================================================================ */
void calculate_discrete_gradient_c(
    const int64_t * restrict v,
    const double * restrict D,
    int32_t n_leaves,
    double * restrict delta_out
) {
    if (ensure_workspace(n_leaves) != 0) return;

    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;

    int32_t *ptr = g_iwork;
    int32_t *pairs = ptr; ptr += 2 * n_pairs;
    int32_t *child1_r = ptr; ptr += n_nodes_r;
    int32_t *child2_r = ptr; ptr += n_nodes_r;
    int32_t *parents_uf = ptr; ptr += n_nodes_r;
    int32_t *min_leaf = ptr; ptr += n_nodes_r;
    int32_t *v_map = ptr; ptr += n_leaves;
    int32_t *destinations = ptr; ptr += (size_t)n_pairs * n_leaves;
    int32_t *parent_u = ptr; ptr += n_nodes_u;
    int32_t *sub_size = ptr; ptr += n_nodes_u;
    int32_t *tin = ptr; ptr += n_nodes_u;
    int32_t *dfs_order = ptr; ptr += n_nodes_u;
    int32_t *adj_u = ptr; ptr += (size_t)n_nodes_u * 3;

    double *dptr = g_dwork;
    double *subdist = dptr; dptr += (size_t)n_nodes_u * n_nodes_u;
    double *nd_u = dptr; dptr += (size_t)n_nodes_u * 3;
    double *edge_delta = dptr; dptr += n_nodes_u;

    int32_t is_cf = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        if (v[i] > i) { is_cf = 0; break; }
    }
    if (is_cf) {
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
            v_map[y] = c1;
            parents_uf[x] = m;
        }
    } else {
        decode_v_to_pairs_fast(v, n_pairs, pairs);

        for (int32_t i = 0; i < n_nodes_r; ++i) {
            parents_uf[i] = i;
            min_leaf[i] = i;
        }
        int32_t next_parent = n_leaves;
        for (int32_t i = 0; i < n_pairs; ++i) {
            int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
            while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
            while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
            child1_r[next_parent] = p2;
            child2_r[next_parent] = p1;
            parents_uf[p1] = next_parent;
            parents_uf[p2] = next_parent;
            next_parent++;
        }

        for (int32_t i = 1; i < n_pairs; ++i) {
            int32_t *row_d = &destinations[i * n_leaves];
            for (int32_t j = 0; j < i; ++j) row_d[j] = j;
        }
        for (int32_t m = n_leaves; m < n_nodes_r; ++m) {
            int32_t c1 = child1_r[m], c2 = child2_r[m];
            int32_t ml1 = min_leaf[c1], ml2 = min_leaf[c2];
            int32_t l_min, l_max;
            if (ml1 < ml2) {
                l_min = ml1; l_max = ml2; v_map[l_max] = c2;
            } else {
                l_min = ml2; l_max = ml1; v_map[l_max] = c1;
            }
            min_leaf[m] = l_min;
            if (m < n_nodes_r - 1) {
                for (int32_t n_idx = l_min + 1; n_idx < l_max; ++n_idx) {
                    destinations[n_idx * n_leaves + l_min] = m;
                }
            }
        }
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

    for (int32_t i = 0; i < n_leaves; ++i) {
        memcpy(&subdist[i * n_nodes_u], &D[i * n_leaves], (size_t)n_leaves * sizeof(double));
    }
    for (int32_t u = n_leaves; u < root_u; ++u) {
        const double * restrict row_c1 = &subdist[child1_r[u] * n_nodes_u];
        const double * restrict row_c2 = &subdist[child2_r[u] * n_nodes_u];
        double * restrict row_u = &subdist[u * n_nodes_u];
        for (int32_t v_node = 0; v_node < u; ++v_node) {
            row_u[v_node] = 0.5 * (row_c1[v_node] + row_c2[v_node]);
        }
        for (int32_t v_node = 0; v_node < u; ++v_node) {
            subdist[v_node * n_nodes_u + u] = row_u[v_node];
        }
    }
    {
        const double * restrict row_c0 = &subdist[root_u_c0 * n_nodes_u];
        const double * restrict row_c1 = &subdist[root_u_c1 * n_nodes_u];
        const double * restrict row_c2 = &subdist[root_u_c2 * n_nodes_u];
        double * restrict row_r = &subdist[root_u * n_nodes_u];
        int32_t end0 = tin[root_u_c0] + sub_size[root_u_c0];
        for (int32_t idx = tin[root_u_c0]; idx < end0; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_r[v_node] = 0.5 * (row_c1[v_node] + row_c2[v_node]);
        }
        int32_t end1 = tin[root_u_c1] + sub_size[root_u_c1];
        for (int32_t idx = tin[root_u_c1]; idx < end1; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_r[v_node] = 0.5 * (row_c0[v_node] + row_c2[v_node]);
        }
        int32_t end2 = tin[root_u_c2] + sub_size[root_u_c2];
        for (int32_t idx = tin[root_u_c2]; idx < end2; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_r[v_node] = 0.5 * (row_c0[v_node] + row_c1[v_node]);
        }
        for (int32_t v_node = 0; v_node < root_u; ++v_node) {
            subdist[v_node * n_nodes_u + root_u] = row_r[v_node];
        }
    }
    for (int32_t p = root_u - 1; p >= n_leaves; --p) {
        int32_t g = parent_u[p];
        int32_t c1 = child1_r[p], c2 = child2_r[p];
        const double * restrict row_c1 = &subdist[c1 * n_nodes_u];
        const double * restrict row_c2 = &subdist[c2 * n_nodes_u];
        const double * restrict row_g = &subdist[g * n_nodes_u];
        double * restrict row_p = &subdist[p * n_nodes_u];

        int32_t end1 = tin[c1] + sub_size[c1];
        for (int32_t idx = tin[c1]; idx < end1; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_p[v_node] = 0.5 * (row_c2[v_node] + row_g[v_node]);
        }
        for (int32_t idx = tin[c1]; idx < end1; ++idx) {
            int32_t v_node = dfs_order[idx];
            subdist[v_node * n_nodes_u + p] = row_p[v_node];
        }
        int32_t end2 = tin[c2] + sub_size[c2];
        for (int32_t idx = tin[c2]; idx < end2; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_p[v_node] = 0.5 * (row_c1[v_node] + row_g[v_node]);
        }
        for (int32_t idx = tin[c2]; idx < end2; ++idx) {
            int32_t v_node = dfs_order[idx];
            subdist[v_node * n_nodes_u + p] = row_p[v_node];
        }
    }
    for (int32_t i = 0; i < n_nodes_u; ++i) subdist[i * n_nodes_u + i] = 0.0;

    for (int32_t u = n_leaves; u <= root_u; ++u) {
        int32_t n0 = adj_u[3 * u + 0];
        int32_t n1 = adj_u[3 * u + 1];
        int32_t n2 = adj_u[3 * u + 2];
        nd_u[3 * u + 0] = subdist[n1 * n_nodes_u + n2];
        nd_u[3 * u + 1] = subdist[n0 * n_nodes_u + n2];
        nd_u[3 * u + 2] = subdist[n0 * n_nodes_u + n1];
    }

    const int32_t max_cols = n_pairs - 1;
    memset(delta_out, 0, (size_t)(n_pairs * max_cols) * sizeof(double));

    if (is_cf) {
        int32_t * restrict dest_row = destinations;
        for (int32_t j = 0; j < n_leaves; ++j) dest_row[j] = j;
        for (int32_t v_index = n_leaves - 1; v_index >= 2; --v_index) {
            int32_t tree_root = v_map[v_index];
            int32_t parent = (tree_root == root_u) ? other_root_child : parent_u[tree_root];
            const double * restrict row_root = &subdist[tree_root * n_nodes_u];

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

                run_half_tree_dfs(
                    other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                    row_root, &subdist[other1 * n_nodes_u], edge_delta
                );
                run_half_tree_dfs(
                    other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                    row_root, &subdist[other0 * n_nodes_u], edge_delta
                );

                int32_t row_idx = v_index - 1;
                double * restrict out_row = &delta_out[(size_t)row_idx * max_cols];
                for (int32_t j = 0; j < row_idx; ++j) {
                    out_row[j] = edge_delta[dest_row[j]];
                }
            }
            int32_t m_active = n_leaves + (n_leaves - 1 - v_index);
            dest_row[min_leaf[m_active]] = m_active;
        }
    } else {
        for (int32_t v_index = 2; v_index < n_leaves; ++v_index) {
            int32_t tree_root = v_map[v_index];
            int32_t parent = (tree_root == root_u) ? other_root_child : parent_u[tree_root];
            const double * restrict row_root = &subdist[tree_root * n_nodes_u];

            int32_t other0 = -1, other1 = -1;
            for (int32_t k = 0; k < 3; ++k) {
                int32_t nb_node = adj_u[3 * parent + k];
                if (nb_node != -1 && nb_node != tree_root) {
                    if (other0 == -1) other0 = nb_node;
                    else other1 = nb_node;
                }
            }
            if (other0 == -1) continue;

            for (int32_t u = 0; u < root_u; ++u) edge_delta[u] = 0.0;

            run_half_tree_dfs(
                other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                row_root, &subdist[other1 * n_nodes_u], edge_delta
            );
            run_half_tree_dfs(
                other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                row_root, &subdist[other0 * n_nodes_u], edge_delta
            );

            int32_t row_idx = v_index - 1;
            double * restrict out_row = &delta_out[(size_t)row_idx * max_cols];
            const int32_t * restrict dest_row = &destinations[row_idx * n_leaves];
            for (int32_t j = 0; j < row_idx; ++j) {
                out_row[j] = edge_delta[dest_row[j]];
            }
        }
    }
}

/* ============================================================================
 * 2. Full 0 <= j <= 2i Discrete Gradient (shape: (N-1) x (2N-3))
 * ============================================================================ */
void calculate_discrete_gradient_full_c(
    const int64_t * restrict v,
    const double * restrict D,
    int32_t n_leaves,
    double * restrict delta_full_out
) {
    if (ensure_workspace(n_leaves) != 0) return;

    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;
    const int32_t max_cols = 2 * n_pairs - 1;

    int32_t *ptr = g_iwork;
    int32_t *pairs = ptr; ptr += 2 * n_pairs;
    int32_t *child1_r = ptr; ptr += n_nodes_r;
    int32_t *child2_r = ptr; ptr += n_nodes_r;
    int32_t *parents_uf = ptr; ptr += n_nodes_r;
    int32_t *min_leaf = ptr; ptr += n_nodes_r;
    int32_t *v_map = ptr; ptr += n_leaves;
    int32_t *dest_row = ptr; ptr += n_leaves;
    int32_t *parent_u = ptr; ptr += n_nodes_u;
    int32_t *sub_size = ptr; ptr += n_nodes_u;
    int32_t *tin = ptr; ptr += n_nodes_u;
    int32_t *dfs_order = ptr; ptr += n_nodes_u;
    int32_t *adj_u = ptr; ptr += (size_t)n_nodes_u * 3;

    double *dptr = g_dwork;
    double *subdist = dptr; dptr += (size_t)n_nodes_u * n_nodes_u;
    double *nd_u = dptr; dptr += (size_t)n_nodes_u * 3;
    double *edge_delta = dptr; dptr += n_nodes_u;

    int32_t is_cf = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        if (v[i] > i) { is_cf = 0; break; }
    }
    if (is_cf) {
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
            v_map[y] = c1;
            parents_uf[x] = m;
        }
    } else {
        decode_v_to_pairs_fast(v, n_pairs, pairs);
        for (int32_t i = 0; i < n_nodes_r; ++i) {
            parents_uf[i] = i;
            min_leaf[i] = i;
        }
        int32_t next_parent = n_leaves;
        for (int32_t i = 0; i < n_pairs; ++i) {
            int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
            while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
            while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
            child1_r[next_parent] = p2;
            child2_r[next_parent] = p1;
            int32_t ml1 = min_leaf[p2], ml2 = min_leaf[p1];
            if (ml1 < ml2) {
                min_leaf[next_parent] = ml1; v_map[ml2] = p1;
            } else {
                min_leaf[next_parent] = ml2; v_map[ml1] = p2;
            }
            parents_uf[p1] = next_parent;
            parents_uf[p2] = next_parent;
            next_parent++;
        }
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

    for (int32_t i = 0; i < n_leaves; ++i) {
        memcpy(&subdist[i * n_nodes_u], &D[i * n_leaves], (size_t)n_leaves * sizeof(double));
    }
    for (int32_t u = n_leaves; u < root_u; ++u) {
        const double * restrict row_c1 = &subdist[child1_r[u] * n_nodes_u];
        const double * restrict row_c2 = &subdist[child2_r[u] * n_nodes_u];
        double * restrict row_u = &subdist[u * n_nodes_u];
        for (int32_t v_node = 0; v_node < u; ++v_node) {
            row_u[v_node] = 0.5 * (row_c1[v_node] + row_c2[v_node]);
        }
        for (int32_t v_node = 0; v_node < u; ++v_node) {
            subdist[v_node * n_nodes_u + u] = row_u[v_node];
        }
    }
    {
        const double * restrict row_c0 = &subdist[root_u_c0 * n_nodes_u];
        const double * restrict row_c1 = &subdist[root_u_c1 * n_nodes_u];
        const double * restrict row_c2 = &subdist[root_u_c2 * n_nodes_u];
        double * restrict row_r = &subdist[root_u * n_nodes_u];
        int32_t end0 = tin[root_u_c0] + sub_size[root_u_c0];
        for (int32_t idx = tin[root_u_c0]; idx < end0; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_r[v_node] = 0.5 * (row_c1[v_node] + row_c2[v_node]);
        }
        int32_t end1 = tin[root_u_c1] + sub_size[root_u_c1];
        for (int32_t idx = tin[root_u_c1]; idx < end1; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_r[v_node] = 0.5 * (row_c0[v_node] + row_c2[v_node]);
        }
        int32_t end2 = tin[root_u_c2] + sub_size[root_u_c2];
        for (int32_t idx = tin[root_u_c2]; idx < end2; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_r[v_node] = 0.5 * (row_c0[v_node] + row_c1[v_node]);
        }
        for (int32_t v_node = 0; v_node < root_u; ++v_node) {
            subdist[v_node * n_nodes_u + root_u] = row_r[v_node];
        }
    }
    for (int32_t p = root_u - 1; p >= n_leaves; --p) {
        int32_t g = parent_u[p];
        int32_t c1 = child1_r[p], c2 = child2_r[p];
        const double * restrict row_c1 = &subdist[c1 * n_nodes_u];
        const double * restrict row_c2 = &subdist[c2 * n_nodes_u];
        const double * restrict row_g = &subdist[g * n_nodes_u];
        double * restrict row_p = &subdist[p * n_nodes_u];

        int32_t end1 = tin[c1] + sub_size[c1];
        for (int32_t idx = tin[c1]; idx < end1; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_p[v_node] = 0.5 * (row_c2[v_node] + row_g[v_node]);
        }
        for (int32_t idx = tin[c1]; idx < end1; ++idx) {
            int32_t v_node = dfs_order[idx];
            subdist[v_node * n_nodes_u + p] = row_p[v_node];
        }
        int32_t end2 = tin[c2] + sub_size[c2];
        for (int32_t idx = tin[c2]; idx < end2; ++idx) {
            int32_t v_node = dfs_order[idx];
            row_p[v_node] = 0.5 * (row_c1[v_node] + row_g[v_node]);
        }
        for (int32_t idx = tin[c2]; idx < end2; ++idx) {
            int32_t v_node = dfs_order[idx];
            subdist[v_node * n_nodes_u + p] = row_p[v_node];
        }
    }
    for (int32_t i = 0; i < n_nodes_u; ++i) subdist[i * n_nodes_u + i] = 0.0;

    for (int32_t u = n_leaves; u <= root_u; ++u) {
        int32_t n0 = adj_u[3 * u + 0];
        int32_t n1 = adj_u[3 * u + 1];
        int32_t n2 = adj_u[3 * u + 2];
        nd_u[3 * u + 0] = subdist[n1 * n_nodes_u + n2];
        nd_u[3 * u + 1] = subdist[n0 * n_nodes_u + n2];
        nd_u[3 * u + 2] = subdist[n0 * n_nodes_u + n1];
    }

    memset(delta_full_out, 0, (size_t)(n_pairs * max_cols) * sizeof(double));
    for (int32_t j = 0; j < n_leaves; ++j) dest_row[j] = j;

    for (int32_t v_index = n_leaves - 1; v_index >= 2; --v_index) {
        int32_t tree_root = v_map[v_index];
        int32_t parent = (tree_root == root_u) ? other_root_child : parent_u[tree_root];
        const double * restrict row_root = &subdist[tree_root * n_nodes_u];

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

            run_half_tree_dfs(
                other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                row_root, &subdist[other1 * n_nodes_u], edge_delta
            );
            run_half_tree_dfs(
                other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                row_root, &subdist[other0 * n_nodes_u], edge_delta
            );

            int32_t i = v_index - 1;
            double * restrict out_row = &delta_full_out[(size_t)i * max_cols];

            // 1. Leaf-clade attachments: 0 <= j <= i
            for (int32_t j = 0; j <= i; ++j) {
                out_row[j] = edge_delta[dest_row[j]];
            }
            // 2. Internal-branch attachments: i < j <= 2i - 2
            for (int32_t j = i + 1; j <= 2 * i - 2; ++j) {
                int32_t dest_u = (2 * n_leaves - 2) - 2 * i + j;
                out_row[j] = edge_delta[dest_u];
            }
            // 3. Root-branch attachments: j in {2i - 1, 2i} (for i = 1, 2i - 1 == 1 is leaf 1)
            if (2 * i - 1 > i) {
                out_row[2 * i - 1] = edge_delta[other_root_child];
            }
            out_row[2 * i] = edge_delta[other_root_child];
        }

        int32_t m_active = n_leaves + (n_leaves - 1 - v_index);
        dest_row[min_leaf[m_active]] = m_active;
    }
}

/* ============================================================================
 * 3. O(N) Cherry-First Taxon Relabeling (v_new[i] <= i)
 * ============================================================================ */
static void relabel_from_rooted_tree(
    const int32_t * restrict child1_r,
    const int32_t * restrict child2_r,
    const int32_t * restrict min_leaf,
    int32_t n_leaves,
    int32_t flip_mode,
    int64_t * restrict v_new,
    int32_t * restrict perm_inout,
    int32_t * restrict scratch
) {
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    int32_t *rep_leaf = scratch;
    int32_t *new_perm = scratch + n_nodes_r;

    int32_t root_r = n_nodes_r - 1;
    int32_t rc1 = child1_r[root_r], rc2 = child2_r[root_r];
    int32_t r_stay = (min_leaf[rc1] < min_leaf[rc2]) ? rc1 : rc2;
    int32_t r_move = (min_leaf[rc1] < min_leaf[rc2]) ? rc2 : rc1;
    if (flip_mode == 1) { int32_t tmp = r_stay; r_stay = r_move; r_move = tmp; }

    rep_leaf[r_stay] = 0;
    rep_leaf[r_move] = 1;
    v_new[0] = 0;

    int32_t next_label = 2;
    for (int32_t m = n_nodes_r - 2; m >= n_leaves; --m) {
        int32_t c1 = child1_r[m], c2 = child2_r[m];
        int32_t c_stay = (min_leaf[c1] < min_leaf[c2]) ? c1 : c2;
        int32_t c_move = (min_leaf[c1] < min_leaf[c2]) ? c2 : c1;
        if (flip_mode == 1) { int32_t tmp = c_stay; c_stay = c_move; c_move = tmp; }

        int32_t x = rep_leaf[m];
        int32_t y = next_label++;
        rep_leaf[c_stay] = x;
        rep_leaf[c_move] = y;
        v_new[y - 1] = x;
    }

    for (int32_t old_l = 0; old_l < n_leaves; ++old_l) {
        new_perm[rep_leaf[old_l]] = perm_inout[old_l];
    }
    memcpy(perm_inout, new_perm, (size_t)n_leaves * sizeof(int32_t));
}

void relabel_cherry_first_c(
    const int64_t * restrict v_in,
    int32_t n_leaves,
    int32_t flip_mode,
    int64_t * restrict v_out,
    int32_t * restrict perm_out
) {
    if (ensure_workspace(n_leaves) != 0) return;
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;

    int32_t *pairs = g_iwork;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *min_leaf = parents_uf + n_nodes_r;
    int32_t *scratch = min_leaf + n_nodes_r;

    decode_v_to_pairs_fast(v_in, n_pairs, pairs);

    for (int32_t i = 0; i < n_nodes_r; ++i) {
        parents_uf[i] = i;
        min_leaf[i] = i;
    }
    int32_t next_parent = n_leaves;
    for (int32_t i = 0; i < n_pairs; ++i) {
        int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
        while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
        while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
        child1_r[next_parent] = p2;
        child2_r[next_parent] = p1;
        int32_t ml1 = min_leaf[p2], ml2 = min_leaf[p1];
        min_leaf[next_parent] = (ml1 < ml2) ? ml1 : ml2;
        parents_uf[p1] = next_parent;
        parents_uf[p2] = next_parent;
        next_parent++;
    }

    for (int32_t i = 0; i < n_leaves; ++i) perm_out[i] = i;
    relabel_from_rooted_tree(child1_r, child2_r, min_leaf, n_leaves, flip_mode, v_out, perm_out, scratch);
}

/* ============================================================================
 * 4. Exact Tree-to-Vector Inverse Bijection (encode_rooted_to_v_c & encode_unrooted_to_v_c)
 * ============================================================================ */
void encode_rooted_to_v_c(
    const int32_t * restrict child1_r,
    const int32_t * restrict child2_r,
    int32_t n_leaves,
    int64_t * restrict v_out
) {
    if (n_leaves < 2) return;
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;

    int32_t *buf = (int32_t *)malloc((size_t)(3 * n_nodes_r + 3 * n_leaves) * sizeof(int32_t));
    if (!buf) return;
    int32_t *ord = buf;
    int32_t *min_leaf = ord + n_leaves;
    int32_t *l_max_node = min_leaf + n_nodes_r;
    int32_t *c_stay_node = l_max_node + n_nodes_r;
    int32_t *node_of_lmax = c_stay_node + n_nodes_r;
    int32_t *pair_lmax_order = node_of_lmax + n_leaves;

    int32_t ord_len = 0;
    ord[ord_len++] = root_r;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        int32_t b0 = child1_r[u], b1 = child2_r[u];
        if (b0 >= n_leaves) ord[ord_len++] = b0;
        if (b1 >= n_leaves) ord[ord_len++] = b1;
    }

    for (int32_t x = 0; x < n_leaves; ++x) {
        min_leaf[x] = x;
        l_max_node[x] = 0;
        node_of_lmax[x] = 0;
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

    v_out[0] = 0;
    int32_t head = n_pairs - 1;
    pair_lmax_order[head] = 1;
    int32_t len_p = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        int32_t next_leaf = i + 1;
        int32_t u = node_of_lmax[next_leaf];
        int32_t w = c_stay_node[u];
        while (w >= n_leaves && l_max_node[w] > next_leaf) {
            w = c_stay_node[w];
        }
        if (w < n_leaves) {
            v_out[i] = w;
            pair_lmax_order[--head] = next_leaf;
        } else {
            int32_t target_lmax = l_max_node[w];
            int32_t idx = 1;
            for (int32_t k = 0; k < len_p; ++k) {
                if (pair_lmax_order[head + k] == target_lmax) {
                    idx = k + 1;
                    break;
                }
            }
            v_out[i] = i + idx;
            memmove(&pair_lmax_order[head - 1], &pair_lmax_order[head], (size_t)idx * sizeof(int32_t));
            --head;
            pair_lmax_order[head + idx] = next_leaf;
        }
        len_p++;
    }
    free(buf);
}

void encode_unrooted_to_v_c(
    const int32_t * restrict adj_u,
    int32_t n_leaves,
    int64_t * restrict v_out
) {
    if (n_leaves < 2) return;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;

    int32_t *buf = (int32_t *)malloc((size_t)(4 * n_nodes_r) * sizeof(int32_t));
    if (!buf) return;
    int32_t *q = buf;
    int32_t *prev = q + n_nodes_r;
    int32_t *c1_r = prev + n_nodes_r;
    int32_t *c2_r = c1_r + n_nodes_r;

    int32_t p0 = adj_u[3 * 0 + 0];
    c1_r[root_r] = 0;
    c2_r[root_r] = p0;
    int32_t head = 0, tail = 0;
    if (p0 >= n_leaves) {
        q[tail] = p0; prev[tail] = 0; tail++;
    }
    while (head < tail) {
        int32_t u = q[head], pr = prev[head]; head++;
        int32_t b0 = -1, b1 = -1;
        for (int32_t k = 0; k < 3; ++k) {
            int32_t nb = adj_u[3 * u + k];
            if (nb != -1 && nb != pr) {
                if (b0 == -1) b0 = nb;
                else b1 = nb;
            }
        }
        c1_r[u] = b0; c2_r[u] = b1;
        if (b0 >= n_leaves) { q[tail] = b0; prev[tail] = u; tail++; }
        if (b1 >= n_leaves) { q[tail] = b1; prev[tail] = u; tail++; }
    }

    encode_rooted_to_v_c(c1_r, c2_r, n_leaves, v_out);
    free(buf);
}

/* ============================================================================
 * Fast O(N^2) BME Length Evaluation (Zero-Copy Leaf-Only Pass + 1D Top-Down)
 * ============================================================================ */
static double evaluate_bme_length_fast(
    const int64_t * restrict v,
    const double * restrict D,
    int32_t n_leaves,
    int32_t * restrict iwork,
    double * restrict dwork
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;

    int32_t *pairs = iwork;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;

    int32_t has_gt_i = 0;
    for (int32_t i = 1; i < n_pairs; ++i) {
        if (v[i] > i) { has_gt_i = 1; break; }
    }
    if (!has_gt_i) {
        for (int32_t x = 0; x < n_leaves; ++x) parents_uf[x] = x;
        for (int32_t k = 0; k < n_pairs; ++k) {
            int32_t i = n_pairs - 1 - k;
            int32_t m = n_leaves + k;
            int32_t x = (int32_t)v[i];
            int32_t y = i + 1;
            child1_r[m] = parents_uf[y];
            child2_r[m] = parents_uf[x];
            parents_uf[x] = m;
        }
    } else {
        decode_v_to_pairs_fast(v, n_pairs, pairs);
        for (int32_t i = 0; i < n_nodes_r; ++i) parents_uf[i] = i;
        int32_t next_parent = n_leaves;
        for (int32_t i = 0; i < n_pairs; ++i) {
            int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
            while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
            while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
            child1_r[next_parent] = p2;
            child2_r[next_parent] = p1;
            parents_uf[p1] = next_parent;
            parents_uf[p2] = next_parent;
            next_parent++;
        }
    }

    int32_t rc1 = child1_r[n_nodes_r - 1], rc2 = child2_r[n_nodes_r - 1];
    int32_t other_root_child = (rc1 < rc2) ? rc1 : rc2;
    int32_t root_u_c0 = child1_r[root_u];
    int32_t root_u_c1 = child2_r[root_u];
    int32_t root_u_c2 = other_root_child;

    double *up = dwork;
    double *ldist = up + n_nodes_r;

    int32_t *leaf_tin = parents_uf + n_nodes_r;
    int32_t *leaf_count = leaf_tin + n_nodes_u;
    int32_t *leaf_order = leaf_count + n_nodes_u;
    for (int32_t u = 0; u < n_leaves; ++u) leaf_count[u] = 1;

    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        leaf_count[u] = leaf_count[c1] + leaf_count[c2];
        const double * restrict r1 = (c1 < n_leaves) ? &D[(size_t)c1 * n_leaves] : &ldist[(size_t)(c1 - n_leaves) * n_leaves];
        const double * restrict r2 = (c2 < n_leaves) ? &D[(size_t)c2 * n_leaves] : &ldist[(size_t)(c2 - n_leaves) * n_leaves];
        double * restrict ru = &ldist[(size_t)(u - n_leaves) * n_leaves];
        for (int32_t x = 0; x < n_leaves; ++x) {
            ru[x] = 0.5 * (r1[x] + r2[x]);
        }
    }

    leaf_tin[root_u_c0] = 0;
    leaf_tin[root_u_c1] = leaf_count[root_u_c0];
    leaf_tin[root_u_c2] = leaf_count[root_u_c0] + leaf_count[root_u_c1];
    for (int32_t u = root_u - 1; u >= n_leaves; --u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        leaf_tin[c1] = leaf_tin[u];
        leaf_tin[c2] = leaf_tin[u] + leaf_count[c1];
    }
    for (int32_t x = 0; x < n_leaves; ++x) leaf_order[leaf_tin[x]] = x;

    const double * restrict rc0_row = (root_u_c0 < n_leaves) ? &D[(size_t)root_u_c0 * n_leaves] : &ldist[(size_t)(root_u_c0 - n_leaves) * n_leaves];
    const double * restrict rc1_row = (root_u_c1 < n_leaves) ? &D[(size_t)root_u_c1 * n_leaves] : &ldist[(size_t)(root_u_c1 - n_leaves) * n_leaves];
    const double * restrict rc2_row = (root_u_c2 < n_leaves) ? &D[(size_t)root_u_c2 * n_leaves] : &ldist[(size_t)(root_u_c2 - n_leaves) * n_leaves];

    int32_t end0 = leaf_tin[root_u_c1];
    for (int32_t idx = 0; idx < end0; ++idx) {
        int32_t x = leaf_order[idx];
        up[x] = 0.5 * (rc1_row[x] + rc2_row[x]);
    }
    int32_t end1 = leaf_tin[root_u_c2];
    for (int32_t idx = end0; idx < end1; ++idx) {
        int32_t x = leaf_order[idx];
        up[x] = 0.5 * (rc0_row[x] + rc2_row[x]);
    }
    for (int32_t idx = end1; idx < n_leaves; ++idx) {
        int32_t x = leaf_order[idx];
        up[x] = 0.5 * (rc0_row[x] + rc1_row[x]);
    }

    for (int32_t p = root_u - 1; p >= n_leaves; --p) {
        int32_t c1 = child1_r[p], c2 = child2_r[p];
        const double * restrict r_c1 = (c1 < n_leaves) ? &D[(size_t)c1 * n_leaves] : &ldist[(size_t)(c1 - n_leaves) * n_leaves];
        const double * restrict r_c2 = (c2 < n_leaves) ? &D[(size_t)c2 * n_leaves] : &ldist[(size_t)(c2 - n_leaves) * n_leaves];

        int32_t s1 = leaf_tin[c1];
        int32_t s2 = leaf_tin[c2];
        int32_t e2 = s2 + leaf_count[c2];
        for (int32_t idx = s1; idx < s2; ++idx) {
            int32_t x = leaf_order[idx];
            up[x] = 0.5 * (r_c2[x] + up[x]);
        }
        for (int32_t idx = s2; idx < e2; ++idx) {
            int32_t x = leaf_order[idx];
            up[x] = 0.5 * (r_c1[x] + up[x]);
        }
    }

    double total_L = 0.0;
    for (int32_t x = 0; x < n_leaves; ++x) total_L += up[x];
    return 0.5 * total_L;
}

/* ============================================================================
 * Apply Unrooted SPR directly on adj_u (in O(1) time)
 * ============================================================================ */
static void apply_unrooted_spr(
    int32_t * restrict adj_u,
    int32_t prune_node,
    int32_t anchor_node,
    int32_t dest_u,
    int32_t dest_v
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

/* ============================================================================
 * 5. Full Unrooted FastME SPR Hill-Climber in C99
 * ============================================================================ */
double optimize_fastme_bme_c(
    const int64_t * restrict v_init,
    const double * restrict D,
    int32_t n_leaves,
    int64_t * restrict v_out,
    int32_t * restrict n_iters_out,
    double * restrict elapsed_ms_out
) {
    if (ensure_workspace(n_leaves) != 0) return -1.0;
    double t0 = now_ms();
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;

    int32_t *pairs = g_iwork;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *parent_u = parents_uf + n_nodes_r;
    int32_t *sub_size = parent_u + n_nodes_u;
    int32_t *tin = sub_size + n_nodes_u;
    int32_t *dfs_order = tin + n_nodes_u;
    int32_t *adj_out = dfs_order + n_nodes_u;
    int32_t *q_node = adj_out + (size_t)3 * n_nodes_u;
    int32_t *q_prev = q_node + n_nodes_u;

    double *subdist = g_dwork;
    double *nd_u = subdist + (size_t)n_nodes_u * n_nodes_u;
    double *edge_delta = nd_u + (size_t)3 * n_nodes_u;
    double *q_pow = edge_delta + n_nodes_u;
    double *q_dkc = q_pow + n_nodes_u;
    double *q_dist = q_dkc + n_nodes_u;

    decode_v_to_pairs_fast(v_init, n_pairs, pairs);
    for (int32_t i = 0; i < n_nodes_r; ++i) parents_uf[i] = i;
    int32_t next_parent = n_leaves;
    for (int32_t i = 0; i < n_pairs; ++i) {
        int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
        while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
        while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
        child1_r[next_parent] = p2;
        child2_r[next_parent] = p1;
        parents_uf[p1] = next_parent;
        parents_uf[p2] = next_parent;
        next_parent++;
    }

    int32_t rc1 = child1_r[n_nodes_r - 1], rc2 = child2_r[n_nodes_r - 1];
    int32_t other_root_child = (rc1 < rc2) ? rc1 : rc2;
    int32_t root_u_c0 = child1_r[root_u];
    int32_t root_u_c1 = child2_r[root_u];
    int32_t root_u_c2 = other_root_child;

    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        parent_u[c1] = u; parent_u[c2] = u;
        adj_out[3 * u + 0] = c1; adj_out[3 * u + 1] = c2;
    }
    parent_u[root_u_c0] = root_u;
    parent_u[root_u_c1] = root_u;
    parent_u[root_u_c2] = root_u;
    parent_u[root_u] = -1;

    for (int32_t u = 0; u < n_leaves; ++u) {
        adj_out[3 * u + 0] = parent_u[u];
        adj_out[3 * u + 1] = -1;
        adj_out[3 * u + 2] = -1;
    }
    for (int32_t u = n_leaves; u < root_u; ++u) adj_out[3 * u + 2] = parent_u[u];
    adj_out[3 * root_u + 0] = root_u_c0;
    adj_out[3 * root_u + 1] = root_u_c1;
    adj_out[3 * root_u + 2] = root_u_c2;

    int32_t n_iters = 0;
    double curr_L = 0.0;

    for (int32_t iter = 0; iter < 500; ++iter) {
        n_iters++;
        int32_t r_c0 = adj_out[3 * root_u + 0];
        int32_t r_c1 = adj_out[3 * root_u + 1];
        int32_t r_c2 = adj_out[3 * root_u + 2];
        parent_u[root_u] = -1;
        parent_u[r_c0] = root_u;
        parent_u[r_c1] = root_u;
        parent_u[r_c2] = root_u;

        int32_t ord_len = 0;
        q_prev[ord_len++] = root_u;
        q_prev[ord_len++] = r_c0;
        q_prev[ord_len++] = r_c1;
        q_prev[ord_len++] = r_c2;
        for (int32_t idx = 1; idx < ord_len; ++idx) {
            int32_t u = q_prev[idx];
            if (u < n_leaves) continue;
            int32_t p = parent_u[u];
            int32_t b0 = -1, b1 = -1;
            for (int32_t k = 0; k < 3; ++k) {
                int32_t nb = adj_out[3 * u + k];
                if (nb != -1 && nb != p) {
                    if (b0 == -1) b0 = nb;
                    else b1 = nb;
                }
            }
            child1_r[u] = b0; child2_r[u] = b1;
            parent_u[b0] = u; parent_u[b1] = u;
            q_prev[ord_len++] = b0;
            q_prev[ord_len++] = b1;
        }

        for (int32_t idx = ord_len - 1; idx >= 1; --idx) {
            int32_t u = q_prev[idx];
            if (u < n_leaves) sub_size[u] = 1;
            else sub_size[u] = 1 + sub_size[child1_r[u]] + sub_size[child2_r[u]];
        }
        tin[root_u] = 0;
        tin[r_c0] = 1;
        tin[r_c1] = 1 + sub_size[r_c0];
        tin[r_c2] = 1 + sub_size[r_c0] + sub_size[r_c1];
        for (int32_t idx = 1; idx < ord_len; ++idx) {
            int32_t u = q_prev[idx];
            if (u >= n_leaves) {
                int32_t c1 = child1_r[u], c2 = child2_r[u];
                tin[c1] = tin[u] + 1;
                tin[c2] = tin[u] + 1 + sub_size[c1];
            }
        }
        for (int32_t u = 0; u < n_nodes_u; ++u) dfs_order[tin[u]] = u;

        for (int32_t i = 0; i < n_leaves; ++i) {
            memcpy(&subdist[i * n_nodes_u], &D[i * n_leaves], (size_t)n_leaves * sizeof(double));
        }
        for (int32_t idx = ord_len - 1; idx >= 1; --idx) {
            int32_t u = q_prev[idx];
            if (u < n_leaves) continue;
            int32_t c1 = child1_r[u], c2 = child2_r[u];
            const double * restrict row_c1 = &subdist[(size_t)c1 * n_nodes_u];
            const double * restrict row_c2 = &subdist[(size_t)c2 * n_nodes_u];
            double * restrict row_u = &subdist[(size_t)u * n_nodes_u];
            for (int32_t leaf = 0; leaf < n_leaves; ++leaf) {
                row_u[leaf] = 0.5 * (row_c1[leaf] + row_c2[leaf]);
            }
            for (int32_t leaf = 0; leaf < n_leaves; ++leaf) {
                subdist[(size_t)leaf * n_nodes_u + u] = row_u[leaf];
            }
            for (int32_t j = ord_len - 1; j > idx; --j) {
                int32_t v_node = q_prev[j];
                if (v_node < n_leaves) continue;
                double val = 0.5 * (row_c1[v_node] + row_c2[v_node]);
                row_u[v_node] = val;
                subdist[(size_t)v_node * n_nodes_u + u] = val;
            }
        }
        {
            const double * restrict row_c0 = &subdist[r_c0 * n_nodes_u];
            const double * restrict row_c1 = &subdist[r_c1 * n_nodes_u];
            const double * restrict row_c2 = &subdist[r_c2 * n_nodes_u];
            double * restrict row_r = &subdist[root_u * n_nodes_u];
            int32_t end0 = tin[r_c0] + sub_size[r_c0];
            for (int32_t idx = tin[r_c0]; idx < end0; ++idx) {
                int32_t v_node = dfs_order[idx];
                row_r[v_node] = 0.5 * (row_c1[v_node] + row_c2[v_node]);
            }
            int32_t end1 = tin[r_c1] + sub_size[r_c1];
            for (int32_t idx = tin[r_c1]; idx < end1; ++idx) {
                int32_t v_node = dfs_order[idx];
                row_r[v_node] = 0.5 * (row_c0[v_node] + row_c2[v_node]);
            }
            int32_t end2 = tin[r_c2] + sub_size[r_c2];
            for (int32_t idx = tin[r_c2]; idx < end2; ++idx) {
                int32_t v_node = dfs_order[idx];
                row_r[v_node] = 0.5 * (row_c0[v_node] + row_c1[v_node]);
            }
            for (int32_t v_node = 0; v_node < root_u; ++v_node) {
                subdist[v_node * n_nodes_u + root_u] = row_r[v_node];
            }
        }
        for (int32_t idx = 1; idx < ord_len; ++idx) {
            int32_t p = q_prev[idx];
            if (p < n_leaves) continue;
            int32_t g = parent_u[p];
            int32_t c1 = child1_r[p], c2 = child2_r[p];
            const double * restrict row_c1 = &subdist[c1 * n_nodes_u];
            const double * restrict row_c2 = &subdist[c2 * n_nodes_u];
            const double * restrict row_g = &subdist[g * n_nodes_u];
            double * restrict row_p = &subdist[p * n_nodes_u];
            int32_t end1 = tin[c1] + sub_size[c1];
            for (int32_t i = tin[c1]; i < end1; ++i) {
                int32_t v_node = dfs_order[i];
                row_p[v_node] = 0.5 * (row_c2[v_node] + row_g[v_node]);
            }
            for (int32_t i = tin[c1]; i < end1; ++i) {
                int32_t v_node = dfs_order[i];
                subdist[v_node * n_nodes_u + p] = row_p[v_node];
            }
            int32_t end2 = tin[c2] + sub_size[c2];
            for (int32_t i = tin[c2]; i < end2; ++i) {
                int32_t v_node = dfs_order[i];
                row_p[v_node] = 0.5 * (row_c1[v_node] + row_g[v_node]);
            }
            for (int32_t i = tin[c2]; i < end2; ++i) {
                int32_t v_node = dfs_order[i];
                subdist[v_node * n_nodes_u + p] = row_p[v_node];
            }
        }
        for (int32_t i = 0; i < n_nodes_u; ++i) subdist[i * n_nodes_u + i] = 0.0;

        for (int32_t u = n_leaves; u <= root_u; ++u) {
            int32_t n0 = adj_out[3 * u + 0];
            int32_t n1 = adj_out[3 * u + 1];
            int32_t n2 = adj_out[3 * u + 2];
            nd_u[3 * u + 0] = subdist[n1 * n_nodes_u + n2];
            nd_u[3 * u + 1] = subdist[n0 * n_nodes_u + n2];
            nd_u[3 * u + 2] = subdist[n0 * n_nodes_u + n1];
        }

        curr_L = 0.0;
        for (int32_t x = 0; x < n_leaves; ++x) {
            curr_L += subdist[x * n_nodes_u + parent_u[x]];
        }
        curr_L *= 0.5;
        double eps_tol = 1e-12 * ((curr_L >= 0.0 ? curr_L : -curr_L) + 1.0);

        double best_delta = -eps_tol;
        int32_t best_prune = -1, best_anchor = -1, best_dest_u = -1;

        for (int32_t u_prune = 0; u_prune < root_u; ++u_prune) {
            int32_t anchor = parent_u[u_prune];
            int32_t other0 = -1, other1 = -1;
            for (int32_t k = 0; k < 3; ++k) {
                int32_t nb = adj_out[3 * anchor + k];
                if (nb != -1 && nb != u_prune) {
                    if (other0 == -1) other0 = nb;
                    else other1 = nb;
                }
            }
            for (int32_t u = 0; u < root_u; ++u) edge_delta[u] = 0.0;
            run_half_tree_dfs(other0, other1, anchor, n_leaves, n_nodes_u, adj_out, parent_u, sub_size, nd_u, &subdist[u_prune * n_nodes_u], &subdist[other1 * n_nodes_u], edge_delta);
            run_half_tree_dfs(other1, other0, anchor, n_leaves, n_nodes_u, adj_out, parent_u, sub_size, nd_u, &subdist[u_prune * n_nodes_u], &subdist[other0 * n_nodes_u], edge_delta);

            for (int32_t dest_u = 0; dest_u < root_u; ++dest_u) {
                if (edge_delta[dest_u] < best_delta) {
                    best_delta = edge_delta[dest_u];
                    best_prune = u_prune;
                    best_anchor = anchor;
                    best_dest_u = dest_u;
                }
            }
        }
        for (int32_t u = n_leaves; u < root_u; ++u) {
            int32_t p_prune = parent_u[u];
            int32_t anchor = u;
            int32_t other0 = child1_r[u], other1 = child2_r[u];
            for (int32_t w = 0; w < root_u; ++w) edge_delta[w] = 0.0;
            run_half_tree_dfs(other0, other1, anchor, n_leaves, n_nodes_u, adj_out, parent_u, sub_size, nd_u, &subdist[p_prune * n_nodes_u], &subdist[other1 * n_nodes_u], edge_delta);
            run_half_tree_dfs(other1, other0, anchor, n_leaves, n_nodes_u, adj_out, parent_u, sub_size, nd_u, &subdist[p_prune * n_nodes_u], &subdist[other0 * n_nodes_u], edge_delta);

            int32_t end_u = tin[u] + sub_size[u];
            for (int32_t idx = tin[u] + 1; idx < end_u; ++idx) {
                int32_t dest_u = dfs_order[idx];
                if (edge_delta[dest_u] < best_delta) {
                    best_delta = edge_delta[dest_u];
                    best_prune = p_prune;
                    best_anchor = anchor;
                    best_dest_u = dest_u;
                }
            }
        }

        if (best_prune == -1) break;
        apply_unrooted_spr(adj_out, best_prune, best_anchor, best_dest_u, parent_u[best_dest_u]);
        curr_L += best_delta;
    }

    encode_unrooted_to_v_c(adj_out, n_leaves, v_out);
    if (n_iters_out) *n_iters_out = n_iters;
    if (elapsed_ms_out) *elapsed_ms_out = now_ms() - t0;
    return curr_L;
}

/* ============================================================================
 * 6. Full Phylo2Vec BME Tree Optimizer in C99
 * ============================================================================ */
double optimize_phylo2vec_bme_c(
    const int64_t * restrict v_init,
    const double * restrict D,
    int32_t n_leaves,
    int64_t * restrict v_out,
    int32_t * restrict n_grad_evals_out,
    double * restrict elapsed_ms_out
) {
    if (ensure_workspace(n_leaves) != 0) return -1.0;
    double t0 = now_ms();
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;

    int32_t *dyn_i = (int32_t *)malloc((size_t)(
        n_leaves +                /* perm */
        3 * n_nodes_u +           /* saved_adj */
        n_leaves +                /* saved_perm */
        3 * n_nodes_u +           /* adj_orig */
        n_leaves +                /* row_best_j */
        n_leaves +                /* row_best_dest */
        n_leaves +                /* improving_rows */
        n_leaves +                /* near_rows */
        n_leaves +                /* near_j */
        n_nodes_r + n_leaves      /* relabel_scratch */
    ) * sizeof(int32_t));
    int64_t *dyn_v = (int64_t *)malloc((size_t)(2 * n_pairs) * sizeof(int64_t));
    double *dyn_d = (double *)malloc((size_t)(2 * n_leaves) * sizeof(double));
    uint8_t *dyn_u8 = (uint8_t *)malloc((size_t)(2 * n_nodes_u) * sizeof(uint8_t));

    if (!dyn_i || !dyn_v || !dyn_d || !dyn_u8) {
        free(dyn_i); free(dyn_v); free(dyn_d); free(dyn_u8);
        return -1.0;
    }

    int32_t *iptr = dyn_i;
    int32_t *perm = iptr; iptr += n_leaves;
    int32_t *saved_adj = iptr; iptr += 3 * n_nodes_u;
    int32_t *saved_perm = iptr; iptr += n_leaves;
    int32_t *adj_orig = iptr; iptr += 3 * n_nodes_u;
    int32_t *row_best_j = iptr; iptr += n_leaves;
    int32_t *row_best_dest = iptr; iptr += n_leaves;
    int32_t *improving_rows = iptr; iptr += n_leaves;
    int32_t *near_rows = iptr; iptr += n_leaves;
    int32_t *near_j = iptr; iptr += n_leaves;
    int32_t *relabel_scratch = iptr; iptr += n_nodes_r + n_leaves;

    int64_t *v_pi = dyn_v;
    int64_t *v_cand = dyn_v + n_pairs;

    double *row_best_val = dyn_d;
    double *near_val = dyn_d + n_leaves;

    uint8_t *used_node = dyn_u8;
    uint8_t *is_vmap = dyn_u8 + n_nodes_u;

    for (int32_t i = 0; i < n_leaves; ++i) perm[i] = i;
    memcpy(v_pi, v_init, (size_t)n_pairs * sizeof(int64_t));

    int32_t *pairs = g_iwork;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *siblings_uf = parents_uf + n_nodes_r;
    int32_t *min_leaf = siblings_uf + n_nodes_r;
    int32_t *v_map = min_leaf + n_nodes_r;
    int32_t *dest_row = v_map + n_leaves;
    int32_t *parent_u = dest_row + n_leaves;
    int32_t *sub_size = parent_u + n_nodes_u;
    int32_t *tin = sub_size + n_nodes_u;
    int32_t *dfs_order = tin + n_nodes_u;
    int32_t *adj_u = dfs_order + n_nodes_u;
    int32_t *eval_iwork = adj_u + 3 * n_nodes_u;

    double *subdist = g_dwork;
    double *nd_u = subdist + (size_t)n_nodes_u * n_nodes_u;
    double *edge_delta = nd_u + 3 * n_nodes_u;
    double *D_perm = edge_delta + n_nodes_u;
    double *eval_dwork = D_perm + (size_t)n_leaves * n_leaves;

    memcpy(D_perm, D, (size_t)n_leaves * (size_t)n_leaves * sizeof(double));
    int32_t perm_dirty = 0;

    int32_t n_evals = 0;
    double curr_L = 0.0;
    double best_saved_L = 1e30;
    int32_t has_saved_best = 0;
    int32_t flipped_already = 0;
    int32_t escape_attempts = 0;

    for (int32_t iter = 0; iter < 500; ++iter) {
        int32_t has_gt_i = 0;
        for (int32_t i = 1; i < n_pairs; ++i) {
            if (v_pi[i] > i) { has_gt_i = 1; break; }
        }
        if (has_gt_i) {
            decode_v_to_pairs_fast(v_pi, n_pairs, pairs);
            for (int32_t i = 0; i < n_nodes_r; ++i) {
                parents_uf[i] = i;
                min_leaf[i] = i;
            }
            int32_t next_parent = n_leaves;
            for (int32_t i = 0; i < n_pairs; ++i) {
                int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
                while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
                while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
                child1_r[next_parent] = p2;
                child2_r[next_parent] = p1;
                int32_t ml1 = min_leaf[p2], ml2 = min_leaf[p1];
                min_leaf[next_parent] = (ml1 < ml2) ? ml1 : ml2;
                parents_uf[p1] = next_parent;
                parents_uf[p2] = next_parent;
                next_parent++;
            }
            relabel_from_rooted_tree(child1_r, child2_r, min_leaf, n_leaves, 0, v_pi, perm, relabel_scratch);
            perm_dirty = 1;
        }

        for (int32_t x = 0; x < n_leaves; ++x) {
            parents_uf[x] = x;
            min_leaf[x] = x;
        }
        for (int32_t k = 0; k < n_pairs; ++k) {
            int32_t i = n_pairs - 1 - k;
            int32_t m = n_leaves + k;
            int32_t x = (int32_t)v_pi[i];
            int32_t y = i + 1;
            int32_t c1 = parents_uf[y];
            int32_t c2 = parents_uf[x];
            child1_r[m] = c1;
            child2_r[m] = c2;
            min_leaf[m] = x;
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

        if (perm_dirty) {
            for (int32_t i = 0; i < n_leaves; ++i) {
                const double * restrict d_row = &D[(size_t)perm[i] * n_leaves];
                double * restrict dp_row = &D_perm[(size_t)i * n_leaves];
                for (int32_t j = 0; j < n_leaves; ++j) {
                    dp_row[j] = d_row[perm[j]];
                }
                memcpy(&subdist[(size_t)i * n_nodes_u], dp_row, (size_t)n_leaves * sizeof(double));
            }
            perm_dirty = 0;
        } else {
            for (int32_t i = 0; i < n_leaves; ++i) {
                memcpy(&subdist[(size_t)i * n_nodes_u], &D_perm[(size_t)i * n_leaves], (size_t)n_leaves * sizeof(double));
            }
        }
        for (int32_t u = n_leaves; u < root_u; ++u) {
            const double * restrict row_c1 = &subdist[(size_t)child1_r[u] * n_nodes_u];
            const double * restrict row_c2 = &subdist[(size_t)child2_r[u] * n_nodes_u];
            double * restrict row_u = &subdist[(size_t)u * n_nodes_u];
            for (int32_t v_node = 0; v_node < u; ++v_node) {
                row_u[v_node] = 0.5 * (row_c1[v_node] + row_c2[v_node]);
            }
            for (int32_t v_node = 0; v_node < u; ++v_node) {
                subdist[(size_t)v_node * n_nodes_u + u] = row_u[v_node];
            }
        }
        {
            const double * restrict row_c0 = &subdist[(size_t)root_u_c0 * n_nodes_u];
            const double * restrict row_c1 = &subdist[(size_t)root_u_c1 * n_nodes_u];
            const double * restrict row_c2 = &subdist[(size_t)root_u_c2 * n_nodes_u];
            double * restrict row_r = &subdist[(size_t)root_u * n_nodes_u];
            int32_t end0 = tin[root_u_c0] + sub_size[root_u_c0];
            for (int32_t idx = tin[root_u_c0]; idx < end0; ++idx) {
                int32_t v_node = dfs_order[idx];
                row_r[v_node] = 0.5 * (row_c1[v_node] + row_c2[v_node]);
            }
            int32_t end1 = tin[root_u_c1] + sub_size[root_u_c1];
            for (int32_t idx = tin[root_u_c1]; idx < end1; ++idx) {
                int32_t v_node = dfs_order[idx];
                row_r[v_node] = 0.5 * (row_c0[v_node] + row_c2[v_node]);
            }
            int32_t end2 = tin[root_u_c2] + sub_size[root_u_c2];
            for (int32_t idx = tin[root_u_c2]; idx < end2; ++idx) {
                int32_t v_node = dfs_order[idx];
                row_r[v_node] = 0.5 * (row_c0[v_node] + row_c1[v_node]);
            }
            for (int32_t v_node = 0; v_node < root_u; ++v_node) {
                subdist[(size_t)v_node * n_nodes_u + root_u] = row_r[v_node];
            }
        }
        for (int32_t p = root_u - 1; p >= n_leaves; --p) {
            int32_t g = parent_u[p];
            int32_t c1 = child1_r[p], c2 = child2_r[p];
            const double * restrict row_c1 = &subdist[(size_t)c1 * n_nodes_u];
            const double * restrict row_c2 = &subdist[(size_t)c2 * n_nodes_u];
            const double * restrict row_g = &subdist[(size_t)g * n_nodes_u];
            double * restrict row_p = &subdist[(size_t)p * n_nodes_u];

            int32_t end1 = tin[c1] + sub_size[c1];
            for (int32_t idx = tin[c1]; idx < end1; ++idx) {
                int32_t v_node = dfs_order[idx];
                row_p[v_node] = 0.5 * (row_c2[v_node] + row_g[v_node]);
            }
            for (int32_t idx = tin[c1]; idx < end1; ++idx) {
                int32_t v_node = dfs_order[idx];
                subdist[(size_t)v_node * n_nodes_u + p] = row_p[v_node];
            }
            int32_t end2 = tin[c2] + sub_size[c2];
            for (int32_t idx = tin[c2]; idx < end2; ++idx) {
                int32_t v_node = dfs_order[idx];
                row_p[v_node] = 0.5 * (row_c1[v_node] + row_g[v_node]);
            }
            for (int32_t idx = tin[c2]; idx < end2; ++idx) {
                int32_t v_node = dfs_order[idx];
                subdist[(size_t)v_node * n_nodes_u + p] = row_p[v_node];
            }
        }
        for (int32_t i = 0; i < n_nodes_u; ++i) subdist[(size_t)i * n_nodes_u + i] = 0.0;

        for (int32_t u = n_leaves; u <= root_u; ++u) {
            int32_t n0 = adj_u[3 * u + 0];
            int32_t n1 = adj_u[3 * u + 1];
            int32_t n2 = adj_u[3 * u + 2];
            nd_u[3 * u + 0] = subdist[(size_t)n1 * n_nodes_u + n2];
            nd_u[3 * u + 1] = subdist[(size_t)n0 * n_nodes_u + n2];
            nd_u[3 * u + 2] = subdist[(size_t)n0 * n_nodes_u + n1];
        }

        curr_L = 0.0;
        for (int32_t x = 0; x < n_leaves; ++x) {
            curr_L += subdist[(size_t)x * n_nodes_u + parent_u[x]];
        }
        curr_L *= 0.5;
        n_evals++;
        double eps_tol = 1e-12 * ((curr_L >= 0.0 ? curr_L : -curr_L) + 1.0);

        int32_t n_imp = 0, n_near = 0;
        double best_vmap_extra_delta = -eps_tol;
        int32_t best_vmap_prune = -1, best_vmap_anchor = -1, best_vmap_dest = -1;
        for (int32_t j = 0; j < n_leaves; ++j) dest_row[j] = j;

        for (int32_t v_index = n_leaves - 1; v_index >= 2; --v_index) {
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
                if (n_imp == 0) {
                    for (int32_t u = 0; u < root_u; ++u) edge_delta[u] = 0.0;
                } else {
                    edge_delta[(parent_u[tree_root] == parent) ? tree_root : parent] = 0.0;
                    edge_delta[(parent_u[other0]    == parent) ? other0    : parent] = 0.0;
                    edge_delta[(parent_u[other1]    == parent) ? other1    : parent] = 0.0;
                }

                run_half_tree_dfs(
                    other0, other1, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                    row_root, &subdist[(size_t)other1 * n_nodes_u], edge_delta
                );
                run_half_tree_dfs(
                    other1, other0, parent, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u,
                    row_root, &subdist[(size_t)other0 * n_nodes_u], edge_delta
                );

                int32_t i = v_index - 1;
                int32_t vi_val = (int32_t)v_pi[i];

                double best_r_val = -eps_tol;
                int32_t best_r_j = -1, best_r_dest = -1;
                double best_pos_val = 1e9;
                int32_t best_pos_j = -1;

                for (int32_t j = 0; j <= i; ++j) {
                    if (j == vi_val) continue;
                    int32_t du = dest_row[j];
                    double val = edge_delta[du];
                    if (val < best_r_val) {
                        best_r_val = val;
                        best_r_j = j;
                        best_r_dest = du;
                    } else if (val > eps_tol && val < best_pos_val) {
                        best_pos_val = val;
                        best_pos_j = j;
                    }
                }
                for (int32_t j = i + 1; j <= 2 * i - 2; ++j) {
                    int32_t du = (2 * n_leaves - 2) - 2 * i + j;
                    double val = edge_delta[du];
                    if (val < best_r_val) {
                        best_r_val = val;
                        best_r_j = j;
                        best_r_dest = du;
                    } else if (val > eps_tol && val < best_pos_val) {
                        best_pos_val = val;
                        best_pos_j = j;
                    }
                }
                {
                    int32_t j = (2 * i - 1 > i) ? (2 * i - 1) : (2 * i);
                    int32_t du = other_root_child;
                    double val = edge_delta[du];
                    if (val < best_r_val) {
                        best_r_val = val;
                        best_r_j = j;
                        best_r_dest = du;
                    } else if (val > eps_tol && val < best_pos_val) {
                        best_pos_val = val;
                        best_pos_j = j;
                    }
                }

                if (best_r_j != -1) {
                    row_best_j[i] = best_r_j;
                    row_best_dest[i] = best_r_dest;
                    row_best_val[i] = best_r_val;
                    improving_rows[n_imp++] = i;
                }
                if (best_r_j == -1 && best_pos_j != -1 && best_pos_val < 0.05) {
                    near_rows[n_near] = i;
                    near_j[n_near] = best_pos_j;
                    near_val[n_near] = best_pos_val;
                    n_near++;
                }

                if (n_imp == 0) {
                    for (int32_t du = 0; du < root_u; ++du) {
                        if (edge_delta[du] < best_vmap_extra_delta) {
                            best_vmap_extra_delta = edge_delta[du];
                            best_vmap_prune = tree_root;
                            best_vmap_anchor = parent;
                            best_vmap_dest = du;
                        }
                    }
                }
            }

            int32_t m_active = n_leaves + (n_leaves - 1 - v_index);
            dest_row[min_leaf[m_active]] = m_active;
        }

        if (n_imp > 0) {
            flipped_already = 0;
            for (int32_t a = 0; a < n_imp / 2; ++a) {
                int32_t tmp = improving_rows[a];
                improving_rows[a] = improving_rows[n_imp - 1 - a];
                improving_rows[n_imp - 1 - a] = tmp;
            }
            for (int32_t a = 1; a < n_imp; ++a) {
                int32_t key_r = improving_rows[a];
                double key_v = row_best_val[key_r];
                int32_t b = a - 1;
                while (b >= 0 && row_best_val[improving_rows[b]] > key_v) {
                    improving_rows[b + 1] = improving_rows[b];
                    b--;
                }
                improving_rows[b + 1] = key_r;
            }

            int32_t best_r = improving_rows[0];
            double single_best_gain = row_best_val[best_r];

            int32_t batch_rows[16];
            int32_t batch_len = 0;
            double sum_pred = 0.0;
            if (n_imp >= 2) {
                double cutoff = 0.05 * single_best_gain;
                memset(used_node, 0, (size_t)n_nodes_u * sizeof(uint8_t));
                for (int32_t m = 0; m < n_imp && batch_len < 8; ++m) {
                    int32_t r = improving_rows[m];
                    if (batch_len > 0 && row_best_val[r] >= cutoff) break;
                    int32_t u_prune = v_map[r + 1];
                    int32_t p_anchor = parent_u[u_prune];
                    int32_t d_dest = row_best_dest[r];
                    int32_t p_dest = (d_dest == root_u) ? other_root_child : parent_u[d_dest];

                    if (used_node[u_prune] || used_node[p_anchor] || used_node[d_dest] || used_node[p_dest]) {
                        continue;
                    }
                    int32_t conflict = 0;
                    int32_t end_sub = tin[u_prune] + sub_size[u_prune];
                    for (int32_t idx = tin[u_prune]; idx < end_sub; ++idx) {
                        if (used_node[dfs_order[idx]]) { conflict = 1; break; }
                    }
                    if (conflict) continue;

                    for (int32_t idx = tin[u_prune]; idx < end_sub; ++idx) {
                        used_node[dfs_order[idx]] = 1;
                    }
                    used_node[p_anchor] = 1;
                    used_node[d_dest] = 1;
                    used_node[p_dest] = 1;
                    batch_rows[batch_len++] = r;
                    sum_pred += row_best_val[r];
                }
            }

            int32_t accepted = 0;
            if (batch_len >= 2 && sum_pred < 1.05 * single_best_gain) {
                int32_t k_try = batch_len;
                while (k_try >= 2) {
                    memcpy(v_cand, v_pi, (size_t)n_pairs * sizeof(int64_t));
                    for (int32_t idx = 0; idx < k_try; ++idx) {
                        int32_t r = batch_rows[idx];
                        v_cand[r] = row_best_j[r];
                    }
                    double L_cand = evaluate_bme_length_fast(v_cand, D_perm, n_leaves, eval_iwork, eval_dwork);
                    if (L_cand < curr_L + single_best_gain - eps_tol) {
                        memcpy(v_pi, v_cand, (size_t)n_pairs * sizeof(int64_t));
                        curr_L = L_cand;
                        accepted = 1;
                        break;
                    }
                    k_try /= 2;
                }
            }
            if (!accepted) {
                v_pi[best_r] = row_best_j[best_r];
                curr_L += single_best_gain;
            }
            continue;
        }

        if (best_vmap_prune != -1) {
            apply_unrooted_spr(adj_u, best_vmap_prune, best_vmap_anchor, best_vmap_dest, parent_u[best_vmap_dest]);
            encode_unrooted_to_v_c(adj_u, n_leaves, v_pi);
            flipped_already = 0;
            continue;
        }

        if (!flipped_already) {
            flipped_already = 1;
            relabel_from_rooted_tree(child1_r, child2_r, min_leaf, n_leaves, 1, v_pi, perm, relabel_scratch);
            perm_dirty = 1;
            continue;
        }

        memset(is_vmap, 0, (size_t)n_nodes_u * sizeof(uint8_t));
        for (int32_t v_idx = 2; v_idx < n_leaves; ++v_idx) is_vmap[v_map[v_idx]] = 1;

        double best_rem_delta = -eps_tol;
        int32_t best_prune = -1, best_anchor = -1, best_dest_u = -1;

        for (int32_t u_prune = 0; u_prune < root_u; ++u_prune) {
            if (is_vmap[u_prune]) continue;
            int32_t anchor = parent_u[u_prune];
            int32_t other0 = -1, other1 = -1;
            for (int32_t k = 0; k < 3; ++k) {
                int32_t nb = adj_u[3 * anchor + k];
                if (nb != -1 && nb != u_prune) {
                    if (other0 == -1) other0 = nb;
                    else other1 = nb;
                }
            }
            for (int32_t u = 0; u < root_u; ++u) edge_delta[u] = 0.0;
            run_half_tree_dfs(other0, other1, anchor, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u, &subdist[(size_t)u_prune * n_nodes_u], &subdist[(size_t)other1 * n_nodes_u], edge_delta);
            run_half_tree_dfs(other1, other0, anchor, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u, &subdist[(size_t)u_prune * n_nodes_u], &subdist[(size_t)other0 * n_nodes_u], edge_delta);
            for (int32_t dest_u = 0; dest_u < root_u; ++dest_u) {
                if (edge_delta[dest_u] < best_rem_delta) {
                    best_rem_delta = edge_delta[dest_u];
                    best_prune = u_prune;
                    best_anchor = anchor;
                    best_dest_u = dest_u;
                }
            }
        }
        for (int32_t u = n_leaves; u < root_u; ++u) {
            int32_t p_prune = parent_u[u];
            int32_t anchor = u;
            int32_t other0 = child1_r[u], other1 = child2_r[u];
            for (int32_t w = 0; w < root_u; ++w) edge_delta[w] = 0.0;
            run_half_tree_dfs(other0, other1, anchor, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u, &subdist[(size_t)p_prune * n_nodes_u], &subdist[(size_t)other1 * n_nodes_u], edge_delta);
            run_half_tree_dfs(other1, other0, anchor, n_leaves, n_nodes_u, adj_u, parent_u, sub_size, nd_u, &subdist[(size_t)p_prune * n_nodes_u], &subdist[(size_t)other0 * n_nodes_u], edge_delta);
            int32_t end_u = tin[u] + sub_size[u];
            for (int32_t idx = tin[u] + 1; idx < end_u; ++idx) {
                int32_t dest_u = dfs_order[idx];
                if (edge_delta[dest_u] < best_rem_delta) {
                    best_rem_delta = edge_delta[dest_u];
                    best_prune = p_prune;
                    best_anchor = anchor;
                    best_dest_u = dest_u;
                }
            }
        }

        if (best_prune != -1) {
            apply_unrooted_spr(adj_u, best_prune, best_anchor, best_dest_u, parent_u[best_dest_u]);
            encode_unrooted_to_v_c(adj_u, n_leaves, v_pi);
            flipped_already = 0;
            continue;
        }

        if (curr_L < best_saved_L) {
            best_saved_L = curr_L;
            memcpy(saved_adj, adj_u, (size_t)(3 * n_nodes_u) * sizeof(int32_t));
            memcpy(saved_perm, perm, (size_t)n_leaves * sizeof(int32_t));
            has_saved_best = 1;
        }

        if (n_near >= 2 && escape_attempts < 2) {
            escape_attempts++;
            for (int32_t a = 0; a < n_near / 2; ++a) {
                int32_t tr = near_rows[a]; near_rows[a] = near_rows[n_near - 1 - a]; near_rows[n_near - 1 - a] = tr;
                int32_t tj = near_j[a]; near_j[a] = near_j[n_near - 1 - a]; near_j[n_near - 1 - a] = tj;
                double tv = near_val[a]; near_val[a] = near_val[n_near - 1 - a]; near_val[n_near - 1 - a] = tv;
            }
            for (int32_t a = 1; a < n_near; ++a) {
                int32_t kr = near_rows[a], kj = near_j[a];
                double kv = near_val[a];
                int32_t b = a - 1;
                while (b >= 0 && near_val[b] > kv) {
                    near_rows[b + 1] = near_rows[b];
                    near_j[b + 1] = near_j[b];
                    near_val[b + 1] = near_val[b];
                    b--;
                }
                near_rows[b + 1] = kr;
                near_j[b + 1] = kj;
                near_val[b + 1] = kv;
            }
            int32_t idx1 = escape_attempts;
            if (idx1 < n_near && near_val[0] + near_val[idx1] < 0.02) {
                v_pi[near_rows[0]] = near_j[0];
                v_pi[near_rows[idx1]] = near_j[idx1];
                flipped_already = 0;
                continue;
            }
        }

        if (has_saved_best && best_saved_L < curr_L) {
            curr_L = best_saved_L;
            memcpy(adj_u, saved_adj, (size_t)(3 * n_nodes_u) * sizeof(int32_t));
            memcpy(perm, saved_perm, (size_t)n_leaves * sizeof(int32_t));
        }
        break;
    }

    for (int32_t u = 0; u < n_nodes_u; ++u) {
        for (int32_t k = 0; k < 3; ++k) {
            int32_t w = adj_u[3 * u + k];
            int32_t w_orig = (w >= 0 && w < n_leaves) ? perm[w] : w;
            int32_t u_orig = (u < n_leaves) ? perm[u] : u;
            adj_orig[3 * u_orig + k] = w_orig;
        }
    }
    encode_unrooted_to_v_c(adj_orig, n_leaves, v_out);

    free(dyn_i);
    free(dyn_v);
    free(dyn_d);
    free(dyn_u8);

    if (n_grad_evals_out) *n_grad_evals_out = n_evals;
    if (elapsed_ms_out) *elapsed_ms_out = now_ms() - t0;
    return curr_L;
}

/* ============================================================================
 * 7. Exact Clade-Condensed Minibatch BME Gradient & Optimizer (Theorem 5.5)
 * ============================================================================ */

static void decode_v_to_rooted_children(
    const int64_t * restrict v,
    int32_t n_leaves,
    int32_t * restrict child1_r,
    int32_t * restrict child2_r,
    int32_t * restrict scratch_i
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    int32_t *pairs = scratch_i;
    int32_t *parents_uf = scratch_i + 2 * n_pairs;

    int32_t is_cf = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        if (v[i] > i) { is_cf = 0; break; }
    }
    if (is_cf) {
        for (int32_t x = 0; x < n_leaves; ++x) parents_uf[x] = x;
        for (int32_t k = 0; k < n_pairs; ++k) {
            int32_t i = n_pairs - 1 - k;
            int32_t m = n_leaves + k;
            int32_t x = (int32_t)v[i];
            int32_t y = i + 1;
            child1_r[m] = parents_uf[y];
            child2_r[m] = parents_uf[x];
            parents_uf[x] = m;
        }
    } else {
        decode_v_to_pairs_fast(v, n_pairs, pairs);
        for (int32_t i = 0; i < n_nodes_r; ++i) parents_uf[i] = i;
        int32_t next_parent = n_leaves;
        for (int32_t i = 0; i < n_pairs; ++i) {
            int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
            while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
            while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
            child1_r[next_parent] = p2;
            child2_r[next_parent] = p1;
            parents_uf[p1] = next_parent;
            parents_uf[p2] = next_parent;
            next_parent++;
        }
    }
}

void condense_tree_bme_c(
    const int64_t * restrict v,
    const double * restrict D,
    int32_t n_leaves,
    const uint8_t * restrict is_condensed_node,
    int32_t * restrict n_eff_out,
    int64_t * restrict v_eff_out,
    int32_t * restrict super_node_map_out,
    int32_t * restrict leaf_to_super_out,
    double * restrict omega_leaf_out,
    double * restrict D_eff_out
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;

    int32_t *buf = (int32_t *)malloc((size_t)(7 * n_nodes_r + 2 * n_pairs) * sizeof(int32_t));
    double *w_node = (double *)malloc((size_t)n_nodes_r * sizeof(double));
    if (!buf || !w_node) { free(buf); free(w_node); return; }

    int32_t *child1_r = buf;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *active_int = child2_r + n_nodes_r;
    int32_t *min_node = active_int + n_nodes_r;
    int32_t *rep_label = min_node + n_nodes_r;
    int32_t *q_sub = rep_label + n_nodes_r;
    int32_t *scratch = q_sub + n_nodes_r;

    decode_v_to_rooted_children(v, n_leaves, child1_r, child2_r, scratch);

    /* 1. Top-down BFS/preorder from root_r stopping at leaves < n_leaves or condensed roots */
    int32_t n_act_int = 0;
    active_int[n_act_int++] = root_r;
    for (int32_t idx = 0; idx < n_act_int; ++idx) {
        int32_t u = active_int[idx];
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        if (c1 >= n_leaves && !is_condensed_node[c1]) active_int[n_act_int++] = c1;
        if (c2 >= n_leaves && !is_condensed_node[c2]) active_int[n_act_int++] = c2;
    }
    const int32_t n_eff = n_act_int + 1;
    *n_eff_out = n_eff;

    /* 2. Bottom-up min_node for deterministic cherry-first ordering on condensed tree */
    for (int32_t u = 0; u < n_nodes_r; ++u) min_node[u] = u;
    for (int32_t idx = n_act_int - 1; idx >= 0; --idx) {
        int32_t u = active_int[idx];
        int32_t m1 = min_node[child1_r[u]], m2 = min_node[child2_r[u]];
        min_node[u] = (m1 < m2) ? m1 : m2;
    }

    /* 3. Top-down cherry-first label assignment on active_int (Theorem 2.1) */
    rep_label[root_r] = 0;
    int32_t next_label = 1;
    for (int32_t idx = 0; idx < n_act_int; ++idx) {
        int32_t u = active_int[idx];
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        int32_t c_stay = (min_node[c1] < min_node[c2]) ? c1 : c2;
        int32_t c_move = (min_node[c1] < min_node[c2]) ? c2 : c1;
        int32_t x = rep_label[u];
        int32_t y = next_label++;
        rep_label[c_stay] = x;
        rep_label[c_move] = y;
        v_eff_out[y - 1] = x;

        if (c_stay < n_leaves || is_condensed_node[c_stay]) super_node_map_out[x] = c_stay;
        if (c_move < n_leaves || is_condensed_node[c_move]) super_node_map_out[y] = c_move;
    }

    /* 4. Compute exact balanced weights omega_x = 2^{-d_{S_z}(r_z, x)} in O(N) time */
    for (int32_t z = 0; z < n_eff; ++z) {
        int32_t r_z = super_node_map_out[z];
        if (r_z < n_leaves) {
            leaf_to_super_out[r_z] = z;
            omega_leaf_out[r_z] = 1.0;
        } else {
            int32_t head = 0, tail = 0;
            q_sub[tail++] = r_z;
            w_node[r_z] = 1.0;
            while (head < tail) {
                int32_t u = q_sub[head++];
                double wu = 0.5 * w_node[u];
                int32_t c1 = child1_r[u], c2 = child2_r[u];
                w_node[c1] = wu;
                w_node[c2] = wu;
                if (c1 < n_leaves) {
                    leaf_to_super_out[c1] = z;
                    omega_leaf_out[c1] = wu;
                } else {
                    q_sub[tail++] = c1;
                }
                if (c2 < n_leaves) {
                    leaf_to_super_out[c2] = z;
                    omega_leaf_out[c2] = wu;
                } else {
                    q_sub[tail++] = c2;
                }
            }
        }
    }

    /* 5. Materialize condensed distance matrix D_eff (N_eff x N_eff) if D is provided */
    if (D && D_eff_out) {
        memset(D_eff_out, 0, (size_t)n_eff * n_eff * sizeof(double));
        for (int32_t x = 0; x < n_leaves; ++x) {
            int32_t zx = leaf_to_super_out[x];
            double wx = omega_leaf_out[x];
            const double * restrict d_row = &D[(size_t)x * n_leaves];
            double * restrict deff_row = &D_eff_out[(size_t)zx * n_eff];
            for (int32_t y = 0; y < n_leaves; ++y) {
                int32_t zy = leaf_to_super_out[y];
                if (zx != zy) {
                    deff_row[zy] += wx * omega_leaf_out[y] * d_row[y];
                }
            }
        }
    }

    free(w_node);
    free(buf);
}

void expand_condensed_tree_to_v_c(
    const int64_t * restrict v_orig,
    int32_t n_leaves,
    const int64_t * restrict v_eff,
    int32_t n_eff,
    const int32_t * restrict super_node_map,
    int64_t * restrict v_expanded_out
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_eff = 2 * n_eff - 1;

    int32_t *buf = (int32_t *)malloc((size_t)(7 * n_nodes_r + 2 * n_nodes_eff + 2 * n_pairs) * sizeof(int32_t));
    uint8_t *in_condensed_sub = (uint8_t *)calloc((size_t)n_nodes_r, sizeof(uint8_t));
    if (!buf || !in_condensed_sub) { free(buf); free(in_condensed_sub); return; }

    int32_t *orig_c1 = buf;
    int32_t *orig_c2 = orig_c1 + n_nodes_r;
    int32_t *full_c1 = orig_c2 + n_nodes_r;
    int32_t *full_c2 = full_c1 + n_nodes_r;
    int32_t *q_sub = full_c2 + n_nodes_r;
    int32_t *eff_c1 = q_sub + n_nodes_r;
    int32_t *eff_c2 = eff_c1 + n_nodes_eff;
    int32_t *scratch = eff_c2 + n_nodes_eff;

    decode_v_to_rooted_children(v_orig, n_leaves, orig_c1, orig_c2, scratch);
    decode_v_to_rooted_children(v_eff, n_eff, eff_c1, eff_c2, scratch);

    memcpy(full_c1, orig_c1, (size_t)n_nodes_r * sizeof(int32_t));
    memcpy(full_c2, orig_c2, (size_t)n_nodes_r * sizeof(int32_t));

    /* Mark all internal nodes strictly inside or at the root of condensed subtrees */
    for (int32_t z = 0; z < n_eff; ++z) {
        int32_t r_z = super_node_map[z];
        if (r_z >= n_leaves) {
            int32_t head = 0, tail = 0;
            q_sub[tail++] = r_z;
            in_condensed_sub[r_z] = 1;
            while (head < tail) {
                int32_t u = q_sub[head++];
                int32_t c1 = orig_c1[u], c2 = orig_c2[u];
                if (c1 >= n_leaves) { in_condensed_sub[c1] = 1; q_sub[tail++] = c1; }
                if (c2 >= n_leaves) { in_condensed_sub[c2] = 1; q_sub[tail++] = c2; }
            }
        }
    }

    /* Map the (n_eff - 1) internal nodes of eff_c1/eff_c2 to the available internal node slots of full_c1/full_c2 */
    int32_t *eff_to_full = q_sub;
    for (int32_t z = 0; z < n_eff; ++z) {
        eff_to_full[z] = super_node_map[z];
    }
    int32_t next_free_u = n_leaves;
    for (int32_t w_eff = n_eff; w_eff < n_nodes_eff - 1; ++w_eff) {
        while (next_free_u < n_nodes_r - 1 && in_condensed_sub[next_free_u]) next_free_u++;
        eff_to_full[w_eff] = next_free_u++;
    }
    eff_to_full[n_nodes_eff - 1] = n_nodes_r - 1;

    for (int32_t w_eff = n_eff; w_eff < n_nodes_eff; ++w_eff) {
        int32_t u_full = eff_to_full[w_eff];
        full_c1[u_full] = eff_to_full[eff_c1[w_eff]];
        full_c2[u_full] = eff_to_full[eff_c2[w_eff]];
    }

    encode_rooted_to_v_c(full_c1, full_c2, n_leaves, v_expanded_out);

    free(in_condensed_sub);
    free(buf);
}

void select_minibatch_clades_c(
    const int64_t * restrict v,
    int32_t n_leaves,
    int32_t target_n_eff,
    int32_t batch_idx,
    uint8_t * restrict is_condensed_node_out
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t root_r = n_nodes_r - 1;
    memset(is_condensed_node_out, 0, (size_t)n_nodes_r * sizeof(uint8_t));
    if (target_n_eff >= n_leaves) return;

    int32_t *buf = (int32_t *)malloc((size_t)(8 * n_nodes_r + 2 * n_pairs) * sizeof(int32_t));
    if (!buf) return;
    int32_t *child1_r = buf;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *dfs_stack = child2_r + n_nodes_r;
    int32_t *dfs_leaves = dfs_stack + n_nodes_r;
    int32_t *has_active = dfs_leaves + n_leaves;
    int32_t *sub_leaves = has_active + n_nodes_r;
    int32_t *scratch = sub_leaves + n_nodes_r;

    decode_v_to_rooted_children(v, n_leaves, child1_r, child2_r, scratch);

    /* Collect leaves in Euler-tour DFS order and post-order internal nodes */
    int32_t sp = 0, n_dfs_leaves = 0, n_pre = 0;
    int32_t *preorder = scratch;
    dfs_stack[sp++] = root_r;
    while (sp > 0) {
        int32_t u = dfs_stack[--sp];
        if (u < n_leaves) {
            dfs_leaves[n_dfs_leaves++] = u;
        } else {
            preorder[n_pre++] = u;
            dfs_stack[sp++] = child2_r[u];
            dfs_stack[sp++] = child1_r[u];
        }
    }

    int32_t win_size = (target_n_eff * 2) / 3;
    if (win_size < 4) win_size = 4;
    if (win_size > target_n_eff - 2) win_size = target_n_eff - 2;
    int32_t stride = (win_size > 2) ? (win_size / 2) : 1;
    int32_t start_idx = (batch_idx * stride) % n_leaves;

    memset(has_active, 0, (size_t)n_nodes_r * sizeof(int32_t));
    for (int32_t k = 0; k < win_size; ++k) {
        int32_t leaf = dfs_leaves[(start_idx + k) % n_leaves];
        has_active[leaf] = 1;
    }
    for (int32_t u = 0; u < n_leaves; ++u) sub_leaves[u] = 1;
    for (int32_t idx = n_pre - 1; idx >= 0; --idx) {
        int32_t u = preorder[idx];
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        has_active[u] = has_active[c1] | has_active[c2];
        sub_leaves[u] = sub_leaves[c1] + sub_leaves[c2];
    }

    /* Collect maximal inactive subtrees (where has_active[u] == 0 and has_active[parent] == 1) */
    int32_t *inactive_roots = dfs_stack;
    int32_t n_inactive = 0;
    for (int32_t idx = 0; idx < n_pre; ++idx) {
        int32_t u = preorder[idx];
        if (has_active[u]) {
            int32_t c1 = child1_r[u], c2 = child2_r[u];
            if (!has_active[c1]) inactive_roots[n_inactive++] = c1;
            if (!has_active[c2]) inactive_roots[n_inactive++] = c2;
        }
    }

    /* Split largest inactive clade roots until win_size + n_inactive reaches target_n_eff */
    while (win_size + n_inactive < target_n_eff) {
        int32_t best_k = -1, best_sz = 1;
        for (int32_t k = 0; k < n_inactive; ++k) {
            int32_t u = inactive_roots[k];
            if (u >= n_leaves && sub_leaves[u] > best_sz) {
                best_sz = sub_leaves[u];
                best_k = k;
            }
        }
        if (best_k == -1) break;
        int32_t u = inactive_roots[best_k];
        inactive_roots[best_k] = child1_r[u];
        inactive_roots[n_inactive++] = child2_r[u];
    }

    for (int32_t k = 0; k < n_inactive; ++k) {
        int32_t u = inactive_roots[k];
        if (u >= n_leaves) is_condensed_node_out[u] = 1;
    }

    free(buf);
}

double optimize_phylo2vec_bme_minibatch_c(
    const int64_t * restrict v_init,
    const double * restrict D,
    int32_t n_leaves,
    int32_t target_n_eff,
    int32_t n_batches,
    int64_t * restrict v_out
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    memcpy(v_out, v_init, (size_t)n_pairs * sizeof(int64_t));

    uint8_t *is_cond = (uint8_t *)malloc((size_t)n_nodes_r * sizeof(uint8_t));
    int64_t *v_eff = (int64_t *)malloc((size_t)n_pairs * sizeof(int64_t));
    int64_t *v_eff_opt = (int64_t *)malloc((size_t)n_pairs * sizeof(int64_t));
    int32_t *super_map = (int32_t *)malloc((size_t)n_leaves * sizeof(int32_t));
    int32_t *leaf_to_super = (int32_t *)malloc((size_t)n_leaves * sizeof(int32_t));
    double *omega = (double *)malloc((size_t)n_leaves * sizeof(double));
    double *D_eff = (double *)malloc((size_t)n_leaves * n_leaves * sizeof(double));

    for (int32_t b = 0; b < n_batches; ++b) {
        select_minibatch_clades_c(v_out, n_leaves, target_n_eff, b, is_cond);
        int32_t n_eff = 0;
        condense_tree_bme_c(v_out, D, n_leaves, is_cond, &n_eff, v_eff, super_map, leaf_to_super, omega, D_eff);
        int32_t evals = 0;
        double ms = 0.0;
        optimize_phylo2vec_bme_c(v_eff, D_eff, n_eff, v_eff_opt, &evals, &ms);
        expand_condensed_tree_to_v_c(v_out, n_leaves, v_eff_opt, n_eff, super_map, v_out);
    }

    if (ensure_workspace(n_leaves) != 0) return -1.0;
    double final_L = evaluate_bme_length_fast(v_out, D, n_leaves, g_iwork, g_dwork);

    free(D_eff); free(omega); free(leaf_to_super); free(super_map);
    free(v_eff_opt); free(v_eff); free(is_cond);
    return final_L;
}

#ifdef __cplusplus
}
#endif
