#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static uint64_t rng_state = 88172645463325252ULL;
static inline uint32_t xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state & 0xFFFFFFFFULL);
}
static inline double rand_uniform(void) {
    return (double)xorshift32() / 4294967296.0;
}

/* ============================================================================
 * Core L1-Cache-Resident Half-Tree BFS (shared by Phylo2Vec & FastME C99)
 * ============================================================================ */
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

static inline void apply_unrooted_spr(
    int32_t * restrict adj_u,
    int32_t prune_node,
    int32_t anchor_node,
    int32_t dest_a,
    int32_t dest_b
) {
    int32_t other0 = -1, other1 = -1;
    for (int32_t k = 0; k < 3; ++k) {
        int32_t nb = adj_u[3 * anchor_node + k];
        if (nb != prune_node) {
            if (other0 == -1) other0 = nb;
            else other1 = nb;
        }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * other0 + k] == anchor_node) {
            adj_u[3 * other0 + k] = other1;
            break;
        }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * other1 + k] == anchor_node) {
            adj_u[3 * other1 + k] = other0;
            break;
        }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * dest_a + k] == dest_b) {
            adj_u[3 * dest_a + k] = anchor_node;
            break;
        }
    }
    for (int32_t k = 0; k < 3; ++k) {
        if (adj_u[3 * dest_b + k] == dest_a) {
            adj_u[3 * dest_b + k] = anchor_node;
            break;
        }
    }
    adj_u[3 * anchor_node + 0] = prune_node;
    adj_u[3 * anchor_node + 1] = dest_a;
    adj_u[3 * anchor_node + 2] = dest_b;
}

/* ============================================================================
 * Fast O(N^2) BME length evaluation (leaf-only 1D top-down pass, zero D copy!)
 * `D_perm` is already permuted as N x N contiguous matrix!
 * ============================================================================ */
static double evaluate_bme_length_fast(
    const int64_t * restrict v_pi,
    const double * restrict D_perm,
    int32_t n_leaves,
    int32_t * restrict iwork,
    double * restrict dwork
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;

    int32_t *child1_r = iwork;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *siblings_uf = parents_uf + n_nodes_r;
    int32_t *sub_size = siblings_uf + n_nodes_r;
    int32_t *tin = sub_size + n_nodes_u;
    int32_t *dfs_order = tin + n_nodes_u;
    int32_t *pairs = dfs_order + n_nodes_u;

    int32_t all_le_i = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        if (v_pi[i] > i) { all_le_i = 0; break; }
    }
    if (all_le_i) {
        for (int32_t k = 0; k < n_pairs; ++k) {
            int32_t i = n_pairs - 1 - k;
            pairs[2 * k] = (int32_t)v_pi[i];
            pairs[2 * k + 1] = i + 1;
        }
    } else {
        pairs[0] = 0; pairs[1] = 1;
        int32_t curr_len = 1;
        for (int32_t i = 1; i < n_pairs; ++i) {
            int32_t next_leaf = i + 1;
            int32_t vi = (int32_t)v_pi[i];
            if (vi <= i) {
                memmove(&pairs[2], &pairs[0], (size_t)(2 * curr_len) * sizeof(int32_t));
                pairs[0] = vi; pairs[1] = next_leaf;
            } else {
                int32_t idx = vi - curr_len;
                int32_t c0 = pairs[2 * (idx - 1)];
                memmove(&pairs[2 * (idx + 1)], &pairs[2 * idx], (size_t)(2 * (curr_len - idx)) * sizeof(int32_t));
                pairs[2 * idx] = c0; pairs[2 * idx + 1] = next_leaf;
            }
            curr_len++;
        }
    }

    for (int32_t i = 0; i < n_nodes_r; ++i) {
        parents_uf[i] = i;
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

    int32_t rc1 = child1_r[n_nodes_r - 1];
    int32_t rc2 = child2_r[n_nodes_r - 1];
    int32_t other_root_child = (rc1 < rc2) ? rc1 : rc2;
    int32_t root_u_c0 = child1_r[root_u];
    int32_t root_u_c1 = child2_r[root_u];
    int32_t root_u_c2 = other_root_child;

    for (int32_t u = 0; u < n_leaves; ++u) sub_size[u] = 1;
    for (int32_t u = n_leaves; u < root_u; ++u) {
        sub_size[u] = sub_size[child1_r[u]] + sub_size[child2_r[u]];
    }
    tin[root_u_c0] = 0;
    tin[root_u_c1] = sub_size[root_u_c0];
    tin[root_u_c2] = sub_size[root_u_c0] + sub_size[root_u_c1];
    for (int32_t u = root_u - 1; u >= n_leaves; --u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        tin[c1] = tin[u];
        tin[c2] = tin[u] + sub_size[c1];
    }
    for (int32_t x = 0; x < n_leaves; ++x) {
        dfs_order[tin[x]] = x;
    }

    // Zero-copy for leaves: u < n_leaves reads directly from D_perm, u >= n_leaves uses down_int!
    double *down_int = dwork;
    double *up = down_int + (root_u - n_leaves) * n_leaves;

    for (int32_t u = n_leaves; u < root_u; ++u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        const double * restrict r1 = (c1 < n_leaves) ? &D_perm[c1 * n_leaves] : &down_int[(c1 - n_leaves) * n_leaves];
        const double * restrict r2 = (c2 < n_leaves) ? &D_perm[c2 * n_leaves] : &down_int[(c2 - n_leaves) * n_leaves];
        double * restrict ru = &down_int[(u - n_leaves) * n_leaves];
        for (int32_t y = 0; y < n_leaves; ++y) {
            ru[y] = 0.5 * (r1[y] + r2[y]);
        }
    }

    {
        const double * restrict r0 = (root_u_c0 < n_leaves) ? &D_perm[root_u_c0 * n_leaves] : &down_int[(root_u_c0 - n_leaves) * n_leaves];
        const double * restrict r1 = (root_u_c1 < n_leaves) ? &D_perm[root_u_c1 * n_leaves] : &down_int[(root_u_c1 - n_leaves) * n_leaves];
        const double * restrict r2 = (root_u_c2 < n_leaves) ? &D_perm[root_u_c2 * n_leaves] : &down_int[(root_u_c2 - n_leaves) * n_leaves];
        int32_t end0 = tin[root_u_c0] + sub_size[root_u_c0];
        for (int32_t idx = tin[root_u_c0]; idx < end0; ++idx) {
            int32_t x = dfs_order[idx];
            up[x] = 0.5 * (r1[x] + r2[x]);
        }
        int32_t end1 = tin[root_u_c1] + sub_size[root_u_c1];
        for (int32_t idx = tin[root_u_c1]; idx < end1; ++idx) {
            int32_t x = dfs_order[idx];
            up[x] = 0.5 * (r0[x] + r2[x]);
        }
        int32_t end2 = tin[root_u_c2] + sub_size[root_u_c2];
        for (int32_t idx = tin[root_u_c2]; idx < end2; ++idx) {
            int32_t x = dfs_order[idx];
            up[x] = 0.5 * (r0[x] + r1[x]);
        }
    }
    for (int32_t u = root_u - 1; u >= n_leaves; --u) {
        int32_t c1 = child1_r[u], c2 = child2_r[u];
        const double * restrict r1 = (c1 < n_leaves) ? &D_perm[c1 * n_leaves] : &down_int[(c1 - n_leaves) * n_leaves];
        const double * restrict r2 = (c2 < n_leaves) ? &D_perm[c2 * n_leaves] : &down_int[(c2 - n_leaves) * n_leaves];
        int32_t end1 = tin[c1] + sub_size[c1];
        for (int32_t idx = tin[c1]; idx < end1; ++idx) {
            int32_t x = dfs_order[idx];
            up[x] = 0.5 * (up[x] + r2[x]);
        }
        int32_t end2 = tin[c2] + sub_size[c2];
        for (int32_t idx = tin[c2]; idx < end2; ++idx) {
            int32_t x = dfs_order[idx];
            up[x] = 0.5 * (up[x] + r1[x]);
        }
    }

    double total = 0.0;
    for (int32_t x = 0; x < n_leaves; ++x) total += up[x];
    return 0.5 * total;
}

/* ============================================================================
 * Encode any unrooted adjacency list adj_u (in original leaf labels 0..N-1)
 * into a Phylo2Vec vector v_out (0 <= v_out[i] <= 2i) in O(N) time.
 * ============================================================================ */
static void encode_unrooted_to_v(
    const int32_t * restrict adj_u,
    int32_t n_leaves,
    int64_t * restrict v_out
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;

    int32_t p0 = adj_u[0];
    int32_t stack[2048], parent_r[2048], order[2048];
    int32_t top = 0, ord_len = 0;
    for (int32_t i = 0; i < n_nodes_r; ++i) parent_r[i] = -1;

    int32_t c1_r[2048], c2_r[2048];
    c1_r[n_nodes_r - 1] = 0;
    c2_r[n_nodes_r - 1] = p0;
    parent_r[0] = n_nodes_r - 1;
    parent_r[p0] = n_nodes_r - 1;

    stack[top++] = p0;
    while (top > 0) {
        int32_t u = stack[--top];
        order[ord_len++] = u;
        if (u >= n_leaves) {
            int32_t ch[2], cnt = 0;
            for (int32_t k = 0; k < 3; ++k) {
                int32_t w = adj_u[3 * u + k];
                if (w != -1 && w != parent_r[u]) {
                    parent_r[w] = u;
                    ch[cnt++] = w;
                    stack[top++] = w;
                }
            }
            c1_r[u] = ch[0];
            c2_r[u] = ch[1];
        }
    }

    int32_t min_leaf[2048], l_max_node[2048], c_stay_node[2048], node_of_lmax[2048];
    for (int32_t u = 0; u < n_leaves; ++u) {
        min_leaf[u] = u;
        l_max_node[u] = 0;
    }
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = order[idx];
        if (u < n_leaves) continue;
        int32_t c1 = c1_r[u], c2 = c2_r[u];
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
    {
        int32_t u = n_nodes_r - 1;
        int32_t c1 = c1_r[u], c2 = c2_r[u];
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
    int32_t pair_lmax_order[2048];
    pair_lmax_order[0] = 1;
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
            memmove(&pair_lmax_order[1], &pair_lmax_order[0], (size_t)len_p * sizeof(int32_t));
            pair_lmax_order[0] = next_leaf;
        } else {
            int32_t target_lmax = l_max_node[w];
            int32_t idx = -1;
            for (int32_t k = 0; k < len_p; ++k) {
                if (pair_lmax_order[k] == target_lmax) {
                    idx = k + 1;
                    break;
                }
            }
            v_out[i] = i + idx;
            memmove(&pair_lmax_order[idx + 1], &pair_lmax_order[idx], (size_t)(len_p - idx) * sizeof(int32_t));
            pair_lmax_order[idx] = next_leaf;
        }
        len_p++;
    }
}

/* ============================================================================
 * O(N) Cherry-First Relabeling from child1_r, child2_r, min_leaf
 * ============================================================================ */
static void relabel_from_rooted_tree(
    const int32_t * restrict child1_r,
    const int32_t * restrict child2_r,
    const int32_t * restrict min_leaf,
    int32_t n_leaves,
    int32_t flip_mode,
    int64_t * restrict v_new,
    int32_t * restrict perm_in_out
) {
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    int32_t rep_leaf[2048];
    int32_t old_perm[1024];
    memcpy(old_perm, perm_in_out, (size_t)n_leaves * sizeof(int32_t));

    int32_t root_r = n_nodes_r - 1;
    int32_t rc1 = child1_r[root_r], rc2 = child2_r[root_r];
    if (min_leaf[rc1] < min_leaf[rc2]) {
        rep_leaf[rc1] = 0;
        rep_leaf[rc2] = 1;
    } else {
        rep_leaf[rc2] = 0;
        rep_leaf[rc1] = 1;
    }
    v_new[0] = 0;

    for (int32_t k = n_pairs - 2; k >= 0; --k) {
        int32_t m = n_leaves + k;
        int32_t r = rep_leaf[m];
        int32_t c1 = child1_r[m], c2 = child2_r[m];
        int32_t cond = (min_leaf[c1] < min_leaf[c2]);
        if (flip_mode && r != 0) cond = !cond;
        int32_t c_stay = cond ? c1 : c2;
        int32_t c_move = cond ? c2 : c1;
        int32_t i = n_pairs - 1 - k;
        rep_leaf[c_stay] = r;
        rep_leaf[c_move] = i + 1;
        v_new[i] = r;
    }

    for (int32_t l = 0; l < n_leaves; ++l) {
        perm_in_out[rep_leaf[l]] = old_perm[l];
    }
}

/* ============================================================================
 * Robinson-Foulds (RF) Distance between two unrooted trees adj1 and adj2
 * Uses 64-bit XOR split hashing of the N-3 internal edges!
 * ============================================================================ */
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static void extract_split_hashes(
    const int32_t * restrict adj_u,
    int32_t n_leaves,
    const uint64_t * restrict leaf_hash,
    uint64_t total_hash,
    uint64_t * restrict splits_out
) {
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;
    int32_t stack[2048], parent[2048], order[2048];
    uint64_t sub_hash[2048];
    for (int32_t i = 0; i < n_nodes_u; ++i) {
        parent[i] = -1;
        sub_hash[i] = (i < n_leaves) ? leaf_hash[i] : 0ULL;
    }
    int32_t top = 0, ord_len = 0;
    stack[top++] = root_u;
    parent[root_u] = root_u;
    while (top > 0) {
        int32_t u = stack[--top];
        order[ord_len++] = u;
        for (int32_t k = 0; k < 3; ++k) {
            int32_t w = adj_u[3 * u + k];
            if (w != -1 && parent[w] == -1) {
                parent[w] = u;
                stack[top++] = w;
            }
        }
    }
    int32_t s_cnt = 0;
    for (int32_t idx = ord_len - 1; idx >= 1; --idx) {
        int32_t u = order[idx];
        sub_hash[parent[u]] ^= sub_hash[u];
        if (u >= n_leaves) {
            uint64_t h1 = sub_hash[u];
            uint64_t h2 = total_hash ^ h1;
            splits_out[s_cnt++] = (h1 < h2) ? h1 : h2;
        }
    }
    qsort(splits_out, (size_t)s_cnt, sizeof(uint64_t), cmp_u64);
}

static double compute_rf_distance(
    const int32_t * restrict adj1,
    const int32_t * restrict adj2,
    int32_t n_leaves
) {
    if (n_leaves <= 3) return 0.0;
    uint64_t leaf_hash[1024], total_hash = 0ULL;
    uint64_t saved_rng = rng_state;
    rng_state = 999999999ULL;
    for (int32_t i = 0; i < n_leaves; ++i) {
        uint64_t h = ((uint64_t)xorshift32() << 32) | (uint64_t)xorshift32();
        leaf_hash[i] = h;
        total_hash ^= h;
    }
    rng_state = saved_rng;

    uint64_t s1[1024], s2[1024];
    extract_split_hashes(adj1, n_leaves, leaf_hash, total_hash, s1);
    extract_split_hashes(adj2, n_leaves, leaf_hash, total_hash, s2);

    int32_t n_splits = n_leaves - 3;
    int32_t shared = 0, i = 0, j = 0;
    while (i < n_splits && j < n_splits) {
        if (s1[i] == s2[j]) { shared++; i++; j++; }
        else if (s1[i] < s2[j]) i++;
        else j++;
    }
    return (double)(n_splits - shared) / (double)n_splits;
}

/* ============================================================================
 * Optimizer 1: Full Unrooted FastME SPR Hill-Climber in C99
 * ============================================================================ */
static double optimize_fastme_c(
    const int64_t * restrict v_init,
    const double * restrict D,
    int32_t n_leaves,
    int32_t * restrict adj_out,
    int32_t * restrict n_iters_out,
    double * restrict elapsed_ms_out,
    int32_t * restrict iwork,
    double * restrict dwork
) {
    double t0 = now_ms();
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;

    int32_t *pairs = iwork;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *siblings_uf = parents_uf + n_nodes_r;
    int32_t *parent_u = siblings_uf + n_nodes_r;
    int32_t *sub_size = parent_u + n_nodes_u;
    int32_t *tin = sub_size + n_nodes_u;
    int32_t *dfs_order = tin + n_nodes_u;
    int32_t *q_node = dfs_order + n_nodes_u;
    int32_t *q_prev = q_node + n_nodes_u;

    double *subdist = dwork;
    double *nd_u = subdist + n_nodes_u * n_nodes_u;
    double *edge_delta = nd_u + 3 * n_nodes_u;
    double *q_pow = edge_delta + n_nodes_u;
    double *q_dkc = q_pow + n_nodes_u;
    double *q_dist = q_dkc + n_nodes_u;

    // Decode v_init into initial unrooted adjacency adj_out
    pairs[0] = 0; pairs[1] = 1;
    int32_t curr_len = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        int32_t next_leaf = i + 1;
        int32_t vi = (int32_t)v_init[i];
        if (vi <= i) {
            memmove(&pairs[2], &pairs[0], (size_t)(2 * curr_len) * sizeof(int32_t));
            pairs[0] = vi; pairs[1] = next_leaf;
        } else {
            int32_t idx = vi - curr_len;
            int32_t c0 = pairs[2 * (idx - 1)];
            memmove(&pairs[2 * (idx + 1)], &pairs[2 * idx], (size_t)(2 * (curr_len - idx)) * sizeof(int32_t));
            pairs[2 * idx] = c0; pairs[2 * idx + 1] = next_leaf;
        }
        curr_len++;
    }
    for (int32_t i = 0; i < n_nodes_r; ++i) { parents_uf[i] = i; }
    int32_t next_parent = n_leaves;
    for (int32_t i = 0; i < n_pairs; ++i) {
        int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
        while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
        while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
        child1_r[next_parent] = p2; child2_r[next_parent] = p1;
        parents_uf[p1] = next_parent; parents_uf[p2] = next_parent;
        next_parent++;
    }
    int32_t rc1 = child1_r[n_nodes_r - 1], rc2 = child2_r[n_nodes_r - 1];
    int32_t other_root_child = (rc1 < rc2) ? rc1 : rc2;
    for (int32_t u = n_leaves; u < root_u; ++u) {
        parent_u[child1_r[u]] = u; parent_u[child2_r[u]] = u;
        adj_out[3 * u + 0] = child1_r[u]; adj_out[3 * u + 1] = child2_r[u];
    }
    parent_u[child1_r[root_u]] = root_u;
    parent_u[child2_r[root_u]] = root_u;
    parent_u[other_root_child] = root_u;
    for (int32_t u = 0; u < n_leaves; ++u) {
        adj_out[3 * u + 0] = parent_u[u]; adj_out[3 * u + 1] = -1; adj_out[3 * u + 2] = -1;
    }
    for (int32_t u = n_leaves; u < root_u; ++u) adj_out[3 * u + 2] = parent_u[u];
    adj_out[3 * root_u + 0] = child1_r[root_u];
    adj_out[3 * root_u + 1] = child2_r[root_u];
    adj_out[3 * root_u + 2] = other_root_child;

    int32_t n_iters = 0;
    double curr_L = 0.0;

    for (int32_t iter = 0; iter < 500; ++iter) {
        n_iters++;
        // 1. Build post-order / DFS intervals from root_u on adj_out
        for (int32_t u = 0; u < n_nodes_u; ++u) parent_u[u] = -2;
        int32_t top = 0, ord_len = 0;
        q_node[top++] = root_u;
        parent_u[root_u] = -1;
        while (top > 0) {
            int32_t u = q_node[--top];
            q_prev[ord_len++] = u;
            if (u >= n_leaves) {
                int32_t cnt = 0;
                for (int32_t k = 0; k < 3; ++k) {
                    int32_t w = adj_out[3 * u + k];
                    if (w != -1 && w != parent_u[u]) {
                        parent_u[w] = u;
                        if (u < root_u) {
                            if (cnt == 0) child1_r[u] = w;
                            else child2_r[u] = w;
                        }
                        cnt++;
                        q_node[top++] = w;
                    }
                }
            }
        }
        int32_t r_c0 = adj_out[3 * root_u + 0];
        int32_t r_c1 = adj_out[3 * root_u + 1];
        int32_t r_c2 = adj_out[3 * root_u + 2];

        for (int32_t u = 0; u < n_leaves; ++u) sub_size[u] = 1;
        for (int32_t idx = ord_len - 1; idx >= 1; --idx) {
            int32_t u = q_prev[idx];
            if (u >= n_leaves) sub_size[u] = 1 + sub_size[child1_r[u]] + sub_size[child2_r[u]];
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

        // 2. Compute subdist in 2 passes (post-order bottom-up + pre-order top-down)
        for (int32_t i = 0; i < n_leaves; ++i) {
            memcpy(&subdist[i * n_nodes_u], &D[i * n_leaves], (size_t)n_leaves * sizeof(double));
        }
        for (int32_t idx = ord_len - 1; idx >= 1; --idx) {
            int32_t u = q_prev[idx];
            if (u < n_leaves) continue;
            const double * restrict rc1_row = &subdist[child1_r[u] * n_nodes_u];
            const double * restrict rc2_row = &subdist[child2_r[u] * n_nodes_u];
            double * restrict ru_row = &subdist[u * n_nodes_u];
            // Compute against all already-computed nodes (leaves + post-order predecessors)
            for (int32_t leaf = 0; leaf < n_leaves; ++leaf) {
                double val = 0.5 * (rc1_row[leaf] + rc2_row[leaf]);
                ru_row[leaf] = val;
                subdist[leaf * n_nodes_u + u] = val;
            }
            for (int32_t j_idx = ord_len - 1; j_idx > idx; --j_idx) {
                int32_t v_node = q_prev[j_idx];
                if (v_node < n_leaves) continue;
                double val = 0.5 * (rc1_row[v_node] + rc2_row[v_node]);
                ru_row[v_node] = val;
                subdist[v_node * n_nodes_u + u] = val;
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
                double val = 0.5 * (row_c2[v_node] + row_g[v_node]);
                row_p[v_node] = val;
                subdist[v_node * n_nodes_u + p] = val;
            }
            int32_t end2 = tin[c2] + sub_size[c2];
            for (int32_t i = tin[c2]; i < end2; ++i) {
                int32_t v_node = dfs_order[i];
                double val = 0.5 * (row_c1[v_node] + row_g[v_node]);
                row_p[v_node] = val;
                subdist[v_node * n_nodes_u + p] = val;
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

        // 3. Full Unrooted FastME SPR Scan (all downward & upward subtrees)
        double best_delta = -1e-12;
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
            run_half_tree_bfs(other0, other1, anchor, n_leaves, adj_out, parent_u, nd_u, &subdist[u_prune * n_nodes_u], &subdist[other1 * n_nodes_u], edge_delta, q_node, q_prev, q_pow, q_dkc, q_dist);
            run_half_tree_bfs(other1, other0, anchor, n_leaves, adj_out, parent_u, nd_u, &subdist[u_prune * n_nodes_u], &subdist[other0 * n_nodes_u], edge_delta, q_node, q_prev, q_pow, q_dkc, q_dist);

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
            run_half_tree_bfs(other0, other1, anchor, n_leaves, adj_out, parent_u, nd_u, &subdist[p_prune * n_nodes_u], &subdist[other1 * n_nodes_u], edge_delta, q_node, q_prev, q_pow, q_dkc, q_dist);
            run_half_tree_bfs(other1, other0, anchor, n_leaves, adj_out, parent_u, nd_u, &subdist[p_prune * n_nodes_u], &subdist[other0 * n_nodes_u], edge_delta, q_node, q_prev, q_pow, q_dkc, q_dist);

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

    *n_iters_out = n_iters;
    *elapsed_ms_out = now_ms() - t0;
    return curr_L;
}

/* ============================================================================
 * Optimizer 3: Full Phylo2Vec BME Optimizer
 * - Full 0 <= j <= 2i Discrete Gradient on cherry-first (v_pi, perm)
 * - Path-disjoint Multi-Coordinate (K-SPR) batch updates in v_pi
 * - O(N) Cherry-First Taxon Relabeling + Complementary Subtree Verification
 * ============================================================================ */
static double optimize_phylo2vec_c(
    const int64_t * restrict v_init,
    const double * restrict D,
    int32_t n_leaves,
    int64_t * restrict v_out,
    int32_t * restrict adj_orig_out,
    int32_t * restrict n_grad_evals_out,
    double * restrict elapsed_ms_out,
    int32_t * restrict iwork,
    double * restrict dwork
) {
    double t0 = now_ms();
    const int32_t n_pairs = n_leaves - 1;
    const int32_t n_nodes_r = 2 * n_leaves - 1;
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;
    const int32_t max_cols = 2 * n_pairs - 1;

    int32_t perm[1024];
    int64_t v_pi[1024], v_cand[1024];
    for (int32_t i = 0; i < n_leaves; ++i) perm[i] = i;
    memcpy(v_pi, v_init, (size_t)n_pairs * sizeof(int64_t));

    int32_t *pairs = iwork;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *siblings_uf = parents_uf + n_nodes_r;
    int32_t *min_leaf = siblings_uf + n_nodes_r;
    int32_t *v_map = min_leaf + n_nodes_r;
    int32_t *destinations = v_map + n_leaves;
    int32_t *parent_u = destinations + n_pairs * n_leaves;
    int32_t *sub_size = parent_u + n_nodes_u;
    int32_t *tin = sub_size + n_nodes_u;
    int32_t *dfs_order = tin + n_nodes_u;
    int32_t *adj_u = dfs_order + n_nodes_u;
    int32_t *q_node = adj_u + 3 * n_nodes_u;
    int32_t *q_prev = q_node + n_nodes_u;
    int32_t *eval_iwork = q_prev + n_nodes_u;

    double *subdist = dwork;
    double *nd_u = subdist + n_nodes_u * n_nodes_u;
    double *edge_delta = nd_u + 3 * n_nodes_u;
    double *q_pow = edge_delta + n_nodes_u;
    double *q_dkc = q_pow + n_nodes_u;
    double *q_dist = q_dkc + n_nodes_u;
    double *D_perm = q_dist + n_nodes_u;
    double *eval_dwork = D_perm + n_leaves * n_leaves;

    memcpy(D_perm, D, (size_t)(n_leaves * n_leaves) * sizeof(double));
    int32_t perm_dirty = 0;

    int32_t n_evals = 0;
    double curr_L = 0.0;
    double best_saved_L = 1e30;
    int32_t has_saved_best = 0;
    int32_t saved_adj[6144], saved_perm[1024];
    int32_t flipped_already = 0;
    int32_t escape_attempts = 0;

    int32_t row_best_j[1024], row_best_dest[1024], improving_rows[1024];
    int32_t row_best_j_le[1024], row_best_dest_le[1024], improving_rows_le[1024];
    double row_best_val[1024], row_best_val_le[1024];
    int32_t near_rows[1024], near_j[1024];
    double near_val[1024];
    uint8_t used_node[2048];

    for (int32_t iter = 0; iter < 500; ++iter) {
        // 1. Decode v_pi into child1_r, child2_r
        int32_t has_gt_i = 0;
        for (int32_t i = 1; i < n_pairs; ++i) {
            if (v_pi[i] > i) { has_gt_i = 1; break; }
        }
        if (!has_gt_i) {
            for (int32_t k = 0; k < n_pairs; ++k) {
                int32_t i = n_pairs - 1 - k;
                pairs[2 * k] = (int32_t)v_pi[i];
                pairs[2 * k + 1] = i + 1;
            }
        } else {
            pairs[0] = 0; pairs[1] = 1;
            int32_t curr_len = 1;
            for (int32_t i = 1; i < n_pairs; ++i) {
                int32_t next_leaf = i + 1;
                int32_t vi = (int32_t)v_pi[i];
                if (vi <= i) {
                    memmove(&pairs[2], &pairs[0], (size_t)(2 * curr_len) * sizeof(int32_t));
                    pairs[0] = vi; pairs[1] = next_leaf;
                } else {
                    int32_t idx = vi - curr_len;
                    int32_t c0 = pairs[2 * (idx - 1)];
                    memmove(&pairs[2 * (idx + 1)], &pairs[2 * idx], (size_t)(2 * (curr_len - idx)) * sizeof(int32_t));
                    pairs[2 * idx] = c0; pairs[2 * idx + 1] = next_leaf;
                }
                curr_len++;
            }
        }

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

        // If v_pi had any v_pi[i] > i, canonicalize to cherry-first v_pi[i] <= i in O(N) time!
        if (has_gt_i) {
            relabel_from_rooted_tree(child1_r, child2_r, min_leaf, n_leaves, 0, v_pi, perm);
            perm_dirty = 1;
            for (int32_t k = 0; k < n_pairs; ++k) {
                int32_t i = n_pairs - 1 - k;
                pairs[2 * k] = (int32_t)v_pi[i];
                pairs[2 * k + 1] = i + 1;
            }
            for (int32_t i = 0; i < n_nodes_r; ++i) {
                parents_uf[i] = i;
                min_leaf[i] = i;
            }
            next_parent = n_leaves;
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
        }
        if (perm_dirty) {
            for (int32_t i = 0; i < n_leaves; ++i) {
                const double * restrict d_row = &D[perm[i] * n_leaves];
                double * restrict dp_row = &D_perm[i * n_leaves];
                for (int32_t j = 0; j < n_leaves; ++j) {
                    dp_row[j] = d_row[perm[j]];
                }
            }
            perm_dirty = 0;
        }

        // 2. Forward pass for v_map and destinations (including j = i!)
        for (int32_t i = 1; i < n_pairs; ++i) {
            int32_t *row_d = &destinations[i * n_leaves];
            for (int32_t j = 0; j <= i; ++j) row_d[j] = j;
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
                for (int32_t n_idx = l_min; n_idx < l_max - 1; ++n_idx) {
                    destinations[n_idx * n_leaves + l_min] = m;
                }
            }
        }

        // 3. Unrooted adjacency & Stackless Forward/Backward Matrix Pass for DFS intervals
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

        // 4. Two-pass O(N^2) matrix DP for subdist (contiguous memcpy from D_perm!)
        for (int32_t i = 0; i < n_leaves; ++i) {
            memcpy(&subdist[i * n_nodes_u], &D_perm[i * n_leaves], (size_t)n_leaves * sizeof(double));
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

        curr_L = 0.0;
        for (int32_t x = 0; x < n_leaves; ++x) {
            curr_L += subdist[x * n_nodes_u + parent_u[x]];
        }
        curr_L *= 0.5;
        n_evals++;

        // 5. Full Phylo2Vec Discrete Gradient Scan (all 0 <= j <= 2i)
        int32_t n_imp = 0, n_imp_le = 0, n_near = 0;
        double best_vmap_extra_delta = -1e-12;
        int32_t best_vmap_prune = -1, best_vmap_anchor = -1, best_vmap_dest = -1;

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

            run_half_tree_bfs(
                other0, other1, parent, n_leaves, adj_u, parent_u, nd_u,
                row_root, &subdist[other1 * n_nodes_u], edge_delta,
                q_node, q_prev, q_pow, q_dkc, q_dist
            );
            run_half_tree_bfs(
                other1, other0, parent, n_leaves, adj_u, parent_u, nd_u,
                row_root, &subdist[other0 * n_nodes_u], edge_delta,
                q_node, q_prev, q_pow, q_dkc, q_dist
            );

            int32_t i = v_index - 1;
            int32_t vi_val = (int32_t)v_pi[i];
            const int32_t * restrict dest_row = &destinations[i * n_leaves];

            double best_r_val = -1e-12;
            int32_t best_r_j = -1, best_r_dest = -1;
            double best_le_val = -1e-12;
            int32_t best_le_j = -1, best_le_dest = -1;
            double best_pos_val = 1e9;
            int32_t best_pos_j = -1;

            // Check 0 <= j <= i (leaf-clade destinations: preserve v_pi[k] <= k!)
            for (int32_t j = 0; j <= i; ++j) {
                if (j == vi_val) continue;
                int32_t du = dest_row[j];
                double val = edge_delta[du];
                if (val < best_le_val) {
                    best_le_val = val;
                    best_le_j = j;
                    best_le_dest = du;
                }
                if (val < best_r_val) {
                    best_r_val = val;
                    best_r_j = j;
                    best_r_dest = du;
                } else if (val > 1e-12 && val < best_pos_val) {
                    best_pos_val = val;
                    best_pos_j = j;
                }
            }
            // Check i < j <= 2i - 2 (internal branch destinations)
            for (int32_t j = i + 1; j <= 2 * i - 2; ++j) {
                int32_t du = (2 * n_leaves - 2) - 2 * i + j;
                double val = edge_delta[du];
                if (val < best_r_val) {
                    best_r_val = val;
                    best_r_j = j;
                    best_r_dest = du;
                } else if (val > 1e-12 && val < best_pos_val) {
                    best_pos_val = val;
                    best_pos_j = j;
                }
            }
            // Check j = 2i - 1 (root branch other_root_child)
            {
                int32_t j = 2 * i - 1;
                int32_t du = other_root_child;
                double val = edge_delta[du];
                if (val < best_r_val) {
                    best_r_val = val;
                    best_r_j = j;
                    best_r_dest = du;
                } else if (val > 1e-12 && val < best_pos_val) {
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
            if (best_le_j != -1) {
                row_best_j_le[i] = best_le_j;
                row_best_dest_le[i] = best_le_dest;
                row_best_val_le[i] = best_le_val;
                improving_rows_le[n_imp_le++] = i;
            }
            if (best_r_j == -1 && best_pos_j != -1 && best_pos_val < 0.05) {
                near_rows[n_near] = i;
                near_j[n_near] = best_pos_j;
                near_val[n_near] = best_pos_val;
                n_near++;
            }

            // Also track if any edge du outside 0..2i has a negative delta (for convergence check)
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

        if (n_imp > 0) {
            flipped_already = 0;
            // Sort improving rows by row_best_val ascending (most negative first)
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

            // Build a non-interfering multi-coordinate batch of moves across 0 <= j <= 2i
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
                    if (L_cand < curr_L + single_best_gain - 1e-12) {
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

        // 6. No improving move in 0 <= j <= 2i!
        if (best_vmap_prune != -1) {
            apply_unrooted_spr(adj_u, best_vmap_prune, best_vmap_anchor, best_vmap_dest, parent_u[best_vmap_dest]);
            encode_unrooted_to_v(adj_u, n_leaves, v_pi);
            flipped_already = 0;
            continue;
        }

        // Next, try O(N) Cherry-First Relabeling with flip_mode = 1 (swapping anchored & movable clades!)
        if (!flipped_already) {
            flipped_already = 1;
            relabel_from_rooted_tree(child1_r, child2_r, min_leaf, n_leaves, 1, v_pi, perm);
            perm_dirty = 1;
            continue;
        }

        // 7. Check remaining unpruned subtrees using already-computed subdist & nd_u
        uint8_t is_vmap[2048];
        memset(is_vmap, 0, (size_t)n_nodes_u * sizeof(uint8_t));
        for (int32_t v_idx = 2; v_idx < n_leaves; ++v_idx) is_vmap[v_map[v_idx]] = 1;

        double best_rem_delta = -1e-12;
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
            run_half_tree_bfs(other0, other1, anchor, n_leaves, adj_u, parent_u, nd_u, &subdist[u_prune * n_nodes_u], &subdist[other1 * n_nodes_u], edge_delta, q_node, q_prev, q_pow, q_dkc, q_dist);
            run_half_tree_bfs(other1, other0, anchor, n_leaves, adj_u, parent_u, nd_u, &subdist[u_prune * n_nodes_u], &subdist[other0 * n_nodes_u], edge_delta, q_node, q_prev, q_pow, q_dkc, q_dist);
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
            run_half_tree_bfs(other0, other1, anchor, n_leaves, adj_u, parent_u, nd_u, &subdist[p_prune * n_nodes_u], &subdist[other1 * n_nodes_u], edge_delta, q_node, q_prev, q_pow, q_dkc, q_dist);
            run_half_tree_bfs(other1, other0, anchor, n_leaves, adj_u, parent_u, nd_u, &subdist[p_prune * n_nodes_u], &subdist[other0 * n_nodes_u], edge_delta, q_node, q_prev, q_pow, q_dkc, q_dist);
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
            encode_unrooted_to_v(adj_u, n_leaves, v_pi);
            flipped_already = 0;
            continue;
        }

        // 8. At 1-SPR local minimum: save best tree found so far, and try a 1-step basin-hopping kick!
        if (curr_L < best_saved_L) {
            best_saved_L = curr_L;
            memcpy(saved_adj, adj_u, (size_t)(3 * n_nodes_u) * sizeof(int32_t));
            memcpy(saved_perm, perm, (size_t)n_leaves * sizeof(int32_t));
            has_saved_best = 1;
        }

        if (n_near >= 2 && escape_attempts < 2) {
            escape_attempts++;
            // Sort near_rows by near_val ascending (smallest positive delta first)
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
            int32_t idx1 = escape_attempts; // 1 on first attempt, 2 on second attempt
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

    // Reconstruct adj_orig_out in original leaf labels (0 .. N-1) and encode to v_out
    for (int32_t u = 0; u < n_nodes_u; ++u) {
        for (int32_t k = 0; k < 3; ++k) {
            int32_t w = adj_u[3 * u + k];
            int32_t w_orig = (w >= 0 && w < n_leaves) ? perm[w] : w;
            int32_t u_orig = (u < n_leaves) ? perm[u] : u;
            adj_orig_out[3 * u_orig + k] = w_orig;
        }
    }
    encode_unrooted_to_v(adj_orig_out, n_leaves, v_out);

    *n_grad_evals_out = n_evals;
    *elapsed_ms_out = now_ms() - t0;
    return curr_L;
}

/* ============================================================================
 * Generate Ground-Truth Tree + Additive Distance Matrix (+ optional noise)
 * ============================================================================ */
static void generate_phylogenetic_dataset(
    int32_t n_leaves,
    double noise_sigma,
    int64_t * restrict true_v,
    int32_t * restrict true_adj,
    double * restrict D_out
) {
    int32_t n_pairs = n_leaves - 1;
    int32_t n_nodes_r = 2 * n_leaves - 1;
    int32_t n_nodes_u = 2 * n_leaves - 2;
    int32_t root_u = n_nodes_u - 1;

    true_v[0] = 0;
    for (int32_t i = 1; i < n_pairs; ++i) {
        true_v[i] = (int64_t)(xorshift32() % (uint32_t)(2 * i + 1));
    }

    int32_t pairs[2048], child1_r[2048], child2_r[2048], parents_uf[2048], siblings_uf[2048], parent_u[2048];
    pairs[0] = 0; pairs[1] = 1;
    int32_t curr_len = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        int32_t next_leaf = i + 1;
        int32_t vi = (int32_t)true_v[i];
        if (vi <= i) {
            memmove(&pairs[2], &pairs[0], (size_t)(2 * curr_len) * sizeof(int32_t));
            pairs[0] = vi; pairs[1] = next_leaf;
        } else {
            int32_t idx = vi - curr_len;
            int32_t c0 = pairs[2 * (idx - 1)];
            memmove(&pairs[2 * (idx + 1)], &pairs[2 * idx], (size_t)(2 * (curr_len - idx)) * sizeof(int32_t));
            pairs[2 * idx] = c0; pairs[2 * idx + 1] = next_leaf;
        }
        curr_len++;
    }
    for (int32_t i = 0; i < n_nodes_r; ++i) { parents_uf[i] = i; }
    int32_t next_parent = n_leaves;
    for (int32_t i = 0; i < n_pairs; ++i) {
        int32_t p1 = pairs[2 * i], p2 = pairs[2 * i + 1];
        while (parents_uf[p1] != p1) { parents_uf[p1] = parents_uf[parents_uf[p1]]; p1 = parents_uf[p1]; }
        while (parents_uf[p2] != p2) { parents_uf[p2] = parents_uf[parents_uf[p2]]; p2 = parents_uf[p2]; }
        child1_r[next_parent] = p2; child2_r[next_parent] = p1;
        parents_uf[p1] = next_parent; parents_uf[p2] = next_parent;
        next_parent++;
    }
    int32_t rc1 = child1_r[n_nodes_r - 1], rc2 = child2_r[n_nodes_r - 1];
    int32_t other_root_child = (rc1 < rc2) ? rc1 : rc2;
    for (int32_t u = n_leaves; u < root_u; ++u) {
        parent_u[child1_r[u]] = u; parent_u[child2_r[u]] = u;
        true_adj[3 * u + 0] = child1_r[u]; true_adj[3 * u + 1] = child2_r[u];
    }
    parent_u[child1_r[root_u]] = root_u;
    parent_u[child2_r[root_u]] = root_u;
    parent_u[other_root_child] = root_u;
    for (int32_t u = 0; u < n_leaves; ++u) {
        true_adj[3 * u + 0] = parent_u[u]; true_adj[3 * u + 1] = -1; true_adj[3 * u + 2] = -1;
    }
    for (int32_t u = n_leaves; u < root_u; ++u) true_adj[3 * u + 2] = parent_u[u];
    true_adj[3 * root_u + 0] = child1_r[root_u];
    true_adj[3 * root_u + 1] = child2_r[root_u];
    true_adj[3 * root_u + 2] = other_root_child;

    // Assign random branch lengths in [0.05, 0.25] to each edge (u, parent_u[u]) for u = 0 .. root_u - 1
    double br_len[2048];
    for (int32_t u = 0; u < root_u; ++u) {
        br_len[u] = 0.05 + 0.20 * rand_uniform();
    }

    // BFS from each leaf to compute true pairwise tree distances
    int32_t q[2048];
    double dist[2048];
    for (int32_t src = 0; src < n_leaves; ++src) {
        for (int32_t k = 0; k < n_nodes_u; ++k) dist[k] = -1.0;
        dist[src] = 0.0;
        q[0] = src;
        int32_t head = 0, tail = 1;
        while (head < tail) {
            int32_t u = q[head++];
            double du = dist[u];
            for (int32_t k = 0; k < 3; ++k) {
                int32_t w = true_adj[3 * u + k];
                if (w != -1 && dist[w] < 0.0) {
                    int32_t edge_idx = (parent_u[u] == w) ? u : w;
                    dist[w] = du + br_len[edge_idx];
                    q[tail++] = w;
                }
            }
        }
        D_out[src * n_leaves + src] = 0.0;
        for (int32_t dst = src + 1; dst < n_leaves; ++dst) {
            double d = dist[dst];
            if (noise_sigma > 0.0) {
                double noise = (rand_uniform() + rand_uniform() + rand_uniform() - 1.5) * 1.632993;
                d *= (1.0 + noise_sigma * noise);
                if (d < 0.001) d = 0.001;
            }
            D_out[src * n_leaves + dst] = d;
            D_out[dst * n_leaves + src] = d;
        }
    }
}

int main(void) {
    int32_t max_N = 250;
    int32_t *iwork = (int32_t *)malloc(32 * 1024 * 1024);
    double *dwork = (double *)malloc(64 * 1024 * 1024);

    const int32_t sizes[] = {20, 50, 100, 150, 200};
    const double noises[] = {0.0, 0.15};
    const char *noise_names[] = {
        "Noiseless Additive Tree Distances (sigma = 0.00, Exact Recovery Test)",
        "Noisy Phylogenetic Distances (sigma = 0.15, Realistic Estimation Noise)"
    };

    for (int32_t ns = 0; ns < 2; ++ns) {
        double sigma = noises[ns];
        printf("\n========================================================================================================\n");
        printf("BENCHMARK %d: %s\n", ns + 1, noise_names[ns]);
        printf("========================================================================================================\n");
        printf("%5s | %11s %11s %8s | %8s %8s | %8s %8s | %9s %9s %8s\n",
               "N", "FastME L", "P2V L", "P2V<=FME", "FME RF", "P2V RF", "FME Iters", "P2V Evals", "FME (ms)", "P2V (ms)", "Speedup");
        printf("--------------------------------------------------------------------------------------------------------\n");

        for (int32_t s = 0; s < 5; ++s) {
            int32_t N = sizes[s];
            int32_t n_trials = (N <= 50) ? 20 : ((N <= 100) ? 10 : 5);

            double sum_L_fme = 0.0, sum_L_p2v = 0.0;
            double sum_rf_fme = 0.0, sum_rf_p2v = 0.0;
            double sum_it_fme = 0.0, sum_ev_p2v = 0.0;
            double sum_ms_fme = 0.0, sum_ms_p2v = 0.0;
            int32_t p2v_wins_or_ties = 0;

            int64_t *true_v = (int64_t *)malloc((size_t)N * sizeof(int64_t));
            int64_t *v_init = (int64_t *)malloc((size_t)N * sizeof(int64_t));
            int64_t *v_p2v = (int64_t *)malloc((size_t)N * sizeof(int64_t));
            int32_t *true_adj = (int32_t *)malloc((size_t)(6 * N) * sizeof(int32_t));
            int32_t *adj_fme = (int32_t *)malloc((size_t)(6 * N) * sizeof(int32_t));
            int32_t *adj_p2v = (int32_t *)malloc((size_t)(6 * N) * sizeof(int32_t));
            double *D = (double *)malloc((size_t)(N * N) * sizeof(double));

            for (int32_t t = 0; t < n_trials; ++t) {
                generate_phylogenetic_dataset(N, sigma, true_v, true_adj, D);
                // Random starting tree v_init
                v_init[0] = 0;
                for (int32_t i = 1; i < N - 1; ++i) {
                    v_init[i] = (int64_t)(xorshift32() % (uint32_t)(i + 1));
                }

                int32_t it_fme = 0, ev_p2v = 0;
                double ms_fme = 0.0, ms_p2v = 0.0;

                double L_fme = optimize_fastme_c(v_init, D, N, adj_fme, &it_fme, &ms_fme, iwork, dwork);
                double L_p2v = optimize_phylo2vec_c(v_init, D, N, v_p2v, adj_p2v, &ev_p2v, &ms_p2v, iwork, dwork);

                double rf_fme = compute_rf_distance(true_adj, adj_fme, N);
                double rf_p2v = compute_rf_distance(true_adj, adj_p2v, N);

                sum_L_fme += L_fme;
                sum_L_p2v += L_p2v;
                sum_rf_fme += rf_fme;
                sum_rf_p2v += rf_p2v;
                sum_it_fme += (double)it_fme;
                sum_ev_p2v += (double)ev_p2v;
                sum_ms_fme += ms_fme;
                sum_ms_p2v += ms_p2v;
                if (L_p2v <= L_fme + 1e-9) p2v_wins_or_ties++;
            }

            printf("%5d | %11.5f %11.5f %6d/%-2d | %8.4f %8.4f | %9.1f %9.1f | %9.3f %9.3f %7.2fx\n",
                   N,
                   sum_L_fme / n_trials,
                   sum_L_p2v / n_trials,
                   p2v_wins_or_ties, n_trials,
                   sum_rf_fme / n_trials,
                   sum_rf_p2v / n_trials,
                   sum_it_fme / n_trials,
                   sum_ev_p2v / n_trials,
                   sum_ms_fme / n_trials,
                   sum_ms_p2v / n_trials,
                   sum_ms_fme / sum_ms_p2v);

            free(true_v); free(v_init); free(v_p2v);
            free(true_adj); free(adj_fme); free(adj_p2v); free(D);
        }
    }

    free(iwork);
    free(dwork);
    return 0;
}
