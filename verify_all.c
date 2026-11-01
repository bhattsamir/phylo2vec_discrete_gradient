#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/*
 * Strict Verification Suite for Phylo2Vec Discrete Gradient & BME Optimizer
 *
 * Enforces finite checks (NaN/Inf detection), strict numerical tolerances,
 * canonical unrooted split (bipartition) isomorphism, fault-injection self-testing,
 * repeated-minimum-leaf destination-mapping edge cases, and N > 1024 boundary tests.
 * Returns non-zero exit status on any failure.
 *
 * Compile (without -ffast-math so IEEE 754 NaN/Inf checks are active):
 *   /usr/bin/clang -O2 -march=native verify_all.c phylo2vec_fastme.c -o verify_all && ./verify_all
 */

extern void calculate_discrete_gradient_c(
    const int64_t *v, const double *D, int32_t n_leaves, double *delta_out
);
extern void calculate_discrete_gradient_full_c(
    const int64_t *v, const double *D, int32_t n_leaves, double *delta_full_out
);
extern void relabel_cherry_first_c(
    const int64_t *v_in, int32_t n_leaves, int32_t flip_mode, int64_t *v_out, int32_t *perm_out
);
extern void encode_rooted_to_v_c(
    const int32_t *child1_r, const int32_t *child2_r, int32_t n_leaves, int64_t *v_out
);
extern void encode_unrooted_to_v_c(
    const int32_t *adj_u, int32_t n_leaves, int64_t *v_out
);
extern double optimize_phylo2vec_bme_c(
    const int64_t *v_init, const double *D, int32_t n_leaves,
    int64_t *v_out, int32_t *n_grad_evals_out, double *elapsed_ms_out
);

static uint32_t rng_state = 20261101U;
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

static inline int is_finite_double(double x) {
    uint64_t bits;
    memcpy(&bits, &x, sizeof(uint64_t));
    return ((bits & 0x7FF0000000000000ULL) != 0x7FF0000000000000ULL);
}

/* Decode any v in V (0 <= v_i <= 2i) into rooted child1_r, child2_r and unrooted adj_u */
static void decode_v_to_trees(
    const int64_t *v, int32_t n_leaves,
    int32_t *child1_r_out, int32_t *child2_r_out, int32_t *adj_u_out
) {
    int32_t n_pairs = n_leaves - 1;
    int32_t n_nodes_r = 2 * n_leaves - 1;
    int32_t n_nodes_u = 2 * n_leaves - 2;
    int32_t root_u = n_nodes_u - 1;

    int32_t *buf = (int32_t *)malloc((size_t)(2 * n_pairs + 4 * n_nodes_r + 3 * n_nodes_u) * sizeof(int32_t));
    int32_t *pairs = buf;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *parent_u = parents_uf + n_nodes_r;
    int32_t *adj_u = parent_u + n_nodes_r;

    pairs[0] = 0; pairs[1] = 1;
    int32_t curr_len = 1;
    for (int32_t i = 1; i < n_pairs; ++i) {
        int32_t next_leaf = i + 1;
        int32_t vi = (int32_t)v[i];
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

    if (child1_r_out) memcpy(child1_r_out, child1_r, (size_t)n_nodes_r * sizeof(int32_t));
    if (child2_r_out) memcpy(child2_r_out, child2_r, (size_t)n_nodes_r * sizeof(int32_t));
    if (adj_u_out) memcpy(adj_u_out, adj_u, (size_t)(3 * n_nodes_u) * sizeof(int32_t));
    free(buf);
}

/* Brute-force Pauplin BME Tree Length L(T(v)) = sum_{x < y} 2^{-(d_T(x,y)-1)} D[x,y] */
static double brute_force_pauplin_bme_length(const int64_t *v, const double *D, int32_t n_leaves, int32_t *adj_u_out) {
    int32_t n_nodes_u = 2 * n_leaves - 2;
    int32_t *adj_u = (int32_t *)malloc((size_t)(3 * n_nodes_u) * sizeof(int32_t));
    int32_t *dist = (int32_t *)malloc((size_t)n_nodes_u * sizeof(int32_t));
    int32_t *q = (int32_t *)malloc((size_t)n_nodes_u * sizeof(int32_t));

    decode_v_to_trees(v, n_leaves, NULL, NULL, adj_u);
    if (adj_u_out) memcpy(adj_u_out, adj_u, (size_t)(3 * n_nodes_u) * sizeof(int32_t));

    double total_L = 0.0;
    for (int32_t src = 0; src < n_leaves; ++src) {
        for (int32_t u = 0; u < n_nodes_u; ++u) dist[u] = -1;
        int32_t head = 0, tail = 0;
        dist[src] = 0;
        q[tail++] = src;
        while (head < tail) {
            int32_t u = q[head++];
            for (int32_t k = 0; k < 3; ++k) {
                int32_t nb = adj_u[3 * u + k];
                if (nb != -1 && dist[nb] == -1) {
                    dist[nb] = dist[u] + 1;
                    q[tail++] = nb;
                }
            }
        }
        for (int32_t dst = src + 1; dst < n_leaves; ++dst) {
            total_L += ldexp(D[(size_t)src * n_leaves + dst], -(dist[dst] - 1));
        }
    }
    free(adj_u);
    free(dist);
    free(q);
    return total_L;
}

/* Exact Canonical Split (Bipartition) Comparison for Unrooted Trees (N <= 64) */
static int cmp_u64(const void *a, const void *b) {
    uint64_t va = *(const uint64_t *)a;
    uint64_t vb = *(const uint64_t *)b;
    return (va > vb) - (va < vb);
}

static void extract_canonical_splits_u64(const int32_t *adj_u, int32_t n_leaves, uint64_t *splits_out) {
    int32_t n_nodes_u = 2 * n_leaves - 2;
    int32_t root_u = n_nodes_u - 1;
    int32_t q[256], parent[256];
    uint64_t mask[256];
    for (int32_t u = 0; u < n_nodes_u; ++u) { parent[u] = -2; mask[u] = 0ULL; }
    int32_t head = 0, tail = 0;
    q[tail++] = root_u;
    parent[root_u] = -1;
    while (head < tail) {
        int32_t u = q[head++];
        for (int32_t k = 0; k < 3; ++k) {
            int32_t nb = adj_u[3 * u + k];
            if (nb != -1 && parent[nb] == -2) {
                parent[nb] = u;
                q[tail++] = nb;
            }
        }
    }
    uint64_t full_mask = (n_leaves == 64) ? ~0ULL : ((1ULL << n_leaves) - 1ULL);
    for (int32_t idx = tail - 1; idx >= 0; --idx) {
        int32_t u = q[idx];
        if (u < n_leaves) {
            mask[u] = (1ULL << u);
        } else {
            uint64_t m = 0ULL;
            for (int32_t k = 0; k < 3; ++k) {
                int32_t nb = adj_u[3 * u + k];
                if (nb != -1 && parent[nb] == u) m |= mask[nb];
            }
            mask[u] = m;
        }
    }
    int32_t n_int = 0;
    for (int32_t u = n_leaves; u < root_u; ++u) {
        uint64_t m = mask[u];
        if ((m & 1ULL) == 0ULL) m = full_mask ^ m;
        splits_out[n_int++] = m;
    }
    qsort(splits_out, (size_t)n_int, sizeof(uint64_t), cmp_u64);
}

/* Validator helper used both for real gradient verification and fault-injection self-test */
static int validate_gradient_matrix(
    const double *delta_pred, const double *delta_exact,
    int32_t n_pairs, int32_t max_cols, double tol, double *max_err_out
) {
    double max_err = 0.0;
    for (int32_t i = 1; i < n_pairs; ++i) {
        for (int32_t j = 0; j <= 2 * i; ++j) {
            double pred = delta_pred[i * max_cols + j];
            double exact = delta_exact[i * max_cols + j];
            if (!is_finite_double(pred) || !is_finite_double(exact)) {
                *max_err_out = INFINITY;
                return 0;
            }
            double err = fabs(pred - exact);
            if (err > max_err) max_err = err;
        }
    }
    *max_err_out = max_err;
    return (max_err <= tol) ? 1 : 0;
}

int main(void) {
    printf("================================================================================\n");
    printf("PHYLO2VEC DISCRETE GRADIENT & BIJECTION STRICT VERIFICATION SUITE\n");
    printf("================================================================================\n\n");

    int all_passed = 1;

    /* TEST 1: Full 0 <= j <= 2i Discrete Gradient vs Brute-Force Pauplin BME + Fault Injection */
    {
        const int32_t N = 20;
        const int32_t n_pairs = N - 1;
        const int32_t max_cols = 2 * n_pairs - 1;
        double D[400], delta_full[1024], delta_exact[1024], delta_lt[1024];
        int64_t v[32], v_mut[32];

        double max_err_full = 0.0;
        double max_err_lt = 0.0;
        int32_t total_entries_checked = 0;
        int test1_ok = 1;

        for (int32_t trial = 0; trial < 100; ++trial) {
            for (int32_t i = 0; i < N; ++i) {
                D[i * N + i] = 0.0;
                for (int32_t j = i + 1; j < N; ++j) {
                    double d = 0.1 + 2.0 * rand_uniform();
                    D[i * N + j] = d;
                    D[j * N + i] = d;
                }
            }
            v[0] = 0;
            for (int32_t i = 1; i < n_pairs; ++i) {
                v[i] = (int64_t)(xorshift32() % (uint32_t)(i + 1));
            }

            calculate_discrete_gradient_full_c(v, D, N, delta_full);
            calculate_discrete_gradient_c(v, D, N, delta_lt);
            double L_base = brute_force_pauplin_bme_length(v, D, N, NULL);

            for (int32_t i = 1; i < n_pairs; ++i) {
                /* Current coordinate delta must be 0 */
                if (fabs(delta_full[i * max_cols + v[i]]) > 1e-12) {
                    test1_ok = 0;
                }
                for (int32_t j = 0; j <= 2 * i; ++j) {
                    memcpy(v_mut, v, (size_t)n_pairs * sizeof(int64_t));
                    v_mut[i] = j;
                    double L_mut = brute_force_pauplin_bme_length(v_mut, D, N, NULL);
                    double exact_diff = L_mut - L_base;
                    delta_exact[i * max_cols + j] = exact_diff;

                    if (j < i) {
                        double pred_lt = delta_lt[i * (n_pairs - 1) + j];
                        if (!is_finite_double(pred_lt)) test1_ok = 0;
                        double err_lt = fabs(pred_lt - exact_diff);
                        if (err_lt > max_err_lt) max_err_lt = err_lt;
                    }
                    total_entries_checked++;
                }
            }
            double trial_err = 0.0;
            if (!validate_gradient_matrix(delta_full, delta_exact, n_pairs, max_cols, 1e-9, &trial_err)) {
                test1_ok = 0;
            }
            if (trial_err > max_err_full) max_err_full = trial_err;
        }
        if (max_err_lt > 1e-9) test1_ok = 0;

        /* Fault-injection self-test: corrupt one entry by +1.0 and another by NaN and verify rejection */
        double corrupted[1024];
        memcpy(corrupted, delta_full, sizeof(delta_full));
        corrupted[1 * max_cols + 0] += 1.0;
        double fault_err = 0.0;
        int fault_caught_plus1 = !validate_gradient_matrix(corrupted, delta_exact, n_pairs, max_cols, 1e-9, &fault_err);
        corrupted[1 * max_cols + 0] = NAN;
        int fault_caught_nan = !validate_gradient_matrix(corrupted, delta_exact, n_pairs, max_cols, 1e-9, &fault_err);
        if (!fault_caught_plus1 || !fault_caught_nan) test1_ok = 0;

        printf("[%s] Test 1: Full 0 <= j <= 2i Discrete Gradient (%d entries across 100 trees, N=%d)\n",
               test1_ok ? "PASS" : "FAIL", total_entries_checked, N);
        printf("       Max absolute error (0 <= j <= 2i): %.3e (tol: 1.000e-09)\n", max_err_full);
        printf("       Max absolute error (0 <= j <  i ): %.3e (tol: 1.000e-09)\n", max_err_lt);
        printf("       Fault-injection self-test (+1.0 and NaN rejected): %s\n\n",
               (fault_caught_plus1 && fault_caught_nan) ? "YES" : "NO");
        if (!test1_ok) all_passed = 0;
    }

    /* TEST 2: Repeated-Minimum-Leaf & Root-Overlap Regression Cases (N = 6) */
    {
        const int32_t N = 6;
        const int32_t n_pairs = N - 1;
        const int32_t max_cols = 2 * n_pairs - 1;
        double D1[36], D2[36], delta1[64], delta2[64];
        int64_t v1[5] = {0, 0, 0, 0, 0};
        int64_t v2[5] = {0, 1, 1, 1, 1};
        int64_t v_mut[5];

        for (int32_t a = 0; a < N; ++a) {
            for (int32_t b = 0; b < N; ++b) {
                D1[a * N + b] = (a == b) ? 0.0 : 1.0;
                D2[a * N + b] = (a == b) ? 0.0 : 1.0;
            }
        }
        D1[0 * N + 2] = D1[2 * N + 0] = 10.0;
        D2[1 * N + 2] = D2[2 * N + 1] = 10.0;

        calculate_discrete_gradient_full_c(v1, D1, N, delta1);
        calculate_discrete_gradient_full_c(v2, D2, N, delta2);

        double max_err_ce = 0.0;
        int ce_ok = 1;
        for (int32_t case_id = 0; case_id < 2; ++case_id) {
            const int64_t *v = (case_id == 0) ? v1 : v2;
            const double *D = (case_id == 0) ? D1 : D2;
            const double *delta = (case_id == 0) ? delta1 : delta2;
            double L_base = brute_force_pauplin_bme_length(v, D, N, NULL);
            for (int32_t i = 1; i < n_pairs; ++i) {
                if (fabs(delta[i * max_cols + v[i]]) > 1e-14) ce_ok = 0;
                for (int32_t j = 0; j <= 2 * i; ++j) {
                    memcpy(v_mut, v, sizeof(v_mut));
                    v_mut[i] = j;
                    double L_mut = brute_force_pauplin_bme_length(v_mut, D, N, NULL);
                    double err = fabs(delta[i * max_cols + j] - (L_mut - L_base));
                    if (err > max_err_ce) max_err_ce = err;
                }
            }
        }
        if (max_err_ce > 1e-12) ce_ok = 0;
        printf("[%s] Test 2: Repeated-Minimum-Leaf & i=1 Root-Overlap Edge Cases (N=6)\n",
               ce_ok ? "PASS" : "FAIL");
        printf("       v=[0,0,0,0,0] and v=[0,1,1,1,1] max error vs Pauplin: %.3e\n\n", max_err_ce);
        if (!ce_ok) all_passed = 0;
    }

    /* TEST 3: Cherry-First Gauge Relabeling Theorem (flip_mode = 0 and flip_mode = 1, N = 25) */
    {
        const int32_t N = 25;
        const int32_t n_pairs = N - 1;
        double D[625], D_perm[625];
        int64_t v_gen[32], v_cf[32];
        int32_t perm[32], seen[32];

        double max_err_mode0 = 0.0, max_err_mode1 = 0.0;
        int test3_ok = 1;

        for (int32_t trial = 0; trial < 200; ++trial) {
            for (int32_t i = 0; i < N; ++i) {
                D[i * N + i] = 0.0;
                for (int32_t j = i + 1; j < N; ++j) {
                    double d = 0.1 + 2.0 * rand_uniform();
                    D[i * N + j] = d;
                    D[j * N + i] = d;
                }
            }
            v_gen[0] = 0;
            for (int32_t i = 1; i < n_pairs; ++i) {
                v_gen[i] = (int64_t)(xorshift32() % (uint32_t)(2 * i + 1));
            }
            double L_orig = brute_force_pauplin_bme_length(v_gen, D, N, NULL);

            for (int32_t mode = 0; mode <= 1; ++mode) {
                relabel_cherry_first_c(v_gen, N, mode, v_cf, perm);
                for (int32_t i = 0; i < n_pairs; ++i) {
                    if (v_cf[i] < 0 || v_cf[i] > i) test3_ok = 0;
                }
                memset(seen, 0, sizeof(seen));
                for (int32_t a = 0; a < N; ++a) {
                    if (perm[a] < 0 || perm[a] >= N || seen[perm[a]]) test3_ok = 0;
                    else seen[perm[a]] = 1;
                }
                for (int32_t a = 0; a < N; ++a) {
                    for (int32_t b = 0; b < N; ++b) {
                        D_perm[a * N + b] = D[perm[a] * N + perm[b]];
                    }
                }
                double L_cf = brute_force_pauplin_bme_length(v_cf, D_perm, N, NULL);
                if (!is_finite_double(L_cf)) test3_ok = 0;
                double err = fabs(L_cf - L_orig);
                if (mode == 0 && err > max_err_mode0) max_err_mode0 = err;
                if (mode == 1 && err > max_err_mode1) max_err_mode1 = err;
            }
        }
        if (max_err_mode0 > 1e-9 || max_err_mode1 > 1e-9) test3_ok = 0;
        printf("[%s] Test 3: Cherry-First Taxon Relabeling Theorem (200 general trees, N=%d)\n",
               test3_ok ? "PASS" : "FAIL", N);
        printf("       All relabeled vectors satisfy 0 <= v_cf[i] <= i and valid permutation: %s\n",
               test3_ok ? "YES (100%)" : "NO");
        printf("       Max BME length discrepancy (flip_mode = 0): %.3e\n", max_err_mode0);
        printf("       Max BME length discrepancy (flip_mode = 1): %.3e\n\n", max_err_mode1);
        if (!test3_ok) all_passed = 0;
    }

    /* TEST 4: Exact Tree-to-Vector Inverse Bijection (Rooted memcmp & Unrooted Split Isomorphism, N = 40) */
    {
        const int32_t N = 40;
        const int32_t n_pairs = N - 1;
        const int32_t n_int_splits = N - 3;
        double D[1600];
        int64_t v_orig[64], v_rec_rooted[64], v_rec_unrooted[64];
        int32_t adj_u_orig[512], adj_u_rec[512], child1_r[128], child2_r[128];
        uint64_t splits_orig[64], splits_rec[64];
        int32_t exact_rooted_matches = 0, exact_unrooted_splits = 0;
        double max_unrooted_L_err = 0.0;
        int test4_ok = 1;

        for (int32_t i = 0; i < N; ++i) {
            D[i * N + i] = 0.0;
            for (int32_t j = i + 1; j < N; ++j) {
                double d = 0.1 + 2.0 * rand_uniform();
                D[i * N + j] = d;
                D[j * N + i] = d;
            }
        }

        for (int32_t trial = 0; trial < 1000; ++trial) {
            v_orig[0] = 0;
            for (int32_t i = 1; i < n_pairs; ++i) {
                v_orig[i] = (int64_t)(xorshift32() % (uint32_t)(2 * i + 1));
            }
            decode_v_to_trees(v_orig, N, child1_r, child2_r, adj_u_orig);

            encode_rooted_to_v_c(child1_r, child2_r, N, v_rec_rooted);
            if (memcmp(v_orig, v_rec_rooted, (size_t)n_pairs * sizeof(int64_t)) == 0) {
                exact_rooted_matches++;
            } else {
                test4_ok = 0;
            }

            double L_orig = brute_force_pauplin_bme_length(v_orig, D, N, adj_u_orig);
            encode_unrooted_to_v_c(adj_u_orig, N, v_rec_unrooted);
            for (int32_t i = 0; i < n_pairs; ++i) {
                if (v_rec_unrooted[i] < 0 || v_rec_unrooted[i] > 2 * i) test4_ok = 0;
            }
            double L_unrooted = brute_force_pauplin_bme_length(v_rec_unrooted, D, N, adj_u_rec);
            double err = fabs(L_orig - L_unrooted);
            if (err > max_unrooted_L_err) max_unrooted_L_err = err;

            extract_canonical_splits_u64(adj_u_orig, N, splits_orig);
            extract_canonical_splits_u64(adj_u_rec, N, splits_rec);
            if (memcmp(splits_orig, splits_rec, (size_t)n_int_splits * sizeof(uint64_t)) == 0 && err < 1e-11) {
                exact_unrooted_splits++;
            } else {
                test4_ok = 0;
            }
        }
        printf("[%s] Test 4: Exact Tree-to-Vector Inverse Bijection (1,000 trees, N=%d)\n",
               test4_ok ? "PASS" : "FAIL", N);
        printf("       Rooted tree -> bit-for-bit v recovery (0 <= v_i <= 2i):   %d / 1000\n", exact_rooted_matches);
        printf("       Unrooted adj_u -> identical canonical split topology:     %d / 1000 (max L diff: %.3e)\n\n",
               exact_unrooted_splits, max_unrooted_L_err);
        if (!test4_ok) all_passed = 0;
    }

    /* TEST 5: Dynamic Allocation Beyond N = 1,024 Boundary (N = 1024, 1025, 1026) */
    {
        int32_t test_sizes[3] = {1024, 1025, 1026};
        int test5_ok = 1;
        for (int s = 0; s < 3; ++s) {
            int32_t N = test_sizes[s];
            int32_t n_pairs = N - 1;
            int32_t n_nodes_r = 2 * N - 1;
            int32_t n_nodes_u = 2 * N - 2;
            int64_t *v_in = (int64_t *)malloc((size_t)n_pairs * sizeof(int64_t));
            int64_t *v_cf = (int64_t *)malloc((size_t)n_pairs * sizeof(int64_t));
            int64_t *v_rec = (int64_t *)malloc((size_t)n_pairs * sizeof(int64_t));
            int32_t *perm = (int32_t *)malloc((size_t)N * sizeof(int32_t));
            int32_t *c1_r = (int32_t *)malloc((size_t)n_nodes_r * sizeof(int32_t));
            int32_t *c2_r = (int32_t *)malloc((size_t)n_nodes_r * sizeof(int32_t));
            int32_t *adj_u = (int32_t *)malloc((size_t)(3 * n_nodes_u) * sizeof(int32_t));

            v_in[0] = 0;
            for (int32_t i = 1; i < n_pairs; ++i) {
                v_in[i] = (int64_t)(xorshift32() % (uint32_t)(2 * i + 1));
            }
            relabel_cherry_first_c(v_in, N, 0, v_cf, perm);
            for (int32_t i = 0; i < n_pairs; ++i) {
                if (v_cf[i] < 0 || v_cf[i] > i) test5_ok = 0;
            }
            decode_v_to_trees(v_in, N, c1_r, c2_r, adj_u);
            encode_rooted_to_v_c(c1_r, c2_r, N, v_rec);
            if (memcmp(v_in, v_rec, (size_t)n_pairs * sizeof(int64_t)) != 0) test5_ok = 0;

            encode_unrooted_to_v_c(adj_u, N, v_rec);
            for (int32_t i = 0; i < n_pairs; ++i) {
                if (v_rec[i] < 0 || v_rec[i] > 2 * i) test5_ok = 0;
            }

            free(v_in); free(v_cf); free(v_rec); free(perm);
            free(c1_r); free(c2_r); free(adj_u);
        }

        /* Also run end-to-end optimize_phylo2vec_bme_c at N = 1025 on a structured distance matrix */
        {
            int32_t N = 1025;
            int32_t n_pairs = N - 1;
            int64_t *v_init = (int64_t *)calloc((size_t)n_pairs, sizeof(int64_t));
            int64_t *v_out  = (int64_t *)calloc((size_t)n_pairs, sizeof(int64_t));
            double  *D      = (double *)malloc((size_t)N * (size_t)N * sizeof(double));
            for (int32_t i = 0; i < N; ++i) {
                if (i < n_pairs) v_init[i] = (i == 0) ? 0 : (i - 1);
                for (int32_t j = 0; j < N; ++j) {
                    D[(size_t)i * N + j] = (i == j) ? 0.0 : 1.0;
                }
            }
            int32_t n_evals = 0;
            double ms = 0.0;
            double L_opt = optimize_phylo2vec_bme_c(v_init, D, N, v_out, &n_evals, &ms);
            if (!is_finite_double(L_opt) || L_opt <= 0.0 || n_evals < 1) test5_ok = 0;
            for (int32_t i = 0; i < n_pairs; ++i) {
                if (v_out[i] < 0 || v_out[i] > 2 * i) test5_ok = 0;
            }
            free(v_init); free(v_out); free(D);
        }

        printf("[%s] Test 5: Dynamic Workspace & Boundary Verification at N = 1024, 1025, 1026\n",
               test5_ok ? "PASS" : "FAIL");
        printf("       Relabel, rooted/unrooted inverse bijections, and N=1025 optimizer passed.\n\n");
        if (!test5_ok) all_passed = 0;
    }

    if (!all_passed) {
        fprintf(stderr, "VERIFICATION FAILED: One or more checks did not pass.\n");
        return 1;
    }
    printf("All verification checks passed strictly (exit code 0).\n");
    return 0;
}
