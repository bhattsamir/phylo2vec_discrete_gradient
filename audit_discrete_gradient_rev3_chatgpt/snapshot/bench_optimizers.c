#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/*
 * Benchmark Suite: Phylo2Vec BME Optimizer vs. Unrooted FastME SPR Hill-Climber
 *
 * Links directly against `phylo2vec_fastme.c` (no duplicated static optimizer implementations).
 * Independently verifies the returned Phylo2Vec vectors `v_fme` and `v_p2v` and reports:
 *   - Strict wins (P2V < FME), exact ties (|P2V - FME| <= 1e-9), and losses (P2V > FME)
 *   - Mean +/- SD and Median [IQR] wall-clock runtime and Robinson-Foulds (RF) recovery
 *
 * Compile:
 *   /usr/bin/clang -O3 -march=native bench_optimizers.c phylo2vec_fastme.c -o bench_optimizers && ./bench_optimizers
 */

extern double optimize_fastme_bme_c(
    const int64_t *v_init, const double *D, int32_t n_leaves,
    int64_t *v_out, int32_t *n_iters_out, double *elapsed_ms_out
);
extern double optimize_phylo2vec_bme_c(
    const int64_t *v_init, const double *D, int32_t n_leaves,
    int64_t *v_out, int32_t *n_grad_evals_out, double *elapsed_ms_out
);

static uint64_t rng_state = 88172645463325252ULL;
static inline uint32_t xorshift32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state & 0xFFFFFFFFULL);
}
static inline double rand_uniform(void) {
    return ((double)(xorshift32() & 0xFFFFFF) + 0.5) / 16777216.0;
}
static inline double rand_normal(void) {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    return sqrt(-2.0 * log(u1)) * cos(6.28318530717958647692 * u2);
}

/* Decode any v in V (0 <= v_i <= 2i) into unrooted adjacency matrix adj_u */
static void decode_v_to_adj_u(const int64_t *v, int32_t n_leaves, int32_t *adj_u) {
    int32_t n_pairs = n_leaves - 1;
    int32_t n_nodes_r = 2 * n_leaves - 1;
    int32_t n_nodes_u = 2 * n_leaves - 2;
    int32_t root_u = n_nodes_u - 1;

    int32_t *buf = (int32_t *)malloc((size_t)(2 * n_pairs + 4 * n_nodes_r) * sizeof(int32_t));
    int32_t *pairs = buf;
    int32_t *child1_r = pairs + 2 * n_pairs;
    int32_t *child2_r = child1_r + n_nodes_r;
    int32_t *parents_uf = child2_r + n_nodes_r;
    int32_t *parent_u = parents_uf + n_nodes_r;

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
        child1_r[next_parent] = p2; child2_r[next_parent] = p1;
        parents_uf[p1] = next_parent; parents_uf[p2] = next_parent;
        next_parent++;
    }
    int32_t rc1 = child1_r[n_nodes_r - 1], rc2 = child2_r[n_nodes_r - 1];
    int32_t other_root_child = (rc1 < rc2) ? rc1 : rc2;
    for (int32_t u = n_leaves; u < root_u; ++u) {
        parent_u[child1_r[u]] = u; parent_u[child2_r[u]] = u;
        adj_u[3 * u + 0] = child1_r[u]; adj_u[3 * u + 1] = child2_r[u];
    }
    parent_u[child1_r[root_u]] = root_u;
    parent_u[child2_r[root_u]] = root_u;
    parent_u[other_root_child] = root_u;
    for (int32_t u = 0; u < n_leaves; ++u) {
        adj_u[3 * u + 0] = parent_u[u]; adj_u[3 * u + 1] = -1; adj_u[3 * u + 2] = -1;
    }
    for (int32_t u = n_leaves; u < root_u; ++u) adj_u[3 * u + 2] = parent_u[u];
    adj_u[3 * root_u + 0] = child1_r[root_u];
    adj_u[3 * root_u + 1] = child2_r[root_u];
    adj_u[3 * root_u + 2] = other_root_child;

    free(buf);
}

/* Robinson-Foulds (RF) distance via 64-bit XOR split hashing of the N-3 internal edges */
static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a;
    uint64_t y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

static void extract_split_hashes(
    const int32_t *adj_u, int32_t n_leaves,
    const uint64_t *leaf_hash, uint64_t total_hash, uint64_t *splits_out
) {
    const int32_t n_nodes_u = 2 * n_leaves - 2;
    const int32_t root_u = n_nodes_u - 1;
    int32_t *buf = (int32_t *)malloc((size_t)(3 * n_nodes_u) * sizeof(int32_t));
    uint64_t *sub_hash = (uint64_t *)malloc((size_t)n_nodes_u * sizeof(uint64_t));
    int32_t *stack = buf, *parent = buf + n_nodes_u, *order = buf + 2 * n_nodes_u;

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
    free(buf);
    free(sub_hash);
}

static double compute_rf_distance(const int32_t *adj1, const int32_t *adj2, int32_t n_leaves) {
    if (n_leaves <= 3) return 0.0;
    uint64_t *leaf_hash = (uint64_t *)malloc((size_t)n_leaves * sizeof(uint64_t));
    uint64_t *s1 = (uint64_t *)malloc((size_t)n_leaves * sizeof(uint64_t));
    uint64_t *s2 = (uint64_t *)malloc((size_t)n_leaves * sizeof(uint64_t));
    uint64_t total_hash = 0ULL;
    uint64_t saved_rng = rng_state;
    rng_state = 999999999ULL;
    for (int32_t i = 0; i < n_leaves; ++i) {
        uint64_t h = ((uint64_t)xorshift32() << 32) | (uint64_t)xorshift32();
        leaf_hash[i] = h;
        total_hash ^= h;
    }
    rng_state = saved_rng;

    extract_split_hashes(adj1, n_leaves, leaf_hash, total_hash, s1);
    extract_split_hashes(adj2, n_leaves, leaf_hash, total_hash, s2);

    int32_t n_splits = n_leaves - 3;
    int32_t shared = 0, i = 0, j = 0;
    while (i < n_splits && j < n_splits) {
        if (s1[i] == s2[j]) { shared++; i++; j++; }
        else if (s1[i] < s2[j]) i++;
        else j++;
    }
    free(leaf_hash); free(s1); free(s2);
    return (double)(n_splits - shared) / (double)n_splits;
}

/* Generate ground-truth phylogenetic tree and distance matrix (Uniform(0.05, 0.55) branches + log-normal noise) */
static void generate_phylogenetic_dataset(
    int32_t n_leaves,
    double noise_sigma,
    int64_t *true_v,
    int32_t *true_adj,
    double *D_out
) {
    int32_t n_pairs = n_leaves - 1;
    int32_t n_nodes_u = 2 * n_leaves - 2;

    true_v[0] = 0;
    for (int32_t i = 1; i < n_pairs; ++i) {
        true_v[i] = (int64_t)(xorshift32() % (uint32_t)(2 * i + 1));
    }
    decode_v_to_adj_u(true_v, n_leaves, true_adj);

    /* Root at root_u = n_nodes_u - 1 to assign edge lengths */
    int32_t root_u = n_nodes_u - 1;
    int32_t *parent_u = (int32_t *)malloc((size_t)n_nodes_u * sizeof(int32_t));
    int32_t *q = (int32_t *)malloc((size_t)n_nodes_u * sizeof(int32_t));
    double *br_len = (double *)malloc((size_t)n_nodes_u * sizeof(double));
    double *dist = (double *)malloc((size_t)n_nodes_u * sizeof(double));

    for (int32_t u = 0; u < n_nodes_u; ++u) parent_u[u] = -2;
    int32_t head = 0, tail = 0;
    q[tail++] = root_u;
    parent_u[root_u] = -1;
    while (head < tail) {
        int32_t u = q[head++];
        for (int32_t k = 0; k < 3; ++k) {
            int32_t w = true_adj[3 * u + k];
            if (w != -1 && parent_u[w] == -2) {
                parent_u[w] = u;
                q[tail++] = w;
            }
        }
    }
    for (int32_t u = 0; u < root_u; ++u) {
        br_len[u] = 0.05 + 0.50 * rand_uniform(); /* Uniform(0.05, 0.55) */
    }

    for (int32_t src = 0; src < n_leaves; ++src) {
        for (int32_t k = 0; k < n_nodes_u; ++k) dist[k] = -1.0;
        dist[src] = 0.0;
        q[0] = src;
        head = 0; tail = 1;
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
        D_out[(size_t)src * n_leaves + src] = 0.0;
        for (int32_t dst = src + 1; dst < n_leaves; ++dst) {
            double d = dist[dst];
            if (noise_sigma > 0.0) {
                d *= exp(noise_sigma * rand_normal());
                if (d < 0.001) d = 0.001;
            }
            D_out[(size_t)src * n_leaves + dst] = d;
            D_out[(size_t)dst * n_leaves + src] = d;
        }
    }
    free(parent_u); free(q); free(br_len); free(dist);
}

static void compute_stats(const double *arr, int32_t n, double *mean_out, double *sd_out, double *med_out, double *iqr_out) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) sum += arr[i];
    double mean = sum / n;
    double var = 0.0;
    for (int32_t i = 0; i < n; ++i) {
        double d = arr[i] - mean;
        var += d * d;
    }
    double sd = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
    double *sorted = (double *)malloc((size_t)n * sizeof(double));
    memcpy(sorted, arr, (size_t)n * sizeof(double));
    qsort(sorted, (size_t)n, sizeof(double), cmp_double);
    double med = (n % 2 == 1) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
    double q25 = sorted[n / 4];
    double q75 = sorted[(3 * n) / 4];
    free(sorted);
    if (mean_out) *mean_out = mean;
    if (sd_out) *sd_out = sd;
    if (med_out) *med_out = med;
    if (iqr_out) *iqr_out = q75 - q25;
}

int main(void) {
    const int32_t sizes[] = {20, 50, 100, 150, 200};
    const double noises[] = {0.0, 0.15};
    const char *noise_names[] = {
        "Noiseless Additive Tree Distances (sigma = 0.00, Exact Recovery Test)",
        "Noisy Phylogenetic Distances (sigma = 0.15, Log-Normal Estimation Noise)"
    };

    for (int32_t ns = 0; ns < 2; ++ns) {
        double sigma = noises[ns];
        printf("\n==================================================================================================================================\n");
        printf("BENCHMARK %d: %s\n", ns + 1, noise_names[ns]);
        printf("==================================================================================================================================\n");
        printf("%5s | %10s %10s %11s | %7s %7s | %8s %8s | %15s %15s | %11s %7s\n",
               "N", "FastME L", "P2V L", "Win/Tie/Los", "FME RF", "P2V RF",
               "FME Iter", "P2V Eval", "FME ms (mean±sd)", "P2V ms (mean±sd)", "P2V med[IQR]", "Speedup");
        printf("----------------------------------------------------------------------------------------------------------------------------------\n");

        for (int32_t s = 0; s < 5; ++s) {
            int32_t N = sizes[s];
            int32_t n_trials = (N <= 50) ? 20 : ((N <= 100) ? 10 : 5);

            double sum_L_fme = 0.0, sum_L_p2v = 0.0;
            double sum_rf_fme = 0.0, sum_rf_p2v = 0.0;
            double sum_it_fme = 0.0, sum_ev_p2v = 0.0;
            double ms_fme_arr[32], ms_p2v_arr[32];
            int32_t strict_wins = 0, exact_ties = 0, losses = 0;

            int64_t *true_v = (int64_t *)malloc((size_t)N * sizeof(int64_t));
            int64_t *v_init = (int64_t *)malloc((size_t)N * sizeof(int64_t));
            int64_t *v_fme  = (int64_t *)malloc((size_t)N * sizeof(int64_t));
            int64_t *v_p2v  = (int64_t *)malloc((size_t)N * sizeof(int64_t));
            int32_t *true_adj = (int32_t *)malloc((size_t)(6 * N) * sizeof(int32_t));
            int32_t *adj_fme  = (int32_t *)malloc((size_t)(6 * N) * sizeof(int32_t));
            int32_t *adj_p2v  = (int32_t *)malloc((size_t)(6 * N) * sizeof(int32_t));
            double *D = (double *)malloc((size_t)N * N * sizeof(double));

            for (int32_t t = 0; t < n_trials; ++t) {
                generate_phylogenetic_dataset(N, sigma, true_v, true_adj, D);
                v_init[0] = 0;
                for (int32_t i = 1; i < N - 1; ++i) {
                    v_init[i] = (int64_t)(xorshift32() % (uint32_t)(i + 1));
                }

                int32_t it_fme = 0, ev_p2v = 0;
                double ms_fme = 0.0, ms_p2v = 0.0;

                double L_fme = optimize_fastme_bme_c(v_init, D, N, v_fme, &it_fme, &ms_fme);
                double L_p2v = optimize_phylo2vec_bme_c(v_init, D, N, v_p2v, &ev_p2v, &ms_p2v);

                decode_v_to_adj_u(v_fme, N, adj_fme);
                decode_v_to_adj_u(v_p2v, N, adj_p2v);

                double rf_fme = compute_rf_distance(true_adj, adj_fme, N);
                double rf_p2v = compute_rf_distance(true_adj, adj_p2v, N);

                sum_L_fme += L_fme;
                sum_L_p2v += L_p2v;
                sum_rf_fme += rf_fme;
                sum_rf_p2v += rf_p2v;
                sum_it_fme += (double)it_fme;
                sum_ev_p2v += (double)ev_p2v;
                ms_fme_arr[t] = ms_fme;
                ms_p2v_arr[t] = ms_p2v;

                if (L_p2v < L_fme - 1e-9) strict_wins++;
                else if (fabs(L_p2v - L_fme) <= 1e-9) exact_ties++;
                else losses++;
            }

            double fme_mean, fme_sd, fme_med, fme_iqr;
            double p2v_mean, p2v_sd, p2v_med, p2v_iqr;
            compute_stats(ms_fme_arr, n_trials, &fme_mean, &fme_sd, &fme_med, &fme_iqr);
            compute_stats(ms_p2v_arr, n_trials, &p2v_mean, &p2v_sd, &p2v_med, &p2v_iqr);

            char wtl_buf[32], fme_ms_buf[32], p2v_ms_buf[32], p2v_med_buf[32];
            snprintf(wtl_buf, sizeof(wtl_buf), "%2d/%2d/%-2d", strict_wins, exact_ties, losses);
            snprintf(fme_ms_buf, sizeof(fme_ms_buf), "%6.2f±%5.2f", fme_mean, fme_sd);
            snprintf(p2v_ms_buf, sizeof(p2v_ms_buf), "%6.2f±%5.2f", p2v_mean, p2v_sd);
            snprintf(p2v_med_buf, sizeof(p2v_med_buf), "%5.2f[%4.2f]", p2v_med, p2v_iqr);

            printf("%5d | %10.5f %10.5f %11s | %7.4f %7.4f | %8.1f %8.1f | %15s %15s | %11s %6.2fx\n",
                   N,
                   sum_L_fme / n_trials,
                   sum_L_p2v / n_trials,
                   wtl_buf,
                   sum_rf_fme / n_trials,
                   sum_rf_p2v / n_trials,
                   sum_it_fme / n_trials,
                   sum_ev_p2v / n_trials,
                   fme_ms_buf,
                   p2v_ms_buf,
                   p2v_med_buf,
                   fme_mean / p2v_mean);

            free(true_v); free(v_init); free(v_fme); free(v_p2v);
            free(true_adj); free(adj_fme); free(adj_p2v); free(D);
        }
    }
    printf("==================================================================================================================================\n");
    return 0;
}
