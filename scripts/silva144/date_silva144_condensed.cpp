/*
 * date_silva144_condensed.cpp
 *
 * O(N) BME Balanced Subtree Condensation & Relative-Rate Dating
 * (Theorem 2.3 & Corollary 2.5).
 *
 * Given an N-leaf binary tree (c1, c2) with Pauplin / BME branch lengths bl[0..2N-3]:
 *   1. Bottom-up pass computes the exact Pauplin balanced crown depth:
 *          h_bal[u] = 0.5 * (h_bal[c1[u]] + l(c1[u])) + 0.5 * (h_bal[c2[u]] + l(c2[u])),
 *      which satisfies exact subtree condensation \widetilde{D}(v, S_u) = d(v, u) + h_bal[u]
 *      and weights sister lineages (1/2, 1/2) at every split (invariant to tip oversampling).
 *   2. Top-down pass computes cumulative root-to-node distance D[u] and assigns each internal
 *      node u its monotone BME relative divergence age \tau_bal[u] \in [0, T_root]:
 *          \tau_bal[u] = min(\tau_bal[parent(u)], T_root * h_bal[u] / (D[u] + h_bal[u])).
 *      Under any lineage-specific local substitution rate r_u, D[u] = r_u * (T_root - t_u)
 *      and h_bal[u] = r_u * t_u, so r_u cancels out and \tau_bal[u] recovers t_u in O(N) time.
 *
 * Compile:
 *   g++ -O3 -march=native -fopenmp date_silva144_condensed.cpp -o date_silva144_condensed
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <algorithm>
#include <vector>

/*
 * bme_balanced_relative_date_o_n
 *
 * Exact O(N) two-pass BME balanced relative-rate dating on a rooted binary tree.
 *
 * Inputs:
 *   N           - Number of leaves (0 .. N-1); internal nodes are N .. 2N-2, root is 2N-2.
 *   c1, c2      - Left and right child arrays of length 2N - 1.
 *   bl_raw      - Pauplin / BME branch lengths of length 2N - 2 (bl_raw[u] is edge parent(u)->u).
 *   T_root      - Root age scale (default 1.0 for dimensionless relative depths in [0, 1]).
 *
 * Outputs:
 *   out_h_bal   - Balanced Pauplin crown depth h_bal[u] for all nodes 0 .. 2N-2.
 *   out_tau_bal - Monotone BME relative node ages \tau_bal[u] \in [0, T_root] for all nodes 0 .. 2N-2.
 */
void bme_balanced_relative_date_o_n(
    int32_t N,
    const int32_t *c1,
    const int32_t *c2,
    const float *bl_raw,
    double T_root,
    std::vector<float> &out_h_bal,
    std::vector<float> &out_tau_bal
) {
    const int32_t n_nodes = 2 * N - 1;
    const int32_t root = n_nodes - 1;
    out_h_bal.assign(n_nodes, 0.0f);
    out_tau_bal.assign(n_nodes, 0.0f);
    if (N <= 1) return;

    // 1. Pre-order traversal of internal nodes in O(N)
    std::vector<int32_t> ord(N - 1);
    int32_t ord_len = 0;
    ord[ord_len++] = root;
    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1[u], b1 = c2[u];
        if (b0 >= N) ord[ord_len++] = b0;
        if (b1 >= N) ord[ord_len++] = b1;
    }

    // 2. Bottom-up pass: exact Pauplin balanced crown depth h_bal[u] (Theorem 2.3 & Corollary 2.5)
    for (int32_t idx = ord_len - 1; idx >= 0; --idx) {
        int32_t u = ord[idx];
        int32_t b0 = c1[u], b1 = c2[u];
        float l0 = (bl_raw[b0] > 0.0f) ? bl_raw[b0] : 0.0f;
        float l1 = (bl_raw[b1] > 0.0f) ? bl_raw[b1] : 0.0f;
        out_h_bal[u] = 0.5f * ((out_h_bal[b0] + l0) + (out_h_bal[b1] + l1));
    }

    // 3. Top-down pass: cumulative root distance D[u] and monotone relative age \tau_bal[u]
    std::vector<float> D_root(n_nodes, 0.0f);
    out_tau_bal[root] = (float)T_root;

    for (int32_t idx = 0; idx < ord_len; ++idx) {
        int32_t u = ord[idx];
        float tau_parent = out_tau_bal[u];
        for (int32_t ch : {c1[u], c2[u]}) {
            float l_ch = (bl_raw[ch] > 0.0f) ? bl_raw[ch] : 0.0f;
            D_root[ch] = D_root[u] + l_ch;
            if (ch < N) {
                out_tau_bal[ch] = 0.0f;
            } else {
                double h_c = (double)out_h_bal[ch];
                double d_c = (double)D_root[ch];
                double denom = d_c + h_c;
                double raw_tau = (denom > 1e-12) ? (T_root * h_c / denom) : 0.0;
                out_tau_bal[ch] = (float)std::max(0.0, std::min((double)tau_parent, raw_tau));
            }
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <tree_rep.bin> [T_root=1.0]\n", argv[0]);
        return 1;
    }
    const char *tree_path = argv[1];
    double T_root = (argc >= 3) ? atof(argv[2]) : 1.0;

    FILE *f = fopen(tree_path, "rb");
    if (!f) {
        fprintf(stderr, "Error: cannot open %s\n", tree_path);
        return 1;
    }

    int32_t N = 0, rep_id = 0;
    double hdr[9];
    if (fread(&N, sizeof(int32_t), 1, f) != 1 ||
        fread(&rep_id, sizeof(int32_t), 1, f) != 1 ||
        fread(hdr, sizeof(double), 9, f) != 9) {
        fclose(f);
        return 1;
    }

    const int32_t n_nodes = 2 * N - 1;
    std::vector<int32_t> c1(n_nodes), c2(n_nodes);
    std::vector<float> bl_raw(n_nodes, 0.0f);
    if (fread(c1.data(), sizeof(int32_t), n_nodes, f) != (size_t)n_nodes ||
        fread(c2.data(), sizeof(int32_t), n_nodes, f) != (size_t)n_nodes ||
        fread(bl_raw.data(), sizeof(float), n_nodes - 1, f) != (size_t)(n_nodes - 1)) {
        fclose(f);
        fprintf(stderr, "Error: incomplete tree binary %s\n", tree_path);
        return 1;
    }
    fclose(f);

    double t0 = omp_get_wtime();
    std::vector<float> h_bal, tau_bal;
    bme_balanced_relative_date_o_n(N, c1.data(), c2.data(), bl_raw.data(), T_root, h_bal, tau_bal);
    double dt = omp_get_wtime() - t0;

    printf("N = %d | rep = %d | root h_bal = %.6f | tau_root = %.4f | elapsed = %.4fs\n",
           N, rep_id, h_bal[n_nodes - 1], tau_bal[n_nodes - 1], dt);
    return 0;
}
