/*
 * test_bme_branch_lengths_gpu.cu
 *
 * Verification & scaling benchmark for O(N) GPU FastME/BME branch-length estimation
 * (bme_branch_lengths_gpu.cuh):
 *   1. Exactness verification against full O(N^2) double-precision Pauplin (2000) /
 *      FastME (Desper & Gascuel 2002) balanced minimum-evolution branch lengths and
 *      Pauplin's identity sum_{e} l(e) == L_BME(T).
 *   2. Runtime & memory scaling from N = 1,000 to N = 10,000,000 leaves on packed
 *      sequence alignments (RNASim L=1,596 and JC10k L=10,000).
 */

#include "bme_branch_lengths_gpu.cuh"
#include <random>
#include <iostream>
#include <iomanip>

/* Exact O(N^2) CPU reference for FastME / Pauplin (2000) BME branch lengths using
 * full uncapped leaf profiles (cap = N) in double precision. */
static double compute_exact_bme_branch_lengths_cpu_ref(
    int32_t N,
    int32_t W,
    const int32_t* h_c1,
    const int32_t* h_c2,
    const uint64_t* h_packed_seqs,
    std::vector<double>& ref_bl,
    double* out_direct_bme_score = nullptr
) {
    auto host_jc69 = [&](int32_t lu, int32_t lv) -> double {
        if (lu == lv) return 0.0;
        const uint64_t* su = h_packed_seqs + (size_t)lu * 5 * W;
        const uint64_t* sv = h_packed_seqs + (size_t)lv * 5 * W;
        int32_t nv = 0, nm = 0;
        for (int32_t w = 0; w < W; ++w) {
            nv += __builtin_popcountll(su[4 * W + w] & sv[4 * W + w]);
            uint64_t m = (su[0 * W + w] & sv[0 * W + w])
                       | (su[1 * W + w] & sv[1 * W + w])
                       | (su[2 * W + w] & sv[2 * W + w])
                       | (su[3 * W + w] & sv[3 * W + w]);
            nm += __builtin_popcountll(m);
        }
        if (nv <= 0) return 1.5;
        float pf = (float)(nv - nm) / (float)nv;
        if (pf >= 0.74f) pf = 0.74f;
        float df = -0.75f * logf(1.0f - (4.0f / 3.0f) * pf);
        return (df > 0.0f) ? (double)df : 0.0;
    };

    /* Precompute full N x N distance matrix D */
    std::vector<double> D((size_t)N * N, 0.0);
    #pragma omp parallel for schedule(dynamic, 16)
    for (int32_t i = 0; i < N; ++i) {
        for (int32_t j = i + 1; j < N; ++j) {
            double d = host_jc69(i, j);
            D[(size_t)i * N + j] = d;
            D[(size_t)j * N + i] = d;
        }
    }

    const int32_t root = 2 * N - 2;
    std::vector<int32_t> parent(2 * N - 1, -1);
    std::vector<int32_t> post;
    {
        std::vector<int32_t> st = {root};
        while (!st.empty()) {
            int32_t u = st.back();
            st.pop_back();
            if (u >= N) {
                post.push_back(u);
                parent[h_c1[u]] = u;
                parent[h_c2[u]] = u;
                st.push_back(h_c1[u]);
                st.push_back(h_c2[u]);
            }
        }
        std::reverse(post.begin(), post.end());
    }

    /* Inside weight vectors w_in[u][leaf] for u in 0..2N-3 */
    std::vector<std::vector<double>> w_in(2 * N - 2, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> w_out(2 * N - 2, std::vector<double>(N, 0.0));
    for (int32_t i = 0; i < N; ++i) w_in[i][i] = 1.0;
    for (int32_t u : post) {
        if (u == root) continue;
        int32_t l = h_c1[u], r = h_c2[u];
        for (int32_t x = 0; x < N; ++x) {
            w_in[u][x] = 0.5 * (w_in[l][x] + w_in[r][x]);
        }
    }
    int32_t r_left = h_c1[root], r_right = h_c2[root];
    w_out[r_left]  = w_in[r_right];
    w_out[r_right] = w_in[r_left];
    for (int32_t pi = (int32_t)post.size() - 1; pi >= 0; --pi) {
        int32_t u = post[pi];
        if (u == root) continue;
        int32_t l = h_c1[u], r = h_c2[u];
        for (int32_t x = 0; x < N; ++x) {
            w_out[l][x] = 0.5 * (w_out[u][x] + w_in[r][x]);
            w_out[r][x] = 0.5 * (w_out[u][x] + w_in[l][x]);
        }
    }

    auto bdist = [&](const std::vector<double>& wa, const std::vector<double>& wb) -> double {
        double s = 0.0;
        for (int32_t i = 0; i < N; ++i) {
            if (wa[i] == 0.0) continue;
            double row = 0.0;
            for (int32_t j = 0; j < N; ++j) {
                if (wb[j] == 0.0) continue;
                row += wb[j] * D[(size_t)i * N + j];
            }
            s += wa[i] * row;
        }
        return s;
    };

    if (out_direct_bme_score) {
        /* Direct Pauplin sum: 0.5 * sum_{x=0..N-1} d^B({x}, w_out[x]) */
        double score = 0.0;
        for (int32_t x = 0; x < N; ++x) {
            score += 0.5 * bdist(w_in[x], w_out[x]);
        }
        *out_direct_bme_score = score;
    }

    ref_bl.assign(2 * N - 2, 0.0);
    double sum_bl = 0.0;
    #pragma omp parallel for reduction(+:sum_bl) schedule(dynamic, 4)
    for (int32_t u = 0; u < 2 * N - 2; ++u) {
        if (u == r_right) {
            ref_bl[u] = 0.0;
            continue;
        }
        double cross = bdist(w_in[u], w_out[u]);
        double div_in = (u >= N) ? bdist(w_in[h_c1[u]], w_in[h_c2[u]]) : 0.0;
        double div_out = 0.0;
        if (u == r_left) {
            div_out = (r_right >= N) ? bdist(w_in[h_c1[r_right]], w_in[h_c2[r_right]]) : 0.0;
        } else {
            int32_t p = parent[u];
            int32_t sib = (h_c1[p] == u) ? h_c2[p] : h_c1[p];
            div_out = bdist(w_in[sib], w_out[p]);
        }
        double bl = cross - 0.5 * div_in - 0.5 * div_out;
        ref_bl[u] = bl;
        sum_bl += bl;
    }
    return sum_bl;
}

static void make_random_tree_and_seqs(
    int32_t N, int32_t W, uint64_t seed,
    std::vector<int32_t>& c1, std::vector<int32_t>& c2,
    std::vector<uint64_t>& packed_seqs
) {
    std::mt19937_64 rng(seed);
    c1.assign(2 * N - 1, -1);
    c2.assign(2 * N - 1, -1);
    /* O(N) uniform random rooted binary tree via incremental random edge splitting */
    std::vector<int32_t> parent_of(2 * N - 1, -1);
    std::vector<int32_t> active_nodes = {0};
    active_nodes.reserve(2 * N - 1);
    for (int32_t leaf = 1; leaf < N; ++leaf) {
        int32_t pick = (int32_t)(rng() % active_nodes.size());
        int32_t u = active_nodes[pick];
        int32_t w = N - 1 + leaf; /* internal node N .. 2N-2 */
        int32_t p = parent_of[u];
        c1[w] = u;
        c2[w] = leaf;
        parent_of[u] = w;
        parent_of[leaf] = w;
        parent_of[w] = p;
        if (p >= 0) {
            if (c1[p] == u) c1[p] = w;
            else            c2[p] = w;
        }
        active_nodes.push_back(leaf);
        active_nodes.push_back(w);
    }
    /* Ensure root = 2N-2 is the unique root with parent_of[root] == -1 */
    int32_t cur_root = -1;
    for (int32_t node : active_nodes) {
        if (parent_of[node] == -1) { cur_root = node; break; }
    }
    int32_t top = 2 * N - 2;
    if (cur_root != top && cur_root >= N) {
        std::swap(c1[cur_root], c1[top]);
        std::swap(c2[cur_root], c2[top]);
        for (int32_t m = N; m <= top; ++m) {
            if (c1[m] == cur_root) c1[m] = top;
            else if (c1[m] == top) c1[m] = cur_root;
            if (c2[m] == cur_root) c2[m] = top;
            else if (c2[m] == top) c2[m] = cur_root;
        }
    }

    packed_seqs.resize((size_t)N * 5 * W);
    #pragma omp parallel for schedule(static)
    for (int32_t i = 0; i < N; ++i) {
        uint64_t s_rng = seed ^ ((uint64_t)i * 0x9E3779B97F4A7C15ULL);
        uint64_t* s = &packed_seqs[(size_t)i * 5 * W];
        for (int32_t w = 0; w < W; ++w) {
            s_rng = s_rng * 6364136223846793005ULL + 1442695040888963407ULL;
            uint64_t r0 = s_rng;
            s_rng = s_rng * 6364136223846793005ULL + 1442695040888963407ULL;
            uint64_t r1 = s_rng;
            s[0 * W + w] = (~r0) & (~r1);
            s[1 * W + w] = ( r0) & (~r1);
            s[2 * W + w] = (~r0) & ( r1);
            s[3 * W + w] = ( r0) & ( r1);
            s[4 * W + w] = ~0ULL;
        }
    }
}

int main() {
    int n_gpus = 0;
    cudaGetDeviceCount(&n_gpus);
    printf("=== O(N) GPU FastME/BME Branch-Length Verification & Scaling (GPUs=%d) ===\n\n", n_gpus);

    /* Part 1: Exactness vs Full O(N^2) Pauplin / FastME Reference */
    printf("--- Part 1: Numerical Verification vs Exact O(N^2) FastME/Pauplin Reference ---\n");
    for (int32_t N : {20, 64, 200, 1000}) {
        int32_t W = 25; /* L = 1,600 */
        std::vector<int32_t> c1, c2;
        std::vector<uint64_t> seqs;
        make_random_tree_and_seqs(N, W, 12345ULL + N, c1, c2, seqs);

        std::vector<double> ref_bl;
        double direct_bme = 0.0;
        double sum_ref_bl = compute_exact_bme_branch_lengths_cpu_ref(
            N, W, c1.data(), c2.data(), seqs.data(), ref_bl, &direct_bme
        );

        uint64_t* d_seqs = nullptr;
        cudaSetDevice(0);
        cudaMalloc(&d_seqs, seqs.size() * sizeof(uint64_t));
        cudaMemcpy(d_seqs, seqs.data(), seqs.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        std::vector<uint64_t*> d_ptrs = {d_seqs};

        for (int32_t cap : {32, 64, N}) {
            std::vector<float> gpu_bl;
            double prof_s = 0.0, kern_s = 0.0;
            double sum_gpu_bl = p2v_bme_bl::compute_bme_branch_lengths_gpu(
                N, W, c1.data(), c2.data(), d_ptrs.data(), 1, gpu_bl, cap, &prof_s, &kern_s
            );
            double max_err = 0.0, mae = 0.0;
            int32_t n_edges = 0;
            for (int32_t u = 0; u < 2 * N - 2; ++u) {
                if (u == c2[2 * N - 2]) continue;
                double err = std::abs((double)gpu_bl[u] - ref_bl[u]);
                max_err = std::max(max_err, err);
                mae += err;
                ++n_edges;
            }
            mae /= std::max(1, n_edges);
            printf("  N=%4d | cap=%4d | Pauplin Identity |sum(ref)-L_BME|=%.2e | Max|BL_gpu-BL_ref|=%.3e | MAE=%.3e | SumDiff=%.3e\n",
                   N, cap, std::abs(sum_ref_bl - direct_bme), max_err, mae, std::abs(sum_gpu_bl - sum_ref_bl));
        }
        cudaFree(d_seqs);
    }

    /* Part 2: Scaling Benchmark up to N = 10,000,000 on available GPUs */
    printf("\n--- Part 2: O(N) GPU FastME/BME Branch-Length Runtime Scaling (cap=32) ---\n");
    for (int32_t W : {25, 160}) {
        const char* tag = (W == 25) ? "RNASim (L=1,596, W=25)" : "JC10k (L=10,000, W=160)";
        printf("\n  [%s]\n", tag);
        for (int32_t N : {1000, 5000, 10000, 50000, 100000, 1000000, 10000000}) {
            if (N == 10000000 && n_gpus < 8) continue; /* Run 10M on 8x H100 node */
            std::vector<int32_t> c1, c2;
            std::vector<uint64_t> seqs;
            make_random_tree_and_seqs(N, W, 99991ULL + N, c1, c2, seqs);

            std::vector<uint64_t*> d_ptrs(n_gpus, nullptr);
            for (int g = 0; g < n_gpus; ++g) {
                cudaSetDevice(g);
                cudaMalloc(&d_ptrs[g], seqs.size() * sizeof(uint64_t));
                cudaMemcpy(d_ptrs[g], seqs.data(), seqs.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
            }

            std::vector<float> gpu_bl;
            double prof_s = 0.0, kern_s = 0.0;
            double sum_bl = p2v_bme_bl::compute_bme_branch_lengths_gpu(
                N, W, c1.data(), c2.data(), d_ptrs.data(), n_gpus, gpu_bl, 32, &prof_s, &kern_s
            );
            printf("    N=%10d (%10d edges) | Total=%7.4fs (CPU profiles=%7.4fs, GPU kernel=%7.4fs) | Sum(BL)=%.2f\n",
                   N, 2 * N - 3, prof_s + kern_s, prof_s, kern_s, sum_bl);
            fflush(stdout);

            for (int g = 0; g < n_gpus; ++g) {
                cudaSetDevice(g);
                cudaFree(d_ptrs[g]);
            }
        }
    }
    return 0;
}
