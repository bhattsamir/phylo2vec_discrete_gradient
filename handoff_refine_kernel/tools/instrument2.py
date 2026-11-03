"""Scratch-only instrumentation, never applied to the repo.
usage: instrument2.py SRC OUT [--hash-only] [--xcheck]
  always : RNASIM_SIZES env, [tree] hash of final tree + full-precision delta L
  default: + [prof] timing counters (needs the split source)
  --xcheck: + unsplit reference launch, CPU FP64 reference for every split pair and a sample of
            unsplit multi-leaf pairs, XCHUNK env override of the chunk size (stress testing)."""
import sys
src, out = sys.argv[1], sys.argv[2]
hash_only = '--hash-only' in sys.argv; xcheck = '--xcheck' in sys.argv
s = open(src).read()
def rep(old, new, count=1):
    global s
    n = s.count(old); assert n == count, (old[:90], n); s = s.replace(old, new)

rep('static double wall_sec(void) {',
    'static double T_prof=0,T_search=0,T_refine=0; static long long C_win=0,C_acc=0,C_tail=0,C_pairs=0,C_heavy=0,C_split_tails=0,C_launch=0; static double C_leafpairs=0,C_maxlp=0;\n'
    'static long long X_pairs=0,X_split_pairs=0,X_windows=0,X_cpu_pairs=0,X_cpu_split=0; static double X_maxrel=0,X_maxabs=0,X_maxrel_unsplit=0,X_cpu_maxrel=0,X_cpu_maxrel_split=0;\n'
    'static double wall_sec(void) {')
rep('''    return total_mb_delta_L;
}
''', '''    { uint64_t h = 1469598103934665603ULL; for (size_t i = 0; i < curr_c1.size(); ++i) { h ^= (uint64_t)(uint32_t)curr_c1[i]; h *= 1099511628211ULL; h ^= (uint64_t)(uint32_t)curr_c2[i]; h *= 1099511628211ULL; }
      fprintf(stderr, "[tree] N=%d hash=%016llx dL=%.17g\\n", N, (unsigned long long)h, total_mb_delta_L); }
    return total_mb_delta_L;
}
''')
rep('''    const int32_t sizes[] = {1000, 5000, 10000, 50000, 100000};''', '''    int32_t sizes[5] = {1000, 5000, 10000, 50000, 100000}; int n_sizes_env = 5;
    if (getenv("RNASIM_SIZES")) { n_sizes_env = 0; char buf[256]; strncpy(buf, getenv("RNASIM_SIZES"), 255); buf[255]=0; for (char *t = strtok(buf, ","); t && n_sizes_env < 5; t = strtok(NULL, ",")) sizes[n_sizes_env++] = atoi(t); }''')
rep('''    for (size_t s_idx = 0; s_idx < sizeof(sizes) / sizeof(sizes[0]); ++s_idx) {''', '''    for (size_t s_idx = 0; s_idx < (size_t)n_sizes_env; ++s_idx) {''')

if not hash_only:
    rep('''    CUDA_CHECK(cudaSetDevice(gpu_ws.dev_id));
    const int32_t R_sk = 2 * K - 1;''', '''    CUDA_CHECK(cudaSetDevice(gpu_ws.dev_id));
    double __ts0 = wall_sec();
    const int32_t R_sk = 2 * K - 1;''')
    rep('''        relabel_from_adj_u(adj_u_cf.data(), K, root_u_sk, r_nb, iter & 1, v_cf, perm);
    }

    if (best_L >= init_L - 1e-6) {''', '''        relabel_from_adj_u(adj_u_cf.data(), K, root_u_sk, r_nb, iter & 1, v_cf, perm);
    }
    T_search += wall_sec() - __ts0;

    if (best_L >= init_L - 1e-6) {''')
    rep('''                gpu_ws.ensure_exact_prof_cap((size_t)total_exact_prof);
''', '''                gpu_ws.ensure_exact_prof_cap((size_t)total_exact_prof);
                double __tr0 = wall_sec(); C_tail++; C_pairs += n_pairs;
                for (int32_t pp = 0; pp < n_pairs; ++pp) { double la = exact_prof_offsets[pair_a[pp]+1]-exact_prof_offsets[pair_a[pp]], lb = exact_prof_offsets[pair_b[pp]+1]-exact_prof_offsets[pair_b[pp]]; C_leafpairs += la*lb; if (la*lb > C_maxlp) C_maxlp = la*lb; if (la > 1 && la*lb > 1e6) C_heavy++; }
''')
    anchor = '''                for (int32_t t = n_pairs; t < n_launch; ++t) h_pair_exact_D[chunk_owner[t - n_pairs]] += h_launch_exact_D[t];
'''
    xc = ''
    if xcheck:
        rep('const double chunk_leaf_pairs = 4194304.0;', 'const double chunk_leaf_pairs = getenv("XCHUNK") ? atof(getenv("XCHUNK")) : 4194304.0;')
        xc = r'''                {   /* xcheck (scratch build only) */
                    CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_exact_offsets, exact_prof_offsets, (size_t)(K + 1) * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
                    CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_a, pair_a.data(), (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
                    CUDA_CHECK(cudaMemcpyAsync(gpu_ws.d_pair_b, pair_b.data(), (size_t)n_pairs * sizeof(int32_t), cudaMemcpyHostToDevice, gpu_ws.stream));
                    k_refine_real_seq_condensed_D_pairs<<<n_pairs, 256, 0, gpu_ws.stream>>>(K, W, n_pairs, gpu_ws.d_pair_a, gpu_ws.d_pair_b, gpu_ws.d_exact_offsets, gpu_ws.d_exact_leaves, gpu_ws.d_exact_weights, d_packed_seqs, gpu_ws.d_D, gpu_ws.d_pair_exact_D);
                    std::vector<double> ref((size_t)n_pairs);
                    CUDA_CHECK(cudaMemcpyAsync(ref.data(), gpu_ws.d_pair_exact_D, (size_t)n_pairs * sizeof(double), cudaMemcpyDeviceToHost, gpu_ws.stream));
                    CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
                    std::vector<uint8_t> was_split((size_t)n_pairs, 0);
                    for (size_t t = 0; t < chunk_owner.size(); ++t) was_split[chunk_owner[t]] = 1;
                    X_windows++;
                    for (int32_t pid = 0; pid < n_pairs; ++pid) {
                        double r = ref[pid], v = h_pair_exact_D[pid];
                        double ad = fabs(r - v), rd = ad / fmax(fabs(r), 1e-300);
                        X_pairs++;
                        if (was_split[pid]) { X_split_pairs++; if (rd > X_maxrel) X_maxrel = rd; if (ad > X_maxabs) X_maxabs = ad; }
                        else if (rd > X_maxrel_unsplit) X_maxrel_unsplit = rd;
                    }
                    /* independent CPU reference: every split pair + up to 24 unsplit pairs per window */
                    static std::vector<uint64_t> h_seq; static const uint64_t *h_seq_src = NULL;
                    int32_t max_leaf = 0;
                    for (int32_t i = 0; i < total_exact_prof; ++i) if (exact_prof_leaves[i] > max_leaf) max_leaf = exact_prof_leaves[i];
                    if (h_seq_src != d_packed_seqs || h_seq.size() < (size_t)(max_leaf + 1) * 125) {
                        h_seq.resize((size_t)(max_leaf + 1) * 125);
                        CUDA_CHECK(cudaMemcpy(h_seq.data(), d_packed_seqs, h_seq.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
                        h_seq_src = d_packed_seqs;
                    }
                    std::vector<int32_t> cpu_pids; int32_t n_unsplit_sampled = 0;
                    for (int32_t pid = 0; pid < n_pairs; ++pid) {
                        int32_t u = pair_a[pid], v = pair_b[pid];
                        double lu = exact_prof_offsets[u + 1] - exact_prof_offsets[u], lv = exact_prof_offsets[v + 1] - exact_prof_offsets[v];
                        if (was_split[pid]) cpu_pids.push_back(pid);
                        else if (n_unsplit_sampled < 24 && (pid % 7 == 0 || (lu > 1 && lv > 1)) && lu * lv < 4e6) { cpu_pids.push_back(pid); n_unsplit_sampled++; }
                    }
                    for (int32_t pid : cpu_pids) {
                        int32_t u = pair_a[pid], v = pair_b[pid];
                        int32_t u0 = exact_prof_offsets[u], u1 = exact_prof_offsets[u + 1], v0 = exact_prof_offsets[v], v1 = exact_prof_offsets[v + 1];
                        double tot = 0.0;
                        #pragma omp parallel for reduction(+:tot) schedule(dynamic, 1) num_threads(32)
                        for (int32_t p = u0; p < u1; ++p) {
                            const uint64_t *sx = &h_seq[(size_t)exact_prof_leaves[p] * 125];
                            long double acc = 0.0L;
                            for (int32_t q = v0; q < v1; ++q) {
                                const uint64_t *sy = &h_seq[(size_t)exact_prof_leaves[q] * 125];
                                int32_t n_valid = 0, n_match = 0;
                                for (int w = 0; w < 25; ++w) {
                                    n_valid += __builtin_popcountll(sx[100 + w] & sy[100 + w]);
                                    n_match += __builtin_popcountll((sx[w] & sy[w]) | (sx[25 + w] & sy[25 + w]) | (sx[50 + w] & sy[50 + w]) | (sx[75 + w] & sy[75 + w]));
                                }
                                float d = 1.5f;
                                if (n_valid > 0) { float pp = (float)(n_valid - n_match) / (float)n_valid; if (pp >= 0.74f) pp = 0.74f; d = -0.75f * logf(1.0f - (4.0f / 3.0f) * pp); if (d < 0.0f) d = 0.0f; }
                                acc += ((long double)exact_prof_weights[p] * (long double)exact_prof_weights[q]) * (long double)d;
                            }
                            tot += (double)acc;
                        }
                        double rd = fabs(tot - h_pair_exact_D[pid]) / fmax(fabs(tot), 1e-300);
                        X_cpu_pairs++;
                        if (was_split[pid]) { X_cpu_split++; if (rd > X_cpu_maxrel_split) X_cpu_maxrel_split = rd; }
                        else if (rd > X_cpu_maxrel) X_cpu_maxrel = rd;
                    }
                }
'''
    rep(anchor, anchor + xc + '''                T_refine += wall_sec() - __tr0; C_launch += n_launch; if (n_launch > n_pairs) C_split_tails++;
''')
    rep('''        relabel_from_rooted_tree(sk_c1.data(), sk_c2.data(), K, 0, v_mb.data(), perm_mb.data());
''', '''        relabel_from_rooted_tree(sk_c1.data(), sk_c2.data(), K, 0, v_mb.data(), perm_mb.data());
        T_prof += wall_sec() - __tw0; C_win++;
''', 2)
    rep('''    const int32_t root_r = 2 * N - 2;
    const int32_t max_prof_leaves = (N <= 2048) ? 0 : 64;''', '''    double __tw0 = wall_sec();
    const int32_t root_r = 2 * N - 2;
    const int32_t max_prof_leaves = (N <= 2048) ? 0 : 64;''')
    rep('if (dL != 0.0) {', 'if (dL != 0.0) { C_acc++;', 3)
    rep('''    { uint64_t h = 1469598103934665603ULL;''', '''    fprintf(stderr,"[prof] N=%d windows=%lld accepted=%lld tails=%lld pairs=%lld launch=%lld split_tails=%lld heavy=%lld leafpairs=%.3e maxlp=%.3e | T_prof=%.2f T_search=%.2f T_refine=%.2f\\n",N,C_win,C_acc,C_tail,C_pairs,C_launch,C_split_tails,C_heavy,C_leafpairs,C_maxlp,T_prof,T_search,T_refine);
    if (X_windows) fprintf(stderr,"[xcheck] N=%d windows=%lld pairs=%lld split_pairs=%lld gpu_split_vs_unsplit: max_rel=%.3e max_abs=%.3e unsplit_max_rel=%.3e | cpu_ref: pairs=%lld split=%lld max_rel_split=%.3e max_rel_unsplit=%.3e\\n",N,X_windows,X_pairs,X_split_pairs,X_maxrel,X_maxabs,X_maxrel_unsplit,X_cpu_pairs,X_cpu_split,X_cpu_maxrel_split,X_cpu_maxrel);
    T_prof=T_search=T_refine=0;C_win=C_acc=C_tail=C_pairs=C_heavy=C_split_tails=C_launch=0;C_leafpairs=C_maxlp=0;X_pairs=X_split_pairs=X_windows=X_cpu_pairs=X_cpu_split=0;X_maxrel=X_maxabs=X_maxrel_unsplit=X_cpu_maxrel=X_cpu_maxrel_split=0;
    { uint64_t h = 1469598103934665603ULL;''')
open(out, 'w').write(s)
print('wrote', out)
