"""Build a bitwise/tolerance cross-check binary source.
usage: python3 make_kcheck.py NEW_SRC ORIG_SRC OUT_CU
Copies the ORIGINAL k_refine_real_seq_condensed_D_pairs from ORIG_SRC as k_refine_ORIG into NEW_SRC, and after the
production launch in optimize_skeleton_on_gpu re-runs k_refine_ORIG on the identical launch list, comparing every
d_pair_exact_D entry. Prints at the end of run_multi_gpu_minibatch_bme:
  [kcheck] N=... launches=... entries=... bitwise_mismatches=... max_abs=... max_rel=...
Also adds the RNASIM_SIZES env var (e.g. RNASIM_SIZES=10000,50000,100000).
Requires the tail's recombination line (see ANCHOR) to exist unchanged."""
import re, sys
new_src, orig_src, out = sys.argv[1:4]
s3 = open(new_src).read(); orig = open(orig_src).read()
def extract(src, name):
    m = re.search(r'^(?:__global__|__device__)[^\n]*\b' + name + r'\(', src, re.M); assert m, name
    b = src.index('{', m.start()); j = b + 1; level = 1
    while level: level += (src[j] == '{') - (src[j] == '}'); j += 1
    return m.start(), j
a, b = extract(orig, 'k_refine_real_seq_condensed_D_pairs')
orig_k = orig[a:b].replace('k_refine_real_seq_condensed_D_pairs', 'k_refine_ORIG')
a3, b3 = extract(s3, 'k_refine_real_seq_condensed_D_pairs')
k = s3[:b3] + '\n\n' + orig_k + '\n' + s3[b3:]
def rep(old, new):
    global k
    assert k.count(old) == 1, (old[:60], k.count(old)); k = k.replace(old, new)
rep('static double wall_sec(void) {', 'static long long KC_launches = 0, KC_entries = 0, KC_mismatch = 0; static double KC_maxabs = 0, KC_maxrel = 0;\nstatic double wall_sec(void) {')
ANCHOR = '''                for (int32_t t = n_pairs; t < n_launch; ++t) h_pair_exact_D[chunk_owner[t - n_pairs]] += h_launch_exact_D[t];
'''
rep(ANCHOR, ANCHOR + '''                {   /* kcheck: original kernel on the identical launch list */
                    double *d_tmp = NULL; float *d_tmpD = NULL;
                    CUDA_CHECK(cudaMalloc(&d_tmp, (size_t)n_launch * sizeof(double)));
                    size_t nD = (size_t)5 * gpu_ws.max_K * ((size_t)gpu_ws.max_K + 1);
                    CUDA_CHECK(cudaMalloc(&d_tmpD, nD * sizeof(float)));
                    k_refine_ORIG<<<n_launch, 256, 0, gpu_ws.stream>>>(K, W, n_launch, gpu_ws.d_pair_a, gpu_ws.d_pair_b, gpu_ws.d_exact_offsets, gpu_ws.d_exact_leaves, gpu_ws.d_exact_weights, d_packed_seqs, d_tmpD, d_tmp);
                    std::vector<double> o((size_t)n_launch);
                    CUDA_CHECK(cudaMemcpyAsync(o.data(), d_tmp, (size_t)n_launch * sizeof(double), cudaMemcpyDeviceToHost, gpu_ws.stream));
                    CUDA_CHECK(cudaStreamSynchronize(gpu_ws.stream));
                    cudaFree(d_tmp); cudaFree(d_tmpD);
                    KC_launches++;
                    for (int32_t t = 0; t < n_launch; ++t) { KC_entries++; double ad = fabs(o[t] - h_launch_exact_D[t]);
                        if (memcmp(&o[t], &h_launch_exact_D[t], sizeof(double)) != 0) KC_mismatch++;
                        if (ad > KC_maxabs) KC_maxabs = ad; double rd = ad / fmax(fabs(o[t]), 1e-300); if (rd > KC_maxrel) KC_maxrel = rd; }
                }
''')
rep('''    return total_mb_delta_L;
}
''', '''    fprintf(stderr, "[kcheck] N=%d launches=%lld entries=%lld bitwise_mismatches=%lld max_abs=%.3e max_rel=%.3e\\n", N, KC_launches, KC_entries, KC_mismatch, KC_maxabs, KC_maxrel);
    KC_launches = KC_entries = KC_mismatch = 0; KC_maxabs = KC_maxrel = 0;
    return total_mb_delta_L;
}
''')
rep('''    const int32_t sizes[] = {1000, 5000, 10000, 50000, 100000};''', '''    int32_t sizes[5] = {1000, 5000, 10000, 50000, 100000}; int n_sizes_env = 5;
    if (getenv("RNASIM_SIZES")) { n_sizes_env = 0; char buf[256]; strncpy(buf, getenv("RNASIM_SIZES"), 255); buf[255]=0; for (char *t = strtok(buf, ","); t && n_sizes_env < 5; t = strtok(NULL, ",")) sizes[n_sizes_env++] = atoi(t); }''')
rep('''    for (size_t s_idx = 0; s_idx < sizeof(sizes) / sizeof(sizes[0]); ++s_idx) {''', '''    for (size_t s_idx = 0; s_idx < (size_t)n_sizes_env; ++s_idx) {''')
open(out, 'w').write(k); print('wrote', out)
