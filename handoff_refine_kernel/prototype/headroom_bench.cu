// headroom_bench.cu -- engineering-headroom microbenchmark for the exact refine.
// Standalone. Uses a VERBATIM copy of k_refine_real_seq_condensed_D_pairs (cur_kernel.cuh,
// extracted from bench_rnasim_gpu.cu lines 184-210 and 252-480) on synthetic one-hot data.
// Build: nvcc -O3 -arch=sm_86 headroom_bench.cu -o headroom_bench
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cuda_runtime.h>
#include "cur_kernel.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    printf("CUDA error '%s' at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); fflush(stdout); exit(1);} } while (0)

static double wall() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static double g_t0 = 0, g_budget = 45.0;
static bool budget_ok(const char *what) {
    double el = wall() - g_t0;
    if (el > g_budget) { printf("SKIP %s (elapsed %.1f s > budget)\n", what, el); return false; }
    return true;
}
static int g_nsm = 0;

__device__ __forceinline__ long long gtimer() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/* ---------------- synthetic one-hot data (same encoding as export_rnasim_bin.py) ----------------
 * seq_cw: [c][w] layout (current: plane c word w at c*25+w); seq_wc: [w][c] layout (w*5+c). */
__global__ void k_gen(uint64_t *seq_cw, uint64_t *seq_wc, int N, uint64_t seed) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= N) return;
    for (int w = 0; w < 25; ++w) {
        uint64_t pl[5] = {0, 0, 0, 0, 0};
        for (int b = 0; b < 64; ++b) {
            int s = w * 64 + b;
            if (s >= 1596) break;
            uint64_t h = splitmix64(seed ^ ((uint64_t)r << 21) ^ (uint64_t)s);
            uint64_t hr = splitmix64(seed * 31ull + (uint64_t)s);
            int st = (int)(hr & 3ull);
            if ((h & 255ull) < 64ull) st = (int)((h >> 8) & 3ull);
            bool valid = ((h >> 16) & 255ull) >= 13ull;
            if (valid) { pl[st] |= 1ull << b; pl[4] |= 1ull << b; }
        }
        for (int c = 0; c < 5; ++c) {
            seq_cw[(size_t)r * 125 + c * 25 + w] = pl[c];
            seq_wc[(size_t)r * 125 + w * 5 + c] = pl[c];
        }
    }
}

/* ---------------- reference exact integer counts (one thread per leaf pair) ----------------
 * checksum per A row r: sum_q (n_valid*2048 + n_match), integer atomics -> order independent, exact. */
__global__ void k_ref_counts(int nA, int nB, const int *A_leaf, const int *B_leaf, const uint64_t *seq_cw,
                             unsigned long long *cs_row) {
    int r = blockIdx.y;
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= nA || q >= nB) return;
    const uint64_t *sx = seq_cw + (size_t)A_leaf[r] * 125;
    const uint64_t *sy = seq_cw + (size_t)B_leaf[q] * 125;
    int nv = 0, nm = 0;
    for (int w = 0; w < 25; ++w) {
        nv += __popcll(sx[100 + w] & sy[100 + w]);
        uint64_t m = (sx[w] & sy[w]) | (sx[25 + w] & sy[25 + w]) | (sx[50 + w] & sy[50 + w]) | (sx[75 + w] & sy[75 + w]);
        nm += __popcll(m);
    }
    atomicAdd(&cs_row[r], (unsigned long long)(nv * 2048 + nm));
}

/* ---------------- candidate kernel: smem-staged, k-sliced, 4x4 register-tiled ----------------
 * Block = 256 threads (16 x 16), tile = 64 A rows x 64 B rows, each thread a 4 x 4 micro-tile with
 * interleaved rows (A: ty + 16 i, B: tx + 16 j). Rows are staged per k-slice (5 words-per-plane x 5 planes
 * = 25 contiguous uint64 in the [w][c] layout) with coalesced loads into transposed, padded smem.
 * Integer counts are exact; d_jc is the identical float expression; per A row FP64 sum S_r = sum_q w_q d,
 * output w_r * S_r per (task, row). Differences from the current kernel: summation order only. */
namespace nk {
constexpr int RI = 4, RJ = 4, TY = 16, TX = 16, NT = TY * TX, TA = RI * TY, TB = RJ * TX;
constexpr int SW = 25, NSL = 5, PA = TA + 1, PB = TB + 1;
}

#define NK_WL_BODY(wl)                                                                         \
    {                                                                                          \
        uint64_t a[RI], b[RJ];                                                                 \
        const int kv = (wl) * 5 + 4;                                                           \
        _Pragma("unroll") for (int i = 0; i < RI; ++i) a[i] = sA[kv * PA + ty + TY * i];       \
        _Pragma("unroll") for (int j = 0; j < RJ; ++j) b[j] = sB[kv * PB + tx + TX * j];       \
        _Pragma("unroll") for (int i = 0; i < RI; ++i)                                         \
            _Pragma("unroll") for (int j = 0; j < RJ; ++j) nv[i][j] += __popcll(a[i] & b[j]);  \
        uint64_t m[RI][RJ];                                                                    \
        _Pragma("unroll") for (int c = 0; c < 4; ++c) {                                        \
            const int kc = (wl) * 5 + c;                                                       \
            _Pragma("unroll") for (int i = 0; i < RI; ++i) a[i] = sA[kc * PA + ty + TY * i];   \
            _Pragma("unroll") for (int j = 0; j < RJ; ++j) b[j] = sB[kc * PB + tx + TX * j];   \
            _Pragma("unroll") for (int i = 0; i < RI; ++i)                                     \
                _Pragma("unroll") for (int j = 0; j < RJ; ++j)                                 \
                    m[i][j] = (c == 0) ? (a[i] & b[j]) : (m[i][j] | (a[i] & b[j]));            \
        }                                                                                      \
        _Pragma("unroll") for (int i = 0; i < RI; ++i)                                         \
            _Pragma("unroll") for (int j = 0; j < RJ; ++j) nm[i][j] += __popcll(m[i][j]);      \
    }

template <bool VAL, bool FULL_UNROLL>
__global__ void __launch_bounds__(256, 2)
k_refine_tiled(const int4 *__restrict__ tasks,
               const int *__restrict__ A_leaf, const float *__restrict__ A_w,
               const int *__restrict__ B_leaf, const float *__restrict__ B_w,
               const uint64_t *__restrict__ seq_wc,
               double *__restrict__ out, unsigned long long *__restrict__ out_cs,
               long long *__restrict__ clk) {
    using namespace nk;
    __shared__ uint64_t sA[SW * PA];
    __shared__ uint64_t sB[SW * PB];
    __shared__ int s_rowA[TA];
    __shared__ int s_rowB[TB];
    __shared__ double s_wB[TB];
    const int tid = threadIdx.x, tx = tid % TX, ty = tid / TX;
    long long c0 = 0, g0 = 0;
    if (clk != nullptr && tid == 0) { c0 = clock64(); g0 = gtimer(); }
    const int4 t = tasks[blockIdx.x];
    const int a_off = t.x, a_len = t.y, b_off = t.z, b_len = t.w;
    if (tid < TA) s_rowA[tid] = (tid < a_len) ? A_leaf[a_off + tid] : -1;
    double acc[RI];
    unsigned long long cs[RI];
#pragma unroll
    for (int i = 0; i < RI; ++i) { acc[i] = 0.0; cs[i] = 0ull; }

    for (int qb = 0; qb < b_len; qb += TB) {
        __syncthreads();
        if (tid < TB) {
            const int q = qb + tid;
            const bool ok = q < b_len;
            s_rowB[tid] = ok ? B_leaf[b_off + q] : -1;
            s_wB[tid] = ok ? (double)B_w[b_off + q] : 0.0;
        }
        uint32_t nv[RI][RJ], nm[RI][RJ];
#pragma unroll
        for (int i = 0; i < RI; ++i)
#pragma unroll
            for (int j = 0; j < RJ; ++j) { nv[i][j] = 0u; nm[i][j] = 0u; }

#pragma unroll 1
        for (int s = 0; s < NSL; ++s) {
            __syncthreads();
            for (int idx = tid; idx < TA * SW; idx += NT) {
                const int r = idx / SW, k = idx - r * SW;
                const int leaf = s_rowA[r];
                sA[k * PA + r] = (leaf >= 0) ? __ldg(&seq_wc[(size_t)leaf * 125 + s * SW + k]) : 0ull;
            }
            for (int idx = tid; idx < TB * SW; idx += NT) {
                const int r = idx / SW, k = idx - r * SW;
                const int leaf = s_rowB[r];
                sB[k * PB + r] = (leaf >= 0) ? __ldg(&seq_wc[(size_t)leaf * 125 + s * SW + k]) : 0ull;
            }
            __syncthreads();
            if (FULL_UNROLL) {
#pragma unroll
                for (int wl = 0; wl < 5; ++wl) NK_WL_BODY(wl)
            } else {
#pragma unroll 1
                for (int wl = 0; wl < 5; ++wl) NK_WL_BODY(wl)
            }
        }
        // epilogue: identical float JC69 expression to the current kernel, FP64 accumulation
#pragma unroll
        for (int j = 0; j < RJ; ++j) {
            const double wq = s_wB[tx + TX * j];
#pragma unroll
            for (int i = 0; i < RI; ++i) {
                const int32_t n_valid = (int32_t)nv[i][j], n_match = (int32_t)nm[i][j];
                float d_jc = 1.5f;
                if (n_valid > 0) {
                    float p = (float)(n_valid - n_match) / (float)n_valid;
                    if (p >= 0.74f) p = 0.74f;
                    d_jc = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
                    if (d_jc < 0.0f) d_jc = 0.0f;
                }
                acc[i] += wq * (double)d_jc;
                if (VAL) { if (wq != 0.0) cs[i] += (unsigned long long)(n_valid * 2048 + n_match); }
            }
        }
    }
    // deterministic butterfly over the 16 lanes sharing ty (one half-warp); addition is commutative,
    // so every lane ends with bit-identical sums.
#pragma unroll
    for (int i = 0; i < RI; ++i) {
        double v = acc[i];
        unsigned long long u = cs[i];
#pragma unroll
        for (int off = TX / 2; off >= 1; off >>= 1) {
            v += __shfl_xor_sync(0xffffffffu, v, off);
            if (VAL) u += __shfl_xor_sync(0xffffffffu, u, off);
        }
        acc[i] = v; cs[i] = u;
    }
    if (tx == 0) {
#pragma unroll
        for (int i = 0; i < RI; ++i) {
            const int r = ty + TY * i;
            if (r < a_len) {
                out[(size_t)blockIdx.x * TA + r] = (double)A_w[a_off + r] * acc[i];
                if (VAL) out_cs[(size_t)blockIdx.x * TA + r] = cs[i];
            }
        }
    }
    if (clk != nullptr && tid == 0) { clk[blockIdx.x * 2 + 0] = clock64() - c0; clk[blockIdx.x * 2 + 1] = gtimer() - g0; }
}

/* ---------------- pipe tests: is POPC on its own quarter-rate pipe or does it share the ALU? ----------------
 * MODE 0: x += popc(x ^ c)  -> LOP3 + POPC + IADD per element
 * MODE 1: x += (x ^ c)      -> LOP3 + IADD per element (no POPC)
 * MODE 2: x += popc(x)      -> POPC + IADD per element */
template <int MODE>
__global__ void k_pipe(int iters, uint32_t *out, long long *clk) {
    uint32_t x[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) x[k] = threadIdx.x * 2654435761u + k * 40503u + blockIdx.x;
    const uint32_t c = 0x5bd1e995u ^ (uint32_t)blockIdx.x;
    long long c0 = 0, g0 = 0;
    if (threadIdx.x == 0) { c0 = clock64(); g0 = gtimer(); }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int u = 0; u < 4; ++u)
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (MODE == 0) x[k] = x[k] + __popc(x[k] ^ c);
                else if (MODE == 1) x[k] = x[k] + (x[k] ^ c);
                else x[k] = x[k] + __popc(x[k]);
            }
    }
    uint32_t s = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) s ^= x[k];
    if (s == 0x9e3779b9u) out[0] = s;
    if (threadIdx.x == 0) { clk[blockIdx.x * 2] = clock64() - c0; clk[blockIdx.x * 2 + 1] = gtimer() - g0; }
}

/* ---------------- memory-pattern tests ----------------
 * scatter: each thread streams its own 1000-byte row in the current kernel's order (25 w x 5 planes, 8-byte loads)
 * coal:    each warp streams one row with 8-byte loads at consecutive addresses */
__global__ void k_mem_scatter(const int *rows, int n, const uint64_t *__restrict__ seq, uint64_t *out) {
    extern __shared__ uint64_t pad[];
    uint64_t acc = 0;
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < n; q += gridDim.x * blockDim.x) {
        const uint64_t *sy = seq + (size_t)rows[q] * 125;
        for (int w = 0; w < 25; ++w) acc += (sy[w] ^ sy[25 + w]) + (sy[50 + w] ^ sy[75 + w]) + sy[100 + w];
    }
    if (acc == 0x123456789ull) { pad[threadIdx.x] = acc; out[0] = pad[(threadIdx.x + 1) % blockDim.x]; }
}
__global__ void k_mem_coal(const int *rows, int n, const uint64_t *__restrict__ seq, uint64_t *out) {
    extern __shared__ uint64_t pad[];
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = (gridDim.x * blockDim.x) >> 5;
    uint64_t acc = 0;
    for (int q = warp; q < n; q += nw) {
        const uint64_t *sy = seq + (size_t)rows[q] * 125;
        for (int k = lane; k < 125; k += 32) acc += sy[k];
    }
    if (acc == 0x123456789ull) { pad[threadIdx.x] = acc; out[0] = pad[(threadIdx.x + 1) % blockDim.x]; }
}

/* ---------------- b1 tensor-core probe (sm_80+ PTX: AND + POPC binary MMA) ---------------- */
__device__ __forceinline__ void bmma_16x8x256(int (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                              uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// A: 16 rows x 8 uint32 (256 bits); B: 8 cols x 8 uint32; D: 16 x 8
__global__ void k_bmma_check(const uint32_t *A, const uint32_t *B, int *D) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    int c[4] = {0, 0, 0, 0};
    bmma_16x8x256(c, A[g * 8 + t], A[(g + 8) * 8 + t], A[g * 8 + 4 + t], A[(g + 8) * 8 + 4 + t], B[g * 8 + t], B[g * 8 + 4 + t]);
    D[g * 8 + 2 * t] = c[0]; D[g * 8 + 2 * t + 1] = c[1]; D[(g + 8) * 8 + 2 * t] = c[2]; D[(g + 8) * 8 + 2 * t + 1] = c[3];
}
__global__ void k_bmma_peak(int iters, int *out, long long *clk) {
    uint32_t a0 = threadIdx.x * 0x9E3779B9u + blockIdx.x, a1 = a0 ^ 0x85ebca6bu, a2 = a0 * 3u + 1u, a3 = ~a0;
    uint32_t b0 = a0 + 0x12345u, b1 = a1 + 77u;
    int c[4][4];
#pragma unroll
    for (int u = 0; u < 4; ++u) { c[u][0] = c[u][1] = c[u][2] = c[u][3] = 0; }
    long long c0 = 0, g0 = 0;
    if (threadIdx.x == 0) { c0 = clock64(); g0 = gtimer(); }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int u = 0; u < 4; ++u) bmma_16x8x256(c[u], a0, a1, a2, a3, b0, b1);
    }
    int s = 0;
#pragma unroll
    for (int u = 0; u < 4; ++u) s += c[u][0] + c[u][1] + c[u][2] + c[u][3];
    if (s == 0x7fffffff) out[0] = s;
    if (threadIdx.x == 0) { clk[blockIdx.x * 2] = clock64() - c0; clk[blockIdx.x * 2 + 1] = gtimer() - g0; }
}

/* ============================== host helpers ============================== */
template <class T> static T *dcopy(const std::vector<T> &h) {
    T *d = nullptr;
    CK(cudaMalloc(&d, std::max<size_t>(16, h.size() * sizeof(T))));
    if (!h.empty()) CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
static double clk_mhz(const std::vector<long long> &clk, int nblocks) {
    double sc = 0, sn = 0;
    for (int b = 0; b < nblocks; ++b) { sc += (double)clk[2 * b]; sn += (double)clk[2 * b + 1]; }
    return sn > 0 ? 1e3 * sc / sn : 0.0;
}

struct CurResult { std::vector<double> D; float best_ms; };
static CurResult run_cur(const char *tag, int K, const std::vector<int> &off, const std::vector<int> &leaf,
                         const std::vector<float> &w, const std::vector<int> &pa, const std::vector<int> &pb,
                         const uint64_t *d_seq_cw, int reps, double ndist) {
    CurResult R; R.best_ms = 1e30f;
    int *d_off = dcopy(off), *d_leaf = dcopy(leaf), *d_pa = dcopy(pa), *d_pb = dcopy(pb);
    float *d_w = dcopy(w);
    const int n = (int)pa.size();
    float *d_D = nullptr; double *d_out = nullptr;
    CK(cudaMalloc(&d_D, (size_t)K * K * sizeof(float)));
    CK(cudaMalloc(&d_out, (size_t)n * sizeof(double)));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int r = 0; r < reps; ++r) {
        CK(cudaMemset(d_out, 0, (size_t)n * sizeof(double)));
        CK(cudaEventRecord(e0));
        k_refine_real_seq_condensed_D_pairs<<<n, 256>>>(K, 25, n, d_pa, d_pb, d_off, d_leaf, d_w, d_seq_cw, d_D, d_out);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        R.best_ms = std::min(R.best_ms, ms);
        printf("  [cur %-22s] rep %d: %9.3f ms  %.3e leafpairs/s\n", tag, r, ms, ndist / (ms * 1e-3));
    }
    R.D.resize(n);
    CK(cudaMemcpy(R.D.data(), d_out, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_off); cudaFree(d_leaf); cudaFree(d_pa); cudaFree(d_pb); cudaFree(d_w); cudaFree(d_D); cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    printf("RESULT cur %-22s ndist=%.4e best_ms=%.3f rate=%.4e leafpairs/s\n", tag, ndist, R.best_ms, ndist / (R.best_ms * 1e-3));
    return R;
}

struct NewResult { std::vector<double> out; std::vector<unsigned long long> cs; float best_ms; double mhz; };
template <bool VAL, bool FU>
static NewResult run_new(const char *tag, const std::vector<int4> &tasks, const std::vector<int> &A_leaf,
                         const std::vector<float> &A_w, const std::vector<int> &B_leaf, const std::vector<float> &B_w,
                         const uint64_t *d_seq_wc, int reps, double ndist) {
    NewResult R; R.best_ms = 1e30f; R.mhz = 0;
    const int nt = (int)tasks.size();
    int4 *d_t = dcopy(tasks);
    int *d_Al = dcopy(A_leaf), *d_Bl = dcopy(B_leaf);
    float *d_Aw = dcopy(A_w), *d_Bw = dcopy(B_w);
    double *d_out = nullptr; unsigned long long *d_cs = nullptr; long long *d_clk = nullptr;
    CK(cudaMalloc(&d_out, (size_t)nt * nk::TA * sizeof(double)));
    CK(cudaMalloc(&d_cs, (size_t)nt * nk::TA * sizeof(unsigned long long)));
    CK(cudaMalloc(&d_clk, (size_t)nt * 2 * sizeof(long long)));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    std::vector<long long> clk((size_t)nt * 2);
    for (int r = 0; r < reps; ++r) {
        CK(cudaMemset(d_out, 0, (size_t)nt * nk::TA * sizeof(double)));
        CK(cudaMemset(d_cs, 0, (size_t)nt * nk::TA * sizeof(unsigned long long)));
        CK(cudaEventRecord(e0));
        k_refine_tiled<VAL, FU><<<nt, 256>>>(d_t, d_Al, d_Aw, d_Bl, d_Bw, d_seq_wc, d_out, d_cs, d_clk);
        CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        CK(cudaMemcpy(clk.data(), d_clk, clk.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        double mhz = clk_mhz(clk, nt);
        if (ms < R.best_ms) { R.best_ms = ms; R.mhz = mhz; }
        printf("  [new %-22s] rep %d: %9.3f ms  %.3e leafpairs/s  sm_clk=%.0f MHz\n", tag, r, ms, ndist / (ms * 1e-3), mhz);
    }
    R.out.resize((size_t)nt * nk::TA);
    R.cs.resize((size_t)nt * nk::TA);
    CK(cudaMemcpy(R.out.data(), d_out, R.out.size() * sizeof(double), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(R.cs.data(), d_cs, R.cs.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    cudaFree(d_t); cudaFree(d_Al); cudaFree(d_Bl); cudaFree(d_Aw); cudaFree(d_Bw); cudaFree(d_out); cudaFree(d_cs); cudaFree(d_clk);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    double rate = ndist / (R.best_ms * 1e-3);
    double clk_per = (R.mhz > 0) ? (double)g_nsm * R.mhz * 1e6 / rate : 0;
    printf("RESULT new %-22s ndist=%.4e best_ms=%.3f rate=%.4e leafpairs/s sm_clk=%.0fMHz SMclk_per_leafpair=%.2f\n",
           tag, ndist, R.best_ms, rate, R.mhz, clk_per);
    return R;
}

static double relerr(double a, double b) { double s = std::max(std::fabs(a), std::fabs(b)); return s > 0 ? std::fabs(a - b) / s : 0.0; }

int main(int argc, char **argv) {
    g_t0 = wall();
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    int clk_khz = 0, memclk_khz = 0, busw = 0;
    cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev);
    cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, dev);
    cudaDeviceGetAttribute(&busw, cudaDevAttrGlobalMemoryBusWidth, dev);
    g_nsm = prop.multiProcessorCount;
    printf("DEVICE %s sm_%d%d SMs=%d max_sm_clock=%d MHz mem_clock=%d MHz bus=%d bit L2=%d KB smem/SM=%zu KB regs/SM=%d\n",
           prop.name, prop.major, prop.minor, g_nsm, clk_khz / 1000, memclk_khz / 1000, busw, prop.l2CacheSize / 1024,
           prop.sharedMemPerMultiprocessor / 1024, prop.regsPerMultiprocessor);
    {
        cudaFuncAttributes fa;
        CK(cudaFuncGetAttributes(&fa, k_refine_real_seq_condensed_D_pairs));
        printf("ATTR cur regs=%d static_smem=%zu local=%zu\n", fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes);
        int nb = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_refine_real_seq_condensed_D_pairs, 256, 0));
        printf("ATTR cur blocks/SM=%d\n", nb);
        CK(cudaFuncGetAttributes(&fa, k_refine_tiled<false, true>));
        printf("ATTR new(full unroll) regs=%d static_smem=%zu local=%zu\n", fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes);
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_refine_tiled<false, true>, 256, 0));
        printf("ATTR new(full unroll) blocks/SM=%d\n", nb);
        CK(cudaFuncGetAttributes(&fa, k_refine_tiled<false, false>));
        printf("ATTR new(unroll1) regs=%d static_smem=%zu local=%zu\n", fa.numRegs, fa.sharedSizeBytes, fa.localSizeBytes);
    }

    /* ---------- data ---------- */
    const int N = 100000;
    uint64_t *d_cw = nullptr, *d_wc = nullptr;
    CK(cudaMalloc(&d_cw, (size_t)N * 125 * sizeof(uint64_t)));
    CK(cudaMalloc(&d_wc, (size_t)N * 125 * sizeof(uint64_t)));
    k_gen<<<(N + 255) / 256, 256>>>(d_cw, d_wc, N, 0x1234abcdULL);
    CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
    printf("DATA N=%d rows x 1000 B, generated (%.2f s)\n", N, wall() - g_t0);

    std::mt19937_64 rng(20261103);
    auto rleaf = [&]() { return (int)(rng() % (uint64_t)N); };
    auto rw = [&]() { return ldexpf(1.0f, -(int)(1 + rng() % 100)); };
    const int nsm = g_nsm;

    /* ---------- 1. pipe tests ---------- */
    if (budget_ok("pipe")) {
        const int blocks = nsm * 6, iters = 8192;
        uint32_t *d_o; long long *d_clk; CK(cudaMalloc(&d_o, 64)); CK(cudaMalloc(&d_clk, (size_t)blocks * 2 * sizeof(long long)));
        std::vector<long long> clk((size_t)blocks * 2);
        double elems = (double)blocks * 256 * iters * 32;
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        for (int mode = 0; mode < 3; ++mode) {
            float best = 1e30f; double mhz = 0;
            for (int r = 0; r < 3; ++r) {
                cudaEventRecord(e0);
                if (mode == 0) k_pipe<0><<<blocks, 256>>>(iters, d_o, d_clk);
                if (mode == 1) k_pipe<1><<<blocks, 256>>>(iters, d_o, d_clk);
                if (mode == 2) k_pipe<2><<<blocks, 256>>>(iters, d_o, d_clk);
                cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                CK(cudaMemcpy(clk.data(), d_clk, clk.size() * sizeof(long long), cudaMemcpyDeviceToHost));
                if (ms < best) { best = ms; mhz = clk_mhz(clk, blocks); }
            }
            double per_sm_clk = elems / ((double)best * 1e-3 * nsm * mhz * 1e6);
            printf("RESULT pipe mode=%d (%s) best_ms=%.3f sm_clk=%.0fMHz elements_per_SM_per_clk=%.2f\n", mode,
                   mode == 0 ? "LOP3+POPC+IADD" : mode == 1 ? "LOP3+IADD" : "POPC+IADD", best, mhz, per_sm_clk);
        }
        cudaFree(d_o); cudaFree(d_clk);
    }

    /* ---------- 2. memory-pattern tests ---------- */
    if (budget_ok("mem")) {
        const int n = 1 << 22;
        std::vector<int> rows(n);
        for (int i = 0; i < n; ++i) rows[i] = rleaf();
        int *d_rows = dcopy(rows);
        uint64_t *d_o; CK(cudaMalloc(&d_o, 64));
        CK(cudaFuncSetAttribute(k_mem_scatter, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        CK(cudaFuncSetAttribute(k_mem_coal, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        struct Cfg { const char *name; int scatter; int bps; size_t smem; };
        Cfg cfgs[] = {{"scatter 1blk/SM smem52K", 1, 1, 52 * 1024}, {"scatter 2blk/SM smem38K", 1, 2, 38400},
                      {"scatter 6blk/SM smem0", 1, 6, 0}, {"coal 1blk/SM smem52K", 0, 1, 52 * 1024},
                      {"coal 6blk/SM smem0", 0, 6, 0}};
        for (const Cfg &c : cfgs) {
            float best = 1e30f;
            for (int r = 0; r < 3; ++r) {
                cudaEventRecord(e0);
                if (c.scatter) k_mem_scatter<<<nsm * c.bps, 256, c.smem>>>(d_rows, n, d_cw, d_o);
                else k_mem_coal<<<nsm * c.bps, 256, c.smem>>>(d_rows, n, d_cw, d_o);
                cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
                float ms; cudaEventElapsedTime(&ms, e0, e1); best = std::min(best, ms);
            }
            int nb = 0;
            if (c.scatter) cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_mem_scatter, 256, c.smem);
            else cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k_mem_coal, 256, c.smem);
            double rows_s = n / (best * 1e-3);
            printf("RESULT mem %-26s occ_blocks/SM=%d best_ms=%.3f rows/s=%.4e useful_GB/s=%.1f\n", c.name, nb, best, rows_s, rows_s * 1000.0 / 1e9);
        }
        cudaFree(d_rows); cudaFree(d_o);
    }

    /* ---------- shared A/B lists for WL-1/WL-2 ---------- */
    const int nA1 = nsm * 4 * 16;   // 5376 rows on 84 SMs
    const int nB1 = 32768;
    std::vector<int> A1(nA1), B1(nB1);
    std::vector<float> A1w(nA1), B1w(nB1);
    for (int i = 0; i < nA1; ++i) { A1[i] = rleaf(); A1w[i] = rw(); }
    for (int i = 0; i < nB1; ++i) { B1[i] = rleaf(); B1w[i] = rw(); }

    /* ---------- 3. validation with unit weights (exact integer counts + reassociation-only FP64 diffs) ---------- */
    if (budget_ok("validation")) {
        const int nA = 64, nB = 4096;
        std::vector<int> Al(A1.begin(), A1.begin() + nA), Bl(B1.begin(), B1.begin() + nB);
        std::vector<float> Aw(nA, 1.0f), Bw(nB, 1.0f);
        // reference counts
        int *d_Al = dcopy(Al), *d_Bl = dcopy(Bl);
        unsigned long long *d_cs; CK(cudaMalloc(&d_cs, nA * sizeof(unsigned long long)));
        CK(cudaMemset(d_cs, 0, nA * sizeof(unsigned long long)));
        k_ref_counts<<<dim3((nB + 255) / 256, nA), 256>>>(nA, nB, d_Al, d_Bl, d_cw, d_cs);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        std::vector<unsigned long long> cs_ref(nA);
        CK(cudaMemcpy(cs_ref.data(), d_cs, nA * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        cudaFree(d_Al); cudaFree(d_Bl); cudaFree(d_cs);
        // current kernel, batched singleton path: K = nA + 1
        std::vector<int> off, leaf, pa, pb; std::vector<float> w;
        off.push_back(0);
        for (int r = 0; r < nA; ++r) { leaf.push_back(Al[r]); w.push_back(1.0f); off.push_back((int)leaf.size()); }
        for (int q = 0; q < nB; ++q) { leaf.push_back(Bl[q]); w.push_back(1.0f); }
        off.push_back((int)leaf.size());
        for (int r = 0; r < nA; ++r) { pa.push_back(r); pb.push_back(nA); }
        CurResult cb = run_cur("V batched(unit w)", nA + 1, off, leaf, w, pa, pb, d_cw, 1, (double)nA * nB);
        // current kernel, tile path: K = 2
        std::vector<int> off2 = {0, nA, nA + nB}, leaf2(Al); leaf2.insert(leaf2.end(), Bl.begin(), Bl.end());
        std::vector<float> w2(nA + nB, 1.0f);
        CurResult ct = run_cur("V tile(unit w)", 2, off2, leaf2, w2, std::vector<int>{0}, std::vector<int>{1}, d_cw, 1, (double)nA * nB);
        // new kernel with validation: 1 A tile x 4 B splits
        std::vector<int4> tasks;
        for (int h = 0; h < 4; ++h) tasks.push_back(make_int4(0, nA, h * (nB / 4), nB / 4));
        NewResult nv = run_new<true, true>("V new(unit w)", tasks, Al, Aw, Bl, Bw, d_wc, 1, (double)nA * nB);
        int cs_bad = 0; double max_rel_row = 0, sum_new = 0, sum_batched = 0;
        for (int r = 0; r < nA; ++r) {
            unsigned long long c = 0; double S = 0;
            for (int h = 0; h < 4; ++h) { c += nv.cs[(size_t)h * nk::TA + r]; S += nv.out[(size_t)h * nk::TA + r]; }
            if (c != cs_ref[r]) cs_bad++;
            max_rel_row = std::max(max_rel_row, relerr(S, cb.D[r]));
            sum_new += S; sum_batched += cb.D[r];
        }
        printf("VALID exact_count_checksum_mismatch_rows=%d/%d  max_rel(new_row vs cur_batched_row)=%.3e  "
               "rel(sum new rows vs cur tile pair)=%.3e  rel(sum cur batched rows vs cur tile pair)=%.3e\n",
               cs_bad, nA, max_rel_row, relerr(sum_new, ct.D[0]), relerr(sum_batched, ct.D[0]));
    }

    /* ---------- 4. WL-1: tile path, 336 pairs of (16-row chunk, 32768-row B) ---------- */
    std::vector<double> rowS_new1;
    if (budget_ok("WL1")) {
        const int nch = nA1 / 16;
        std::vector<int> off, leaf(A1), pa, pb; std::vector<float> w(A1w);
        for (int i = 0; i <= nch; ++i) off.push_back(16 * i);
        leaf.insert(leaf.end(), B1.begin(), B1.end()); w.insert(w.end(), B1w.begin(), B1w.end());
        off.push_back((int)leaf.size());
        for (int i = 0; i < nch; ++i) { pa.push_back(i); pb.push_back(nch); }
        const double nd = (double)nA1 * nB1;
        CurResult c1 = run_cur("WL1 tile16 x 32768", nch + 1, off, leaf, w, pa, pb, d_cw, 3, nd);
        std::vector<int4> tasks;
        for (int at = 0; at < nA1 / 64; ++at)
            for (int bs = 0; bs < 4; ++bs) tasks.push_back(make_int4(at * 64, 64, bs * (nB1 / 4), nB1 / 4));
        NewResult n1 = run_new<false, true>("WL1 64x64 tiles", tasks, A1, A1w, B1, B1w, d_wc, 4, nd);
        NewResult n1b = run_new<false, false>("WL1 64x64 (wl unroll1)", tasks, A1, A1w, B1, B1w, d_wc, 3, nd);
        rowS_new1.assign(nA1, 0.0);
        for (int at = 0; at < nA1 / 64; ++at)
            for (int bs = 0; bs < 4; ++bs)
                for (int r = 0; r < 64; ++r) rowS_new1[at * 64 + r] += n1.out[(size_t)(at * 4 + bs) * 64 + r];
        double mx = 0, mxb = 0;
        for (int i = 0; i < nch; ++i) {
            double s = 0, sb = 0;
            for (int r = 0; r < 16; ++r) s += rowS_new1[16 * i + r];
            for (int at = 0; at < 1; ++at) {}
            // unroll1 variant per chunk
            for (int r = 0; r < 16; ++r) {
                int R = 16 * i + r, atile = R / 64, rr = R % 64;
                for (int bs = 0; bs < 4; ++bs) sb += n1b.out[(size_t)(atile * 4 + bs) * 64 + rr];
            }
            mx = std::max(mx, relerr(s, c1.D[i]));
            mxb = std::max(mxb, relerr(sb, s));
        }
        printf("VALID WL1 max_rel(new chunk vs cur chunk)=%.3e  max_rel(unroll1 vs full-unroll)=%.3e\n", mx, mxb);
        printf("SPEEDUP WL1 new/cur = %.2fx\n", c1.best_ms / n1.best_ms);
    }

    /* ---------- 5. WL-2: batched singleton path, 5376 singletons sharing a 32768-row B ---------- */
    if (budget_ok("WL2")) {
        std::vector<int> off, leaf, pa, pb; std::vector<float> w;
        off.push_back(0);
        for (int r = 0; r < nA1; ++r) { leaf.push_back(A1[r]); w.push_back(A1w[r]); off.push_back((int)leaf.size()); }
        leaf.insert(leaf.end(), B1.begin(), B1.end()); w.insert(w.end(), B1w.begin(), B1w.end());
        off.push_back((int)leaf.size());
        for (int r = 0; r < nA1; ++r) { pa.push_back(r); pb.push_back(nA1); }
        const double nd = (double)nA1 * nB1;
        CurResult c2 = run_cur("WL2 batched16 x 32768", nA1 + 1, off, leaf, w, pa, pb, d_cw, 3, nd);
        if (!rowS_new1.empty()) {
            double mx = 0;
            for (int r = 0; r < nA1; ++r) mx = std::max(mx, relerr(rowS_new1[r], c2.D[r]));
            printf("VALID WL2 max_rel(new row vs cur batched pair)=%.3e (new kernel timing = WL1 new)\n", mx);
        }
    }

    /* ---------- 6. WL-3: unbatched singleton path (len_a == 1, distinct B per pair) ---------- */
    if (budget_ok("WL3")) {
        const int np = nsm * 4, lb = 16384;
        std::vector<int> off, leaf, pa, pb; std::vector<float> w;
        off.push_back(0);
        for (int i = 0; i < np; ++i) { leaf.push_back(rleaf()); w.push_back(rw()); off.push_back((int)leaf.size()); }
        for (int i = 0; i < np; ++i) {
            for (int q = 0; q < lb; ++q) { leaf.push_back(rleaf()); w.push_back(rw()); }
            off.push_back((int)leaf.size());
        }
        for (int i = 0; i < np; ++i) { pa.push_back(i); pb.push_back(np + i); }
        run_cur("WL3 single(len_a=1) x16384", 2 * np, off, leaf, w, pa, pb, d_cw, 3, (double)np * lb);
    }

    /* ---------- 7. WL-4: small tiles (len_a = 4), 2560 pairs, B shared by groups of 16 pairs ---------- */
    if (budget_ok("WL4")) {
        const int ng = 160, ppg = 16, la = 4, lb = 8192, np = ng * ppg;
        std::vector<int> Al, Bl; std::vector<float> Aw, Bw;
        for (int i = 0; i < np * la; ++i) { Al.push_back(rleaf()); Aw.push_back(rw()); }
        for (int i = 0; i < ng * lb; ++i) { Bl.push_back(rleaf()); Bw.push_back(rw()); }
        std::vector<int> off, leaf(Al), pa, pb; std::vector<float> w(Aw);
        for (int u = 0; u <= np; ++u) off.push_back(la * u);
        leaf.insert(leaf.end(), Bl.begin(), Bl.end()); w.insert(w.end(), Bw.begin(), Bw.end());
        for (int g = 1; g <= ng; ++g) off.push_back(np * la + g * lb);
        for (int u = 0; u < np; ++u) { pa.push_back(u); pb.push_back(np + u / ppg); }
        const double nd = (double)np * la * lb;
        CurResult c4 = run_cur("WL4 tile4 x 8192", np + ng, off, leaf, w, pa, pb, d_cw, 3, nd);
        std::vector<int4> tasks;   // pack the 16 pairs sharing B_g into one 64-row A tile
        for (int g = 0; g < ng; ++g)
            for (int h = 0; h < 2; ++h) tasks.push_back(make_int4(g * 64, 64, g * lb + h * (lb / 2), lb / 2));
        NewResult n4 = run_new<false, true>("WL4 packed 64x64", tasks, Al, Aw, Bl, Bw, d_wc, 4, nd);
        double mx = 0;
        for (int u = 0; u < np; ++u) {
            int g = u / ppg, base = (u % ppg) * la;
            double s = 0;
            for (int r = 0; r < la; ++r)
                for (int h = 0; h < 2; ++h) s += n4.out[(size_t)(g * 2 + h) * 64 + base + r];
            mx = std::max(mx, relerr(s, c4.D[u]));
        }
        printf("VALID WL4 max_rel(new pair vs cur pair)=%.3e\n", mx);
        printf("SPEEDUP WL4 new/cur = %.2fx\n", c4.best_ms / n4.best_ms);
    }

    /* ---------- 8. b1 tensor-core probe (last: an illegal-instruction fault would end the process) ---------- */
    if (budget_ok("bmma")) {
        std::vector<uint32_t> A(16 * 8), B(8 * 8);
        for (auto &x : A) x = (uint32_t)rng();
        for (auto &x : B) x = (uint32_t)rng();
        uint32_t *d_A = dcopy(A), *d_B = dcopy(B); int *d_D; CK(cudaMalloc(&d_D, 16 * 8 * sizeof(int)));
        k_bmma_check<<<1, 32>>>(d_A, d_B, d_D);
        cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) { printf("BMMA check failed: %s\n", cudaGetErrorString(e)); return 0; }
        std::vector<int> D(16 * 8);
        CK(cudaMemcpy(D.data(), d_D, D.size() * sizeof(int), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 8; ++c) {
                int ref = 0;
                for (int k = 0; k < 8; ++k) ref += __builtin_popcount(A[r * 8 + k] & B[c * 8 + k]);
                if (ref != D[r * 8 + c]) bad++;
            }
        printf("BMMA layout check (row-k words t, 4+t; col-k words t, 4+t): mismatches=%d/128\n", bad);
        const int blocks = nsm * 4;
        long long *d_clk; int *d_o; CK(cudaMalloc(&d_clk, (size_t)blocks * 2 * sizeof(long long))); CK(cudaMalloc(&d_o, 64));
        std::vector<long long> clk((size_t)blocks * 2);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        int iters = 2000;
        cudaEventRecord(e0); k_bmma_peak<<<blocks, 256>>>(iters, d_o, d_clk); cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        int iters2 = (int)std::min(400000.0, std::max(2000.0, iters * 150.0 / std::max(ms, 0.01f)));
        cudaEventRecord(e0); k_bmma_peak<<<blocks, 256>>>(iters2, d_o, d_clk); cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        cudaEventElapsedTime(&ms, e0, e1);
        CK(cudaMemcpy(clk.data(), d_clk, clk.size() * sizeof(long long), cudaMemcpyDeviceToHost));
        double mhz = clk_mhz(clk, blocks);
        double nmma = (double)blocks * 8 * 4 * iters2;
        double mma_per_sm_clk = nmma / (ms * 1e-3 * nsm * mhz * 1e6);
        double bitmac_s = nmma * 16 * 8 * 256 / (ms * 1e-3);
        printf("RESULT bmma m16n8k256.and.popc iters=%d ms=%.3f sm_clk=%.0fMHz mma/SM/clk=%.4f bitMAC/s=%.4e "
               "=> count-only bound %.3e leafpairs/s at 32 k-steps per 128 leaf pairs\n",
               iters2, ms, mhz, mma_per_sm_clk, bitmac_s, nmma / (ms * 1e-3) * 128.0 / 32.0);
    }
    printf("DONE total %.2f s\n", wall() - g_t0);
    return 0;
}
