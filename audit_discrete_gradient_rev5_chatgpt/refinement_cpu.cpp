#include <cstdint>
#include <cmath>
#define __device__
#define __forceinline__ inline
#define __shared__
#define __syncthreads() ((void)0)
#define __popcll __builtin_popcountll
struct Idx {int x;};static Idx blockIdx={0},blockDim={256},threadIdx={0};
__device__ __forceinline__ float dev_jc69_dist_popc(
    int32_t x, int32_t y,
    const uint64_t * __restrict__ d_packed_seqs,
    int32_t W
) {
    if (x == y) return 0.0f;
    const uint64_t *sx = &d_packed_seqs[(size_t)x * 5 * W];
    const uint64_t *sy = &d_packed_seqs[(size_t)y * 5 * W];
    int32_t n_valid = 0;
    int32_t n_match = 0;
    #pragma unroll
    for (int w = 0; w < 25; ++w) {
        uint64_t vx = sx[4 * 25 + w];
        uint64_t vy = sy[4 * 25 + w];
        n_valid += __popcll(vx & vy);
        uint64_t m = (sx[0 * 25 + w] & sy[0 * 25 + w])
                   | (sx[1 * 25 + w] & sy[1 * 25 + w])
                   | (sx[2 * 25 + w] & sy[2 * 25 + w])
                   | (sx[3 * 25 + w] & sy[3 * 25 + w]);
        n_match += __popcll(m);
    }
    if (n_valid <= 0) return 1.5f;
    float p = (float)(n_valid - n_match) / (float)n_valid;
    if (p >= 0.74f) p = 0.74f;
    float d = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
    return (d > 0.0f) ? d : 0.0f;
}
static void k_refine_real_seq_condensed_D_pairs(
    int32_t K, int32_t W, int32_t n_pairs,
    const int32_t  * __restrict__ d_pair_a,
    const int32_t  * __restrict__ d_pair_b,
    const int32_t  * __restrict__ d_exact_offsets,
    const int32_t  * __restrict__ d_exact_leaves,
    const float    * __restrict__ d_exact_weights,
    const uint64_t * __restrict__ d_packed_seqs,
    float  * __restrict__ d_D_cond,
    double * __restrict__ d_pair_exact_D
) {
    int32_t pid = blockIdx.x;
    if (pid >= n_pairs) return;
    int32_t a = d_pair_a[pid];
    int32_t b = d_pair_b[pid];
    int32_t off_a = d_exact_offsets[a], end_a = d_exact_offsets[a + 1];
    int32_t off_b = d_exact_offsets[b], end_b = d_exact_offsets[b + 1];
    int32_t len_a = end_a - off_a;
    int32_t len_b = end_b - off_b;
    if (len_a > len_b) {
        int32_t to = off_a; off_a = off_b; off_b = to;
        int32_t te = end_a; end_a = end_b; end_b = te;
        int32_t tl = len_a; len_a = len_b; len_b = tl;
    }

    __shared__ uint64_t s_seq_a[256];
    double s_sum[256];
    for (threadIdx.x=0;threadIdx.x<256;threadIdx.x++) {
    double my_sum = 0.0;
    if (len_a == 1 && 5 * W <= 256) {
        int32_t la = d_exact_leaves[off_a];
        double wa = (double)d_exact_weights[off_a];
        const uint64_t *sx = d_packed_seqs + (size_t)la * (5 * W);
        for (int k=0;k<5*W;k++) s_seq_a[k]=sx[k];
        __syncthreads();
        if (wa != 0.0) {
            for (int32_t q = off_b + threadIdx.x; q < end_b; q += blockDim.x) {
                float wq = d_exact_weights[q];
                if (wq == 0.0f) continue;
                int32_t lb = d_exact_leaves[q];
                const uint64_t *sy = d_packed_seqs + (size_t)lb * (5 * W);
                int32_t n_valid = 0, n_match = 0;
                #pragma unroll
                for (int32_t w = 0; w < 25; ++w) {
                    uint64_t v = s_seq_a[4 * 25 + w] & sy[4 * 25 + w];
                    n_valid += __popcll(v);
                    uint64_t m = (s_seq_a[0 * 25 + w] & sy[0 * 25 + w])
                               | (s_seq_a[1 * 25 + w] & sy[1 * 25 + w])
                               | (s_seq_a[2 * 25 + w] & sy[2 * 25 + w])
                               | (s_seq_a[3 * 25 + w] & sy[3 * 25 + w]);
                    n_match += __popcll(m);
                }
                float d_jc = 1.5f;
                if (n_valid > 0) {
                    float p = (float)(n_valid - n_match) / (float)n_valid;
                    if (p >= 0.74f) p = 0.74f;
                    d_jc = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
                    if (d_jc < 0.0f) d_jc = 0.0f;
                }
                my_sum += (wa * (double)wq) * (double)d_jc;
            }
        }
    } else {
        int64_t total_pq = (int64_t)len_a * (int64_t)len_b;
        for (int64_t idx = threadIdx.x; idx < total_pq; idx += blockDim.x) {
            int32_t p = off_a + (int32_t)(idx / len_b);
            int32_t q = off_b + (int32_t)(idx % len_b);
            double w = (double)d_exact_weights[p] * (double)d_exact_weights[q];
            if (w == 0.0) continue;
            int32_t la = d_exact_leaves[p];
            int32_t lb = d_exact_leaves[q];
            double d = (double)dev_jc69_dist_popc(la, lb, d_packed_seqs, W);
            my_sum += w * d;
        }
    }
    s_sum[threadIdx.x] = my_sum;
    }
    __syncthreads();
    for (int32_t s = 128; s > 0; s >>= 1) {
        for(int t=0;t<s;t++) s_sum[t] += s_sum[t+s];
        __syncthreads();
    }
    if (true) {
        double exact_val = s_sum[0];
        d_pair_exact_D[pid] = exact_val;
        d_D_cond[(size_t)a * K + b] = (float)exact_val;
        d_D_cond[(size_t)b * K + a] = (float)exact_val;
    }
}
extern "C" double refine(int na,int nb,const float*w,const uint64_t*seq){int K=2,pa=0,pb=1;int off[3]={0,na,na+nb};int*leaves=new int[na+nb];for(int i=0;i<na+nb;i++)leaves[i]=i;float D[4]={0};double result=0;blockIdx.x=0;k_refine_real_seq_condensed_D_pairs(K,25,1,&pa,&pb,off,leaves,w,seq,D,&result);delete[]leaves;return result;}
extern "C" double independent_pair_sum(int na,int nb,const float*w,const uint64_t*seq){double total=0;for(int a=0;a<na;a++)for(int b=na;b<na+nb;b++)total+=(double)w[a]*(double)w[b]*(double)dev_jc69_dist_popc(a,b,seq,25);return total;}
