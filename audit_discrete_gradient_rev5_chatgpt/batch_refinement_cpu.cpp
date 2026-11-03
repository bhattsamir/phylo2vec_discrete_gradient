#include <cstdint>
#include <cmath>
#include <vector>
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

    const int32_t base_pid = pid & ~15;
    const int32_t b0 = d_pair_b[base_pid];
    const int32_t off_b0 = d_exact_offsets[b0], end_b0 = d_exact_offsets[b0 + 1];
    const int32_t len_b0 = end_b0 - off_b0;
    int32_t batch_sz = 0;
    if (5 * W == 125 && len_b0 > 1) {
        int32_t max_k = n_pairs - base_pid;
        if (max_k > 16) max_k = 16;
        for (int32_t k = 0; k < max_k; ++k) {
            int32_t ak = d_pair_a[base_pid + k];
            int32_t bk = d_pair_b[base_pid + k];
            int32_t lak = d_exact_offsets[ak + 1] - d_exact_offsets[ak];
            if (bk == b0 && lak == 1) batch_sz++;
            else break;
        }
    }
    if (batch_sz >= 2 && pid < base_pid + batch_sz) {
        if (pid != base_pid) return;
        __shared__ uint64_t s_batch_seq_a[16][125];
        __shared__ double s_wa[16];
        for (int32_t idx = 0; idx < batch_sz * 125; ++idx) {
            int32_t k = idx / 125;
            int32_t w = idx % 125;
            int32_t ak = d_pair_a[base_pid + k];
            int32_t la = d_exact_leaves[d_exact_offsets[ak]];
            s_batch_seq_a[k][w] = d_packed_seqs[(size_t)la * 125 + w];
        }
        for (threadIdx.x=0;threadIdx.x<batch_sz;threadIdx.x++) {
            int32_t ak = d_pair_a[base_pid + threadIdx.x];
            s_wa[threadIdx.x] = (double)d_exact_weights[d_exact_offsets[ak]];
        }
        __syncthreads();

        double all_sums[256][16];
        for(threadIdx.x=0;threadIdx.x<256;threadIdx.x++){
        double*my_sums=all_sums[threadIdx.x];
        #pragma unroll
        for (int32_t k = 0; k < 16; ++k) my_sums[k] = 0.0;

        for (int32_t q = off_b0 + threadIdx.x; q < end_b0; q += blockDim.x) {
            float wq = d_exact_weights[q];
            if (wq == 0.0f) continue;
            int32_t lb = d_exact_leaves[q];
            const uint64_t *sy = d_packed_seqs + (size_t)lb * 125;
            int32_t n_valid[16], n_match[16];
            #pragma unroll
            for (int32_t k = 0; k < 16; ++k) { n_valid[k] = 0; n_match[k] = 0; }
            for (int32_t w = 0; w < 25; ++w) {
                uint64_t sy0 = sy[0 * 25 + w];
                uint64_t sy1 = sy[1 * 25 + w];
                uint64_t sy2 = sy[2 * 25 + w];
                uint64_t sy3 = sy[3 * 25 + w];
                uint64_t sy4 = sy[4 * 25 + w];
                for (int32_t k = 0; k < batch_sz; ++k) {
                    n_valid[k] += __popcll(s_batch_seq_a[k][4 * 25 + w] & sy4);
                    uint64_t m = (s_batch_seq_a[k][0 * 25 + w] & sy0)
                               | (s_batch_seq_a[k][1 * 25 + w] & sy1)
                               | (s_batch_seq_a[k][2 * 25 + w] & sy2)
                               | (s_batch_seq_a[k][3 * 25 + w] & sy3);
                    n_match[k] += __popcll(m);
                }
            }
            for (int32_t k = 0; k < batch_sz; ++k) {
                float d_jc = 1.5f;
                if (n_valid[k] > 0) {
                    float p = (float)(n_valid[k] - n_match[k]) / (float)n_valid[k];
                    if (p >= 0.74f) p = 0.74f;
                    d_jc = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
                    if (d_jc < 0.0f) d_jc = 0.0f;
                }
                my_sums[k] += (s_wa[k] * (double)wq) * (double)d_jc;
            }
        }
        }
        __shared__ double s_red[256];
        for (int32_t k = 0; k < batch_sz; ++k) {
            for(int t=0;t<256;t++)s_red[t] = all_sums[t][k];
            __syncthreads();
            for (int32_t s = 128; s > 0; s >>= 1) {
                for(int t=0;t<s;t++)s_red[t] += s_red[t+s];
                __syncthreads();
            }
            if (true) {
                double exact_val = s_red[0];
                int32_t ak = d_pair_a[base_pid + k];
                d_pair_exact_D[base_pid + k] = exact_val;
                d_D_cond[(size_t)ak * K + b0] = (float)exact_val;
                d_D_cond[(size_t)b0 * K + ak] = (float)exact_val;
            }
            __syncthreads();
        }
        return;
    }

}

extern "C" void run(int K,int n,const int*pa,const int*pb,const int*off,const float*w,const uint64_t*seq,double*out,double*ref){std::vector<int>ls(off[K]);for(int i=0;i<off[K];i++)ls[i]=i;std::vector<float>D(K*K);for(int p=0;p<n;p++){blockIdx.x=p;k_refine_real_seq_condensed_D_pairs(K,25,n,pa,pb,off,ls.data(),w,seq,D.data(),out);double sum=0;for(int a=off[pa[p]];a<off[pa[p]+1];a++)for(int b=off[pb[p]];b<off[pb[p]+1];b++)sum+=(double)w[a]*(double)w[b]*(double)dev_jc69_dist_popc(a,b,seq,25);ref[p]=sum;}}
