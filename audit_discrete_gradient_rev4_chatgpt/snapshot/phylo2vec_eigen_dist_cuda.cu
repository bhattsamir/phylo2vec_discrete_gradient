/*
 * phylo2vec_eigen_dist_cuda.cu
 *
 * Multi-GPU CUDA Implementation of the Penn et al. (2025) Eigenvalue Distance
 * and Bayesian MAP GTR+\Gamma Estimator for DNA/RNA Alignments.
 *
 * Architectural Optimizations:
 *   1. Coalesced Shared-Memory Bitplane Tiling (16x16 taxon tiles x 8-word chunks)
 *      with +1 inner-dimension padding to eliminate shared-memory bank conflicts.
 *   2. Upper-Triangular Tile Pruning (block_row <= block_col):
 *      Skips strictly lower-triangular 16x16 blocks and evaluates SVD only for
 *      i < j, mirroring D[j, i] = D[i, j] for exact symmetry and 2x speedup.
 *   3. Exact 64-bit POPCNT (__popcll) + FP64 In-Register 4x4 Jacobi SVD,
 *      reduced-subspace projection for missing states, and log-space iteratively
 *      reweighted SymWLS matching the CPU reference implementation.
 *   4. Pair-Balanced Multi-GPU OpenMP Host Dispatch across 1x - 8x GPUs.
 */

#include "phylo2vec_eigen_dist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define CUDA_CHECK_RET(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -100; \
    } \
} while (0)

#define DEV_PSEUDOCOUNT_EPS   1e-8
#define DEV_SIGMA_FLOOR       1e-4
#define DEV_SIGMA_CEIL        (1.0 - 1e-12)
#define DEV_MIN_STATE_FREQ    1e-3
#define MAX_EIGEN_GPUS        16
#define WORD_CHUNK            8

static inline uint64_t host_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ============================================================================
 * 1. Device FP64 4x4 Symmetric Jacobi Eigensolver & SVD
 * ============================================================================ */
__device__ __forceinline__ void dev_sym4x4_jacobi_eigen_f64(
    const double A_in[16],
    double evals[4],
    double V[16]
) {
    double A[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) A[i] = A_in[i];

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            V[i * 4 + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    const int p0[6] = {0, 0, 0, 1, 1, 2};
    const int p1[6] = {1, 2, 3, 2, 3, 3};

    for (int sweep = 0; sweep < 6; ++sweep) {
        double off_norm2 = 0.0;
        #pragma unroll
        for (int p = 0; p < 6; ++p) {
            double apq = A[p0[p] * 4 + p1[p]];
            off_norm2 += apq * apq;
        }
        if (off_norm2 < 1e-30) break;

        #pragma unroll
        for (int p = 0; p < 6; ++p) {
            int ip = p0[p];
            int iq = p1[p];
            double apq = A[ip * 4 + iq];
            if (fabs(apq) < 1e-18) continue;

            double app = A[ip * 4 + ip];
            double aqq = A[iq * 4 + iq];
            double tau = (aqq - app) / (2.0 * apq);
            double t = (tau >= 0.0)
                ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
            double c = rsqrt(1.0 + t * t);
            double s = t * c;

            A[ip * 4 + ip] = app - t * apq;
            A[iq * 4 + iq] = aqq + t * apq;
            A[ip * 4 + iq] = 0.0;
            A[iq * 4 + ip] = 0.0;

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r != ip && r != iq) {
                    double arp = A[r * 4 + ip];
                    double arq = A[r * 4 + iq];
                    A[r * 4 + ip] = c * arp - s * arq;
                    A[ip * 4 + r] = A[r * 4 + ip];
                    A[r * 4 + iq] = s * arp + c * arq;
                    A[iq * 4 + r] = A[r * 4 + iq];
                }
            }

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                double vip = V[r * 4 + ip];
                double viq = V[r * 4 + iq];
                V[r * 4 + ip] = c * vip - s * viq;
                V[r * 4 + iq] = s * vip + c * viq;
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) evals[i] = A[i * 4 + i];

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        int best = i;
        for (int j = i + 1; j < 4; ++j) {
            if (evals[j] > evals[best]) best = j;
        }
        if (best != i) {
            double tmp = evals[i];
            evals[i] = evals[best];
            evals[best] = tmp;
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                double vtmp = V[r * 4 + i];
                V[r * 4 + i] = V[r * 4 + best];
                V[r * 4 + best] = vtmp;
            }
        }
    }
}

/* ============================================================================
 * 2. Device Eigenvalue-Only 4x4 Symmetric Jacobi (No Eigenvector Tracking)
 * ============================================================================ */
__device__ __forceinline__ void dev_sym4x4_jacobi_evals_only_f64(
    const double G_in[16],
    double evals[4]
) {
    double A[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) A[i] = G_in[i];

    const int p0[6] = {0, 0, 0, 1, 1, 2};
    const int p1[6] = {1, 2, 3, 2, 3, 3};

    for (int sweep = 0; sweep < 6; ++sweep) {
        double off_norm2 = 0.0;
        #pragma unroll
        for (int p = 0; p < 6; ++p) {
            double apq = A[p0[p] * 4 + p1[p]];
            off_norm2 += apq * apq;
        }
        if (off_norm2 < 1e-30) break;

        #pragma unroll
        for (int p = 0; p < 6; ++p) {
            int ip = p0[p];
            int iq = p1[p];
            double apq = A[ip * 4 + iq];
            if (fabs(apq) < 1e-18) continue;

            double app = A[ip * 4 + ip];
            double aqq = A[iq * 4 + iq];
            double tau = (aqq - app) / (2.0 * apq);
            double t = (tau >= 0.0)
                ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
            double c = rsqrt(1.0 + t * t);
            double s = t * c;

            A[ip * 4 + ip] = app - t * apq;
            A[iq * 4 + iq] = aqq + t * apq;
            A[ip * 4 + iq] = 0.0;
            A[iq * 4 + ip] = 0.0;

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                if (r != ip && r != iq) {
                    double arp = A[r * 4 + ip];
                    double arq = A[r * 4 + iq];
                    A[r * 4 + ip] = c * arp - s * arq;
                    A[ip * 4 + r] = A[r * 4 + ip];
                    A[r * 4 + iq] = s * arp + c * arq;
                    A[iq * 4 + r] = A[r * 4 + iq];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) evals[i] = A[i * 4 + i];

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        int best = i;
        for (int j = i + 1; j < 4; ++j) {
            if (evals[j] > evals[best]) best = j;
        }
        if (best != i) {
            double tmp = evals[i];
            evals[i] = evals[best];
            evals[best] = tmp;
        }
    }
}

__device__ __forceinline__ void dev_svd4x4_singular_values(
    const double M[16],
    double sigma_out[4]
) {
    double G[16];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = i; j < 4; ++j) {
            double s = 0.0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                s += M[k * 4 + i] * M[k * 4 + j];
            }
            G[i * 4 + j] = s;
            G[j * 4 + i] = s;
        }
    }
    double evals[4];
    dev_sym4x4_jacobi_evals_only_f64(G, evals);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        sigma_out[i] = (evals[i] > 0.0) ? sqrt(evals[i]) : 0.0;
    }
}

/*
 * Reduced-subspace singular value computation when one or both sequences
 * are missing 1 or 2 nucleotide states (e.g., all 'T's masked or absent).
 */
__device__ __forceinline__ void dev_compute_reduced_subspace_sigmas(
    const double K[16],
    const double Fa[4],
    const double Fb[4],
    int32_t mode,
    const int active[4],
    int n_act,
    double sigma_out[4]
) {
    double K_sub[16];
    for (int i = 0; i < 16; ++i) K_sub[i] = 0.0;

    int n_one_sided = 0;
    for (int c = 0; c < 4; ++c) {
        if (!active[c] && (Fa[c] >= DEV_MIN_STATE_FREQ || Fb[c] >= DEV_MIN_STATE_FREQ)) {
            n_one_sided++;
        }
    }

    double total_sub = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (!active[i]) continue;
        double missing_off = 0.0;
        for (int c = 0; c < 4; ++c) {
            if (!active[c]) {
                missing_off += K[i * 4 + c] + K[c * 4 + i];
            }
        }
        double add_off = missing_off / (double)(n_act - 1);
        for (int j = 0; j < 4; ++j) {
            if (!active[j]) continue;
            double v = (i == j) ? K[i * 4 + i] : (K[i * 4 + j] + add_off);
            K_sub[i * 4 + j] = v;
            total_sub += v;
        }
    }

    if (total_sub <= 1e-12) {
        sigma_out[0] = 1.0;
        sigma_out[1] = DEV_SIGMA_FLOOR;
        sigma_out[2] = DEV_SIGMA_FLOOR;
        sigma_out[3] = DEV_SIGMA_FLOOR;
        return;
    }

    double Fa_sub[4] = {0.0, 0.0, 0.0, 0.0};
    double Fb_sub[4] = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
        if (!active[i]) continue;
        for (int j = 0; j < 4; ++j) {
            if (!active[j]) continue;
            K_sub[i * 4 + j] /= total_sub;
            Fa_sub[i] += K_sub[i * 4 + j];
            Fb_sub[j] += K_sub[i * 4 + j];
        }
    }

    double M[16];
    for (int i = 0; i < 16; ++i) M[i] = 0.0;

    if (mode == EIGEN_MODE_PAPER_ORIG) {
        for (int i = 0; i < 4; ++i) {
            if (!active[i]) continue;
            double sfa_i = sqrt(Fa_sub[i]);
            double fb_m32_i = 1.0 / (Fb_sub[i] * sqrt(Fb_sub[i]));
            for (int j = 0; j < 4; ++j) {
                if (!active[j]) continue;
                double fa_m32_j = 1.0 / (Fa_sub[j] * sqrt(Fa_sub[j]));
                double sfb_j = sqrt(Fb_sub[j]);
                double t1 = sfa_i * K_sub[i * 4 + j] * fa_m32_j;
                double t2 = fb_m32_i * K_sub[i * 4 + j] * sfb_j;
                M[i * 4 + j] = 0.5 * (t1 + t2);
            }
        }
        double sig_raw[4];
        dev_svd4x4_singular_values(M, sig_raw);
        sigma_out[0] = sig_raw[0];
        double scale = (4.0 * (double)(n_act - 1)) / (3.0 * (double)n_act);
        double mean_sub = 0.0;
        for (int k = 1; k < n_act; ++k) {
            double s_sub = sig_raw[k];
            if (s_sub < 0.0) s_sub = 0.0;
            if (s_sub > 1.0) s_sub = 1.0;
            double s_adj;
            if (n_one_sided > 0) {
                s_adj = 1.0 - scale * (1.0 - s_sub);
            } else {
                double denom = 4.0 - (double)(4 - n_act) * s_sub;
                s_adj = (denom > 1e-12) ? (((double)n_act * s_sub) / denom) : DEV_SIGMA_FLOOR;
            }
            if (s_adj < DEV_SIGMA_FLOOR) s_adj = DEV_SIGMA_FLOOR;
            if (s_adj > 1.0) s_adj = 1.0;
            sigma_out[k] = s_adj;
            mean_sub += s_adj;
        }
        mean_sub /= (double)(n_act - 1);
        for (int k = n_act; k < 4; ++k) {
            sigma_out[k] = mean_sub;
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            if (!active[i]) continue;
            double sfa_i = sqrt(Fa_sub[i]);
            double inv_sfa_i = 1.0 / sfa_i;
            for (int j = 0; j < 4; ++j) {
                if (!active[j]) continue;
                double sfb_j = sqrt(Fb_sub[j]);
                double inv_sfb_j = 1.0 / sfb_j;
                double l_ij = inv_sfa_i * K_sub[i * 4 + j] * inv_sfb_j;
                M[i * 4 + j] = l_ij - sfa_i * sfb_j;
            }
        }
        double sig_defl[4];
        dev_svd4x4_singular_values(M, sig_defl);
        sigma_out[0] = 1.0;
        double scale = (4.0 * (double)(n_act - 1)) / (3.0 * (double)n_act);
        double mean_sub = 0.0;
        for (int k = 0; k < n_act - 1; ++k) {
            double s_sub = sig_defl[k];
            if (s_sub < 0.0) s_sub = 0.0;
            if (s_sub > 1.0) s_sub = 1.0;
            double s_adj;
            if (n_one_sided > 0) {
                s_adj = 1.0 - scale * (1.0 - s_sub);
            } else {
                double denom = 4.0 - (double)(4 - n_act) * s_sub;
                s_adj = (denom > 1e-12) ? (((double)n_act * s_sub) / denom) : DEV_SIGMA_FLOOR;
            }
            if (s_adj < DEV_SIGMA_FLOOR) s_adj = DEV_SIGMA_FLOOR;
            if (s_adj > 1.0) s_adj = 1.0;
            sigma_out[k + 1] = s_adj;
            mean_sub += s_adj;
        }
        mean_sub /= (double)(n_act - 1);
        for (int k = n_act; k < 4; ++k) {
            sigma_out[k] = mean_sub;
        }
    }
    for (int i = 1; i < 3; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if (sigma_out[j] > sigma_out[i]) {
                double tmp = sigma_out[i];
                sigma_out[i] = sigma_out[j];
                sigma_out[j] = tmp;
            }
        }
    }
}

__device__ __forceinline__ void dev_stats_from_counts(
    const uint32_t counts[16],
    int32_t mode,
    PairwiseSuffStats *st
) {
    uint32_t total_valid = 0;
    uint32_t off_diag_valid = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t c = counts[i * 4 + j];
            total_valid += c;
            if (i != j) off_diag_valid += c;
        }
    }

    st->valid_sites = (double)total_valid;
    if (total_valid == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                st->K[i * 4 + j] = (i == j) ? 0.25 : 0.0;
            }
            st->Fa[i] = 0.25;
            st->Fb[i] = 0.25;
            st->xi[i] = 0.25;
            st->sigma[i] = NAN;
        }
        return;
    }

    double denom = (double)total_valid + 16.0 * DEV_PSEUDOCOUNT_EPS;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        st->Fa[i] = 0.0;
        st->Fb[i] = 0.0;
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            double kij = ((double)counts[i * 4 + j] + DEV_PSEUDOCOUNT_EPS) / denom;
            st->K[i * 4 + j] = kij;
            st->Fa[i] += kij;
            st->Fb[j] += kij;
        }
    }

    double xi_sum = 0.0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        double val = sqrt(st->Fa[i] * st->Fb[i]);
        st->xi[i] = val;
        xi_sum += val;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) st->xi[i] /= xi_sum;

    int active[4];
    int n_act = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        double min_f = (st->Fa[i] < st->Fb[i]) ? st->Fa[i] : st->Fb[i];
        active[i] = (min_f >= DEV_MIN_STATE_FREQ) ? 1 : 0;
        n_act += active[i];
    }

    if (n_act < 2) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) st->sigma[i] = NAN;
        return;
    }

    if (off_diag_valid == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) st->sigma[i] = 1.0;
        return;
    }

    if (n_act < 4) {
        double act_xi_sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            if (active[i]) act_xi_sum += st->xi[i];
        }
        double act_mass = 0.25 * (double)n_act;
        for (int i = 0; i < 4; ++i) {
            if (active[i]) {
                st->xi[i] = (act_xi_sum > 1e-12) ? (act_mass * st->xi[i] / act_xi_sum) : 0.25;
            } else {
                st->xi[i] = 0.25;
            }
        }
        dev_compute_reduced_subspace_sigmas(st->K, st->Fa, st->Fb, mode, active, n_act, st->sigma);
        return;
    }

    if (mode == EIGEN_MODE_PAPER_ORIG) {
        double Lbar[16];
        double fa_p12[4], fa_m32[4], fb_p12[4], fb_m32[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            double sfa = sqrt(st->Fa[i]);
            double sfb = sqrt(st->Fb[i]);
            fa_p12[i] = sfa;
            fa_m32[i] = 1.0 / (st->Fa[i] * sfa);
            fb_p12[i] = sfb;
            fb_m32[i] = 1.0 / (st->Fb[i] * sfb);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                double t1 = fa_p12[i] * st->K[i * 4 + j] * fa_m32[j];
                double t2 = fb_m32[i] * st->K[i * 4 + j] * fb_p12[j];
                Lbar[i * 4 + j] = 0.5 * (t1 + t2);
            }
        }
        dev_svd4x4_singular_values(Lbar, st->sigma);
    } else {
        double Ltilde[16];
        double sfa[4], inv_sfa[4], sfb[4], inv_sfb[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sfa[i] = sqrt(st->Fa[i]);
            inv_sfa[i] = 1.0 / sfa[i];
            sfb[i] = sqrt(st->Fb[i]);
            inv_sfb[i] = 1.0 / sfb[i];
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                double l_ij = inv_sfa[i] * st->K[i * 4 + j] * inv_sfb[j];
                Ltilde[i * 4 + j] = l_ij - sfa[i] * sfb[j];
            }
        }
        double sig_defl[4];
        dev_svd4x4_singular_values(Ltilde, sig_defl);
        st->sigma[0] = 1.0;
        st->sigma[1] = sig_defl[0];
        st->sigma[2] = sig_defl[1];
        st->sigma[3] = sig_defl[2];
    }
}

__device__ __forceinline__ void dev_compute_pair_suff_stats(
    const uint64_t * __restrict__ planes_a,
    const uint64_t * __restrict__ planes_b,
    int32_t n_words,
    int32_t mode,
    PairwiseSuffStats *st
) {
    uint32_t counts[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) counts[i] = 0;

    for (int32_t w = 0; w < n_words; ++w) {
        uint64_t a0 = planes_a[0 * n_words + w];
        uint64_t a1 = planes_a[1 * n_words + w];
        uint64_t a2 = planes_a[2 * n_words + w];
        uint64_t a3 = planes_a[3 * n_words + w];

        uint64_t b0 = planes_b[0 * n_words + w];
        uint64_t b1 = planes_b[1 * n_words + w];
        uint64_t b2 = planes_b[2 * n_words + w];
        uint64_t b3 = planes_b[3 * n_words + w];

        counts[0]  += (uint32_t)__popcll(a0 & b0);
        counts[1]  += (uint32_t)__popcll(a0 & b1);
        counts[2]  += (uint32_t)__popcll(a0 & b2);
        counts[3]  += (uint32_t)__popcll(a0 & b3);

        counts[4]  += (uint32_t)__popcll(a1 & b0);
        counts[5]  += (uint32_t)__popcll(a1 & b1);
        counts[6]  += (uint32_t)__popcll(a1 & b2);
        counts[7]  += (uint32_t)__popcll(a1 & b3);

        counts[8]  += (uint32_t)__popcll(a2 & b0);
        counts[9]  += (uint32_t)__popcll(a2 & b1);
        counts[10] += (uint32_t)__popcll(a2 & b2);
        counts[11] += (uint32_t)__popcll(a2 & b3);

        counts[12] += (uint32_t)__popcll(a3 & b0);
        counts[13] += (uint32_t)__popcll(a3 & b1);
        counts[14] += (uint32_t)__popcll(a3 & b2);
        counts[15] += (uint32_t)__popcll(a3 & b3);
    }

    dev_stats_from_counts(counts, mode, st);
}

__device__ __forceinline__ void dev_build_gtr_sym_matrix(
    const double s_rates[6],
    const double xi[4],
    double R_out[16],
    double *out_neg_tr_Q
) {
    double S[16] = {
        0.0,        s_rates[0], s_rates[1], s_rates[2],
        s_rates[0], 0.0,        s_rates[3], s_rates[4],
        s_rates[1], s_rates[3], 0.0,        s_rates[5],
        s_rates[2], s_rates[4], s_rates[5], 0.0
    };

    double row_sum[4];
    double mu = 0.0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        double rs = 0.0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            rs += S[i * 4 + j] * xi[j];
        }
        row_sum[i] = rs;
        mu += xi[i] * rs;
    }
    double inv_mu = (mu > 1e-30) ? (1.0 / mu) : 1.0;

    double sxi[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) sxi[i] = sqrt(xi[i]);

    double neg_tr = 0.0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (i == j) {
                double q_ii = -row_sum[i] * inv_mu;
                R_out[i * 4 + i] = q_ii;
                neg_tr -= q_ii;
            } else {
                R_out[i * 4 + j] = S[i * 4 + j] * sxi[i] * sxi[j] * inv_mu;
            }
        }
    }
    *out_neg_tr_Q = neg_tr;
}

/*
 * Log-space + iteratively reweighted WLS distance estimator (device FP64).
 */
__device__ __forceinline__ double dev_compute_wls_distance_logspace(
    const double sigma[4],
    const double evals[4],
    double alpha
) {
    double inv_alpha = 1.0 / alpha;
    double y[4] = {0.0, 0.0, 0.0, 0.0};
    double neg_lam[4] = {0.0, 0.0, 0.0, 0.0};
    double lw[4] = {-1e300, -1e300, -1e300, -1e300};
    double max_lw = -1e300;

    #pragma unroll
    for (int k = 1; k < 4; ++k) {
        double sig = sigma[k];
        if (sig > DEV_SIGMA_CEIL) sig = DEV_SIGMA_CEIL;
        if (sig < DEV_SIGMA_FLOOR) sig = DEV_SIGMA_FLOOR;

        double nl = -evals[k];
        if (nl < 1e-8) nl = 1e-8;
        neg_lam[k] = nl;

        double log_sig = log(sig);
        double u = -log_sig * inv_alpha;
        if (u > 30.0) u = 30.0;
        y[k] = alpha * expm1(u);

        lw[k] = (2.0 + 2.0 * inv_alpha) * log_sig - log(1.0 - sig * sig + 1e-4);
        if (lw[k] > max_lw) max_lw = lw[k];
    }

    double num = 0.0, den = 0.0;
    #pragma unroll
    for (int k = 1; k < 4; ++k) {
        double w_k = exp(lw[k] - max_lw);
        num += w_k * neg_lam[k] * y[k];
        den += w_k * neg_lam[k] * neg_lam[k];
    }
    double d_ab = (den > 1e-20) ? (num / den) : 0.0;
    if (d_ab < 0.0) d_ab = 0.0;

    /* 3 iterations of model-implied weight refinement to eliminate plug-in bias */
    #pragma unroll
    for (int it = 0; it < 3; ++it) {
        double d_curr = (d_ab > EIGEN_D_MAX) ? EIGEN_D_MAX : d_ab;
        max_lw = -1e300;
        #pragma unroll
        for (int k = 1; k < 4; ++k) {
            double base = 1.0 + (d_curr * neg_lam[k]) * inv_alpha;
            if (base < 1.0) base = 1.0;
            double log_mu = -alpha * log(base);
            if (log_mu > log(DEV_SIGMA_CEIL)) log_mu = log(DEV_SIGMA_CEIL);
            if (log_mu < log(DEV_SIGMA_FLOOR)) log_mu = log(DEV_SIGMA_FLOOR);
            double mu = exp(log_mu);
            lw[k] = (2.0 + 2.0 * inv_alpha) * log_mu - log(1.0 - mu * mu + 1e-4);
            if (lw[k] > max_lw) max_lw = lw[k];
        }
        num = 0.0;
        den = 0.0;
        #pragma unroll
        for (int k = 1; k < 4; ++k) {
            double w_k = exp(lw[k] - max_lw);
            num += w_k * neg_lam[k] * y[k];
            den += w_k * neg_lam[k] * neg_lam[k];
        }
        d_ab = (den > 1e-20) ? (num / den) : d_ab;
        if (d_ab < 0.0) d_ab = 0.0;
    }

    if (d_ab > EIGEN_D_MAX) d_ab = EIGEN_D_MAX;
    return d_ab;
}

__device__ __forceinline__ double dev_eval_pair_distance(
    const PairwiseSuffStats *st,
    const double s_rates[6],
    double alpha,
    int32_t mode
) {
    if (st->valid_sites <= 0.0 || isnan(st->sigma[1])) {
        return NAN;
    }

    int start_idx = (mode == EIGEN_MODE_PAPER_ORIG) ? 0 : 1;
    int all_ceil = 1;
    for (int k = start_idx; k < 4; ++k) {
        if (st->sigma[k] < DEV_SIGMA_CEIL) {
            all_ceil = 0;
            break;
        }
    }
    if (all_ceil) {
        return 0.0;
    }

    double inv_alpha = 1.0 / alpha;
    double R[16], neg_tr_Q = 3.0;
    dev_build_gtr_sym_matrix(s_rates, st->xi, R, &neg_tr_Q);

    double d_ab = 0.0;
    if (mode == EIGEN_MODE_SYMMETRIC_WLS) {
        double evals[4];
        dev_sym4x4_jacobi_evals_only_f64(R, evals);
        d_ab = dev_compute_wls_distance_logspace(st->sigma, evals, alpha);
    } else {
        double sum_gamma = 0.0;
        for (int k = start_idx; k < 4; ++k) {
            double sig = st->sigma[k];
            if (sig > 1.0) sig = 1.0;
            if (sig < DEV_SIGMA_FLOOR) sig = DEV_SIGMA_FLOOR;
            double u = -log(sig) * inv_alpha;
            if (u > 30.0) u = 30.0;
            sum_gamma += alpha * expm1(u);
        }
        d_ab = (neg_tr_Q > 1e-12) ? (sum_gamma / neg_tr_Q) : 0.0;
    }
    if (d_ab < 0.0) d_ab = 0.0;
    if (d_ab > EIGEN_D_MAX) d_ab = EIGEN_D_MAX;
    return d_ab;
}

/* ============================================================================
 * 3. GPU Kernels for MAP Parameter Estimation & Tiled Distance Matrix
 * ============================================================================ */
__global__ void k_extract_sampled_pairs_svd(
    const uint64_t * __restrict__ d_bitplanes,
    const int32_t * __restrict__ d_pair_a,
    const int32_t * __restrict__ d_pair_b,
    int32_t n_words,
    int32_t n_samples,
    int32_t mode,
    PairwiseSuffStats * __restrict__ d_samples
) {
    int32_t m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < n_samples) {
        int32_t a = d_pair_a[m];
        int32_t b = d_pair_b[m];
        const uint64_t *pa = &d_bitplanes[(size_t)a * 4 * (size_t)n_words];
        const uint64_t *pb = &d_bitplanes[(size_t)b * 4 * (size_t)n_words];
        PairwiseSuffStats st;
        dev_compute_pair_suff_stats(pa, pb, n_words, mode, &st);
        d_samples[m] = st;
    }
}

__global__ void k_eval_map_loglik_batch(
    const PairwiseSuffStats * __restrict__ d_samples,
    int32_t n_samples,
    int32_t mode,
    double s0, double s1, double s2, double s3, double s4, double s5,
    double alpha,
    double * __restrict__ d_block_sums
) {
    __shared__ double s_ll[256];
    int32_t tid = threadIdx.x;
    int32_t m = blockIdx.x * blockDim.x + tid;

    double my_ll = 0.0;
    if (m < n_samples) {
        PairwiseSuffStats st = d_samples[m];
        if (st.valid_sites >= EIGEN_MIN_OVERLAP_SITES && !isnan(st.sigma[1])) {
            double s_rates[6] = {s0, s1, s2, s3, s4, s5};
            double inv_alpha = 1.0 / alpha;

            double R[16], neg_tr_Q = 3.0;
            dev_build_gtr_sym_matrix(s_rates, st.xi, R, &neg_tr_Q);

            double evals[4], V[16];
            dev_sym4x4_jacobi_eigen_f64(R, evals, V);

            double d_ab = 0.0;
            if (mode == EIGEN_MODE_SYMMETRIC_WLS) {
                d_ab = dev_compute_wls_distance_logspace(st.sigma, evals, alpha);
            } else {
                int start_idx = (mode == EIGEN_MODE_PAPER_ORIG) ? 0 : 1;
                int all_ceil = 1;
                for (int k = start_idx; k < 4; ++k) {
                    if (st.sigma[k] < DEV_SIGMA_CEIL) {
                        all_ceil = 0;
                        break;
                    }
                }
                if (!all_ceil) {
                    double sum_gamma = 0.0;
                    for (int k = start_idx; k < 4; ++k) {
                        double sig = st.sigma[k];
                        if (sig > 1.0) sig = 1.0;
                        if (sig < DEV_SIGMA_FLOOR) sig = DEV_SIGMA_FLOOR;
                        double u = -log(sig) * inv_alpha;
                        if (u > 30.0) u = 30.0;
                        sum_gamma += alpha * expm1(u);
                    }
                    d_ab = (neg_tr_Q > 1e-12) ? (sum_gamma / neg_tr_Q) : 0.0;
                }
            }
            if (d_ab < 0.0) d_ab = 0.0;
            if (d_ab > EIGEN_D_MAX) d_ab = EIGEN_D_MAX;

            double mu_k[4];
            mu_k[0] = 1.0;
            #pragma unroll
            for (int k = 1; k < 4; ++k) {
                double base = 1.0 - (d_ab * evals[k]) * inv_alpha;
                if (base < 1.0) base = 1.0;
                mu_k[k] = pow(base, -alpha);
            }

            double sxi[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) sxi[i] = sqrt(st.xi[i]);

            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    double sym_p = 0.0;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        sym_p += V[i * 4 + k] * V[j * 4 + k] * mu_k[k];
                    }
                    double p_ij = (sxi[j] / sxi[i]) * sym_p;
                    if (p_ij < 1e-15) p_ij = 1e-15;
                    my_ll += st.K[i * 4 + j] * log(p_ij);
                }
            }
            my_ll *= st.valid_sites;
        }
    }

    s_ll[tid] = my_ll;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) s_ll[tid] += s_ll[tid + stride];
        __syncthreads();
    }
    if (tid == 0) d_block_sums[blockIdx.x] = s_ll[0];
}

/*
 * Shared-Memory Tiled Upper-Triangular Pairwise Distance Kernel:
 *   1. Prunes 16x16 tiles that lie strictly below the main diagonal
 *      (row_start + r_base > c_base + 15).
 *   2. Pads the innermost shared-memory dimension by +1 (WORD_CHUNK + 1 = 9)
 *      to eliminate 16-way shared-memory bank conflicts.
 *   3. Evaluates FP64 SVD + distance only for upper-triangular entries (i < j)
 *      and sets diagonal entries (i == j) to 0.0.
 */
__global__ void k_compute_eigen_dist_rows_tiled_gpu(
    const uint64_t * __restrict__ d_bitplanes,
    int32_t row_start,
    int32_t row_count,
    int32_t n_sub,
    int32_t n_words,
    int32_t mode,
    double s0, double s1, double s2, double s3, double s4, double s5,
    double alpha,
    double * __restrict__ d_D_rows
) {
    int32_t r_base = blockIdx.y * 16;
    int32_t c_base = blockIdx.x * 16;

    /* Prune tiles strictly in the lower triangle (all threads have i_global > j_global) */
    if (row_start + r_base > c_base + 15) {
        return;
    }

    __shared__ uint64_t sh_a[16][4][WORD_CHUNK + 1];
    __shared__ uint64_t sh_b[16][4][WORD_CHUNK + 1];

    int32_t tx = threadIdx.x; /* 0..15 (column in tile) */
    int32_t ty = threadIdx.y; /* 0..15 (row in tile)    */
    int32_t tid = ty * 16 + tx;

    int32_t r_local = r_base + ty;
    int32_t i_global = row_start + r_local;
    int32_t j_global = c_base + tx;

    uint32_t counts[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) counts[k] = 0;

    int32_t load_tax = tid >> 4;        /* 0..15 */
    int32_t load_plane = (tid >> 2) & 3;/* 0..3  */
    int32_t load_w_sub = tid & 3;       /* 0..3  */

    for (int32_t w_base = 0; w_base < n_words; w_base += WORD_CHUNK) {
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            int32_t w_off = pass * 4 + load_w_sub;
            int32_t w_idx = w_base + w_off;

            int32_t ta = row_start + r_base + load_tax;
            uint64_t va = 0ULL;
            if (ta < row_start + row_count && ta < n_sub && w_idx < n_words) {
                va = d_bitplanes[((size_t)ta * 4 + (size_t)load_plane) * (size_t)n_words + (size_t)w_idx];
            }
            sh_a[load_tax][load_plane][w_off] = va;

            int32_t tb = c_base + load_tax;
            uint64_t vb = 0ULL;
            if (tb < n_sub && w_idx < n_words) {
                vb = d_bitplanes[((size_t)tb * 4 + (size_t)load_plane) * (size_t)n_words + (size_t)w_idx];
            }
            sh_b[load_tax][load_plane][w_off] = vb;
        }
        __syncthreads();

        int32_t w_lim = (w_base + WORD_CHUNK <= n_words) ? WORD_CHUNK : (n_words - w_base);
        for (int32_t w = 0; w < w_lim; ++w) {
            uint64_t a0 = sh_a[ty][0][w];
            uint64_t a1 = sh_a[ty][1][w];
            uint64_t a2 = sh_a[ty][2][w];
            uint64_t a3 = sh_a[ty][3][w];

            uint64_t b0 = sh_b[tx][0][w];
            uint64_t b1 = sh_b[tx][1][w];
            uint64_t b2 = sh_b[tx][2][w];
            uint64_t b3 = sh_b[tx][3][w];

            counts[0]  += (uint32_t)__popcll(a0 & b0);
            counts[1]  += (uint32_t)__popcll(a0 & b1);
            counts[2]  += (uint32_t)__popcll(a0 & b2);
            counts[3]  += (uint32_t)__popcll(a0 & b3);

            counts[4]  += (uint32_t)__popcll(a1 & b0);
            counts[5]  += (uint32_t)__popcll(a1 & b1);
            counts[6]  += (uint32_t)__popcll(a1 & b2);
            counts[7]  += (uint32_t)__popcll(a1 & b3);

            counts[8]  += (uint32_t)__popcll(a2 & b0);
            counts[9]  += (uint32_t)__popcll(a2 & b1);
            counts[10] += (uint32_t)__popcll(a2 & b2);
            counts[11] += (uint32_t)__popcll(a2 & b3);

            counts[12] += (uint32_t)__popcll(a3 & b0);
            counts[13] += (uint32_t)__popcll(a3 & b1);
            counts[14] += (uint32_t)__popcll(a3 & b2);
            counts[15] += (uint32_t)__popcll(a3 & b3);
        }
        __syncthreads();
    }

    if (r_local < row_count && j_global < n_sub) {
        if (i_global == j_global) {
            d_D_rows[(size_t)r_local * (size_t)n_sub + (size_t)j_global] = 0.0;
            return;
        }
        if (i_global > j_global) {
            return;
        }
        PairwiseSuffStats st;
        dev_stats_from_counts(counts, mode, &st);
        double s_rates[6] = {s0, s1, s2, s3, s4, s5};
        double d_ab = dev_eval_pair_distance(&st, s_rates, alpha, mode);
        d_D_rows[(size_t)r_local * (size_t)n_sub + (size_t)j_global] = d_ab;
    }
}

/* ============================================================================
 * 4. Host Wrapper: GPU-Accelerated MAP (\alpha, S) Optimization
 * ============================================================================ */
static double eval_gpu_map_objective(
    const double theta[7],
    const PairwiseSuffStats *d_samples,
    double *d_block_sums,
    double *h_block_sums,
    int32_t n_samples,
    int32_t valid_samples,
    int32_t n_eff,
    int32_t n_blocks,
    int32_t mode
) {
    double mean_log_s = 0.0;
    for (int i = 0; i < 6; ++i) mean_log_s += theta[i];
    mean_log_s /= 6.0;

    double s_rates[6];
    double s_sum = 0.0;
    for (int i = 0; i < 6; ++i) {
        double v = theta[i] - mean_log_s;
        if (v > 10.0) v = 10.0;
        if (v < -10.0) v = -10.0;
        s_rates[i] = exp(v);
        s_sum += s_rates[i];
    }
    for (int i = 0; i < 6; ++i) s_rates[i] *= (6.0 / s_sum);

    double log_alpha = theta[6];
    if (log_alpha > 5.0) log_alpha = 5.0;
    if (log_alpha < -3.0) log_alpha = -3.0;
    double alpha = exp(log_alpha);

    double log_prior = -0.5 * (mean_log_s * mean_log_s) / 0.01;
    for (int i = 0; i < 6; ++i) {
        double diff_s = s_rates[i] - 1.0;
        log_prior -= 0.5 * (diff_s * diff_s);
    }
    log_prior -= 0.5 * (alpha * alpha);

    if (n_samples > 0 && valid_samples > 0) {
        k_eval_map_loglik_batch<<<n_blocks, 256>>>(
            d_samples, n_samples, mode,
            s_rates[0], s_rates[1], s_rates[2], s_rates[3], s_rates[4], s_rates[5],
            alpha, d_block_sums
        );
        if (cudaMemcpy(h_block_sums, d_block_sums, (size_t)n_blocks * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
            return -1e300;
        }
    }

    double total_ll = 0.0;
    for (int32_t b = 0; b < n_blocks; ++b) total_ll += h_block_sums[b];

    double weighted_ll = (valid_samples > 0) ? (((double)n_eff / (double)valid_samples) * total_ll) : 0.0;
    return weighted_ll + log_prior;
}

int fit_eigen_gtr_map_gpu_c(
    const DnaBitplaneAlignment *aln,
    int32_t n_sample_pairs,
    int32_t mode,
    uint64_t seed,
    EigenGtrParams *out_params
) {
    if (!aln || aln->n_taxa < 2 || !out_params) return -1;
    mode = (mode >= 0 && mode <= 2) ? mode : EIGEN_MODE_SYMMETRIC_WLS;

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        return fit_eigen_gtr_map_c(aln, n_sample_pairs, mode, seed, out_params);
    }
    CUDA_CHECK_RET(cudaSetDevice(0));

    int32_t n_eff = aln->n_taxa - 1;
    int64_t total_pairs = ((int64_t)aln->n_taxa * (int64_t)(aln->n_taxa - 1)) / 2;
    int32_t M = n_sample_pairs;
    if (M <= 0 || (int64_t)M > total_pairs) {
        M = (total_pairs > 50000) ? 50000 : (int32_t)total_pairs;
    }

    int32_t *h_pair_a = (int32_t *)malloc((size_t)M * sizeof(int32_t));
    int32_t *h_pair_b = (int32_t *)malloc((size_t)M * sizeof(int32_t));
    if (!h_pair_a || !h_pair_b) {
        free(h_pair_a);
        free(h_pair_b);
        return -2;
    }

    if ((int64_t)M == total_pairs) {
        int32_t idx = 0;
        for (int32_t a = 0; a < aln->n_taxa; ++a) {
            for (int32_t b = a + 1; b < aln->n_taxa; ++b) {
                h_pair_a[idx] = a;
                h_pair_b[idx] = b;
                idx++;
            }
        }
    } else {
        uint64_t rng = seed ? seed : 0xCAFEBABE12345678ULL;
        for (int32_t m = 0; m < M; ++m) {
            int32_t a = (int32_t)(host_splitmix64(&rng) % (uint64_t)aln->n_taxa);
            int32_t b = (int32_t)(host_splitmix64(&rng) % (uint64_t)(aln->n_taxa - 1));
            if (b >= a) b++;
            if (a > b) { int32_t t = a; a = b; b = t; }
            h_pair_a[m] = a;
            h_pair_b[m] = b;
        }
    }

    size_t bp_bytes = (size_t)aln->n_taxa * 4 * (size_t)aln->n_words * sizeof(uint64_t);
    uint64_t *d_bitplanes = NULL;
    int32_t *d_pair_a = NULL, *d_pair_b = NULL;
    PairwiseSuffStats *d_samples = NULL;

    int32_t n_blocks = (M + 255) / 256;
    double *d_block_sums = NULL;
    double *h_block_sums = (double *)malloc((size_t)n_blocks * sizeof(double));
    PairwiseSuffStats *h_samples = (PairwiseSuffStats *)malloc((size_t)M * sizeof(PairwiseSuffStats));
    if (!h_block_sums || !h_samples) {
        free(h_pair_a);
        free(h_pair_b);
        free(h_block_sums);
        free(h_samples);
        return -2;
    }

    if (cudaMalloc(&d_bitplanes, bp_bytes) != cudaSuccess ||
        cudaMalloc(&d_pair_a, (size_t)M * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_pair_b, (size_t)M * sizeof(int32_t)) != cudaSuccess ||
        cudaMalloc(&d_samples, (size_t)M * sizeof(PairwiseSuffStats)) != cudaSuccess ||
        cudaMalloc(&d_block_sums, (size_t)n_blocks * sizeof(double)) != cudaSuccess ||
        cudaMemcpy(d_bitplanes, aln->bitplanes, bp_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_pair_a, h_pair_a, (size_t)M * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_pair_b, h_pair_b, (size_t)M * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        free(h_samples);
        free(h_block_sums);
        free(h_pair_a);
        free(h_pair_b);
        if (d_bitplanes) cudaFree(d_bitplanes);
        if (d_pair_a) cudaFree(d_pair_a);
        if (d_pair_b) cudaFree(d_pair_b);
        if (d_samples) cudaFree(d_samples);
        if (d_block_sums) cudaFree(d_block_sums);
        return -100;
    }

    k_extract_sampled_pairs_svd<<<n_blocks, 256>>>(
        d_bitplanes, d_pair_a, d_pair_b, aln->n_words, M, mode, d_samples
    );
    if (cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(h_samples, d_samples, (size_t)M * sizeof(PairwiseSuffStats), cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(h_samples);
        free(h_block_sums);
        free(h_pair_a);
        free(h_pair_b);
        cudaFree(d_bitplanes);
        cudaFree(d_pair_a);
        cudaFree(d_pair_b);
        cudaFree(d_samples);
        cudaFree(d_block_sums);
        return -100;
    }

    int32_t valid_samples = 0;
    for (int32_t m = 0; m < M; ++m) {
        if (h_samples[m].valid_sites >= EIGEN_MIN_OVERLAP_SITES && !isnan(h_samples[m].sigma[1])) {
            valid_samples++;
        }
    }

    cudaFree(d_pair_a);
    cudaFree(d_pair_b);
    cudaFree(d_bitplanes);

    /* Grid search initialization for alpha */
    double theta[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    static const double alpha_grid[7] = {-1.0, -0.5, -0.2, 0.0, 0.3, 0.7, 1.2};
    double best_init_f = -1e300;
    double best_log_alpha = 0.0;
    for (int g = 0; g < 7; ++g) {
        theta[6] = alpha_grid[g];
        double f = eval_gpu_map_objective(theta, d_samples, d_block_sums, h_block_sums, M, valid_samples, n_eff, n_blocks, mode);
        if (f > best_init_f) {
            best_init_f = f;
            best_log_alpha = alpha_grid[g];
        }
    }
    theta[6] = best_log_alpha;

    /* BFGS Optimization on GPU Objective */
    double H[49];
    for (int i = 0; i < 49; ++i) H[i] = 0.0;
    for (int i = 0; i < 7; ++i) H[i * 7 + i] = 1.0;

    double f_curr = eval_gpu_map_objective(theta, d_samples, d_block_sums, h_block_sums, M, valid_samples, n_eff, n_blocks, mode);
    double grad[7];
    const double eps_fd = 1e-5;

    for (int d = 0; d < 7; ++d) {
        double th_p[7], th_m[7];
        memcpy(th_p, theta, sizeof(theta));
        memcpy(th_m, theta, sizeof(theta));
        th_p[d] += eps_fd;
        th_m[d] -= eps_fd;
        double fp = eval_gpu_map_objective(th_p, d_samples, d_block_sums, h_block_sums, M, valid_samples, n_eff, n_blocks, mode);
        double fm = eval_gpu_map_objective(th_m, d_samples, d_block_sums, h_block_sums, M, valid_samples, n_eff, n_blocks, mode);
        grad[d] = (fp - fm) / (2.0 * eps_fd);
    }

    int32_t iter = 0;
    const int32_t max_iters = 60;
    for (; iter < max_iters; ++iter) {
        double gnorm = 0.0;
        for (int d = 0; d < 7; ++d) gnorm += grad[d] * grad[d];
        if (sqrt(gnorm) < 1e-6 * (1.0 + fabs(f_curr))) break;

        double p[7];
        double dir_deriv = 0.0;
        for (int i = 0; i < 7; ++i) {
            double s = 0.0;
            for (int j = 0; j < 7; ++j) s += H[i * 7 + j] * grad[j];
            p[i] = s;
            dir_deriv += s * grad[i];
        }
        if (dir_deriv <= 0.0) {
            for (int i = 0; i < 49; ++i) H[i] = 0.0;
            for (int i = 0; i < 7; ++i) {
                H[i * 7 + i] = 1.0;
                p[i] = grad[i];
            }
            dir_deriv = gnorm;
        }

        double step = 1.0;
        double p_max = 0.0;
        for (int i = 0; i < 7; ++i) {
            if (fabs(p[i]) > p_max) p_max = fabs(p[i]);
        }
        if (p_max > 1.0) step = 1.0 / p_max;

        double theta_next[7];
        double f_next = f_curr;
        int accepted = 0;
        for (int ls = 0; ls < 20; ++ls) {
            for (int i = 0; i < 7; ++i) theta_next[i] = theta[i] + step * p[i];
            f_next = eval_gpu_map_objective(theta_next, d_samples, d_block_sums, h_block_sums, M, valid_samples, n_eff, n_blocks, mode);
            if (f_next >= f_curr + 1e-4 * step * dir_deriv) {
                accepted = 1;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) break;

        double grad_next[7];
        for (int d = 0; d < 7; ++d) {
            double th_p[7], th_m[7];
            memcpy(th_p, theta_next, sizeof(theta_next));
            memcpy(th_m, theta_next, sizeof(theta_next));
            th_p[d] += eps_fd;
            th_m[d] -= eps_fd;
            double fp = eval_gpu_map_objective(th_p, d_samples, d_block_sums, h_block_sums, M, valid_samples, n_eff, n_blocks, mode);
            double fm = eval_gpu_map_objective(th_m, d_samples, d_block_sums, h_block_sums, M, valid_samples, n_eff, n_blocks, mode);
            grad_next[d] = (fp - fm) / (2.0 * eps_fd);
        }

        double s_vec[7], y_vec[7], sy = 0.0, yy = 0.0;
        for (int i = 0; i < 7; ++i) {
            s_vec[i] = theta_next[i] - theta[i];
            y_vec[i] = grad[i] - grad_next[i];
            sy += s_vec[i] * y_vec[i];
            yy += y_vec[i] * y_vec[i];
        }

        if (sy > 1e-12) {
            if (iter == 0 && yy > 1e-20) {
                double scale = sy / yy;
                for (int i = 0; i < 7; ++i) H[i * 7 + i] = scale;
            }
            double Hy[7], yHy = 0.0;
            for (int i = 0; i < 7; ++i) {
                double acc = 0.0;
                for (int j = 0; j < 7; ++j) acc += H[i * 7 + j] * y_vec[j];
                Hy[i] = acc;
                yHy += y_vec[i] * acc;
            }
            double coef1 = (sy + yHy) / (sy * sy);
            for (int i = 0; i < 7; ++i) {
                for (int j = 0; j < 7; ++j) {
                    H[i * 7 + j] += coef1 * s_vec[i] * s_vec[j]
                                  - (Hy[i] * s_vec[j] + s_vec[i] * Hy[j]) / sy;
                }
            }
        }

        memcpy(theta, theta_next, sizeof(theta));
        memcpy(grad, grad_next, sizeof(grad));
        f_curr = f_next;
    }

    double mean_log_s = 0.0;
    for (int i = 0; i < 6; ++i) mean_log_s += theta[i];
    mean_log_s /= 6.0;

    double s_sum = 0.0;
    for (int i = 0; i < 6; ++i) {
        double v = theta[i] - mean_log_s;
        if (v > 10.0) v = 10.0;
        if (v < -10.0) v = -10.0;
        out_params->s_rates[i] = exp(v);
        s_sum += out_params->s_rates[i];
    }
    for (int i = 0; i < 6; ++i) out_params->s_rates[i] *= (6.0 / s_sum);

    double log_alpha = theta[6];
    if (log_alpha > 5.0) log_alpha = 5.0;
    if (log_alpha < -3.0) log_alpha = -3.0;
    out_params->alpha = exp(log_alpha);
    out_params->log_post = f_curr;
    out_params->n_iters = iter;
    out_params->q_near_degenerate = check_eigen_q_near_degenerate_c(
        out_params->s_rates, h_samples, M, out_params->alpha
    );
    out_params->alpha_sd = compute_eigen_bootstrap_alpha_sd_c(
        aln, h_pair_a, h_pair_b, M, n_eff, mode, seed, theta, H
    );

    free(h_samples);
    free(h_pair_a);
    free(h_pair_b);
    cudaFree(d_samples);
    cudaFree(d_block_sums);
    free(h_block_sums);
    return 0;
}

/* ============================================================================
 * 5. Pair-Balanced Multi-GPU Dispatch for Distance Matrix Computation
 * ============================================================================ */
int compute_eigen_subset_distance_matrix_gpu_c(
    const DnaBitplaneAlignment *aln,
    const int32_t *taxon_indices,
    int32_t n_sub,
    const EigenGtrParams *params,
    int32_t mode,
    int32_t num_gpus,
    double *out_D
) {
    if (!aln || n_sub <= 0 || !out_D) return -1;
    mode = (mode >= 0 && mode <= 2) ? mode : EIGEN_MODE_SYMMETRIC_WLS;

    int avail_gpus = 0;
    if (cudaGetDeviceCount(&avail_gpus) != cudaSuccess || avail_gpus <= 0) {
        if (taxon_indices) {
            compute_eigen_subset_distance_matrix_cpu_c(aln, taxon_indices, n_sub, params, mode, out_D);
        } else {
            compute_eigen_distance_matrix_cpu_c(aln, params, mode, out_D, NULL);
        }
        return 0;
    }

    int G = (num_gpus > 0 && num_gpus <= avail_gpus) ? num_gpus : avail_gpus;
    if (G > MAX_EIGEN_GPUS) G = MAX_EIGEN_GPUS;
    if (n_sub < 64) G = 1;

    const double default_s[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const double *s_rates = params ? params->s_rates : default_s;
    double alpha = (params && params->alpha > 1e-4) ? params->alpha : 1.0;

    int32_t n_words = aln->n_words;
    size_t sub_bp_words = (size_t)n_sub * 4 * (size_t)n_words;
    size_t sub_bp_bytes = sub_bp_words * sizeof(uint64_t);

    uint64_t *h_sub_bp = NULL;
    const uint64_t *src_bp_ptr = aln->bitplanes;
    if (taxon_indices) {
        h_sub_bp = (uint64_t *)malloc(sub_bp_bytes);
        if (!h_sub_bp) return -2;
        #pragma omp parallel for schedule(static)
        for (int32_t i = 0; i < n_sub; ++i) {
            int32_t x = taxon_indices[i];
            memcpy(
                &h_sub_bp[(size_t)i * 4 * (size_t)n_words],
                &aln->bitplanes[(size_t)x * 4 * (size_t)n_words],
                4 * (size_t)n_words * sizeof(uint64_t)
            );
        }
        src_bp_ptr = h_sub_bp;
    }

    /* Partition rows across GPUs balanced by upper-triangular pair count (n_sub - i) */
    int32_t row_start[MAX_EIGEN_GPUS];
    int32_t row_count[MAX_EIGEN_GPUS];
    int64_t total_upper_elems = ((int64_t)n_sub * (int64_t)(n_sub + 1)) / 2;
    int32_t cur_row = 0;
    int64_t accum_elems = 0;
    for (int g = 0; g < G; ++g) {
        row_start[g] = cur_row;
        if (g == G - 1) {
            row_count[g] = n_sub - cur_row;
            cur_row = n_sub;
        } else {
            int64_t target_elems = ((int64_t)(g + 1) * total_upper_elems) / G;
            int32_t end_row = cur_row;
            while (end_row < n_sub && accum_elems < target_elems) {
                accum_elems += (int64_t)(n_sub - end_row);
                end_row++;
            }
            /* Align split to 16-row tile boundary when possible */
            int32_t aligned = ((end_row + 8) / 16) * 16;
            if (aligned > cur_row && aligned < n_sub) {
                while (end_row < aligned) {
                    accum_elems += (int64_t)(n_sub - end_row);
                    end_row++;
                }
                while (end_row > aligned) {
                    end_row--;
                    accum_elems -= (int64_t)(n_sub - end_row);
                }
            }
            row_count[g] = end_row - cur_row;
            cur_row = end_row;
        }
    }

    int err_flag = 0;
    #pragma omp parallel for num_threads(G) schedule(static, 1)
    for (int g = 0; g < G; ++g) {
        if (row_count[g] <= 0) continue;
        if (cudaSetDevice(g) != cudaSuccess) {
            #pragma omp atomic write
            err_flag = 1;
            continue;
        }

        uint64_t *d_bp = NULL;
        double *d_D_rows = NULL;
        size_t d_bytes = (size_t)row_count[g] * (size_t)n_sub * sizeof(double);

        if (cudaMalloc(&d_bp, sub_bp_bytes) != cudaSuccess) {
            #pragma omp atomic write
            err_flag = 1;
            continue;
        }
        if (cudaMalloc(&d_D_rows, d_bytes) != cudaSuccess) {
            cudaFree(d_bp);
            #pragma omp atomic write
            err_flag = 1;
            continue;
        }
        if (cudaMemcpy(d_bp, src_bp_ptr, sub_bp_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(d_D_rows);
            cudaFree(d_bp);
            #pragma omp atomic write
            err_flag = 1;
            continue;
        }

        dim3 block(16, 16);
        dim3 grid((n_sub + 15) / 16, (row_count[g] + 15) / 16);
        k_compute_eigen_dist_rows_tiled_gpu<<<grid, block>>>(
            d_bp, row_start[g], row_count[g], n_sub, n_words, mode,
            s_rates[0], s_rates[1], s_rates[2], s_rates[3], s_rates[4], s_rates[5],
            alpha, d_D_rows
        );
        if (cudaGetLastError() != cudaSuccess ||
            cudaMemcpy(
                &out_D[(size_t)row_start[g] * (size_t)n_sub],
                d_D_rows,
                d_bytes,
                cudaMemcpyDeviceToHost
            ) != cudaSuccess) {
            #pragma omp atomic write
            err_flag = 1;
        }

        cudaFree(d_D_rows);
        cudaFree(d_bp);
    }

    if (h_sub_bp) free(h_sub_bp);
    if (err_flag) return -100;

    /* Symmetrize lower triangle from computed upper triangle (i < j) */
    #pragma omp parallel for schedule(dynamic, 16)
    for (int32_t i = 0; i < n_sub; ++i) {
        out_D[(size_t)i * (size_t)n_sub + (size_t)i] = 0.0;
        for (int32_t j = i + 1; j < n_sub; ++j) {
            out_D[(size_t)j * (size_t)n_sub + (size_t)i] = out_D[(size_t)i * (size_t)n_sub + (size_t)j];
        }
    }

    return 0;
}

int compute_eigen_distance_matrix_gpu_c(
    const DnaBitplaneAlignment *aln,
    const EigenGtrParams *params,
    int32_t mode,
    int32_t num_gpus,
    double *out_D
) {
    if (!aln) return -1;
    return compute_eigen_subset_distance_matrix_gpu_c(
        aln, NULL, aln->n_taxa, params, mode, num_gpus, out_D
    );
}
