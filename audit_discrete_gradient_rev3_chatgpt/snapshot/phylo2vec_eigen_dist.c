/*
 * phylo2vec_eigen_dist.c
 *
 * High-Performance C99 + OpenMP Implementation of:
 *   "Bayesian Inference of Phylogenetic Distances: Revisiting the Eigenvalue Approach"
 *   (Penn, Scheidwasser, Donnelly, Duchene & Bhatt, Bull. Math. Biol. 87:32, 2025)
 *
 * Features:
 *   1. Streaming FASTA parser & 4-bitplane 64-bit sequence encoding (with strict
 *      length and byte validation).
 *   2. Hardware POPCNT (__builtin_popcountll) 4x4 pairwise frequency matrix K^{ab},
 *      with reduced-alphabet subspace deflation when a nucleotide state is missing
 *      in either sequence, and NAN sentinel for zero-overlap pairs.
 *   3. Exact 4x4 Jacobi SVD and Lemma-1 symmetric GTR eigensolver in registers.
 *   4. Supports all 3 distance modes:
 *        - EIGEN_MODE_PAPER_ORIG    (Symmetrized & scale-corrected Eq. 5-6)
 *        - EIGEN_MODE_SYMMETRIC_SVD (Canonical (-1/2, -1/2) + exact rank-1 deflation)
 *        - EIGEN_MODE_SYMMETRIC_WLS (Canonical (-1/2, -1/2) + log-space model-iterated WLS)
 *   5. Fast MAP optimization of GTR exchangeabilities S and Gamma shape \alpha (Eq. 7)
 *      via Shanno-Phua scaled BFGS over sampled pairs.
 */

#include "phylo2vec_eigen_dist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define PSEUDOCOUNT_EPS   1e-8
#define SIGMA_FLOOR       1e-4
#define SIGMA_CEIL        (1.0 - 1e-12)
#define MISSING_STATE_TOL 1e-3

/* ============================================================================
 * 1. Fast SplitMix64 PRNG for Pair Subsampling
 * ============================================================================ */
static inline uint64_t eigen_splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* ============================================================================
 * 2. In-Register 4x4 Real Symmetric Jacobi Eigensolver
 *    Computes A = V * diag(evals) * V^T for 4x4 symmetric matrix A.
 *    Eigenvalues and columns of V are sorted in descending order:
 *    evals[0] >= evals[1] >= evals[2] >= evals[3].
 * ============================================================================ */
static inline void sym4x4_jacobi_eigen(
    const double A_in[16],
    double evals[4],
    double V[16]
) {
    double A[16];
    for (int i = 0; i < 16; ++i) A[i] = A_in[i];

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            V[i * 4 + j] = (i == j) ? 1.0 : 0.0;
        }
    }

    static const int pairs[6][2] = {
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}
    };

    for (int sweep = 0; sweep < 6; ++sweep) {
        double off_norm2 = 0.0;
        for (int p = 0; p < 6; ++p) {
            int p_i = pairs[p][0];
            int p_j = pairs[p][1];
            double apq = A[p_i * 4 + p_j];
            off_norm2 += apq * apq;
        }
        if (off_norm2 < 1e-30) break;

        for (int p = 0; p < 6; ++p) {
            int ip = pairs[p][0];
            int iq = pairs[p][1];
            double apq = A[ip * 4 + iq];
            if (fabs(apq) < 1e-18) continue;

            double app = A[ip * 4 + ip];
            double aqq = A[iq * 4 + iq];
            double tau = (aqq - app) / (2.0 * apq);
            double t = (tau >= 0.0)
                ? (1.0 / (tau + sqrt(1.0 + tau * tau)))
                : (-1.0 / (-tau + sqrt(1.0 + tau * tau)));
            double c = 1.0 / sqrt(1.0 + t * t);
            double s = t * c;

            A[ip * 4 + ip] = app - t * apq;
            A[iq * 4 + iq] = aqq + t * apq;
            A[ip * 4 + iq] = 0.0;
            A[iq * 4 + ip] = 0.0;

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

            for (int r = 0; r < 4; ++r) {
                double vip = V[r * 4 + ip];
                double viq = V[r * 4 + iq];
                V[r * 4 + ip] = c * vip - s * viq;
                V[r * 4 + iq] = s * vip + c * viq;
            }
        }
    }

    for (int i = 0; i < 4; ++i) evals[i] = A[i * 4 + i];

    /* Sort eigenvalues and eigenvector columns in descending order */
    for (int i = 0; i < 3; ++i) {
        int best = i;
        for (int j = i + 1; j < 4; ++j) {
            if (evals[j] > evals[best]) best = j;
        }
        if (best != i) {
            double tmp = evals[i];
            evals[i] = evals[best];
            evals[best] = tmp;
            for (int r = 0; r < 4; ++r) {
                double vtmp = V[r * 4 + i];
                V[r * 4 + i] = V[r * 4 + best];
                V[r * 4 + best] = vtmp;
            }
        }
    }
}

/* ============================================================================
 * 3. Singular Values of 4x4 Matrix M via Gram Matrix M^T M
 * ============================================================================ */
static inline void svd4x4_singular_values(
    const double M[16],
    double sigma_out[4]
) {
    double G[16];
    for (int i = 0; i < 4; ++i) {
        for (int j = i; j < 4; ++j) {
            double s = 0.0;
            for (int k = 0; k < 4; ++k) {
                s += M[k * 4 + i] * M[k * 4 + j];
            }
            G[i * 4 + j] = s;
            G[j * 4 + i] = s;
        }
    }
    double evals[4], V[16];
    sym4x4_jacobi_eigen(G, evals, V);
    for (int i = 0; i < 4; ++i) {
        sigma_out[i] = (evals[i] > 0.0) ? sqrt(evals[i]) : 0.0;
    }
}

/* ============================================================================
 * 4. Sequence Bitplane Encoding & Streaming FASTA Loader
 * ============================================================================ */
static inline int char_to_dna_plane(unsigned char ch) {
    switch (ch) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': case 'U': case 'u': return 3;
        default: return -1;
    }
}

int encode_sequences_dna_bitplanes_c(
    const char **seqs,
    const char **names,
    int32_t n_taxa,
    int32_t n_sites,
    DnaBitplaneAlignment *out_aln
) {
    if (!seqs || n_taxa <= 0 || n_sites <= 0 || !out_aln) return -1;

    /* Validate that every sequence is non-null and has exact length n_sites */
    for (int32_t x = 0; x < n_taxa; ++x) {
        if (!seqs[x]) return -4;
        size_t slen = strlen(seqs[x]);
        if (slen != (size_t)n_sites) return -4;
    }

    int32_t n_words = (n_sites + 63) / 64;
    size_t total_words = (size_t)n_taxa * 4 * (size_t)n_words;

    uint64_t *bp = (uint64_t *)calloc(total_words, sizeof(uint64_t));
    if (!bp) return -2;

    char **t_names = (char **)calloc((size_t)n_taxa, sizeof(char *));
    if (!t_names) {
        free(bp);
        return -3;
    }

    for (int32_t x = 0; x < n_taxa; ++x) {
        if (names && names[x]) {
            size_t len = strlen(names[x]);
            t_names[x] = (char *)malloc(len + 1);
            if (t_names[x]) memcpy(t_names[x], names[x], len + 1);
        } else {
            t_names[x] = (char *)malloc(24);
            if (t_names[x]) snprintf(t_names[x], 24, "taxon_%d", x);
        }
    }

    #pragma omp parallel for schedule(static)
    for (int32_t x = 0; x < n_taxa; ++x) {
        const unsigned char *s = (const unsigned char *)seqs[x];
        uint64_t *dst_x = &bp[(size_t)x * 4 * (size_t)n_words];
        for (int32_t k = 0; k < n_sites; ++k) {
            int c = char_to_dna_plane(s[k]);
            if (c >= 0) {
                int32_t w = k >> 6;
                int32_t b = k & 63;
                dst_x[(size_t)c * (size_t)n_words + (size_t)w] |= (1ULL << b);
            }
        }
    }

    out_aln->n_taxa = n_taxa;
    out_aln->n_sites = n_sites;
    out_aln->n_words = n_words;
    out_aln->bitplanes = bp;
    out_aln->taxon_names = t_names;
    return 0;
}

int load_fasta_dna_bitplanes_c(
    const char *fasta_path,
    int32_t max_taxa,
    DnaBitplaneAlignment *out_aln
) {
    if (!fasta_path || !out_aln) return -1;
    FILE *fp = fopen(fasta_path, "rb");
    if (!fp) return -2;

    /* Pass 1: Determine n_taxa, n_sites, and verify uniform sequence lengths */
    char line[65536];
    int32_t n_taxa = 0;
    int32_t n_sites = 0;
    int32_t curr_len = 0;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '>') {
            if (n_taxa >= 1) {
                if (n_taxa == 1) {
                    n_sites = curr_len;
                } else if (curr_len != n_sites) {
                    fclose(fp);
                    return -5; /* Unequal sequence length */
                }
            }
            if (max_taxa > 0 && n_taxa >= max_taxa) {
                curr_len = n_sites;
                break;
            }
            n_taxa++;
            curr_len = 0;
        } else if (n_taxa >= 1) {
            for (const unsigned char *p = (const unsigned char *)line; *p; ++p) {
                if (*p > 0x20 && *p < 0x80) {
                    curr_len++;
                } else if (*p >= 0x80) {
                    fclose(fp);
                    return -6; /* Invalid non-ASCII byte */
                }
            }
        }
    }
    if (n_taxa >= 1) {
        if (n_taxa == 1 && n_sites == 0) n_sites = curr_len;
        else if (curr_len != n_sites) {
            fclose(fp);
            return -5;
        }
    }
    if (n_taxa <= 0 || n_sites <= 0) {
        fclose(fp);
        return -3;
    }

    int32_t n_words = (n_sites + 63) / 64;
    size_t total_words = (size_t)n_taxa * 4 * (size_t)n_words;
    uint64_t *bp = (uint64_t *)calloc(total_words, sizeof(uint64_t));
    char **t_names = (char **)calloc((size_t)n_taxa, sizeof(char *));
    if (!bp || !t_names) {
        free(bp);
        free(t_names);
        fclose(fp);
        return -4;
    }

    /* Pass 2: Stream and bit-pack sequences directly */
    rewind(fp);
    int32_t x = -1;
    int32_t site_idx = 0;
    uint64_t *dst_x = NULL;

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '>') {
            x++;
            if (x >= n_taxa) break;
            site_idx = 0;
            dst_x = &bp[(size_t)x * 4 * (size_t)n_words];

            char *name_start = line + 1;
            while (*name_start == ' ' || *name_start == '\t') name_start++;
            char *name_end = name_start;
            while (*name_end && *name_end != '\n' && *name_end != '\r' &&
                   *name_end != ' ' && *name_end != '\t') {
                name_end++;
            }
            size_t nlen = (size_t)(name_end - name_start);
            t_names[x] = (char *)malloc(nlen + 1);
            if (t_names[x]) {
                memcpy(t_names[x], name_start, nlen);
                t_names[x][nlen] = '\0';
            }
        } else if (x >= 0 && x < n_taxa) {
            for (const unsigned char *p = (const unsigned char *)line; *p; ++p) {
                unsigned char ch = *p;
                if (ch <= 0x20) continue;
                if (site_idx < n_sites) {
                    int c = char_to_dna_plane(ch);
                    if (c >= 0) {
                        int32_t w = site_idx >> 6;
                        int32_t b = site_idx & 63;
                        dst_x[(size_t)c * (size_t)n_words + (size_t)w] |= (1ULL << b);
                    }
                    site_idx++;
                }
            }
        }
    }

    fclose(fp);
    out_aln->n_taxa = n_taxa;
    out_aln->n_sites = n_sites;
    out_aln->n_words = n_words;
    out_aln->bitplanes = bp;
    out_aln->taxon_names = t_names;
    return 0;
}

void free_dna_bitplane_alignment_c(DnaBitplaneAlignment *aln) {
    if (!aln) return;
    if (aln->bitplanes) {
        free(aln->bitplanes);
        aln->bitplanes = NULL;
    }
    if (aln->taxon_names) {
        for (int32_t i = 0; i < aln->n_taxa; ++i) {
            free(aln->taxon_names[i]);
        }
        free(aln->taxon_names);
        aln->taxon_names = NULL;
    }
    aln->n_taxa = 0;
    aln->n_sites = 0;
    aln->n_words = 0;
}

/* ============================================================================
 * 5. Per-Pair Sufficient Statistics & Reduced-Alphabet SVD Extraction
 * ============================================================================ */
static inline void compute_reduced_subspace_sigmas(
    const double K_raw[16],
    const double Fa_raw[4],
    const double Fb_raw[4],
    const int active[4],
    int c_act,
    int32_t mode,
    double sigma_out[4]
) {
    /*
     * When 1 or 2 nucleotide states are missing in sequence a or b (c_act in {2, 3}),
     * restrict K to the active states A = {c : active[c] == 1}.
     * For any missing state c that is present in one sequence (so transitions
     * K_{i,c} or K_{c,i} are observed), fold those observed out-of-subspace
     * mismatches (K_{i,c} + K_{c,i}) / (c_act - 1) into the active off-diagonals
     * of row i so the survival probability K_{i,i} / F_i is preserved exactly,
     * then rescale 1 - sigma_sub by 4*(c_act - 1)/(3*c_act).
     * If a missing state is absent from both sequences, apply the Möbius relation.
     */
    double K_sub[16];
    for (int i = 0; i < 16; ++i) K_sub[i] = 0.0;

    int n_one_sided = 0;
    for (int c = 0; c < 4; ++c) {
        if (!active[c] && (Fa_raw[c] >= MISSING_STATE_TOL || Fb_raw[c] >= MISSING_STATE_TOL)) {
            n_one_sided++;
        }
    }

    double total_sub = 0.0;
    for (int i = 0; i < 4; ++i) {
        if (!active[i]) continue;
        double missing_off = 0.0;
        for (int c = 0; c < 4; ++c) {
            if (!active[c]) {
                missing_off += K_raw[i * 4 + c] + K_raw[c * 4 + i];
            }
        }
        double add_off = missing_off / (double)(c_act - 1);
        for (int j = 0; j < 4; ++j) {
            if (!active[j]) continue;
            double v = (i == j) ? K_raw[i * 4 + i] : (K_raw[i * 4 + j] + add_off);
            K_sub[i * 4 + j] = v;
            total_sub += v;
        }
    }

    if (total_sub <= 0.0) {
        sigma_out[0] = 1.0;
        sigma_out[1] = 1.0;
        sigma_out[2] = 1.0;
        sigma_out[3] = 1.0;
        return;
    }

    double Fa_s[4] = {0.0, 0.0, 0.0, 0.0};
    double Fb_s[4] = {0.0, 0.0, 0.0, 0.0};
    for (int i = 0; i < 4; ++i) {
        if (!active[i]) continue;
        for (int j = 0; j < 4; ++j) {
            if (!active[j]) continue;
            K_sub[i * 4 + j] /= total_sub;
            Fa_s[i] += K_sub[i * 4 + j];
            Fb_s[j] += K_sub[i * 4 + j];
        }
    }

    double M[16];
    for (int i = 0; i < 16; ++i) M[i] = 0.0;

    if (mode == EIGEN_MODE_PAPER_ORIG) {
        for (int i = 0; i < 4; ++i) {
            if (!active[i]) continue;
            for (int j = 0; j < 4; ++j) {
                if (!active[j]) continue;
                double t1 = sqrt(Fa_s[i]) * K_sub[i * 4 + j] / (Fa_s[j] * sqrt(Fa_s[j]));
                double t2 = K_sub[i * 4 + j] * sqrt(Fb_s[j]) / (Fb_s[i] * sqrt(Fb_s[i]));
                M[i * 4 + j] = 0.5 * (t1 + t2);
            }
        }
    } else {
        for (int i = 0; i < 4; ++i) {
            if (!active[i]) continue;
            for (int j = 0; j < 4; ++j) {
                if (!active[j]) continue;
                double l_ij = K_sub[i * 4 + j] / sqrt(Fa_s[i] * Fb_s[j]);
                M[i * 4 + j] = l_ij - sqrt(Fa_s[i] * Fb_s[j]);
            }
        }
    }

    double sig_raw[4];
    svd4x4_singular_values(M, sig_raw);

    double scale = (4.0 * (double)(c_act - 1)) / (3.0 * (double)c_act);
    int n_modes = c_act - 1;
    double active_sigs[3];
    double mean_sig = 0.0;
    int start_offset = (mode == EIGEN_MODE_PAPER_ORIG) ? 1 : 0;

    for (int k = 0; k < n_modes; ++k) {
        double s_sub = sig_raw[start_offset + k];
        if (s_sub < 0.0) s_sub = 0.0;
        if (s_sub > 1.0) s_sub = 1.0;
        double s_4;
        if (n_one_sided > 0) {
            s_4 = 1.0 - scale * (1.0 - s_sub);
        } else {
            double denom = 4.0 - (double)(4 - c_act) * s_sub;
            s_4 = (denom > 1e-12) ? (((double)c_act * s_sub) / denom) : SIGMA_FLOOR;
        }
        if (s_4 < SIGMA_FLOOR) s_4 = SIGMA_FLOOR;
        if (s_4 > 1.0) s_4 = 1.0;
        active_sigs[k] = s_4;
        mean_sig += s_4;
    }
    mean_sig /= (double)n_modes;

    sigma_out[0] = (mode == EIGEN_MODE_PAPER_ORIG) ? sig_raw[0] : 1.0;
    for (int k = 0; k < 3; ++k) {
        sigma_out[1 + k] = (k < n_modes) ? active_sigs[k] : mean_sig;
    }
    /* Sort sigma_out[1..3] descending */
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

static inline int32_t normalize_eigen_mode(int32_t mode) {
    return (mode >= 0 && mode <= 2) ? mode : EIGEN_MODE_SYMMETRIC_WLS;
}

static void compute_pair_suff_stats_from_counts_c(
    const uint64_t counts[16],
    int32_t mode,
    PairwiseSuffStats *out_stats
) {
    mode = normalize_eigen_mode(mode);
    uint64_t total_valid = 0;
    uint64_t off_diag_valid = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            uint64_t c = counts[i * 4 + j];
            total_valid += c;
            if (i != j) off_diag_valid += c;
        }
    }

    out_stats->valid_sites = (double)total_valid;
    if (total_valid == 0) {
        for (int i = 0; i < 16; ++i) out_stats->K[i] = 0.0;
        for (int i = 0; i < 4; ++i) {
            out_stats->Fa[i] = 0.25;
            out_stats->Fb[i] = 0.25;
            out_stats->xi[i] = 0.25;
            out_stats->sigma[i] = NAN;
        }
        return;
    }

    double denom = (double)total_valid + 16.0 * PSEUDOCOUNT_EPS;
    for (int i = 0; i < 4; ++i) {
        out_stats->Fa[i] = 0.0;
        out_stats->Fb[i] = 0.0;
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            double kij = ((double)counts[i * 4 + j] + PSEUDOCOUNT_EPS) / denom;
            out_stats->K[i * 4 + j] = kij;
            out_stats->Fa[i] += kij;
            out_stats->Fb[j] += kij;
        }
    }

    int active[4];
    int c_act = 0;
    for (int i = 0; i < 4; ++i) {
        active[i] = (out_stats->Fa[i] >= MISSING_STATE_TOL && out_stats->Fb[i] >= MISSING_STATE_TOL) ? 1 : 0;
        c_act += active[i];
    }

    double xi_sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        double val = sqrt(out_stats->Fa[i] * out_stats->Fb[i]);
        out_stats->xi[i] = val;
        xi_sum += val;
    }
    for (int i = 0; i < 4; ++i) {
        out_stats->xi[i] /= xi_sum;
    }

    if (c_act < 2) {
        for (int i = 0; i < 4; ++i) out_stats->sigma[i] = NAN;
        return;
    }

    if (off_diag_valid == 0) {
        for (int i = 0; i < 4; ++i) out_stats->sigma[i] = 1.0;
        return;
    }

    if (c_act < 4) {
        double act_xi_sum = 0.0;
        for (int i = 0; i < 4; ++i) {
            if (active[i]) act_xi_sum += out_stats->xi[i];
        }
        double act_mass = 0.25 * (double)c_act;
        for (int i = 0; i < 4; ++i) {
            if (active[i]) {
                out_stats->xi[i] = (act_xi_sum > 1e-12) ? (act_mass * out_stats->xi[i] / act_xi_sum) : 0.25;
            } else {
                out_stats->xi[i] = 0.25;
            }
        }
        compute_reduced_subspace_sigmas(
            out_stats->K, out_stats->Fa, out_stats->Fb, active, c_act, mode, out_stats->sigma
        );
        return;
    }

    if (mode == EIGEN_MODE_PAPER_ORIG) {
        double Lbar[16];
        double fa_p12[4], fa_m32[4], fb_p12[4], fb_m32[4];
        for (int i = 0; i < 4; ++i) {
            double sfa = sqrt(out_stats->Fa[i]);
            double sfb = sqrt(out_stats->Fb[i]);
            fa_p12[i] = sfa;
            fa_m32[i] = 1.0 / (out_stats->Fa[i] * sfa);
            fb_p12[i] = sfb;
            fb_m32[i] = 1.0 / (out_stats->Fb[i] * sfb);
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double t1 = fa_p12[i] * out_stats->K[i * 4 + j] * fa_m32[j];
                double t2 = fb_m32[i] * out_stats->K[i * 4 + j] * fb_p12[j];
                Lbar[i * 4 + j] = 0.5 * (t1 + t2);
            }
        }
        svd4x4_singular_values(Lbar, out_stats->sigma);
    } else {
        double Ltilde[16];
        double sfa[4], inv_sfa[4], sfb[4], inv_sfb[4];
        for (int i = 0; i < 4; ++i) {
            sfa[i] = sqrt(out_stats->Fa[i]);
            inv_sfa[i] = 1.0 / sfa[i];
            sfb[i] = sqrt(out_stats->Fb[i]);
            inv_sfb[i] = 1.0 / sfb[i];
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double l_ij = inv_sfa[i] * out_stats->K[i * 4 + j] * inv_sfb[j];
                Ltilde[i * 4 + j] = l_ij - sfa[i] * sfb[j];
            }
        }
        double sig_defl[4];
        svd4x4_singular_values(Ltilde, sig_defl);
        out_stats->sigma[0] = 1.0;
        out_stats->sigma[1] = sig_defl[0];
        out_stats->sigma[2] = sig_defl[1];
        out_stats->sigma[3] = sig_defl[2];
    }
}

void compute_pair_suff_stats_c(
    const uint64_t *planes_a,
    const uint64_t *planes_b,
    int32_t n_words,
    int32_t mode,
    PairwiseSuffStats *out_stats
) {
    mode = normalize_eigen_mode(mode);
    uint64_t counts[16];

    for (int i = 0; i < 4; ++i) {
        const uint64_t *pa = &planes_a[(size_t)i * (size_t)n_words];
        for (int j = 0; j < 4; ++j) {
            const uint64_t *pb = &planes_b[(size_t)j * (size_t)n_words];
            uint64_t acc = 0;
            for (int32_t w = 0; w < n_words; ++w) {
                acc += (uint64_t)__builtin_popcountll(pa[w] & pb[w]);
            }
            counts[i * 4 + j] = acc;
        }
    }

    compute_pair_suff_stats_from_counts_c(counts, mode, out_stats);
}

/* ============================================================================
 * 6. GTR Rate Matrix Construction & Lemma-1 Symmetric Eigendecomposition
 * ============================================================================ */
static inline void build_gtr_sym_matrix(
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
    for (int i = 0; i < 4; ++i) {
        double rs = 0.0;
        for (int j = 0; j < 4; ++j) {
            rs += S[i * 4 + j] * xi[j];
        }
        row_sum[i] = rs;
        mu += xi[i] * rs;
    }
    double inv_mu = (mu > 1e-30) ? (1.0 / mu) : 1.0;

    double sxi[4];
    for (int i = 0; i < 4; ++i) sxi[i] = sqrt(xi[i]);

    double neg_tr = 0.0;
    for (int i = 0; i < 4; ++i) {
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

static inline double compute_wls_distance_logspace(
    const double sigma[4],
    const double evals[4],
    double alpha,
    double *out_den
) {
    if (sigma[1] >= SIGMA_CEIL && sigma[2] >= SIGMA_CEIL && sigma[3] >= SIGMA_CEIL) {
        if (out_den) *out_den = 1.0;
        return 0.0;
    }
    double inv_alpha = 1.0 / alpha;
    double neg_lam[3], y_k[3], log_w[3];
    double max_log_w = -1e300;

    for (int idx = 0; idx < 3; ++idx) {
        int k = idx + 1;
        double sig = sigma[k];
        if (sig > SIGMA_CEIL) sig = SIGMA_CEIL;
        if (sig < SIGMA_FLOOR) sig = SIGMA_FLOOR;

        double nl = -evals[k];
        if (nl < 1e-8) nl = 1e-8;
        neg_lam[idx] = nl;

        double exponent = -log(sig) * inv_alpha;
        if (exponent > 30.0) exponent = 30.0;
        y_k[idx] = alpha * expm1(exponent);

        double lw = (2.0 + 2.0 * inv_alpha) * log(sig) - log(1.0 - sig * sig + 1e-4);
        log_w[idx] = lw;
        if (lw > max_log_w) max_log_w = lw;
    }

    double num = 0.0, den = 0.0;
    for (int idx = 0; idx < 3; ++idx) {
        double w = exp(log_w[idx] - max_log_w);
        num += w * neg_lam[idx] * y_k[idx];
        den += w * neg_lam[idx] * neg_lam[idx];
    }
    double d_ab = (den > 0.0) ? (num / den) : 0.0;

    /*
     * 3-step iteratively reweighted WLS using model-implied singular values
     * mu_k(D) = (1 + D * |lambda_k| / alpha)^{-alpha} to remove plug-in weight bias.
     */
    for (int it = 0; it < 3; ++it) {
        if (d_ab <= 0.0 || d_ab >= EIGEN_D_MAX) break;
        max_log_w = -1e300;
        for (int idx = 0; idx < 3; ++idx) {
            double log_mu = -alpha * log1p((d_ab * neg_lam[idx]) * inv_alpha);
            double mu = exp(log_mu);
            if (mu > SIGMA_CEIL) mu = SIGMA_CEIL;
            if (mu < SIGMA_FLOOR) mu = SIGMA_FLOOR;
            double lw = (2.0 + 2.0 * inv_alpha) * log(mu) - log(1.0 - mu * mu + 1e-4);
            log_w[idx] = lw;
            if (lw > max_log_w) max_log_w = lw;
        }
        num = 0.0;
        den = 0.0;
        for (int idx = 0; idx < 3; ++idx) {
            double w = exp(log_w[idx] - max_log_w);
            num += w * neg_lam[idx] * y_k[idx];
            den += w * neg_lam[idx] * neg_lam[idx];
        }
        d_ab = (den > 0.0) ? (num / den) : d_ab;
    }

    if (out_den) *out_den = den;
    if (d_ab < 0.0) d_ab = 0.0;
    if (d_ab > EIGEN_D_MAX) d_ab = EIGEN_D_MAX;
    return d_ab;
}

double eval_pair_distance_from_stats_c(
    const PairwiseSuffStats *st,
    const EigenGtrParams *params,
    int32_t mode,
    double *out_var
) {
    mode = normalize_eigen_mode(mode);
    if (!st || st->valid_sites <= 0.0 || isnan(st->sigma[0])) {
        if (out_var) *out_var = NAN;
        return NAN;
    }

    if (st->sigma[1] >= SIGMA_CEIL && st->sigma[2] >= SIGMA_CEIL && st->sigma[3] >= SIGMA_CEIL &&
        (mode != EIGEN_MODE_PAPER_ORIG || st->sigma[0] >= SIGMA_CEIL)) {
        if (out_var) *out_var = 0.0;
        return 0.0;
    }

    double alpha = (params && params->alpha > 1e-4) ? params->alpha : 1.0;
    double inv_alpha = 1.0 / alpha;

    const double default_s[6] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    const double *s_rates = (params) ? params->s_rates : default_s;

    double R[16], neg_tr_Q = 3.0;
    build_gtr_sym_matrix(s_rates, st->xi, R, &neg_tr_Q);

    double d_ab = 0.0;
    double var_ab = 0.0;

    if (mode == EIGEN_MODE_SYMMETRIC_WLS) {
        double evals[4], V[16];
        sym4x4_jacobi_eigen(R, evals, V);
        double den = 0.0;
        d_ab = compute_wls_distance_logspace(st->sigma, evals, alpha, &den);
        if (out_var) {
            double L_eff = (st->valid_sites > 1.0) ? st->valid_sites : 1.0;
            var_ab = (den > 1e-20) ? (1.0 / (L_eff * den)) : 0.0;
        }
    } else {
        int start_idx = (mode == EIGEN_MODE_PAPER_ORIG) ? 0 : 1;
        double sum_gamma = 0.0;
        double sum_var = 0.0;
        for (int k = start_idx; k < 4; ++k) {
            double sig = st->sigma[k];
            if (sig > 1.0) sig = 1.0;
            if (sig < SIGMA_FLOOR) sig = SIGMA_FLOOR;
            double exponent = -log(sig) * inv_alpha;
            if (exponent > 30.0) exponent = 30.0;
            sum_gamma += alpha * expm1(exponent);
            if (out_var && sig < SIGMA_CEIL) {
                double p_m = exp(exponent);
                sum_var += (p_m * p_m) * (1.0 / (sig * sig) - 1.0);
            }
        }
        d_ab = (neg_tr_Q > 1e-12) ? (sum_gamma / neg_tr_Q) : 0.0;
        if (out_var) {
            double L_eff = (st->valid_sites > 1.0) ? st->valid_sites : 1.0;
            var_ab = sum_var / (L_eff * neg_tr_Q * neg_tr_Q);
        }
    }

    if (d_ab < 0.0) d_ab = 0.0;
    if (d_ab > EIGEN_D_MAX) d_ab = EIGEN_D_MAX;
    if (out_var) *out_var = var_ab;
    return d_ab;
}

/* ============================================================================
 * 7. Composite Log-Posterior Evaluation (Eq. 7 + Priors) for MAP Estimation
 * ============================================================================ */
static double eval_map_objective_from_theta(
    const double theta[7],
    const PairwiseSuffStats *samples,
    int32_t n_samples,
    int32_t n_eff,
    int32_t mode
) {
    EigenGtrParams params;
    double mean_log_s = 0.0;
    for (int i = 0; i < 6; ++i) mean_log_s += theta[i];
    mean_log_s /= 6.0;

    double s_sum = 0.0;
    for (int i = 0; i < 6; ++i) {
        double v = theta[i] - mean_log_s;
        if (v > 10.0) v = 10.0;
        if (v < -10.0) v = -10.0;
        params.s_rates[i] = exp(v);
        s_sum += params.s_rates[i];
    }
    for (int i = 0; i < 6; ++i) {
        params.s_rates[i] *= (6.0 / s_sum);
    }

    double log_alpha = theta[6];
    if (log_alpha > 5.0) log_alpha = 5.0;
    if (log_alpha < -3.0) log_alpha = -3.0;
    params.alpha = exp(log_alpha);
    double alpha = params.alpha;
    double inv_alpha = 1.0 / alpha;

    /*
     * DNA Adaptation of the Section 2.1 Prior (Penn et al. 2025):
     *   The paper uses S ~ Normal^+(LG, 1) for proteins and alpha ~ HalfNormal(1).
     *   For DNA we center S at the Jukes-Cantor baseline S_r ~ Normal(1, 1) with a
     *   soft gauge-fixing penalty N(0, 0.1^2) on mean(log S), plus alpha ~ HalfNormal(1).
     */
    double log_prior = -0.5 * (mean_log_s * mean_log_s) / 0.01;
    for (int i = 0; i < 6; ++i) {
        double diff_s = params.s_rates[i] - 1.0;
        log_prior -= 0.5 * (diff_s * diff_s);
    }
    log_prior -= 0.5 * (alpha * alpha);

    double total_ll = 0.0;
    int32_t valid_pairs = 0;

    #pragma omp parallel for reduction(+:total_ll, valid_pairs) schedule(static)
    for (int32_t m = 0; m < n_samples; ++m) {
        const PairwiseSuffStats *st = &samples[m];
        if (st->valid_sites < EIGEN_MIN_OVERLAP_SITES || isnan(st->sigma[0])) continue;
        valid_pairs++;

        double R[16], neg_tr_Q = 3.0;
        build_gtr_sym_matrix(params.s_rates, st->xi, R, &neg_tr_Q);

        double evals[4], V[16];
        sym4x4_jacobi_eigen(R, evals, V);

        double d_ab = 0.0;
        if (mode == EIGEN_MODE_SYMMETRIC_WLS) {
            d_ab = compute_wls_distance_logspace(st->sigma, evals, alpha, NULL);
        } else {
            int start_idx = (mode == EIGEN_MODE_PAPER_ORIG) ? 0 : 1;
            double sum_gamma = 0.0;
            for (int k = start_idx; k < 4; ++k) {
                double sig = st->sigma[k];
                if (sig > 1.0) sig = 1.0;
                if (sig < SIGMA_FLOOR) sig = SIGMA_FLOOR;
                double exponent = -log(sig) * inv_alpha;
                if (exponent > 30.0) exponent = 30.0;
                sum_gamma += alpha * expm1(exponent);
            }
            d_ab = (neg_tr_Q > 1e-12) ? (sum_gamma / neg_tr_Q) : 0.0;
            if (d_ab < 0.0) d_ab = 0.0;
            if (d_ab > EIGEN_D_MAX) d_ab = EIGEN_D_MAX;
        }

        double mu_k[4];
        mu_k[0] = 1.0;
        for (int k = 1; k < 4; ++k) {
            double base = 1.0 - (d_ab * evals[k]) * inv_alpha;
            if (base < 1.0) base = 1.0;
            mu_k[k] = pow(base, -alpha);
        }

        double sxi[4];
        for (int i = 0; i < 4; ++i) sxi[i] = sqrt(st->xi[i]);

        double pair_ll = 0.0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double sym_p = 0.0;
                for (int k = 0; k < 4; ++k) {
                    sym_p += V[i * 4 + k] * V[j * 4 + k] * mu_k[k];
                }
                double p_ij = (sxi[j] / sxi[i]) * sym_p;
                if (p_ij < 1e-15) p_ij = 1e-15;
                pair_ll += st->K[i * 4 + j] * log(p_ij);
            }
        }
        total_ll += st->valid_sites * pair_ll;
    }

    double weighted_ll = (valid_pairs > 0) ? (((double)n_eff / (double)valid_pairs) * total_ll) : 0.0;
    return weighted_ll + log_prior;
}

/* ============================================================================
 * 8. Shanno-Phua Scaled BFGS Optimizer for MAP (\alpha, S) Inference
 * ============================================================================ */
int32_t check_eigen_q_near_degenerate_c(
    const double s_rates[6],
    const PairwiseSuffStats *samples,
    int32_t n_samples,
    double alpha
) {
    double mean_xi[4] = {0.0, 0.0, 0.0, 0.0};
    int32_t n_xi = 0;
    for (int32_t m = 0; m < n_samples; ++m) {
        if (samples[m].valid_sites >= EIGEN_MIN_OVERLAP_SITES && !isnan(samples[m].sigma[0])) {
            for (int i = 0; i < 4; ++i) mean_xi[i] += samples[m].xi[i];
            n_xi++;
        }
    }
    if (n_xi > 0) {
        double sum_xi = mean_xi[0] + mean_xi[1] + mean_xi[2] + mean_xi[3];
        for (int i = 0; i < 4; ++i) mean_xi[i] = (sum_xi > 1e-12) ? (mean_xi[i] / sum_xi) : 0.25;
    } else {
        for (int i = 0; i < 4; ++i) mean_xi[i] = 0.25;
    }

    double R[16], neg_tr_Q = 3.0;
    build_gtr_sym_matrix(s_rates, mean_xi, R, &neg_tr_Q);
    double evals[4], V[16];
    sym4x4_jacobi_eigen(R, evals, V);

    double r_min = 1e300, r_max = 0.0;
    for (int k = 1; k < 4; ++k) {
        double rk = -evals[k];
        if (rk < 0.0) rk = 0.0;
        if (rk < r_min) r_min = rk;
        if (rk > r_max) r_max = rk;
    }
    double rel_spread = (r_max > 1e-12) ? ((r_max - r_min) / r_max) : 0.0;
    if (rel_spread < EIGEN_Q_DEGENERATE_TOL) {
        fprintf(stderr,
            "[phylo2vec_eigen_dist WARNING] Fitted Q spectrum is near-degenerate "
            "(non-trivial eigenvalues [%.4f, %.4f, %.4f], relative spread %.2f%% < %.1f%%): "
            "Gamma shape alpha (%.4f) is weakly or un-identifiable from pairwise data (JC69/F81 regime).\n",
            -evals[1], -evals[2], -evals[3], 100.0 * rel_spread, 100.0 * EIGEN_Q_DEGENERATE_TOL, alpha);
        return 1;
    }
    return 0;
}

#define BOOT_MAX_MULT 8

double compute_eigen_bootstrap_alpha_sd_c(
    const DnaBitplaneAlignment *aln,
    const int32_t *pair_a,
    const int32_t *pair_b,
    int32_t n_pairs,
    int32_t n_eff,
    int32_t mode,
    uint64_t seed,
    const double theta_map[7],
    const double H_map[49]
) {
    mode = normalize_eigen_mode(mode);
    if (!aln || aln->n_sites < 2 || n_pairs <= 0) return 0.0;

    int32_t L = aln->n_sites;
    int32_t n_words = aln->n_words;
    int32_t M_boot = (n_pairs <= 256) ? n_pairs : 256;
    const int32_t B = EIGEN_DEFAULT_BOOT_REPS;

    uint8_t *site_mult = (uint8_t *)malloc((size_t)L * sizeof(uint8_t));
    uint64_t *mask_ge = (uint64_t *)malloc((size_t)BOOT_MAX_MULT * (size_t)n_words * sizeof(uint64_t));
    PairwiseSuffStats *boot_samples = (PairwiseSuffStats *)malloc((size_t)M_boot * sizeof(PairwiseSuffStats));
    if (!site_mult || !mask_ge || !boot_samples) {
        free(site_mult);
        free(mask_ge);
        free(boot_samples);
        return 0.0;
    }

    uint64_t boot_rng = (seed ? seed : 0xCAFEBABE12345678ULL) ^ 0x9E3779B97F4A7C15ULL;
    double alpha_boot[EIGEN_DEFAULT_BOOT_REPS];
    const double eps_fd = 1e-5;

    for (int32_t b = 0; b < B; ++b) {
        memset(site_mult, 0, (size_t)L * sizeof(uint8_t));
        int max_mult = 0;
        for (int32_t s = 0; s < L; ++s) {
            int32_t idx = (int32_t)(eigen_splitmix64(&boot_rng) % (uint64_t)L);
            if (site_mult[idx] < BOOT_MAX_MULT) {
                site_mult[idx]++;
                if ((int)site_mult[idx] > max_mult) max_mult = (int)site_mult[idx];
            }
        }
        if (max_mult < 1) max_mult = 1;

        memset(mask_ge, 0, (size_t)max_mult * (size_t)n_words * sizeof(uint64_t));
        for (int32_t s = 0; s < L; ++s) {
            int c = (int)site_mult[s];
            if (c <= 0) continue;
            int32_t w = s >> 6;
            uint64_t bit = 1ULL << (s & 63);
            for (int r = 0; r < c; ++r) {
                mask_ge[(size_t)r * (size_t)n_words + (size_t)w] |= bit;
            }
        }

        #pragma omp parallel for schedule(static)
        for (int32_t m = 0; m < M_boot; ++m) {
            int32_t p_idx = (M_boot == n_pairs) ? m : (int32_t)(((int64_t)m * (int64_t)n_pairs) / (int64_t)M_boot);
            int32_t ta = pair_a[p_idx];
            int32_t tb = pair_b[p_idx];
            const uint64_t *pa_base = &aln->bitplanes[(size_t)ta * 4 * (size_t)n_words];
            const uint64_t *pb_base = &aln->bitplanes[(size_t)tb * 4 * (size_t)n_words];

            uint64_t counts[16];
            for (int i = 0; i < 4; ++i) {
                const uint64_t *pa = &pa_base[(size_t)i * (size_t)n_words];
                for (int j = 0; j < 4; ++j) {
                    const uint64_t *pb = &pb_base[(size_t)j * (size_t)n_words];
                    uint64_t acc = 0;
                    for (int32_t w = 0; w < n_words; ++w) {
                        uint64_t ab = pa[w] & pb[w];
                        if (!ab) continue;
                        for (int r = 0; r < max_mult; ++r) {
                            acc += (uint64_t)__builtin_popcountll(
                                ab & mask_ge[(size_t)r * (size_t)n_words + (size_t)w]
                            );
                        }
                    }
                    counts[i * 4 + j] = acc;
                }
            }
            compute_pair_suff_stats_from_counts_c(counts, mode, &boot_samples[m]);
        }

        /* Warm-start BFGS from MAP solution theta_map and inverse Hessian H_map */
        double theta[7];
        double H[49];
        memcpy(theta, theta_map, sizeof(theta));
        memcpy(H, H_map, sizeof(H));

        double f_curr = eval_map_objective_from_theta(theta, boot_samples, M_boot, n_eff, mode);
        double grad[7];
        for (int d = 0; d < 7; ++d) {
            double th_p[7];
            memcpy(th_p, theta, sizeof(theta));
            th_p[d] += eps_fd;
            double fp = eval_map_objective_from_theta(th_p, boot_samples, M_boot, n_eff, mode);
            grad[d] = (fp - f_curr) / eps_fd;
        }

        for (int32_t iter = 0; iter < 6; ++iter) {
            double gnorm = 0.0;
            for (int d = 0; d < 7; ++d) gnorm += grad[d] * grad[d];
            if (sqrt(gnorm) < 1e-5 * (1.0 + fabs(f_curr))) break;

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
            for (int ls = 0; ls < 8; ++ls) {
                for (int i = 0; i < 7; ++i) theta_next[i] = theta[i] + step * p[i];
                f_next = eval_map_objective_from_theta(theta_next, boot_samples, M_boot, n_eff, mode);
                if (f_next >= f_curr + 1e-4 * step * dir_deriv) {
                    accepted = 1;
                    break;
                }
                step *= 0.5;
            }
            if (!accepted) break;

            double grad_next[7];
            for (int d = 0; d < 7; ++d) {
                double th_p[7];
                memcpy(th_p, theta_next, sizeof(theta_next));
                th_p[d] += eps_fd;
                double fp = eval_map_objective_from_theta(th_p, boot_samples, M_boot, n_eff, mode);
                grad_next[d] = (fp - f_next) / eps_fd;
            }

            double s_vec[7], y_vec[7], sy = 0.0;
            for (int i = 0; i < 7; ++i) {
                s_vec[i] = theta_next[i] - theta[i];
                y_vec[i] = grad[i] - grad_next[i];
                sy += s_vec[i] * y_vec[i];
            }
            if (sy > 1e-12) {
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

        double log_a = theta[6];
        if (log_a > 5.0) log_a = 5.0;
        if (log_a < -3.0) log_a = -3.0;
        alpha_boot[b] = exp(log_a);
    }

    free(site_mult);
    free(mask_ge);
    free(boot_samples);

    double mean_a = 0.0;
    for (int32_t b = 0; b < B; ++b) mean_a += alpha_boot[b];
    mean_a /= (double)B;

    double var_a = 0.0;
    for (int32_t b = 0; b < B; ++b) {
        double da = alpha_boot[b] - mean_a;
        var_a += da * da;
    }
    return (B > 1) ? sqrt(var_a / (double)(B - 1)) : 0.0;
}

int fit_eigen_gtr_map_c(
    const DnaBitplaneAlignment *aln,
    int32_t n_sample_pairs,
    int32_t mode,
    uint64_t seed,
    EigenGtrParams *out_params
) {
    if (!aln || aln->n_taxa < 2 || !out_params) return -1;
    mode = normalize_eigen_mode(mode);

    int32_t n_eff = aln->n_taxa - 1;
    int64_t total_pairs = ((int64_t)aln->n_taxa * (int64_t)(aln->n_taxa - 1)) / 2;
    int32_t M = n_sample_pairs;
    if (M <= 0 || (int64_t)M > total_pairs) {
        M = (total_pairs > 50000) ? 50000 : (int32_t)total_pairs;
    }

    PairwiseSuffStats *samples = (PairwiseSuffStats *)malloc((size_t)M * sizeof(PairwiseSuffStats));
    int32_t *pair_a = (int32_t *)malloc((size_t)M * sizeof(int32_t));
    int32_t *pair_b = (int32_t *)malloc((size_t)M * sizeof(int32_t));
    if (!samples || !pair_a || !pair_b) {
        free(samples);
        free(pair_a);
        free(pair_b);
        return -2;
    }

    if ((int64_t)M == total_pairs) {
        int32_t idx = 0;
        for (int32_t a = 0; a < aln->n_taxa; ++a) {
            for (int32_t b = a + 1; b < aln->n_taxa; ++b) {
                pair_a[idx] = a;
                pair_b[idx] = b;
                idx++;
            }
        }
    } else {
        uint64_t rng = seed ? seed : 0xCAFEBABE12345678ULL;
        for (int32_t m = 0; m < M; ++m) {
            int32_t a = (int32_t)(eigen_splitmix64(&rng) % (uint64_t)aln->n_taxa);
            int32_t b = (int32_t)(eigen_splitmix64(&rng) % (uint64_t)(aln->n_taxa - 1));
            if (b >= a) b++;
            if (a > b) { int32_t t = a; a = b; b = t; }
            pair_a[m] = a;
            pair_b[m] = b;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int32_t m = 0; m < M; ++m) {
        const uint64_t *pa = &aln->bitplanes[(size_t)pair_a[m] * 4 * (size_t)aln->n_words];
        const uint64_t *pb = &aln->bitplanes[(size_t)pair_b[m] * 4 * (size_t)aln->n_words];
        compute_pair_suff_stats_c(pa, pb, aln->n_words, mode, &samples[m]);
    }

    /*
     * Initialize exchangeabilities at Jukes-Cantor (log S_r = 0) and perform a
     * 7-point 1D grid scan over log(alpha) for warm-starting BFGS.
     */
    double theta[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    static const double alpha_grid[7] = {-1.0, -0.5, -0.2, 0.0, 0.3, 0.7, 1.2};
    double best_init_f = -1e300;
    double best_log_alpha = 0.0;
    for (int g = 0; g < 7; ++g) {
        theta[6] = alpha_grid[g];
        double f = eval_map_objective_from_theta(theta, samples, M, n_eff, mode);
        if (f > best_init_f) {
            best_init_f = f;
            best_log_alpha = alpha_grid[g];
        }
    }
    theta[6] = best_log_alpha;

    double H[49];
    for (int i = 0; i < 49; ++i) H[i] = 0.0;
    for (int i = 0; i < 7; ++i) H[i * 7 + i] = 1.0;

    double f_curr = eval_map_objective_from_theta(theta, samples, M, n_eff, mode);
    double grad[7];
    const double eps_fd = 1e-5;

    for (int d = 0; d < 7; ++d) {
        double th_p[7], th_m[7];
        memcpy(th_p, theta, sizeof(theta));
        memcpy(th_m, theta, sizeof(theta));
        th_p[d] += eps_fd;
        th_m[d] -= eps_fd;
        double fp = eval_map_objective_from_theta(th_p, samples, M, n_eff, mode);
        double fm = eval_map_objective_from_theta(th_m, samples, M, n_eff, mode);
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
            f_next = eval_map_objective_from_theta(theta_next, samples, M, n_eff, mode);
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
            double fp = eval_map_objective_from_theta(th_p, samples, M, n_eff, mode);
            double fm = eval_map_objective_from_theta(th_m, samples, M, n_eff, mode);
            grad_next[d] = (fp - fm) / (2.0 * eps_fd);
        }

        /* BFGS inverse Hessian update (for maximization: y = grad_old - grad_new) */
        double s_vec[7], y_vec[7];
        double sy = 0.0, yy = 0.0;
        for (int i = 0; i < 7; ++i) {
            s_vec[i] = theta_next[i] - theta[i];
            y_vec[i] = grad[i] - grad_next[i];
            sy += s_vec[i] * y_vec[i];
            yy += y_vec[i] * y_vec[i];
        }

        if (sy > 1e-12) {
            /* Shanno-Phua scaling on initial step */
            if (iter == 0 && yy > 1e-12) {
                double sp_scale = sy / yy;
                for (int i = 0; i < 7; ++i) H[i * 7 + i] = sp_scale;
            }
            double Hy[7];
            double yHy = 0.0;
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
    for (int i = 0; i < 6; ++i) {
        out_params->s_rates[i] *= (6.0 / s_sum);
    }

    double log_alpha = theta[6];
    if (log_alpha > 5.0) log_alpha = 5.0;
    if (log_alpha < -3.0) log_alpha = -3.0;
    out_params->alpha = exp(log_alpha);
    out_params->log_post = f_curr;
    out_params->n_iters = iter;
    out_params->q_near_degenerate = check_eigen_q_near_degenerate_c(
        out_params->s_rates, samples, M, out_params->alpha
    );
    out_params->alpha_sd = compute_eigen_bootstrap_alpha_sd_c(
        aln, pair_a, pair_b, M, n_eff, mode, seed, theta, H
    );

    free(pair_a);
    free(pair_b);
    free(samples);
    return 0;
}

/* ============================================================================
 * 9. Full & Subset Distance Matrix Evaluation on Multi-Core CPU (OpenMP)
 * ============================================================================ */
void compute_eigen_distance_matrix_cpu_c(
    const DnaBitplaneAlignment *aln,
    const EigenGtrParams *params,
    int32_t mode,
    double *out_D,
    double *out_Var
) {
    mode = normalize_eigen_mode(mode);
    int32_t N = aln->n_taxa;
    int32_t n_words = aln->n_words;

    for (int32_t a = 0; a < N; ++a) {
        out_D[(size_t)a * (size_t)N + (size_t)a] = 0.0;
        if (out_Var) out_Var[(size_t)a * (size_t)N + (size_t)a] = 0.0;
    }

    #pragma omp parallel for schedule(dynamic, 4)
    for (int32_t a = 0; a < N; ++a) {
        const uint64_t *pa = &aln->bitplanes[(size_t)a * 4 * (size_t)n_words];
        for (int32_t b = a + 1; b < N; ++b) {
            const uint64_t *pb = &aln->bitplanes[(size_t)b * 4 * (size_t)n_words];
            PairwiseSuffStats st;
            compute_pair_suff_stats_c(pa, pb, n_words, mode, &st);
            double var_ab = 0.0;
            double d_ab = eval_pair_distance_from_stats_c(
                &st, params, mode, out_Var ? &var_ab : NULL
            );
            out_D[(size_t)a * (size_t)N + (size_t)b] = d_ab;
            out_D[(size_t)b * (size_t)N + (size_t)a] = d_ab;
            if (out_Var) {
                out_Var[(size_t)a * (size_t)N + (size_t)b] = var_ab;
                out_Var[(size_t)b * (size_t)N + (size_t)a] = var_ab;
            }
        }
    }
}

void compute_eigen_subset_distance_matrix_cpu_c(
    const DnaBitplaneAlignment *aln,
    const int32_t *taxon_indices,
    int32_t n_sub,
    const EigenGtrParams *params,
    int32_t mode,
    double *out_D
) {
    mode = normalize_eigen_mode(mode);
    int32_t n_words = aln->n_words;

    for (int32_t i = 0; i < n_sub; ++i) {
        out_D[(size_t)i * (size_t)n_sub + (size_t)i] = 0.0;
    }

    #pragma omp parallel for schedule(dynamic, 4)
    for (int32_t i = 0; i < n_sub; ++i) {
        int32_t a = taxon_indices[i];
        const uint64_t *pa = &aln->bitplanes[(size_t)a * 4 * (size_t)n_words];
        for (int32_t j = i + 1; j < n_sub; ++j) {
            int32_t b = taxon_indices[j];
            const uint64_t *pb = &aln->bitplanes[(size_t)b * 4 * (size_t)n_words];
            PairwiseSuffStats st;
            compute_pair_suff_stats_c(pa, pb, n_words, mode, &st);
            double d_ab = eval_pair_distance_from_stats_c(&st, params, mode, NULL);
            out_D[(size_t)i * (size_t)n_sub + (size_t)j] = d_ab;
            out_D[(size_t)j * (size_t)n_sub + (size_t)i] = d_ab;
        }
    }
}
