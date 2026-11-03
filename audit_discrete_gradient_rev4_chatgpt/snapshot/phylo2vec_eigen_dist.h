#ifndef PHYLO2VEC_EIGEN_DIST_H
#define PHYLO2VEC_EIGEN_DIST_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Maximum finite evolutionary distance cap (in expected substitutions per site)
 * returned when a sequence pair is completely saturated.
 *
 * Edge-case & reduced-alphabet behavior:
 *   - Zero shared unambiguous sites (valid_sites == 0) or fewer than 2 active
 *     nucleotide states (C_act < 2 with min(F^a_c, F^b_c) >= 1e-3, e.g. poly-A):
 *     sigma[i] = NAN and distance returns NAN (unestimable).
 *   - Identical sequences on shared sites (zero off-diagonal mismatch counts
 *     with C_act >= 2): sigma[i] = 1.0 and distance returns exact 0.0.
 *   - Reduced alphabet (C_act in {2, 3}, when 1 or 2 bases are missing/masked):
 *     compute_pair_suff_stats_c projects onto the C_act x C_act active subspace
 *     prior to knowing S, folding one-sided observed out-of-subspace mismatches
 *     to preserve survival probabilities K_{ii}/F_i and applying the isotropic
 *     4-state rescaling (or Möbius inversion when a base is absent in both
 *     sequences) with stationary weight 0.25 imputed for absent states.
 *     This approximation is asymptotically unbiased as L -> inf under isotropic
 *     exchangeabilities (<2% at L = 200,000), with finite-sample / rate-asymmetry
 *     deviations at L = 3,000 of ~3-7% when 1 base is missing and ~12-18% when
 *     2 bases are missing (leaving only a 2x2 active block).
 *   - In MAP parameter optimization, pairs with valid_sites < EIGEN_MIN_OVERLAP_SITES
 *     or C_act < 2 (isnan(sigma[1])) are skipped from the composite log-likelihood.
 */
#define EIGEN_D_MAX             10.0
#define EIGEN_MIN_OVERLAP_SITES 30.0
#define EIGEN_DEFAULT_BOOT_REPS 20
#define EIGEN_Q_DEGENERATE_TOL  0.05

/* ============================================================================
 * Distance Estimator Variants (Penn et al. 2025, Bull. Math. Biol. 87:32)
 * ============================================================================ */
typedef enum {
    /*
     * Default mode for MAP (S, \alpha) estimation and distance evaluation:
     * resolves to EIGEN_MODE_SYMMETRIC_WLS (Mode 2).
     */
    EIGEN_MODE_DEFAULT = -1,

    /*
     * Mode 0: Symmetrized & Scale-Corrected Variant of Penn et al. (2025) Eq. (5)-(6)
     *   Note: The paper as printed writes Eq. (5) with diag(F^b)^{1/2} K diag(F^b)^{-3/2}
     *   in the second term (which is asymmetric under a <-> b) and includes an extra
     *   factor of 1/2 in Eq. (6) (which biases distances by ~ -50%). Here Mode 0
     *   uses the transpose-consistent (+1/2, -3/2) average:
     *     \bar{L}^{ab} = 0.5 * [ diag(F^a)^{1/2} K^{ab} diag(F^a)^{-3/2}
     *                          + diag(F^b)^{-3/2} K^{ab} diag(F^b)^{1/2} ]
     *   and omits the duplicate 1/2 factor in Eq. (6):
     *     D_{ab} = (1 / -tr(Q^{ab})) * \sum_{i=0}^3 \alpha * ((Sigma_{ii})^{-1/\alpha} - 1)
     */
    EIGEN_MODE_PAPER_ORIG = 0,

    /*
     * Mode 1: Canonical Symmetric Contingency SVD (-1/2, -1/2) + Rank-1 Deflation
     *   L_{sym}^{ab} = diag(F^a)^{-1/2} K^{ab} diag(F^b)^{-1/2}
     *   Deflates exact unit singular mode u_0 = sqrt(F^a), v_0 = sqrt(F^b) (Sigma_0 = 1):
     *   \widetilde{L}^{ab} = L_{sym}^{ab} - sqrt(F^a) sqrt(F^b)^T
     *   D_{ab} = (1 / -tr(Q^{ab})) * \sum_{i=1}^3 \alpha * ((Sigma_{ii})^{-1/\alpha} - 1)
     */
    EIGEN_MODE_SYMMETRIC_SVD = 1,

    /*
     * Mode 2 (DEFAULT for MAP fitting): Canonical Symmetric SVD + Iteratively Reweighted WLS
     *   Weights each deflated eigenmode i = 1..3 by its inverse delta-method
     *   variance w_i = mu_i^{2 + 2/\alpha} / (1 - mu_i^2 + eps) (computed in
     *   log-space to prevent underflow and refined with model-implied mu_i(D)
     *   to eliminate plug-in weight bias) against the sorted non-zero eigenvalues
     *   \lambda_i^{ab} of Q^{ab}:
     *   D_{ab}^{WLS} = (\sum_{i=1}^3 w_i (-\lambda_i^{ab}) \alpha (Sigma_{ii}^{-1/\alpha} - 1))
     *                / (\sum_{i=1}^3 w_i (\lambda_i^{ab})^2)
     *
     *   Why Mode 2 is the default for fit_eigen_gtr_map_{c,gpu_c}:
     *     In joint (S, \alpha) MAP estimation at finite L (e.g. L = 2000), Mode 1
     *     (SVD-sum) weights all three non-trivial singular values equally in the
     *     plug-in distance D_{ab}. Because fast transition modes (for \pi_R = \pi_Y = 0.5,
     *     HKY/TN93 has one transversion eigenvalue and two faster transition eigenvalues)
     *     have smaller, noisier singular values \hat{\sigma}_k, equal weighting inflates
     *     \hat{D}_{ab} at small \alpha and induces a finite-sample upward bias on \hat{\alpha}
     *     (+33% at \alpha=0.5, +18% at \alpha=2.0 across 12 replicates). Mode 2 downweights
     *     noisy saturated modes via w_i, reducing joint-fit \hat{\alpha} bias to +10% at
     *     \alpha=0.5 and -4% at \alpha=2.0.
     */
    EIGEN_MODE_SYMMETRIC_WLS = 2
} EigenDistMode;

/* ============================================================================
 * Bit-Packed 4-Bitplane DNA/RNA Alignment Container
 *   Each sequence x in 0..n_taxa-1 is stored as 4 contiguous bitplanes
 *   (A=0, C=1, G=2, T/U=3) of n_words = (n_sites + 63) / 64 uint64_t words.
 *   Memory layout: bitplanes[((size_t)x * 4 + c) * n_words + w]
 * ============================================================================ */
typedef struct {
    int32_t n_taxa;
    int32_t n_sites;
    int32_t n_words;
    uint64_t *bitplanes;   /* length: (size_t)n_taxa * 4 * n_words */
    char **taxon_names;    /* length: n_taxa (null-terminated strings) */
} DnaBitplaneAlignment;

/* ============================================================================
 * Fitted GTR + Gamma(\alpha, \alpha) Model Parameters
 *   s_rates[6]:        symmetric off-diagonal exchangeabilities in order:
 *                      (AC, AG, AT, CG, CT, GT), normalized so mean(s_rates) = 1.
 *   alpha:             Gamma(\alpha, \alpha) shape/scale parameter (> 0).
 *   log_post:          Final MAP log-posterior value on the training pair set.
 *   n_iters:           Optimizer iterations taken.
 *   q_near_degenerate: 1 if the non-trivial eigenvalues of the fitted Q matrix
 *                      have relative spread < 5% (EIGEN_Q_DEGENERATE_TOL), in
 *                      which case \alpha is structurally unidentifiable from
 *                      pairwise data (as in JC69/F81) and prior-dominated; 0 otherwise.
 *   alpha_sd:          Bootstrap standard deviation of \hat{\alpha} across
 *                      EIGEN_DEFAULT_BOOT_REPS (20) site-level resamples.
 * ============================================================================ */
typedef struct {
    double s_rates[6];          /* (S_AC, S_AG, S_AT, S_CG, S_CT, S_GT) */
    double alpha;               /* Gamma shape parameter \alpha > 0 */
    double log_post;            /* Final MAP log-posterior */
    int32_t n_iters;            /* Optimizer iterations taken */
    int32_t q_near_degenerate;  /* 1 if Q spectrum spread < 5% (\alpha unidentifiable) */
    double alpha_sd;            /* Site-bootstrap SD of \hat{\alpha} (20 resamples) */
} EigenGtrParams;

/* ============================================================================
 * Precomputed Pairwise Sufficient Statistics for Fast MAP Optimization
 * ============================================================================ */
typedef struct {
    double K[16];          /* 4x4 normalized transition frequency matrix K^{ab} */
    double Fa[4];          /* Row marginals F^a */
    double Fb[4];          /* Column marginals F^b */
    double xi[4];          /* Normalized pair-specific stationary dist \xi^{ab} */
    double sigma[4];       /* Sorted singular values \Sigma_{00} >= ... >= \Sigma_{33} */
    double valid_sites;    /* Number of jointly unambiguous sites L_{ab} */
} PairwiseSuffStats;

/* Alignment I/O and encoding */
int encode_sequences_dna_bitplanes_c(
    const char **seqs,
    const char **names,
    int32_t n_taxa,
    int32_t n_sites,
    DnaBitplaneAlignment *out_aln
);

int load_fasta_dna_bitplanes_c(
    const char *fasta_path,
    int32_t max_taxa,
    DnaBitplaneAlignment *out_aln
);

void free_dna_bitplane_alignment_c(DnaBitplaneAlignment *aln);

/* Core per-pair computations */
void compute_pair_suff_stats_c(
    const uint64_t *planes_a,
    const uint64_t *planes_b,
    int32_t n_words,
    int32_t mode,
    PairwiseSuffStats *out_stats
);

double eval_pair_distance_from_stats_c(
    const PairwiseSuffStats *st,
    const EigenGtrParams *params,
    int32_t mode,
    double *out_var
);

/*
 * MAP Parameter Estimation (Eq. 7 + Priors)
 *
 * Composite-likelihood weighting heuristic:
 *   Summing all M sampled pairs would overstate Fisher information by ~O(n_taxa)
 *   because pairwise paths share internal tree branches, whereas dividing by M
 *   (weight 1/M) treats the entire alignment as a single pair. Both CPU and GPU
 *   MAP estimators weight the pairwise composite log-likelihood by an effective
 *   number of independent pairs N_eff = n_taxa - 1:
 *     objective = (N_eff / M_valid) * \sum_{m=1}^{M_valid} ll_m + log_prior.
 *   Note: This N_eff = n_taxa - 1 scaling is a fast heuristic; for calibrated
 *   posterior covariance (e.g., in HMC/NUTS sampling), the principled approach
 *   is a Godambe information / sandwich magnitude adjustment (Ribatet et al. 2012).
 *
 * Default estimator mode:
 *   Passing mode = EIGEN_MODE_DEFAULT (-1) (or any value outside {0, 1, 2})
 *   selects Mode 2 (EIGEN_MODE_SYMMETRIC_WLS), which down-weights noisy rapid
 *   transition modes relative to transversion modes and has substantially lower
 *   finite-sample upward bias and variance on \hat{\alpha} in joint (S, \alpha)
 *   fits than Mode 1.
 *
 * Diagnostics & uncertainty populated in out_params:
 *   - out_params->q_near_degenerate: set to 1 (with a warning to stderr) if all
 *     3 non-trivial eigenvalues of the fitted Q matrix are within
 *     EIGEN_Q_DEGENERATE_TOL (5%) of one another (near-JC69/F81 regime where
 *     \alpha is unidentifiable from pairwise data).
 *   - out_params->alpha_sd: site-level bootstrap standard deviation of \hat{\alpha}
 *     over EIGEN_DEFAULT_BOOT_REPS (20) multinomial site resamples.
 */
int fit_eigen_gtr_map_c(
    const DnaBitplaneAlignment *aln,
    int32_t n_sample_pairs,
    int32_t mode,
    uint64_t seed,
    EigenGtrParams *out_params
);

int32_t check_eigen_q_near_degenerate_c(
    const double s_rates[6],
    const PairwiseSuffStats *samples,
    int32_t n_samples,
    double alpha
);

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
);

/* Full N x N and Subset Distance Matrix Evaluation (OpenMP Multi-Core CPU) */
void compute_eigen_distance_matrix_cpu_c(
    const DnaBitplaneAlignment *aln,
    const EigenGtrParams *params,
    int32_t mode,
    double *out_D,
    double *out_Var
);

void compute_eigen_subset_distance_matrix_cpu_c(
    const DnaBitplaneAlignment *aln,
    const int32_t *taxon_indices,
    int32_t n_sub,
    const EigenGtrParams *params,
    int32_t mode,
    double *out_D
);

/* Multi-GPU CUDA Accelerated Entry Points */
int fit_eigen_gtr_map_gpu_c(
    const DnaBitplaneAlignment *aln,
    int32_t n_sample_pairs,
    int32_t mode,
    uint64_t seed,
    EigenGtrParams *out_params
);

int compute_eigen_distance_matrix_gpu_c(
    const DnaBitplaneAlignment *aln,
    const EigenGtrParams *params,
    int32_t mode,
    int32_t num_gpus,
    double *out_D
);

int compute_eigen_subset_distance_matrix_gpu_c(
    const DnaBitplaneAlignment *aln,
    const int32_t *taxon_indices,
    int32_t n_sub,
    const EigenGtrParams *params,
    int32_t mode,
    int32_t num_gpus,
    double *out_D
);

#ifdef __cplusplus
}
#endif

#endif /* PHYLO2VEC_EIGEN_DIST_H */
