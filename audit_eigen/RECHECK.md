# Recheck of the eigen-distance fixes (follow-up to HANDOFF.md)

I checked the versions of `phylo2vec_eigen_dist.{c,h}` and `phylo2vec_eigen_dist_cuda.cu` saved at 13:49 on 2026-11-02. I rebuilt both libraries from source and reran every script in this folder, plus two new ones: `edge.py` and `fit.py`.

**Summary:** all six bugs from HANDOFF.md are fixed, and CPU and GPU now give identical results. Thank you. Some work is left: one fix is only approximate, there are two new small issues, and a few low-priority items are still open.

## Results

| Bug | Status | Evidence |
|---|---|---|
| 1. No shared sites | Fixed | Returns NaN in every mode, on CPU and GPU. `fit.py` runs cleanly with an all-gap taxon included. |
| 2. WLS gave 0 for saturated pairs | Fixed | The log-space, max-normalised weights work. Unrelated pairs never give 0 (`ctests.py` line B). |
| 3. Missing base | Fixed, but approximate | No blow-up any more, but see task 1 below. |
| 4. Absurd saturated distances | Fixed | Capped at `EIGEN_D_MAX = 10`, and computed with `expm1`. |
| 5. Mode 0 vs the paper | Documented | The header comment is now accurate. Samir will decide whether to rename the mode. Leave the name as it is for now. |
| 6. GPU asymmetry | Fixed | 0 of 32,640 pairs differ. Upper-triangle tiles only, rows balanced across GPUs, FP64. GPU matches CPU to about 1e-14 (`gputest.py`; the "f32" label in its output is out of date). |

Also fixed:
- the prior comment
- relative gradient tolerance and Shanno–Phua scaling in BFGS
- a default of 50,000 sampled pairs on both CPU and GPU
- FASTA checks for unequal lengths and non-ASCII bytes
- padding for the shared-memory bank conflict
- error checks after the kernel launches

## Remaining tasks

1. **Bias in the reduced-alphabet path (bug 3).** `compute_reduced_subspace_sigmas` (C) and `dev_compute_reduced_subspace_sigmas` (`cu:256`) avoid the blow-up, but they use heuristics:
   - mass in the missing states is moved onto the off-diagonals
   - sigma is rescaled with `n_act*s/(4-(4-n_act)*s)` or `1 - scale*(1-s)`
   - modes that are missing are filled with the mean sigma
   - ξ is set to 0.25 for *absent* states (`cu:466-479`), which changes Q.

   **Measured bias** (`edge.py`, 10% diverged pair, true D ≈ 0.102):

   | Case | D | Bias |
   |---|---|---|
   | T missing in one sequence | 0.1055 | +3% |
   | T missing in both sequences | 0.0950 | −7% |
   | Only A and C in both | 0.121 | +18% |

   **Better approach:** do the exact computation on the C_act states.
   - Run the deflated SVD on the C_act × C_act block. It has C_act − 1 non-trivial modes.
   - Build Q restricted to the active states and use the matching trace: sum only over k = 1..C_act−1, with −tr(Q_act).
   - Do not rescale the sigmas, and do not set ξ for absent states.

   If that turns out to be awkward, document the current behaviour as an approximation in the header and give its size of bias.

2. **New: a single-state pair returns the cap instead of NaN.** All-A against 1%-mutated all-A returns D = 10 (`edge.py`, pair `polyA-polyA2`). When `n_act < 2`, the reduced path is skipped and the pair is treated as saturated. When `n_act < 2`, return NaN (not estimable), the same as for no overlap.

3. **New, needs investigation: α is probably prior-dominated.** `fit.py` simulates data with no rate variation, so the true α is infinite. Both CPU and GPU return α ≈ 0.35. My guess is that pairwise data barely identify α once each pair's D is fitted, so the HalfNormal(1) prior dominates.

   Please check this:
   - Profile the log-posterior over log α with S fixed, with and without the prior. Is the data term nearly flat?
   - Rerun on data simulated with a known finite α, such as 0.5 or 2.

   Report what you find to Samir before changing the prior or the objective.

4. **Small:**
   - Identical sequences give D ≈ 4e-11 rather than exactly 0, because the 1e-8 pseudocount pulls sigma just below 1. Return 0 when K has no off-diagonal counts.
   - The static per-GPU buffers `g_d_bp` / `g_d_D_rows` are neither thread-safe nor ever freed. Add a cleanup function, or document that the library must not be called from more than one thread at once.
   - The `CUDA_CHECK_RET` early returns in `fit_eigen_gtr_map_gpu_c` still leak host and device memory.

## How to rerun

Run from this folder. `edge.py`, `fit.py` and `gputest.py` load `./libeig_gpu.so`.

```bash
cd audit_eigen
PY=~/anaconda3/envs/rcpms-jax/bin/python
SRC=..
gcc -O2 -fopenmp -shared -fPIC -I$SRC $SRC/phylo2vec_eigen_dist.c -o libeig.so -lm
nvcc -O2 -arch=sm_86 -Xcompiler -fopenmp,-fPIC -shared -I$SRC \
     $SRC/phylo2vec_eigen_dist_cuda.cu $SRC/phylo2vec_eigen_dist.c -o libeig_gpu.so -lgomp
$PY ctests.py                            # bugs 1-4: expect nan / 10 / ~0.08
CUDA_VISIBLE_DEVICES=1 $PY gputest.py    # expect 0 asymmetric pairs, CPU == GPU
CUDA_VISIBLE_DEVICES=1 $PY edge.py       # tasks 1 and 2: compare missing-state rows with a-b
CUDA_VISIBLE_DEVICES=1 $PY fit.py        # task 3: fitted alpha (the data have no rate variation)
```

**Done when:**
- In `edge.py`, every missing-state case is within about 3% of `a-b`, or the bias is documented, and `polyA-polyA2` gives `nan`.
- Task 3 has a short written finding.
