# Discrete-gradient fixes: independent recheck

Reviewed 2 November 2026 against the working source captured at 20:10:03 UTC. HEAD remains 838d97da5b3035b0612320b1849093438c795e17. The fixes are uncommitted changes, so [the source manifest](audit_discrete_gradient_followup_chatgpt/source_manifest.json) identifies the exact version tested.

**Nine of the ten original findings are fixed in the rechecks. F2 is partly addressed through clearer documentation, but the representative minibatch calculations remain approximate and the original sign-reversal counterexample is unchanged. Two new source comments also overstate when that approximation is exact.**

This was a read-only review of the implementation. All builds, diagnostic copies and outputs are under [audit_discrete_gradient_followup_chatgpt](audit_discrete_gradient_followup_chatgpt/). The original review and its evidence were preserved. Tests used local CPUs and GPU 0, an RTX A6000. No cluster jobs, commits or pushes were made.

The live sources still matched the captured hashes at 20:14:29 UTC. See [source_recheck.json](audit_discrete_gradient_followup_chatgpt/source_recheck.json).

## Finding-by-finding results

| Original finding | Status | Current result |
|---|---|---|
| F1: general vectors accepted by an invalid lower-gradient path | **Fixed for the supported Python API** | Both backends now reject all 915 non-ordered vectors in the N = 4, 5, 6 enumeration. The 150 ordered vectors are accepted. Maximum error in the general-domain enumeration's accepted subset is 1.887379141862766e-15. The minimal counterexample raises ValueError through C-backed Python, Numba-backed Python and direct Numba. |
| F2: representative condensation can reverse the full-objective gain | **Partly fixed: documentation improved, numerical behavior unchanged** | Full BME change remains +0.029296875 while eight representatives predict -0.03125 and four predict -0.09375. See the detailed section below. |
| F3: competing writes in benchmark GPU kernels | **Fixed** | The transposed write is removed in both copies. Across 4,980 valid selected candidates, repeated-input gain difference is now 0, previously 9.5367431640625e-7. Maximum gain error against independent graph scoring remains 3.7440045161929447e-6. |
| F4: concurrent C workspace corruption | **Fixed in the exercised paths** | Both original 100-call Python threads now have zero mismatches and maximum error 0. Previously they had 10 and 20 bad calls, with errors up to 2.538661660343395. Workspace is thread-local. The expanded eight-thread sanitizer regression also passes. |
| F5: tree/score mismatch at iteration cap | **Fixed in the reproduced exit paths** | Original forced-cap case now returns score 7.842162528665195 and a tree independently scoring exactly 7.842162528665195. Across 120 cases with caps 1, 2, 3, 5, 10 and 20, maximum discrepancy is 4.440892098500626e-15. |
| F6: 32-bit quadratic indexing | **Fixed for the reported expressions and reviewed subdist indexing** | Current promoted expressions return diagonal index 2147580963 at N = 23172 and gradient entry count 2147581953 at N = 32770, without UBSan errors. No unpromoted subdist × n_nodes_u expressions remained in the source scan. These are arithmetic tests, not full large allocations. |
| F7: CUDA top-1 validator accepts invalid outputs | **Fixed** | Separate injections of NaN gains, illegal indices and legal-but-wrong selected columns each produce validator_ok = 0. The unmodified case produces 1. |
| F8: paper's factor-of-two normalization | **Fixed** | The paper now defines returned gradients as ordinary ΔL. The original example still scores 4.25 → 6.5 and returns 2.25, matching the corrected description. |
| F9: component-pass claim of exposing every internal node | **Fixed as a documentation finding** | The comment now promises all original leaves in leaf-bearing components. The original balanced cases still expose 4992/4999 and 8184/8191 internal nodes in that pass, and all 5000/8192 leaves. The narrower claim matches these results. |
| F10: silently truncating non-integer vectors | **Fixed** | [0, 0.9, 1.9] now raises ValueError. Validation requires integer dtype before conversion. This also rejects integer-valued floating arrays, which is a stricter input contract. |

## F2 remains an approximation, with an additional small-clade wording error

The numerical representative and outside-profile algorithms did not change. The new paper remark at [paper/phylo2vec_bme_paper.tex:333](/home/sjbhatt/phylo2vec_discrete_gradient/paper/phylo2vec_bme_paper.tex:333) explicitly distinguishes exact full-profile condensation from constant-width representative approximations. It also describes the outside representation's folding of the remaining tail mass into the third sibling clade. Those clarifications are useful.

However, they do not turn the benchmark method into exact full-tree descent. I rebuilt the host helper from the current source and reran the original additive-distance example:

| Calculation | Before | After | Change |
|---|---:|---:|---:|
| Full tree | 19.9354248046875 | 19.9647216796875 | **+0.029296875** |
| Eight-representative skeleton | 10.99609375 | 10.96484375 | **-0.03125** |
| Four-representative skeleton | 11.0546875 | 10.9609375 | **-0.09375** |

The move still improves the representative objective and worsens the full objective. This is the same 17-leaf tree-metric counterexample described in the original report. The calculation uses independent graph-distance Pauplin scores, not a complete sequence-data optimizer run.

There is also a new incorrect unqualified statement in the source comments:

- [bench_minibatch_recovery.cu:462](/home/sjbhatt/phylo2vec_discrete_gradient/bench_minibatch_recovery.cu:462): exact for clades of size ≤4.
- [bench_rnasim_gpu.cu:213](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:213): exact for clades of size ≤8.

Calling the actual current representative helpers on small ladders gives:

| Clade and representative count | Exact weights, in leaf order | Representative weights |
|---|---|---|
| Four-leaf ladder, R = 4 | (1/8, 1/8, 1/4, 1/2) | (1/4, **0**, 1/4, 1/2) |
| Five-leaf ladder, R = 8 | (1/16, 1/16, 1/8, 1/4, 1/2) | (1/8, **0**, 1/8, 1/4, 1/2) |

Thus a four-leaf clade already breaks the four-representative claim, and a five-leaf clade breaks the eight-representative claim. Maximum weight errors are 0.125 and 0.0625 respectively. Leaf count alone is insufficient. A sufficient condition for this representative construction is that every leaf be at depth at most log2(R) within the clade. Deeper clades can sometimes give the same distances by coincidence, but that is not general exactness.

The paper includes the qualifier “with balanced top splits”, which the source comments omit. An explicit depth condition would be clearer than a leaf-count threshold.

**Decision needed for the method description:** retaining the approximation is a legitimate algorithm choice, provided the benchmark is described as approximate and full-objective monotonicity is not claimed. If the requirement is exact full-tree descent, F2 still needs an algorithmic change, such as exact condensed distances or full-objective verification of accepted proposals. Neither was added in this revision.

Evidence: [current additive results](audit_discrete_gradient_followup_chatgpt/representative_additive_check.json), [small-clade results](audit_discrete_gradient_followup_chatgpt/small_clade_check.json), [small-clade harness](audit_discrete_gradient_followup_chatgpt/small_clade_check.py).

## Source locations and implementation checks

- **F1:** ordered-domain checks are at [discrete_gradient.py:576](/home/sjbhatt/phylo2vec_discrete_gradient/discrete_gradient.py:576), direct Numba at line 408, and C at [phylo2vec_fastme.c:403](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:403). The unsupported general C mapping was removed. The raw void C API zeroes its output and returns on invalid input. It does not provide a distinguishable error status, so the explicit rejection claim above refers to the Python entry points. The original scratch-poisoning C harness now returns zeros, with no leaked sentinel values.
- **F3:** own-column stores are at [bench_rnasim_gpu.cu:317](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:317) and [bench_minibatch_recovery.cu:569](/home/sjbhatt/phylo2vec_discrete_gradient/bench_minibatch_recovery.cu:569). Actual device execution tested the RNASim copy. Inspection confirms the same correction in the recovery copy.
- **F4:** [phylo2vec_fastme.c:40](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:40) makes workspace and capacity thread-local. pthread cleanup and main-thread cleanup were added. The ASan/UBSan suite passed with leak detection enabled on this Linux build. Windows cleanup behavior was not tested.
- **F5:** [phylo2vec_fastme.c:1964](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:1964) marks coordinate edits as unsynchronized. Adjacency-based changes now update curr_L at lines 1971 and 2036. The final reconstruction/recalculation starts at [line 2089](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:2089). The diagnostic test changes only the audit copy's cap and corresponding terminal-hop guard. It does not claim that these small inputs naturally hit the production cap of 500.
- **F6:** output sizes are promoted before multiplication at [phylo2vec_fastme.c:403](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:403) and [line 616](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:616). The reported subdist indices were similarly promoted.
- **F7:** [phylo2vec_cuda_8xh100.cu:2211](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_cuda_8xh100.cu:2211) checks finite gains, legal indices and the selected entry's value. Fault injection was performed immediately before validation in an audit-only source copy.
- **F8:** [paper/phylo2vec_bme_paper.tex:59](/home/sjbhatt/phylo2vec_discrete_gradient/paper/phylo2vec_bme_paper.tex:59), line 78 and line 181 consistently use ΔL. The additional root-level paper source matches the paper-directory source in this snapshot.
- **F9:** [bench_rnasim_gpu.cu:1599](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:1599) now describes a leaf-complete component pass. The measured scheduling behavior is unchanged. This recheck does not prove connectivity of the complete proposal policy or guarantee reachability under greedy acceptance.
- **F10:** [discrete_gradient.py:541](/home/sjbhatt/phylo2vec_discrete_gradient/discrete_gradient.py:541) checks dtype before conversion.

## Regression results and evidence

The expanded CPU verification suite passes all nine groups under ASan/UBSan, including 36,000 full-gradient entries at N = 20 with maximum error 3.553e-14, exhaustive topology/relabel checks, 16,231 exact-condensation comparisons with maximum error 1.564e-13, and the added eight-thread test. The independent ordered-gradient checks also pass.

The separate exact SPR-neighborhood test also passes: 90 P2V and 90 comparison-optimizer outputs have zero reported-score mismatches and zero SPR-nonoptimal outputs.

The current main CUDA full/fallback/tiled implementations pass the original 20-tree GPU test, with maximum FP64-reference discrepancy 5.9620707459906441e-6. That value is unchanged from the first review. The benchmark repetition test now has zero variation across its repeated inputs. These are FP32 numerical checks, not proofs of a uniform error bound for arbitrary tree sizes.

The original ladder scheduling check remains fixed: global K = 1201, five component waves at N = 5000, zero size skips, all 5000 leaves and 4999 internal nodes exposed in the component test.

Key outputs:

- [API, domain, concurrency and normalization](audit_discrete_gradient_followup_chatgpt/followup_api.json)
- [120 iteration-cap cases](audit_discrete_gradient_followup_chatgpt/followup_boundary.json)
- [Validator fault injections and corrected index arithmetic](audit_discrete_gradient_followup_chatgpt/followup_validation.out)
- [Sanitizer regression suite](audit_discrete_gradient_followup_chatgpt/verify_all_asan.out)
- [Exact SPR-neighborhood regression](audit_discrete_gradient_followup_chatgpt/verify_spr_coverage.out)
- [GPU repeated-input check](audit_discrete_gradient_followup_chatgpt/rnasim_gpu_checks.out)
- [Main CUDA device tests](audit_discrete_gradient_followup_chatgpt/cuda_checks.out)
- [Main CUDA host simulation](audit_discrete_gradient_followup_chatgpt/cuda_host_verify.out)
- [Scheduler recheck](audit_discrete_gradient_followup_chatgpt/rnasim_host_checks.out)

Reproduction commands are in [the follow-up audit README](audit_discrete_gradient_followup_chatgpt/README.md). No implementation changes were made during this recheck.
