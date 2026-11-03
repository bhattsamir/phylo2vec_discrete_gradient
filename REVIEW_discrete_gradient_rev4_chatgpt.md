# Fourth discrete-gradient review: changed-pair acceptance

Reviewed 3 November 2026. Sources captured at 08:39:14 CET, above HEAD `838d97da5b3035b0612320b1849093438c795e17`. All captured files still matched at the final source check. [Hashes](audit_discrete_gradient_rev4_chatgpt/source_manifest.json) and [changes since review 3](audit_discrete_gradient_rev4_chatgpt/changes.diff) identify the version.

**The previous capped-profile sign-flip finding is fixed in the host acceptance tests. A separate wave-level issue still prevents a general guarantee of full-tree BME descent.** Individual windows can each pass an exact check against the same frozen surroundings, while their combined update increases BME. This is an interaction between updates, not a failure of the newly added single-window formula.

No implementation files were changed. Tests used isolated source copies and local CPUs. GPU 0 remained occupied, so I did not run native CUDA tests. Both current CUDA benchmark sources compiled successfully with NVCC 12.8 for `sm_86`. The acceptance tests below execute extracted production host code with GPU transfers replaced by host copies and refinement supplied by independently computed full-profile distances. They do **not** validate native GPU refinement execution.

| Item | Status | Measured result |
|---|---|---|
| Earlier cap-64 sign flip | Fixed in host acceptance test | Surrogate −0.25, checked full change +0.2421875. Returns 0 and restores both initial vector and permutation. |
| Earlier cap-128 sign flip | Fixed in host acceptance test | Surrogate −0.25, checked full change +0.234375. Returns 0 and restores both initial vector and permutation. |
| New uncapped DFS clade/outside builders | Pass | 8,208 clade checks and 4,048 outside checks, maximum weight error 0. |
| New skeleton pair-distance calculation | Pass | 224 comparisons covering both implementations, both relabelling modes and random trees through 129 leaves. All integer distances match independent graph traversal. |
| Changed-pair acceptance controls | Pass | 240 checks, 98 accepted and 142 rejected, maximum computed delta error 0 on integer-distance controls. |
| Previous API/concurrency/iteration-boundary regressions | Still pass | 915 unsupported vectors rejected by each backend. Both concurrent threads: 0 errors in 100 calls. 120 forced iteration-cap cases: maximum score/tree discrepancy 4.440892098500626e−15. |
| Current sanitizer suite | Pass | All nine groups pass ASan/UBSan with leak detection. |
| Full-tree descent across a subtree wave | **Not fixed generally** | Controlled replay reports −0.02874351479113102 while full-tree BME actually changes by **+0.050525834783911705**. |
| Native execution of the new CUDA refinement kernels | Not rerun | GPU 0 occupied throughout. Compile checks pass. |

**What the new fix correctly does.**

The capped profiles remain approximations, which is now accurately documented. The change adds a separate acceptance check using uncapped profiles. For fixed super-leaf contents, only pairs whose skeleton distance changes contribute to the objective difference:

\[
\Delta L=\sum_{a<b}\left(2^{1-d_{\rm new}(a,b)}-2^{1-d_{\rm old}(a,b)}\right)\widetilde D_{ab}.
\]

Unchanged pairs cancel. Refining changed pairs involving a capped profile is sufficient to remove that profile approximation, provided the surrounding topology stays fixed. The implementation handles permutations and root suppression correctly in my tests. Ordinary floating-point limitations still apply.

The relevant implementation is [bench_minibatch_recovery.cu:1695](/home/sjbhatt/phylo2vec_discrete_gradient/bench_minibatch_recovery.cu:1695), with rejection/restoration at line 1770, and [bench_rnasim_gpu.cu:1740](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:1740), with rejection/restoration at line 1816. Proper-subtree callers supply complete outside profiles at [bench_rnasim_gpu.cu:1989](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:1989).

I held the original adversarial marked leaf sets fixed, reconstructed the uncapped matrices using current profile builders, and passed them through the extracted production acceptance blocks. Both old failures are rejected. Reverse-move controls return −0.2421875 and −0.234375, demonstrating that the check also permits genuine improvements. Each original case requires refinement of two changed pairs. The earlier 14- and 17-leaf examples still match full BME exactly.

Evidence: [acceptance_checks.json](audit_discrete_gradient_rev4_chatgpt/acceptance_checks.json), [acceptance harness construction](audit_discrete_gradient_rev4_chatgpt/build_acceptance_checks.py), [earlier profile reproductions](audit_discrete_gradient_rev4_chatgpt/profile_checks.json).

**Remaining finding: P1 — frozen outside profiles do not make separate windows independent.**

At [bench_rnasim_gpu.cu:2142](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:2142), the controller computes outside profiles before processing the wave. At line 2152 it snapshots the surrounding topology. Lines 2153–2163 process the windows and add their locally computed changes. Similar processing occurs at lines 2198–2209 and 2251–2263. The total is returned without a joint objective check at line 2271 and printed as the BME change at line 2564.

Disjoint internal node sets prevent competing topology writes. They do not make BME changes additive: changing one subtree can change the leaf weights seen by another subtree. The snapshot makes each outside traversal internally consistent, but also ensures the other window's accepted changes are absent from its acceptance calculation. With one worker, the precomputed outside profiles are still stale for later windows in that wave.

I constructed a reproducible sequence-distance example using 1,596 binary characters, evaluated by the current `dev_jc69_dist_popc` body compiled for the CPU. It uses ten distinct sequence labels. Six labels are each expanded to a balanced group of 256 identical copies, giving 1,540 leaves. Distances within each identical group are zero, so this expansion preserves the relevant ten-label BME scores.

The starting top-level tree is a ladder whose first item is `(A,B)`, followed by four outside leaves. The two subtrees are:

```
A: ((group0, group1), group2) -> ((group0, group2), group1)
B: ((group3, group4), group5) -> ((group3, group5), group4)
```

The actual production scheduler, with its unchanged 1,050-node component limits, puts both 768-leaf subtrees in level 0. Their node IDs are 2306 and 3073. The root connector is in level 1. This is therefore a partition the scheduler actually produces, not an arbitrary incompatible pair of windows.

| Update | Full-tree BME | Change from initial tree |
|---|---:|---:|
| Initial tree | 3.3833344113081694 | 0 |
| A alone | 3.380039880052209 | −0.0032945312559604645 |
| B alone | 3.357885427772999 | −0.025448983535170555 |
| A and B together | 3.433860246092081 | **+0.050525834783911705** |

Both individual changes are well below the −1e−6 acceptance threshold. Their sum is −0.02874351479113102, but the omitted cross-window interaction is +0.07926934957504272. Profile caps and floating-point noise are not responsible: this case uses uncapped profiles, and the discrepancy is much larger than rounding error.

I also executed the **unchanged source wave controller** with one worker and a controlled proposal callback replacing the GPU optimizer. The callback proposes only the two swaps above, computes their exact local changes using the outside profiles actually supplied by the controller, and accepts only improving proposals. Other windows return no change. Both proposals are committed. The controller returns the negative sum above, while independent graph-distance scoring of the complete expanded tree gives the positive change above.

This is a controlled proposal replay, **not a claim that the native GPU optimizer selects these exact proposals in an unconstrained run**. It establishes that the wave's acceptance/composition rule permits worsening updates even when every individual local check is correct. I have not measured how often this occurs in benchmark data. This wave-composition problem predates the current changed-pair fix.

Evidence: [sequence construction and all distances](audit_discrete_gradient_rev4_chatgpt/wave_sequence_case.json), [source scheduler/profile results](audit_discrete_gradient_rev4_chatgpt/wave_checks.json), [controlled controller replay](audit_discrete_gradient_rev4_chatgpt/wave_driver_check.json), [driver extraction](audit_discrete_gradient_rev4_chatgpt/build_wave_driver.py).

To guarantee descent across a wave, either commit windows sequentially and rebuild their outside profiles after each accepted update, or evaluate the combined candidate wave against the initial full objective and reject/roll back a worsening combination. Restricting batches to updates whose attachment profiles are provably unchanged is another possible route, but disjoint internal nodes alone are insufficient. The unconditional guarantee in [paper/phylo2vec_bme_paper.tex:333](/home/sjbhatt/phylo2vec_discrete_gradient/paper/phylo2vec_bme_paper.tex:333) needs this qualification until wave composition is addressed.

Reproduction commands and test scope are in [the audit README](audit_discrete_gradient_rev4_chatgpt/README.md). Native CUDA runtime validation and benchmark frequency remain outstanding.
