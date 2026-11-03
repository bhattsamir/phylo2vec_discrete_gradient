# Fifth discrete-gradient review: live outside profiles

Reviewed 3 November 2026 against the source captured at 10:03:23 CET, above HEAD `838d97da5b3035b0612320b1849093438c795e17`. [Source hashes](audit_discrete_gradient_rev5_chatgpt/source_manifest.json) and [changes since review 4](audit_discrete_gradient_rev5_chatgpt/changes.diff) identify the tested version.

A further CUDA batching optimization landed during this review. I captured it at 10:08:19 CET in [the late source snapshot](audit_discrete_gradient_rev5_chatgpt/late_snapshot/bench_rnasim_gpu.cu), compiled it and tested its added arithmetic path separately. The wave controller, window caller, outside builder, summary updater and host acceptance function are byte-identical to the initially tested versions. [Final source check](audit_discrete_gradient_rev5_chatgpt/source_recheck.json) confirms no further changes. Source links below identify the captured versions.

**The frozen-outside finding is fixed in my controlled replay of the actual wave controller. Both processing orders now reject the conflicting second proposal, and the reported BME change exactly matches independent scoring of the complete tree. I found no new correctness failure in the checks run here.**

The earlier capped-profile acceptance checks still pass. Native CUDA execution remains untested in this round because both GPUs were occupied. All execution was on local CPUs, and both current CUDA benchmark sources compiled successfully for `sm_86`. No implementation files were edited.

| Item | Result | Evidence |
|---|---|---|
| Frozen-outside wave composition | **Fixed in controlled controller replay** | Six cases: both window orders, each with configured GPU counts 1, 2 and 8. Exactly one proposal accepted per case. Maximum reported-versus-actual BME change discrepancy 0. |
| Capped-profile sign-flip acceptance | **Still fixed in CPU host tests** | Both old worsening proposals rejected. Checked changes +0.2421875 and +0.234375, returned change 0, initial vector and permutation restored. |
| Changed-pair acceptance controls | Pass | 240 checks, 98 accepted and 142 rejected, maximum delta error 0 on the integer-distance controls. |
| Uncapped profile builders | Pass | 8,208 clade checks and 4,048 outside checks, maximum weight error 0. |
| Skeleton-distance calculation | Pass | 224 comparisons with independent graph traversal, all exact. |
| Updated refinement kernel arithmetic | Pass in CPU emulation | 60 cases, maximum discrepancy 3.552713678800501e−15 against an independent pair-sum loop. Native GPU execution not tested. |
| Earlier API, concurrency and iteration-limit regressions | Pass | Unsupported vectors rejected, concurrent-call errors 0, 120 iteration-limit cases with maximum score/tree discrepancy 4.440892098500626e−15. |
| ASan/UBSan suite | Pass | All nine groups, including the added wave example, pass with leak detection. |
| Late shared-input batching branch | Pass in CPU emulation | 50 cases, 610 batched outputs, maximum pair-sum discrepancy 1.687538997430238e−14. |
| CUDA compilation | Pass | Both complete benchmark sources, including the late update, compile with NVCC 12.8 for `sm_86`. |

**What changed and why it fixes the counterexample.**

At [bench_rnasim_gpu.cu:2195](/home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_rev5_chatgpt/snapshot/bench_rnasim_gpu.cu:2195), each component window now rebuilds its outside profile from the current tree immediately before optimization. An accepted change is followed by propagation of ancestor summaries at line 2212. The old wave-wide outside snapshots and parallel commits have been removed.

The same sequence appears in the staggered component pass at [bench_rnasim_gpu.cu:2234](/home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_rev5_chatgpt/snapshot/bench_rnasim_gpu.cu:2234) and the medium-window pass at [bench_rnasim_gpu.cu:2288](/home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_rev5_chatgpt/snapshot/bench_rnasim_gpu.cu:2288). Those callers no longer pass frozen topology arrays into the exact outside-profile builder. Consequently, the second window's acceptance calculation includes the first window's accepted change.

I reran the same 1,540-leaf sequence-distance construction from review 4. The source scheduler still puts the two 768-leaf subtrees in the same initial wave. The test executes the current source controller with a controlled proposal callback that evaluates the two prescribed swaps against the outside profiles actually supplied by the controller. It independently scores the entire expanded tree before and after. This is not an unconstrained native GPU optimizer run.

| Processing order | First proposal | Recomputed second proposal | Accepted total change | Actual full-tree change |
|---|---:|---:|---:|---:|
| B then A, source's original order | −0.025448983535170555 | **+0.07597481831908226, rejected** | −0.025448983535170555 | −0.025448983535170555 |
| A then B, reversed sibling traversal | −0.0032945312559604645 | **+0.05382036603987217, rejected** | −0.0032945312559604645 | −0.0032945312559604645 |

The initial BME is 3.3833344113081694. The final scores are respectively 3.357885427772999 and 3.380039880052209. Previously both proposals were accepted against stale surroundings, resulting in an actual increase of +0.050525834783911705. That failure no longer occurs in either order.

Configured GPU counts 1, 2 and 8 give identical results for a given order because the new controller uses workspace 0 for every window. These are CPU controller configurations, not runs on those numbers of physical GPUs.

Evidence: [wave_fix_checks.json](audit_discrete_gradient_rev5_chatgpt/wave_fix_checks.json), [test](audit_discrete_gradient_rev5_chatgpt/wave_fix_checks.py), [controller extraction and controlled callback](audit_discrete_gradient_rev5_chatgpt/build_wave_driver.py). The original test's assertion that two proposals must be accepted now fails, as expected. The updated regression asserts that exactly one is accepted and that the reported and actual changes agree.

**Other changes checked.**

The recovery benchmark's additional minibatch caller now constructs and supplies uncapped profiles when needed, then skips a rejected update at [bench_minibatch_recovery.cu:2020](/home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_rev5_chatgpt/snapshot/bench_minibatch_recovery.cu:2020). Both capped minibatch call sites now pass the exact-profile arguments. The earlier CPU host acceptance tests reject the two original capped-profile failures and accept their genuinely improving reverse moves.

The sequence refinement kernel adds a singleton-profile fast path at [bench_rnasim_gpu.cu:282](/home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_rev5_chatgpt/snapshot/bench_rnasim_gpu.cu:282). I checked its accumulation using CPU emulation of the current kernel body and its barrier-separated reduction. Cases cover either operand being the singleton, profile sizes crossing the 256-thread boundary through 2,048 entries, non-singleton pairs, missing observations, and zero weights. All 60 cases agree with a separate full pair-sum loop to at most 3.552713678800501e−15. This verifies the arithmetic under the emulation, not native CUDA synchronization, code generation or runtime behavior. [Results](audit_discrete_gradient_rev5_chatgpt/refinement_cpu_check.json).

The late kernel revision additionally groups up to 16 singleton-versus-profile pairs sharing an operand, at [the captured kernel, line 271](/home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_rev5_chatgpt/late_snapshot/bench_rnasim_gpu.cu:271). I separately emulated this branch with its cooperative loads and reduction barriers represented as serial phases. Tests cover partial groups, 16-pair boundaries, interrupted common-operand runs, non-singleton interruptions, missing observations and zero weights. All 50 cases select the expected outputs. The 610 computed outputs differ from scalar pair sums by at most 1.687538997430238e−14. The fallback body is unchanged from the earlier 60-case check. [Late branch results](audit_discrete_gradient_rev5_chatgpt/batch_refinement_cpu_check.json). These are CPU arithmetic/control-flow tests, not native GPU synchronization tests.

Both duplicated SPR kernels now write an explicit invalid-result sentinel before their no-destination early return. This avoids retaining the preceding call's output for that row. I inspected this change and compiled both CUDA sources, but did not exercise that branch on a GPU.

**Remaining limitations and performance consequence.**

No unresolved correctness finding from review 4 remains in the tested controller and host acceptance paths. Native execution of the acceptance/refinement kernels and an end-to-end GPU optimizer regression remain outstanding because GPU 0 was busy throughout. CPU emulation and compile checks do not replace those tests.

The correctness fix deliberately serializes the three window loops and sends all their work to `d_packed_seqs[0]` / `mb_ws[0]`, for example at [bench_rnasim_gpu.cu:2207](/home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_rev5_chatgpt/snapshot/bench_rnasim_gpu.cu:2207). Additional GPUs no longer parallelize these windows. The old multi-GPU timings and parallel-wave description therefore need remeasurement or qualification for this implementation. I did not benchmark the speed cost.

Profiles above their caps remain approximate proposal mechanisms, with a full-profile acceptance check. The fix does not make those proposal gradients exact or guarantee finding the globally optimal tree.

Reproduction commands and test boundaries are in [the audit README](audit_discrete_gradient_rev5_chatgpt/README.md).
