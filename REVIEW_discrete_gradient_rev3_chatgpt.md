# Third discrete-gradient review: weighted profiles

Reviewed 2 November 2026 against the source captured at 20:51:28 UTC. The fixes are working-tree changes above HEAD 838d97da5b3035b0612320b1849093438c795e17. [Source hashes](audit_discrete_gradient_rev3_chatgpt/source_manifest.json) and [the diff from the preceding review](audit_discrete_gradient_rev3_chatgpt/changes.diff) identify the tested version.

**The original small counterexamples are now fixed. The remaining exactness issue is only partly fixed: the production profile caps still permit an apparent improvement that worsens full-tree BME. I reproduced this with the new profile construction and with the actual recovery CUDA distance kernel.**

No implementation files were edited. Tests and builds used audit copies, local CPUs and GPU 0 only. No cluster jobs, commits or pushes were made.

At the final source check at 20:57:48 UTC, all tested implementation files still matched the snapshot. Both paper copies had changed during the review. I captured those updates under [paper_at_finish](audit_discrete_gradient_rev3_chatgpt/paper_at_finish/). The updated profile paragraph strengthens the ladder-tail expression to 2^(-(M_prof-1)), but does not address the balanced-clade counterexample below. Updated benchmark numbers were not independently verified here.

## Results

| Item | Status | Evidence |
|---|---|---|
| Original 14-leaf counterexample | **Fixed** | Both current profile implementations give +0.0146484375, exactly matching full-tree BME. Previously the four/eight-slot approximations gave negative changes. |
| Original 17-leaf additive-distance counterexample | **Fixed** | Both current profiles give +0.029296875 with zero discrepancy. The actual recovery CUDA kernel also gives +0.029296875. |
| Four- and five-leaf unbalanced-clade counterexamples | **Fixed** | Every leaf is retained with its correct dyadic weight. Maximum weight discrepancy is 0. Nine-leaf ladders also pass. |
| Exact clade profiles within the tested sizes | **Pass** | 3,072 comparisons against independently computed weights, maximum error 0. |
| Full outside traversal with no profile cap | **Fixed in the tested cases** | 1,000 comparisons with an independent traversal of the unrooted graph, maximum weight error 0. This includes suppression of the degree-2 root. |
| Exact full-tree descent with the production profile caps | **Not fixed generally** | A balanced-clade example gives predicted change -0.25 but true change +0.2421875 with cap 64, or +0.234375 with cap 128. |
| Prior API, concurrency and iteration-cap fixes | **Still pass** | Domain guards reject the 915 unsupported vectors. Both concurrent threads have zero errors. All 120 forced iteration-cap cases pass, maximum tree/score discrepancy 4.440892098500626e-15. |
| Current CPU sanitizer suite | **Pass** | All nine groups pass under ASan/UBSan with leak detection enabled, including the newly added condensation case. |

The small-clade changes are substantive numerical fixes, rather than only documentation changes. The weighted GPU accumulation also computes its supplied profiles correctly in the tests below.

## Remaining issue: a cap can collapse substantial mass in a balanced clade

The new builders expand a priority queue until the profile capacity is reached. They then assign each remaining internal subtree's entire mass to one representative leaf:

- [bench_rnasim_gpu.cu:866](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:866) stops expansion. Lines 884–888 collapse the queued masses onto representatives.
- [bench_minibatch_recovery.cu:1134](/home/sjbhatt/phylo2vec_discrete_gradient/bench_minibatch_recovery.cu:1134) and lines 1152–1156 implement the same rule.
- The sequence benchmark uses no cap for N ≤ 2048 and cap 64 above that size, at [bench_rnasim_gpu.cu:1558](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:1558) and line 1726.
- The recovery benchmark sets cap 128 at [bench_minibatch_recovery.cu:1017](/home/sjbhatt/phylo2vec_discrete_gradient/bench_minibatch_recovery.cu:1017).

This is exact when every leaf is expanded, up to floating-point arithmetic. Above the cap, the stored weight is a remaining subtree's mass, not necessarily the true weight of the representative leaf receiving it.

On a ladder, repeated expansion makes the remaining mass small quickly. A balanced tree behaves differently. In a balanced 4096-leaf clade, a cap of 64 leaves 64 internal frontier subtrees, each with weight 1/64. The code assigns all that mass to 64 selected leaves. Each selected leaf's true weight is only 1/4096. Both profiles sum to 1, but their total-variation distance is 0.984375. With cap 128, it is 0.96875. Unit total mass does not imply an accurate distribution over leaves.

### Reproducer and measured BME error

Use 4102 leaves. Let A be a balanced clade on leaves 0–4095, B = 4096, C = 4097, and G the balanced four-leaf clade on 4098–4101. Compare:

    before = ((A,B),(C,G))
    after  = ((A,C),(B,G))

Condense A only, leaving B, C and G's four leaves explicit. This gives a seven-leaf skeleton.

Let P be the leaves selected by the actual capped profile builder for A. Define S as P together with C and leaves 4098 and 4099. Set every diagonal distance to 0 and, for distinct leaves:

    D[i,j] = 2 + indicator(exactly one of i,j belongs to S)

This is an additive tree metric with unit pendant lengths and one length-1 internal split S versus its complement. Remaining internal lengths are zero. The metric is chosen adversarially to expose the approximation. No claim is made that this particular case occurs in the supplied benchmark dataset.

| Profile cap | True fraction of A's weight in S | Capped fraction | True full-tree BME change | Capped change | Absolute error |
|---:|---:|---:|---:|---:|---:|
| 64 | 1/64 | 1 | **+0.2421875** | **-0.25** | 0.4921875 |
| 128 | 1/32 | 1 | **+0.234375** | **-0.25** | 0.484375 |

For this construction, the change is one half of the difference between G's marked fraction and A's marked fraction. G's fraction is 1/2. I also evaluated the full tree independently using a Pauplin recursion for the split metric. That recursion was checked against explicit graph-distance scoring on the small examples.

The cap-128 case was run through the actual current k_build_true_condensed_D on GPU 0, using its HLD distance oracle populated with this tree metric. Its condensed matrix matched an independent weighted sum exactly, and scoring the resulting matrix gave **-0.25**. Thus the discrepancy is caused by profile truncation, not the CUDA accumulation or FP32 rounding.

This tests a supplied clade/skeleton through the actual profile and distance-kernel path. It is not an end-to-end optimizer run, nor evidence that the scheduler necessarily selects this partition on the constructed starting tree. It disproves general exactness of the capped calculation.

Evidence: [profile_checks.py](audit_discrete_gradient_rev3_chatgpt/profile_checks.py), [profile results](audit_discrete_gradient_rev3_chatgpt/profile_checks.json), [recovery GPU harness](audit_discrete_gradient_rev3_chatgpt/recovery_profile_gpu.cu), [GPU output](audit_discrete_gradient_rev3_chatgpt/recovery_profile_gpu.out).

## GPU accumulation and outside-message checks

I compiled the actual current benchmark sources under renamed main functions and called their kernels from independent harnesses.

- Recovery distance kernel, original 17-leaf example: maximum condensed-matrix discrepancy 0, correct objective change +0.029296875.
- Recovery distance kernel, 4102-leaf capped example: maximum discrepancy from the supplied weighted profile 0, objective change -0.25 versus full-tree +0.234375.
- Sequence distance kernel, 17 leaves and uncapped profiles: maximum discrepancy from independently evaluated weighted CPU Jukes–Cantor distances 6.5380246994095614e-8.
- Sequence distance kernel, 4102 leaves and cap 64: maximum discrepancy from that same weighted-profile reference 1.5821299026441693e-7. This checks accumulation of the supplied approximate profile, not exactness against every original leaf.

The independent outside check builds the undirected tree, suppresses its degree-2 root, removes the edge above the chosen clade and recursively halves weights at each remaining internal split. All 1000 uncapped profiles matched. Capped outside profiles still collapse unresolved directed components onto representative leaves at [bench_rnasim_gpu.cu:950](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:950), so those carry the same general approximation qualification.

The previous numerical fixes unrelated to condensation were largely unchanged. The CPU minibatch function additionally copies its input vector before expansion at [phylo2vec_fastme.c:2545](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:2545). The current ASan/UBSan suite and independent API/iteration-boundary tests pass. I did not repeat large benchmark timings, native Metal execution or multi-GPU communication tests.

## What remains to resolve

If the intended method is an approximate large-tree search, the new weighted profiles are a real improvement. The description should explicitly state that profiles above the cap remain approximate, and that monotonicity of the condensed objective is not a guarantee for the full objective. The exact-monotonicity wording at [bench_minibatch_recovery.cu:24](/home/sjbhatt/phylo2vec_discrete_gradient/bench_minibatch_recovery.cu:24) needs that scope qualification. The ladder-tail argument in [paper/phylo2vec_bme_paper.tex:333](/home/sjbhatt/phylo2vec_discrete_gradient/paper/phylo2vec_bme_paper.tex:333) does not bound the balanced-clade case above.

If exact full-tree descent is required, a finite fixed profile cap with unconditional representative collapse does not provide it. It needs exact profiles, a full-objective acceptance check, or a valid approximation-error bound that triggers refinement/rejection when it cannot establish the sign. Merely increasing the cap moves the counterexample to a larger clade.

## Reproduction

All work is under [audit_discrete_gradient_rev3_chatgpt](audit_discrete_gradient_rev3_chatgpt/). The primary commands are in its [README](audit_discrete_gradient_rev3_chatgpt/README.md). Key additional outputs are [the sanitizer suite](audit_discrete_gradient_rev3_chatgpt/verify_all_asan.out), [API checks](audit_discrete_gradient_rev3_chatgpt/followup_api.out), [iteration-cap checks](audit_discrete_gradient_rev3_chatgpt/followup_boundary.out), and [sequence GPU checks](audit_discrete_gradient_rev3_chatgpt/rna_profile_gpu.out).
