# Discrete-gradient code review

Reviewed 2 November 2026. No implementation or existing paper files were changed.

The ordered-vector CPU gradients, relabelling and exact CPU clade condensation pass the reference tests. The main CUDA gradient implementations also pass the tests run here, within FP32 error. However, there are confirmed problems in the general-vector API, representative-leaf minibatches, duplicated benchmark kernels, concurrent CPU calls and termination at the iteration cap. The previously discussed 5,000-leaf ladder scheduling failure is fixed in the version reviewed.

## Scope and reproducibility

This review covers the discrete-gradient and optimizer paths in phylo2vec_fastme.c, discrete_gradient.py, phylo2vec_cuda_8xh100.cu, bench_parallel_gpu.m, bench_minibatch_recovery.cu and bench_rnasim_gpu.cu, together with their tests, benchmark callers and the mathematical claims in paper/phylo2vec_bme_paper.tex. The eigen-distance files supply input distances. This is not a new audit of their statistical distance estimator.

HEAD was 838d97da5b3035b0612320b1849093438c795e17, with pre-existing uncommitted changes. Tests therefore use a captured source snapshot, not an assertion that HEAD alone reproduces the review. Capture time was 2026-11-02 17:35:36 UTC. All captured source hashes still matched the working files at 17:52:05 UTC.

Source hashes and the initial working-tree status are in [source_manifest.json](audit_discrete_gradient_chatgpt/source_manifest.json) and [initial_git_status.txt](audit_discrete_gradient_chatgpt/initial_git_status.txt). All harnesses, diagnostic copies, binaries and outputs are under [audit_discrete_gradient_chatgpt](audit_discrete_gradient_chatgpt/). Line references below refer to that source snapshot, which matched the working files when checked.

Tests used local CPUs and GPU 0, an RTX A6000. GPU 1 was left alone. No cluster jobs or commits were made. Native Metal execution and communication between multiple GPUs were not tested. Most diagnostic builds use GCC/G++ -O2 or NVCC -O2, rather than the production Makefile's fast-math settings. Passing results below are numerical checks, not a promise of bitwise agreement across compilers or devices.

## Findings

P1 means a wrong objective or gradient in an accepted execution path. P2 means a conditional correctness issue, reproducibility defect or important validation gap. P3 means a smaller input-validation or documentation issue.

### F1 — P1: the lower-triangular API accepts general vectors but does not compute their coordinate differences correctly

Locations: [discrete_gradient.py:564](/home/sjbhatt/phylo2vec_discrete_gradient/discrete_gradient.py:564), [phylo2vec_fastme.c:409](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:409), [phylo2vec_fastme.c:607](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:607), and [phylo2vec_fastme.c:637](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:637).

calculate_discrete_gradient validates against the general domain, 0 ≤ v[i] ≤ 2i. Its general-vector destination mapping does not give the correct literal coordinate edits. Both the C and Numba backends fail. The C path can additionally read scratch values that were not written for the requested destination.

A four-leaf reproducer is v = [0, 0, 3], with zero diagonal, off-diagonal distances 1, except D[0,2] = D[2,0] = 10. Set v[2] to 0:

| Quantity | Value |
|---|---:|
| Independently scored BME change | -2.25 |
| Numba returned change | 0 |
| C returned change after explicitly seeding scratch with 1234567 | 1234567 |

The scratch-seeding harness calls the unchanged function. It demonstrates dependence on unwritten workspace, rather than relying on a particular malloc result. Ordinary calls also fail without seeding. Across all 1,065 general vectors at N = 4, 5, 6, the recorded maximum C error was 3.3535446739677193 and the maximum Numba error was 1.2335204667854907. For example, at v = [0,2,4,5], edit (2,0), truth was -1.2335204667854907, C returned 2.1200242071822286 and Numba returned 0. C's exact bad number can vary with call history.

By contrast, the corresponding 150 ordered vectors passed in both backends, with maximum error 1.7763568394002505e-15.

Recommended correction: reject non-ordered input for this API until its general-coordinate semantics are implemented and tested. Merely relabelling the tree and returning the resulting matrix would change what the caller's row and column indices mean. The full-gradient Python API already restricts its input to ordered vectors.

Evidence: [api_checks.py](audit_discrete_gradient_chatgpt/api_checks.py), [api_checks.json](audit_discrete_gradient_chatgpt/api_checks.json), [minimal_general_check.py](audit_discrete_gradient_chatgpt/minimal_general_check.py), [general_poison.c](audit_discrete_gradient_chatgpt/general_poison.c), [general_poison.out](audit_discrete_gradient_chatgpt/general_poison.out).

### F2 — P1 for exactness claims: representative-leaf minibatches can reverse the sign of the full-tree BME change

Locations: [bench_rnasim_gpu.cu:212](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:212), [bench_rnasim_gpu.cu:841](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:841), [bench_minibatch_recovery.cu:462](/home/sjbhatt/phylo2vec_discrete_gradient/bench_minibatch_recovery.cu:462), [bench_minibatch_recovery.cu:1110](/home/sjbhatt/phylo2vec_discrete_gradient/bench_minibatch_recovery.cu:1110).

The exact condensation theorem requires each clade's full dyadic leaf-weight profile. The benchmark implementations instead retain four or eight representative leaves and average their pairwise distances uniformly. These are approximate profiles. An exact gradient of that condensed matrix need not be the full tree's gradient.

I constructed a 17-leaf additive-distance example. Let A be the left-associated ladder on leaves 0 through 8, and G the left-associated ladder on leaves 11 through 16. Compare:

    before = ((A,9),(10,G))
    after  = ((A,10),(9,G))

Use the tree metric with unit pendant edges and one internal edge of length 1 separating {0,10,13} from the other leaves. All other internal edges have length zero. Equivalently, D[i,j] is 2 plus the indicator that i and j lie on opposite sides of this split, for i ≠ j. Small positive resolutions of the zero-length edges preserve the strict signs below.

Condense only A, leaving the other eight leaves explicit. Its true weight on leaf 0 is 1/256. The current eight-representative rule returns [0,6,7,7,8,8,8,8], giving leaf 0 weight 1/8. The four-representative rule gives [0,7,8,8], hence weight 1/4.

| Objective | Before | After | Change |
|---|---:|---:|---:|
| Full-tree BME | 19.9354248046875 | 19.9647216796875 | **+0.029296875** |
| Eight-representative skeleton | 10.99609375 | 10.96484375 | **-0.03125** |
| Four-representative skeleton | 11.0546875 | 10.9609375 | **-0.09375** |

The absolute skeleton scores omit within-clade constants, so the changes are the relevant comparison. The move looks improving in both representative objectives and worsens true BME. A separate 14-leaf metric example gives true change +0.0146484375 versus -0.015625 and -0.046875.

The actual RNASim host representative helper was called to verify the eight-leaf profile. The scores use an independent graph-distance Pauplin calculation. These are explicit distance-matrix counterexamples to exact condensation. I did not claim to observe this particular wrong move in the supplied sequence dataset or run a complete large benchmark.

There is a related outside-profile approximation at [bench_rnasim_gpu.cu:1391](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:1391): compute_outgroup_rep8 visits only the first three sibling clades above a subtree, assigning representative masses 1/2, 1/4, 1/4. For deeper attachment paths, that last quarter is not the entire remaining outside component. This also needs qualification if minibatch differences are described as exact full-tree differences.

The exact CPU implementation at [phylo2vec_fastme.c:2280](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:2280) does accumulate the full dyadic weights. Its 16,231-entry regression passed, with maximum error 1.564e-13. That pass does not validate the representative approximation.

Recommended correction: either use exact profiles/distances or explicitly describe these benchmark paths as approximate search. If full-objective descent is required, verify proposed changes against the full objective before committing them.

Evidence: [representative_check.cpp](audit_discrete_gradient_chatgpt/representative_check.cpp), [representative_check.py](audit_discrete_gradient_chatgpt/representative_check.py), [representative_additive_check.py](audit_discrete_gradient_chatgpt/representative_additive_check.py), [additive results](audit_discrete_gradient_chatgpt/representative_additive_check.json).

### F3 — P2: the duplicated benchmark internal-distance kernels contain competing writes

Locations: [bench_rnasim_gpu.cu:306](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:306) and [bench_minibatch_recovery.cu:559](/home/sjbhatt/phylo2vec_discrete_gradient/bench_minibatch_recovery.cu:559).

Each thread owns column v, but also writes the transposed entry, subdist[v*M+u]. Another thread computes and writes that same entry while progressing through its own recurrence. Symmetry in exact arithmetic does not make these concurrent floating-point writes safe.

On GPU 0, I tested 32 trees at N = 6, 17, 64, 129, with three identical-input repetitions. Across 4,980 valid selected SPR candidates:

| Measurement | Original kernel | Diagnostic own-column-only kernel |
|---|---:|---:|
| Maximum repeated-input gain difference | 9.5367431640625e-7 | 0 |
| Maximum gain error against graph BME | 3.7440045161929447e-6 | 3.7440045161929447e-6 |
| Candidates exceeding the test's 1e-4 tolerance | 0 | 0 |

The maximum original BME-length error was 1.1487063602544367e-6. Thus the measured problem is reproducibility and possible near-tie/threshold sensitivity, not a demonstrated large-error failure in these cases. The actual GPU execution was of the RNASim copy. The recovery benchmark has the same competing-write statements, identified by source inspection.

The main CUDA implementation at [phylo2vec_cuda_8xh100.cu:159](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_cuda_8xh100.cu:159) already writes only the owned column. Use that ownership rule consistently. The replacement used for diagnosis exists only in the audit harness.

Evidence: [rnasim_checks.cu](audit_discrete_gradient_chatgpt/rnasim_checks.cu), [original output](audit_discrete_gradient_chatgpt/rnasim_gpu_checks.out), [diagnostic output](audit_discrete_gradient_chatgpt/rnasim_gpu_safe_checks.out).

### F4 — P2: concurrent Python calls share and corrupt the C workspace

Locations: [phylo2vec_fastme.c:26](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:26), [discrete_gradient.py:475](/home/sjbhatt/phylo2vec_discrete_gradient/discrete_gradient.py:475), [discrete_gradient.py:590](/home/sjbhatt/phylo2vec_discrete_gradient/discrete_gradient.py:590).

g_iwork, g_dwork and their capacity are process-global. The ctypes CDLL calls do not serialize access to them. Two Python threads, using the same N = 64 topology and two different immutable distance matrices, produced incorrect gradients compared with their respective serial references:

| Thread | Bad calls / 100 | Maximum absolute error |
|---|---:|---:|
| 0 | 10 | 1.4639611187374728 |
| 1 | 20 | 2.538661660343395 |

This is a demonstrated concurrent-call issue. Serial calls passed. Different Python DiscreteGradient objects do not create separate C globals. Concurrent resizing could be more dangerous because ensure_workspace frees the shared buffers, but I did not attempt that additional failure.

Recommended correction: per-call/context-owned or thread-local workspace, or a shared lock covering all entry points that use it. Document the concurrency contract. Separate processes have separate globals.

Evidence: the final test in [api_checks.py](audit_discrete_gradient_chatgpt/api_checks.py), with recorded results in [api_checks.json](audit_discrete_gradient_chatgpt/api_checks.json).

### F5 — P2: reaching the optimizer's iteration cap can return a mismatched tree and score

Locations: [phylo2vec_fastme.c:1654](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:1654), [phylo2vec_fastme.c:2022](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:2022), [phylo2vec_fastme.c:2143](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:2143).

An accepted coordinate move changes v_pi and curr_L, then continues. The adjacency is rebuilt on the next iteration. If the loop terminates at its fixed 500-iteration cap, final output encoding uses the previous adjacency while returning the updated score. Other exits that apply an adjacency move defer the score recomputation instead. Termination needs to synchronize both representations and the objective.

I reproduced the coordinate-path failure with a diagnostic copy changing only this loop's limit from 500 to 1. On a 12-leaf input:

| Quantity | BME |
|---|---:|
| Initial tree | 8.997850513353793 |
| Returned score | 7.842162528665195 |
| Independently evaluated returned tree | 8.997850513353791 |

The mismatch is approximately 1.155687984688596. This is a forced-cap reproducer of the exit path. I did not find or claim a natural input that reaches iteration 500. The ordinary optimizer checks passed.

Recommended correction: perform a final synchronization/recomputation on cap exhaustion, and return an explicit termination reason. Add a configurable small-cap regression.

Evidence: [boundary_checks.py](audit_discrete_gradient_chatgpt/boundary_checks.py), [boundary_checks.json](audit_discrete_gradient_chatgpt/boundary_checks.json).

### F6 — P2: 32-bit arithmetic remains in large matrix indexing

Locations: [phylo2vec_fastme.c:31](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:31), [phylo2vec_fastme.c:553](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:553), [phylo2vec_fastme.c:830](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:830), [phylo2vec_fastme.c:841](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:841).

ensure_workspace admits up to 100,000 leaves and allocates with size_t, but several subdist indices still multiply int32_t operands. At N = 23,172, the last diagonal index should be 2,147,580,963. The isolated source expression triggers UBSan signed-overflow diagnostics and evaluates to -2,147,386,333 in the tested build.

The full-gradient memset casts only after multiplying n_pairs * max_cols. At N = 32,770, the intended 2,147,581,953 entries become 18,446,744,071,562,166,273 after the overflowing signed result is converted to size_t in this build. Signed overflow is undefined, so those wrapped numbers are observations, not portable behavior.

These are arithmetic reproducers, not full allocations at those sizes. The existing N = 1024–1026 boundary tests cannot expose the issue. Correct every quadratic index/count by promoting an operand before multiplication, or reject sizes that the implementation cannot safely handle.

Evidence: [validation_checks.py](audit_discrete_gradient_chatgpt/validation_checks.py), [index_expression_check.c](audit_discrete_gradient_chatgpt/index_expression_check.c), [validation_checks.out](audit_discrete_gradient_chatgpt/validation_checks.out).

### F7 — P2: the CUDA top-1 validator accepts NaN gains and invalid selected indices

Location: [phylo2vec_cuda_8xh100.cu:2187](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_cuda_8xh100.cu:2187), especially line 2209.

The full gradient arrays receive NaN checks, but the top-1 check is only fabsf(best_delta - row_min) > 1e-6. That expression is false for NaN. h_best_j is copied back but never validated against its legal domain or the gain at the chosen column.

In a diagnostic copy, immediately before validation, I replaced every top-1 gain with NaN and every selected index with -999. The validator still returned 1. The full-gradient errors remained zero, so the surrounding error thresholds would not repair this omission.

Recommended correction: require finite gains, validate the index or the documented no-move sentinel, and check that the selected entry realizes the reported minimum. Retain fault injection for both values and indices.

This is a validator defect, not evidence that the current top-1 kernel returns invalid indices. My separate direct tests of full, tiled and sharded top-1 paths on one GPU checked 18 selections, with zero failures, zero gap from the FP64 minimum and maximum gain error 4.3670096028236571e-7.

Evidence: [validation_checks.py](audit_discrete_gradient_chatgpt/validation_checks.py), [cuda_top1_checks.cu](audit_discrete_gradient_chatgpt/cuda_top1_checks.cu), [cuda_top1_checks.out](audit_discrete_gradient_chatgpt/cuda_top1_checks.out).

### F8 — P2 for the method description: the paper gives the standalone gradient the wrong normalization

Locations: [paper/phylo2vec_bme_paper.tex:59](/home/sjbhatt/phylo2vec_discrete_gradient/paper/phylo2vec_bme_paper.tex:59), also line 78, and the check at [verify_all.c:313](/home/sjbhatt/phylo2vec_discrete_gradient/verify_all.c:313).

The paper states that the standalone routines return ΔL_ord = 2ΔL. They return the ordinary unordered-pair BME difference ΔL.

For v = [0,0,0], change v[2] to 1, using off-diagonal distances 1 except D[0,2] = D[2,0] = 10. Independent BME changes from 4.25 to 6.5. The standalone routine returns 2.25, not 4.5. Existing verification also compares directly with L_mut - L_base.

Correct the mathematical convention and affected statements in the paper. Doubling the implementation to match the prose would require corresponding changes throughout objective/gain handling.

Evidence: normalization case in [api_checks.json](audit_discrete_gradient_chatgpt/api_checks.json).

### F9 — P3: the component pass's “100% of internals” comment is too strong

Locations: [bench_rnasim_gpu.cu:1060](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:1060), [bench_rnasim_gpu.cu:1595](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:1595).

build_component_waves excludes components having no direct original leaves. On balanced trees, the component pass exposed 4,992 of 4,999 internal nodes at N = 5,000, and 8,184 of 8,191 at N = 8,192. It did expose every leaf in both cases.

This contradicts the comment about that pass. It does not prove the complete optimizer misses those nodes, because the global and medium passes can cover them. Nor does counting exposed nodes establish that every required move across component boundaries is proposed. Keep these distinct when documenting reachability.

Evidence: [rnasim_host_checks.out](audit_discrete_gradient_chatgpt/rnasim_host_checks.out).

### F10 — P3: non-integer vector entries are silently truncated

Location: [discrete_gradient.py:534](/home/sjbhatt/phylo2vec_discrete_gradient/discrete_gradient.py:534).

Validation converts to int64 before checking values. Input [0,0.9,1.9] is silently accepted as [0,0,1]. If a caller accidentally supplies continuous coordinates, the API evaluates a different vector without reporting the mistake. Validate integrality before casting.

Evidence: fractional_vector_accepted in [api_checks.json](audit_discrete_gradient_chatgpt/api_checks.json).

## Recheck of the previously identified minibatch scheduling bug

The old counterexample was a 5,000-leaf ladder whose global and medium skeletons exceeded capacity, leaving only a lower part editable. The current [bounded extraction](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:852) and [component scheduler](/home/sjbhatt/phylo2vec_discrete_gradient/bench_rnasim_gpu.cu:954) repair that counterexample:

| Topology | N | Global skeleton leaves | Component waves | Component K range, including outgroup | Skipped for size | Explicit original leaves | Active internal nodes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ladder | 5000 | 1201 | 5 | 800–1052 | 0 | 5000 | 4999 |
| Ladder | 8192 | 1201 | 8 | 842–1052 | 0 | 8192 | 8191 |
| Balanced | 5000 | 1024 | 1 | 626 | 0 | 5000 | 4992 |
| Balanced | 8192 | 1024 | 1 | 1025 | 0 | 8192 | 8184 |

These are independent host calls to the current extraction/scheduling helpers. They test exposed skeletons, not a complete optimization run. The original ladder failure is fixed. A proof that every tree is reachable under the complete minibatch proposal policy remains a separate task. Improvement-only acceptance would still restrict trajectories even if the proposal graph were connected.

For the full CPU optimizer, the two fixed relabelling modes alone do not enumerate all SPR neighbors. The existing coverage test measures their union at 55.6% for N = 8 and 51.5% for N = 20. The remaining-SPR scans at [phylo2vec_fastme.c:2043](/home/sjbhatt/phylo2vec_discrete_gradient/phylo2vec_fastme.c:2043) matter. The tested optimizer outputs were all SPR-local optima.

## Passing checks and their limits

| Check | Result |
|---|---|
| Existing CPU verification, ASan + UBSan | All eight groups pass |
| Full ordered gradient, 100 trees at N = 20 | 36,000 entries, max error 3.553e-14 |
| Repeated-minimum and i = 1 root-overlap cases | Zero error in the recorded cases |
| Relabelling, 200 general trees at N = 25 | Max BME discrepancy 1.776e-14 / 2.665e-14 for the two modes |
| Rooted/unrooted inverse | 1,000/1,000 cases pass each |
| Small-tree exhaustive suite | 11,460 rooted trees and 870 ordered-gradient cases pass, latter 28,488 entries with max error 2.776e-15 |
| Independent lower-gradient checks on ordered vectors | 150 vectors, both backends, max error 1.7763568394002505e-15 |
| Exact CPU condensation | 16,231 entries, max error 1.564e-13 |
| Exact neighborhood checks of optimizer outputs | 90 P2V and 90 comparison-optimizer outputs, zero score mismatches or SPR-nonoptimal outputs |
| Main CUDA host simulation | Forward, reverse and randomized thread orders pass |
| Main CUDA on A6000 | 20 trees, N = 6–256, full/fallback/tiled paths, max FP64 discrepancy 5.9620707459906441e-6 |
| Main CUDA full/tiled/sharded top-1, one GPU | 18 selections, all correct minima, max gain discrepancy 4.3670096028236571e-7 |
| Apple benchmark CPU full/tiled recurrences via a serial dispatch adapter | 20 trees, max errors 3.7747582837255322e-15 / 4.4408920985006262e-15 |

The Apple adapter preserves the recurrence code but replaces dispatch with serial calls. It checks the CPU arithmetic, not native dispatch concurrency, Objective-C integration or Metal hardware execution. The CUDA tests do not establish error bounds at millions of leaves, validate multi-GPU transfers, or establish that all approximate minibatch proposals descend full-tree BME.

Logs: [CPU suite](audit_discrete_gradient_chatgpt/verify_all_asan.out), [SPR coverage](audit_discrete_gradient_chatgpt/verify_spr_coverage.out), [CUDA host suite](audit_discrete_gradient_chatgpt/cuda_host_verify.out), [CUDA device checks](audit_discrete_gradient_chatgpt/cuda_checks.out), [Apple CPU adapter](audit_discrete_gradient_chatgpt/mac_host_checks.out).

## Suggested repair order

1. Reject unsupported general vectors in the lower-gradient API, and make workspace ownership safe for concurrent calls.
2. Decide explicitly whether the representative minibatch algorithms optimize an approximation or must preserve the exact full-tree objective.
3. Remove competing writes from the two benchmark kernels, then test near-tie decisions and reproducibility.
4. Synchronize tree and score at optimizer termination, promote all quadratic index arithmetic, and strengthen top-1 validation.
5. Correct the paper's normalization and the scheduler coverage wording, then add the small reproducers to the regression suite.

Reproduction commands and artifact notes are in [the audit README](audit_discrete_gradient_chatgpt/README.md).
