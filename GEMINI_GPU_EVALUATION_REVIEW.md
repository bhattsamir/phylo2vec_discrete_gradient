# Gemini handoff: review of the Phylo2Vec GPU BME evaluation

Review date: 1 November 2026. Reviewed source: commit `d4719c3` and the files present during this review. The GPU sources and manuscript matched the copies under `/Users/Sam/Desktop/phylo2vec_discrete_gradient`. Line references describe this snapshot; use the named functions if subsequent edits move them.

## Assessment

The CPU mathematical results and small synthetic optimizer comparisons are encouraging and reproducible. The M5 also shows useful acceleration of a gradient sweep. However, the current evidence establishes a CPU optimizer and a Metal-assisted gradient benchmark. A complete, validated GPU optimizer and an executable multi-GPU implementation remain to be demonstrated. The 350,000–400,000-taxon optimization claims go beyond the implementation and measurements in this repository.

Prioritize the correctness and evaluation gaps below before tuning kernels or expanding the performance claims. This review did not change the implementation, paper, or existing binaries. Instrumentation and diagnostic programs were built separately in `/private/tmp/phylo2vec_gpu_review/`.

**Evidence labels:** “confirmed” means directly supported by source or an executed check; “needs measurement” identifies a performance hypothesis or a hardware-dependent question. A source-level CUDA defect is not a claim that a CUDA run was performed.

## Checks performed

| Check | Observation | Interpretation |
|---|---|---|
| Rebuilt and ran `verify_all.c` with the C library | Full gradient: 36,000 entries, maximum error `3.553e-14`; lower triangle: `3.375e-14` | Supports the tested small-tree CPU arithmetic, subject to the test-harness defects below |
| Rebuilt and ran `bench_optimizers.c` | Reproduced the reported mean objectives, RF means, evaluation counts, and 120/120 win-or-tie counts at the benchmark tolerance | Reproduces these synthetic experiments against the local baseline; rounded aggregate output does not establish individual objective equality or validate against official FastME |
| Rebuilt and ran the unmodified Metal benchmark on the M5 | At N=4,000: serial full gradient `136.906 ms`, multicore `56.545 ms`, GPU `subdist` `6.275 ms`, hybrid full gradient `25.214 ms`, tiled CPU `44.740 ms` | About `6.49x` for the component and `5.43x` for the hybrid gradient versus their respective serial baselines; a single run, not a statistical estimate |
| Temporarily instrumented the hybrid output on the same seven benchmark inputs | Legal-coordinate maximum absolute errors versus FP64 ranged from `4.075e-7` at N=100 to `1.083e-6` at N=4,000 | The GPU-assisted output was checked directly during this review; the repository benchmark does not perform this check |
| Injected `+1.0` into the predicted full-gradient values in a temporary verification source | Printed error `1.000e+00`, still printed `[PASS]` and “All verification checks passed,” exited 0 | Confirmed verification-harness failure |
| Called the exported optimizer at N=1,025 under UndefinedBehaviorSanitizer | `phylo2vec_fastme.c:1404:44: index 1024 out of bounds for type int32_t[1024]` | Confirmed optimizer size limit and memory-safety bug |
| Executed two small C harnesses using the CUDA DFS arithmetic on CPU | Repeated destination writers produced different values for the same output cell | Confirms the competing values behind the CUDA races; GPU scheduling was not tested |

Timing caveat: the instrumented rerun gave GPU `subdist` `21.179 ms` at N=4,000, versus `6.275 ms` in the first run, while the hybrid remained around `25–26 ms`. These timings do not establish a new speedup claim. They show why two repetitions and one reported mean are inadequate.

No NVIDIA GPU or `nvcc` was available locally. The official `fastme` binary, NumPy, and Numba were absent from the default Python environment. The CLI exporter issue below was checked through an isolated shape-contract probe, not a complete FastME invocation.

## P0: correctness and implementation blockers

### 1. The CUDA program does not execute the algorithm

**Confirmed.** [`phylo2vec_cuda_8xh100.cu:475`](phylo2vec_cuda_8xh100.cu#L475), `main`, only enumerates devices and prints properties. There are no kernel launches, device allocations, input transfers, OpenMP execution, CUDA streams, sharding, timing, or optimizer calls. `dev_compute_autonomous_row` is an unused helper, and the tiled kernels are not connected to an SPR pipeline.

The header and README describe implemented multi-GPU sharding and an automatic global-memory fallback above N=28,000. Neither runtime path exists. The projection table is explicitly labeled projected, but the abstract/conclusion describe very large optimizations as enabled results.

**Improvement:** first build a complete single-GPU gradient executable with deterministic inputs, reference checks, checked launches, memory accounting, and CUDA-event timings. Implement and test the global-memory fallback. Then connect the optimizer and add multi-GPU execution. Until then, describe this file as a CUDA kernel prototype and the large-node estimates as unvalidated projections.

**Acceptance:** running the CUDA executable must compute and validate a gradient, emit measured stage times and memory use, and return nonzero on errors. A successful device listing is not an algorithm benchmark.

### 2. CUDA destination mapping has conflicting writers

**Confirmed source defect, with executed arithmetic counterexamples.** In [`k_spr_rows_smem`, CUDA:335](phylo2vec_cuda_8xh100.cu#L335), threads independently execute:

```cpp
out_row[m_lmin[k]] = s_edge_delta[n_leaves + k];
```

Nested merges can have the same minimum leaf. The CPU code intentionally applies those merges in increasing order and retains the last destination, for example [`bench_parallel_gpu.m:478`](bench_parallel_gpu.m#L478). Parallel float stores do not preserve that order.

Reproducer: N=6, `v=[0,0,0,0,0]`, symmetric D with diagonal 0 and off-diagonal 1 except `D[0,2]=D[2,0]=10`. For `v_index=2`, row `i=1`, three CUDA writers target `out_row[0]`:

| Merge k | Minimum leaf | Proposed stored value |
|---|---:|---:|
| 0 | 0 | `edge_delta[6] = 1.6875` |
| 1 | 0 | `edge_delta[7] = 0.5625` |
| 2 | 0 | `edge_delta[8] = 0` |

The correct ordered destination is node 8, so the current-coordinate delta is zero. CUDA can retain a different writer. A second case, `v=[0,1,1,1,1]` with the exceptional distance at `(1,2)`, targets column 1. At `i=1`, the final root assignment at CUDA:347 also writes column `2*i-1=1` without a barrier after the merge scatter.

**Improvement:** construct deterministic destination indices, then gather deltas. An integer `atomicMax` on active merge/node indices followed by synchronization and a gather is one possible implementation. An atomic float exchange does not select the correct last merge. Handle the `i=1` root overlap explicitly. A barrier alone fixes neither the repeated-minimum winner nor the general mapping semantics.

**Acceptance:** both examples match the CPU oracle on repeated runs, all current-coordinate deltas are zero, and GPU race/memory diagnostics plus numerical comparisons pass. Do not rely on a sanitizer alone to prove ordering correctness.

### 3. The exported optimizer overflows fixed arrays above 1,024 taxa

**Confirmed by UBSan.** [`phylo2vec_fastme.c:1402`](phylo2vec_fastme.c#L1402) allocates `perm[1024]` and then initializes `n_leaves` entries. N=1,025 immediately writes out of bounds. Other fixed arrays include `v_pi`, `v_cand`, per-row candidates, `saved_perm`, `used_node`, and [`is_vmap[2048]` at line 1874](phylo2vec_fastme.c#L1874). Relabeling and inverse encoding have similar limits at lines 704–705, 806–808, and 877. The standalone optimizer benchmark duplicates fixed-size storage.

The successful N=4,000 gradient benchmark does not exercise this end-to-end optimizer.

**Improvement:** allocate all size-dependent buffers from a checked workspace; return explicit errors for unsupported sizes and allocation failures. Audit all index/allocation products: `ensure_workspace` casts expressions to `size_t` after several signed 32-bit multiplications have already occurred. Cast operands before multiplication and check overflow. Replace global `g_iwork`/`g_dwork` with an explicit context if concurrent calls are supported.

**Acceptance:** sanitizer checks at N=1,024, 1,025, 1,026 and larger sizes; relabel/encode round trips; a complete optimization beyond the previous cap; clear failure behavior for oversized allocations.

### 4. Verification can report success without testing the claimed result

**Confirmed.** [`verify_all.c:190`](verify_all.c#L190), :237, :309 and :314–315 print success unconditionally and return 0. Numerical error, invalid relabeling, or recovery-count failures do not fail the executable. NaNs can evade `err > max_err` comparisons. The temporary `+1.0` injection demonstrated this directly.

Separately, the GPU benchmark's `Max Diff` at [`bench_parallel_gpu.m:940`](bench_parallel_gpu.m#L940) compares the two FP64 CPU outputs. The additional comparison at line 1161 checks the tiled FP64 CPU output. Neither reads `delta_hyb`. Therefore the paper's `~1e-16` verification figure is not a GPU error bound.

**Improvement:** enforce finite values, absolute/relative tolerances, recovery counts, valid coordinate bounds, and nonzero exit status. Compile correctness checks with floating-point settings that preserve NaN/Inf detection, or use explicit bit classification. Keep validation outside timing regions. Compare only legal coordinates unless padding has been initialized; `delta_hyb` is allocated without initialization and does not fill the whole rectangular output.

**Acceptance:** deliberate corruption makes the suite fail; GPU/hybrid and tiled errors are reported separately; small-tree GPU results match independently evaluated BME differences.

## P1: evaluation and mathematical claims to repair

### 5. The external FastME comparison is broken and incomplete

**Confirmed.** [`compare_with_fastme_cli.py:29`](compare_with_fastme_cli.py#L29) and :32 read `M[k,2]`, but [`v_to_ancestry_matrix`, discrete_gradient.py:41](discrete_gradient.py#L41) constructs a two-column matrix. The exporter raises `IndexError` when the external CLI path is used. Parent identifiers should be derived from `N+k`, consistent with the producer's contract, rather than a nonexistent column.

Even after that fix, lines 94–102 only time the CLI. They do not parse the result, independently score BME, compare RF, or validate leaf labels. The default-initialization run promised by the docstring is missing. PHYLIP export rounds to eight decimals, so the external and in-memory optimizers also receive slightly different matrices.

`bench_optimizers.c` calls its own static optimizer implementations at lines 579, 862 and 1649–1650. This establishes comparison against an in-house BME-SPR implementation, not a pinned official FastME release or necessarily the exported C implementation.

**Improvement:** repair and test Newick export; pin the official release/build/options; retain diagnostics and parse its final tree; independently score all outputs on the same matrix. Export sufficient precision or use the same rounded matrix for every method. Include both matched starting trees and practical initialization choices such as BME taxon addition, NJ, and BioNJ. Pin thread counts. Consolidate duplicated optimizer code.

Use two distinct timing comparisons: equal-scope search computation and complete program runtime including I/O/initialization. The official [FastME guide](https://www.atgc-montpellier.fr/fastme-user-guidefastme/) documents user trees, initialization choices, SPR, branch-length estimation, and thread selection; specify these settings instead of assuming defaults.

### 6. FP32 acceleration needs decision-level validation

**Confirmed evaluation gap.** The CPU reference uses FP64; Metal and the hybrid scan use FP32. Speedup therefore combines hardware, precision, traversal, and layout changes.

The temporary direct comparison found:

| N | Hybrid maximum absolute error versus FP64 |
|---:|---:|
| 100 | `4.07517291e-7` |
| 250 | `5.39103333e-7` |
| 500 | `6.74416786e-7` |
| 1,000 | `8.01960299e-7` |
| 2,000 | `8.37855415e-7` |
| 3,000 | `1.00524922e-6` |
| 4,000 | `1.08301364e-6` |

On these inputs, current-coordinate deltas were zero and no sign disagreements were observed for reference deltas with magnitude above `1e-5`. This does not validate near-zero move decisions, selected minima, converged trees, or other distance scales. The instrumentation retained the original fast-math build, so it does not establish a strict NaN/Inf check. In particular, the CPU optimizer's `1e-12` improvement thresholds cannot simply be transferred to FP32.

**Improvement:** add matched CPU FP32 and FP64 baselines using the same algorithm. Measure sign errors near zero, agreement of best moves with tie handling, final independently recomputed BME, RF, and convergence behavior. Use FP32 to screen candidates and recompute shortlisted gains or the joint candidate objective in FP64 before accepting marginal moves. Define tolerances relative to distance scale and accumulated error, not one arbitrary universal epsilon.

### 7. Large-node memory and communication projections omit essential state

**Confirmed accounting gap; runtime feasibility needs measurement.** The tiled representation stores the input D plus an approximately N-by-N internal-to-leaf table. The CUDA SPR interface also writes a full rectangular gradient. In FP32:

```text
D:                     4*N*N bytes
internal-to-leaf table: 4*(N-1)*N bytes
full gradient output:  4*(N-1)*(2*N-3) bytes
subtotal:              approximately 16*N*N bytes
```

| N | D | Internal-to-leaf | Full gradient | Subtotal before other workspace |
|---:|---:|---:|---:|---:|
| 80,000 | 25.6 GB | 25.6 GB | 51.2 GB | 102.4 GB |
| 180,000 | 129.6 GB | 129.6 GB | 259.2 GB | 518.4 GB |
| 350,000 | 490.0 GB | 490.0 GB | 980.0 GB | 1,960.0 GB |
| 400,000 | 640.0 GB | 640.0 GB | 1,280.0 GB | 2,560.0 GB |

These are decimal GB and describe the stated buffers, not an unavoidable lower bound for every future optimizer design. Streaming/reduction can remove the full output, and recomputation can trade storage for time. Those alternatives need an implemented schedule and a new runtime model. At N=350,000, D alone is 490 GB; at N=400,000, D alone consumes the quoted 640 GB aggregate H100 capacity with no room for other state.

Column independence in the DP does not make the complete SPR scan communication-free. Autonomous reconstruction requires an entire leaf-distance row for a selected node. A GPU holding only a column shard lacks that full row. Replication avoids those gathers at the cost of per-device capacity; sharding requires communication or recomputation. Specify data ownership separately for D, the internal-to-leaf table, SPR work, candidate reduction, and topology updates.

**Improvement:** reduce each row to its best few moves during scanning; keep only candidates needed by batching, with their deltas and destination identifiers. Publish per-GPU and host memory formulas, peak allocations, transfer bytes per sweep, topology, and an explicit tile/communication schedule. Measure 1/2/4/8-GPU strong scaling on identical instances. For size scaling, distinguish feasibility, gradient throughput, optimization time, and final tree quality.

Do not present the official FastME `int32` infeasibility threshold as established without auditing and testing a specific release. This repository itself also has signed 32-bit allocation/index arithmetic to repair.

### 8. Several linear/logarithmic complexity claims do not describe the code

**Confirmed.** The ancestor walk costs `O(sum_v depth(v))`. The conclusion `O(N log N)` in [`paper/phylo2vec_bme_paper.tex:247`](paper/phylo2vec_bme_paper.tex#L247) needs a balance assumption or a specified random-tree expectation. A caterpillar has `Theta(N^2)` total depth. Individual walks can be length `Theta(N)`.

The exported relabeling and inverse encoders also contain quadratic operations:

- [`relabel_cherry_first_c`, C:756](phylo2vec_fastme.c#L756) repeatedly shifts the merge-pair array. For `v` identically zero it moves `2*i` integers per iteration, totaling `(N-2)*(N-1)` integers.
- [`encode_rooted_to_v_c`, C:841](phylo2vec_fastme.c#L841) repeatedly shifts `pair_lmax_order` at line 850. The same family moves `(N-2)*(N-1)/2` integers. Linear searches at line 855 and repeated ancestor walks need their own bounds.
- `encode_unrooted_to_v_c` calls the rooted encoder at line 901 and inherits this cost.

The relabel helper on an already decoded tree may be linear; that does not establish a linear full vector-to-vector API. Dense distance-matrix permutation is another quadratic operation in the optimizer and should appear in end-to-end cost accounting.

**Improvement:** correct implementation complexity statements now. Optimize decoding/order maintenance separately and prove the new bound before restoring linear claims. Test balanced, caterpillar, and random tree families, and report depth statistics. This qualification does not invalidate the overall quadratic gradient bound; it changes the claimed component bounds and scaling model.

### 9. Paper normalization, datasets, and verification descriptions disagree with code

**Confirmed.** The BME equation at [`paper:51`](paper/phylo2vec_bme_paper.tex#L51) sums ordered pairs `x != y` with weight `2^(-number_of_internal_nodes)`. The code at [`verify_all.c:131`](verify_all.c#L131) sums unordered pairs `x < y` with that weight. The displayed equation is twice the code's length. A three-taxon star with unit branches gives 6 from the paper equation and 3 from the implemented convention. Use `x < y` or add a factor of one half, then check the derivations against the selected convention.

Other discrepancies:

- Paper line 321: branch lengths `Uniform(0.05,0.55)`; [`bench_optimizers.c:1562`](bench_optimizers.c#L1562): `Uniform(0.05,0.25)`.
- Paper line 322: lognormal Gaussian noise; C:1589–1590: bounded sum-of-three-uniforms noise applied through `D*(1+sigma*noise)` with a floor.
- Paper lines 314–315: different test sizes/trial counts from the current N=25 relabeling and N=40 inverse tests.
- Paper claims bit-for-bit rooted and unrooted vector recovery; only rooted recovery uses `memcmp`. Unrooted recovery at verify_all.c:302–307 checks one BME score, which does not prove labeled topology equality. Unrooting also loses the information needed to recover an arbitrary original rooted vector.

**Improvement:** align the methods with the actual generator or change the generator and regenerate results. Compare canonical labeled splits for unrooted round trips. Generate paper tables and verification summaries from saved machine-readable outputs.

### 10. Timing methodology needs stronger controls

**Confirmed.** The GPU benchmark uses one topology and one matrix per size, with only two timing repetitions at N>=3,000 (`bench_parallel_gpu.m:896,910–921`). Its i.i.d. symmetric distances differ from the additive/noisy phylogenetic matrices in the CPU optimizer study. The optimizer study uses only 5–20 instances per size, fixed algorithm order, and means without dispersion.

The standalone serial `subdist` timing at [`bench_parallel_gpu.m:964`](bench_parallel_gpu.m#L964) omits the root recurrence and reads a root row inherited from preceding multicore calls. It is not an independent complete stage benchmark. The serial full-gradient implementation does reuse a cached workspace, so per-call allocation overhead should not be alleged.

**Improvement:** put each timed stage behind a self-contained function; validate with freshly initialized or poisoned workspace; include the same work on both sides. Separate GPU kernel execution time from command submission/synchronization, setup/copying, SPR scanning, and the complete pipeline. Warm up all paths, randomize/interleave timing order, use multiple inputs and seeds, and collect enough samples to report median/IQR and paired uncertainty. Save raw rows and machine/compiler metadata. Do not infer memory-bandwidth saturation from peak specifications alone.

### 11. Gradient throughput is not full optimizer performance or general quality superiority

**Confirmed scope gap.** Metal runs the `subdist` stage; [`bench_parallel_gpu.m:1119`](bench_parallel_gpu.m#L1119) performs SPR scans on CPU. The benchmark does not run a full GPU-backed descent with relabeling, batches, line-search checks, polish, kicks, or a stopping decision. The abstract's `6.79x` is a component speedup; the paper's full hybrid gradient figure is `5.69x`.

The CPU results show matching reported mean objectives/RF and 120/120 win-or-tie counts against the local baseline. Full-precision paired scores are needed to distinguish individual wins from exact ties. These results do not establish global optimality on noisy data, identical labeled topologies, or superiority on general data. The statement that every FastME optimization requires `Theta(N)` moves is also too strong; use a measured iteration count `k` and cost `O(k*N^2)`. The [FastME 2.0 paper](https://academic.oup.com/mbe/article/32/10/2798/1212138) describes SPR cost in terms of that iteration count.

**Improvement:** benchmark the actual optimizer with CPU and GPU backends, counting all expensive evaluations, objective checks, polish scans, and accepted moves. Use matched seeds/initializations and stop conditions; save final trees and independently score them. Include empirical or alignment-derived distances, stronger noise, short internal branches, heterogeneous branch lengths, and difficult tree shapes. Report BME and RF separately. Compare equal-time quality as well as time to a specified score.

Ablate single-move descent, batching, gauge rotation, escape kicks, full unrooted polish, CPU parallelism, GPU DP, and GPU SPR. Measure batch acceptance rate and actual accepted moves per sweep. Retain exact joint objective checks for proposed batches; do not infer gain additivity solely from local endpoint-disjointness.

## P2: targeted performance and usability improvements

### 12. Profile the actual GPU bottlenecks before assuming bandwidth scaling

**Confirmed structure; performance impact needs measurement.**

- CUDA:313–324 gives traversal work only to threads 0 and 1. These are lanes of the same warp in different branches. This is not evidence of efficient concurrent half-tree traversal; most block lanes do no traversal work. Explore separate warps/cooperative traversal or more row-level concurrency, and benchmark the trade-off.
- At N≈28,000, `edge_delta` alone needs approximately 224 KB per block. That heavily limits residency. Hopper has 228 KiB per SM but at most 227 KiB per block, and large dynamic allocations require opt-in. Query the actual device limit, set `cudaFuncAttributeMaxDynamicSharedMemorySize`, check launch errors, and measure occupancy. See NVIDIA's [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/).
- Three autonomous FP32 rows plus an edge buffer need approximately `32*N` bytes per block if all are retained. At N=28,000 this is roughly 896 KB, already beyond a single H100 block's shared memory. The tiled CUDA implementation needs a documented global/shared-memory layout rather than assuming all rows fit on chip.
- The dynamically indexed 32-element stack arrays are not guaranteed to occupy registers. Inspect `nvcc -Xptxas=-v`, generated code, and Nsight local-memory traffic before claiming a register stack.
- The column kernels launch only approximately N threads, each with a serial loop over internal nodes. At N=4,000 there are only about sixteen 256-thread groups. Measure achieved occupancy, memory traffic, and warp stalls. Batching independent trees, restarts, or bootstrap instances may improve utilization.
- The transpose kernels are direct copies, not shared/threadgroup-memory tiled transposes. Metal:790–801 and CUDA:72–81 have a strided side. Implement a real tile and measure its benefit.
- Top-down symmetric writes at Metal:849/:858 and CUDA:143–144 still scatter; ancestor walks also diverge. Restrict “100% coalesced” claims to the applicable access pattern. Alignment and row stride must also be checked.

### 13. Report total peak memory, not only the main table

**Confirmed.** “4x smaller” describes the internal distance table, not the whole application. For the FP64 CPU methods at N=4,000, input D, full gradient, main distance state, and tiled scratch give approximately 594 MB tiled versus 896 MB untiled: around 1.5x reduction for those components. These are per-method accounting estimates, not measured process peaks.

The benchmark actually retains multiple methods' buffers concurrently, including the library's cached workspace. Metal `newBuffer...` objects are not explicitly released and the documented build does not enable ARC, so resource lifetime also affects the observed process footprint.

**Improvement:** measure peak CPU/GPU memory for each backend in isolation, release per-size resources, and report input/output/workspace separately. Tune tile width against cache size. Reducing the full gradient to per-row candidates is likely more consequential at very large N than shrinking only `subdist`.

### 14. Make failure handling and the public interface suitable for a replacement tool

**Confirmed API gaps.** Check Metal device creation, every pipeline/buffer creation, and command-buffer completion/error status; currently only library compilation is checked. A failed dispatch must not produce a plausible timing row.

`discrete_gradient.py` passes raw pointers without checking D's shape, finiteness, symmetry, vector length/range, or minimum supported N. Explicit `backend="numba"` leaves `_c_lib` unset, but the full-gradient, relabeling, and optimizer methods call it anyway. The Numba lower-gradient output has shape `(N-1,N-1)` while the C wrapper promises `(N-1,N-2)`.

The wrapper hard-codes `/usr/bin/clang` and a macOS dylib, with no dependency/environment specification. Locally, default Python is x86_64 while the tracked dylib is arm64. These are portability/runtime problems, separate from the mathematical method.

**Improvement:** validate contracts before entering C, raise explicit unsupported-backend errors, unify output shapes, use platform-appropriate builds, and provide a reproducible environment. Replace shared global workspace if parallel calls are intended. For a practical FastME replacement, define the scope—distance-matrix topology search versus broader sequence/model/bootstrap support—and test PHYLIP/Newick compatibility, leaf-label preservation, branch-length estimation, and stop/error reporting within that scope.

## Suggested implementation order for Gemini

| Step | Concrete deliverable | Completion criterion |
|---|---|---|
| 1 | Enforcing CPU/GPU correctness tests; fixed optimizer storage and Newick export | Fault injection fails, size-boundary sanitizer tests pass, round-trip topology and legal-coordinate checks pass |
| 2 | Honest manuscript and README claims | Formula, complexity, datasets, precision, implementation status, and measured/projected results agree with source and saved outputs |
| 3 | Validated single-GPU CUDA gradient | Deterministic mapping, both counterexamples pass, checked memory/launches, complete stage timings, real fallback |
| 4 | Controlled baseline experiment | Pinned official FastME; matched matrices/starts/settings; parsed and independently scored trees; repeated timing data |
| 5 | Complete GPU-backed optimizer | Includes all search stages and stopping costs; final BME/RF and total memory/time compared with the CPU path |
| 6 | Multi-GPU implementation and scaling | Explicit ownership/communication schedule; measured 1/2/4/8-GPU results; no unsupported capacity or runtime claims |

Keep successful CPU gradient/reference checks as regression oracles while implementing these changes. Treat the large-node timing table as a model to test, not an acceptance target that results must reproduce.

## Reproduction notes and source identity

The review used the documented Clang builds with outputs redirected to temporary binaries. The optimizer boundary probe used dynamically allocated `v`, output, and a symmetric N=1,025 matrix with diagonal zero and all off-diagonal entries one:

```bash
/usr/bin/clang -O1 -g -fsanitize=undefined -fno-sanitize-recover=all \
  /private/tmp/phylo2vec_gpu_review/optimizer_asan_probe.c phylo2vec_fastme.c \
  -o /private/tmp/phylo2vec_gpu_review/optimizer_ubsan_probe
/private/tmp/phylo2vec_gpu_review/optimizer_ubsan_probe
```

The filename `optimizer_asan_probe.c` is historical: the completed diagnostic above used UBSan. An ASan attempt did not complete and was interrupted; no ASan result is claimed.

Temporary evidence, available in this local session:

- `verification.log`, `verification_fault_injection.log`, and `optimizer_benchmark.log`.
- `bench_parallel_gpu_validated.m` and `metal_validation.log`: the extra comparison checks rows `1..N-2`, columns `0..2*i`; padding is excluded. Validation is after the timed hybrid call.
- `cuda_scatter_counterexample.c` and `cuda_scatter_root_overlap.c`: use production C tree/distance tables and the CUDA DFS body with CUDA qualifiers removed. These execute CUDA arithmetic on CPU, not CUDA kernels.
- `optimizer_asan_probe.c`: the small boundary-call harness described above.

The temporary files are diagnostic artifacts, not an installed test suite. Recreate or move the relevant regressions into the repository when implementing fixes.

Reviewed SHA-256 values:

```text
bench_parallel_gpu.m:
6cd737fbd64e1ab2f9e9feb16a6efbc023061860e9adc2250a6e81798173c665
phylo2vec_cuda_8xh100.cu:
2b6665c0f9b86fc484a2a4f2739b4ed1e6f31efaa8d19be772d657c49ace46db
phylo2vec_fastme.c:
2fff230abf7e79683ed309ac216655da6ab4a52469b7504849413dfb681ef9d3
paper/phylo2vec_bme_paper.tex:
fbd5ac7fe3b2902e309c3de6a172f0154b011b88a10c24bf9129d01bf79439aa
```
