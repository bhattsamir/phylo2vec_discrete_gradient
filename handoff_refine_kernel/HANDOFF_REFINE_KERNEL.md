# Handoff: rewrite the exact-refinement kernel `k_refine_real_seq_condensed_D_pairs`

## Goal

Make the exact acceptance check in `bench_rnasim_gpu.cu` 4–6× faster without changing a single acceptance decision.

The kernel computes, for each changed capped super-leaf pair (a, b), the FP64 sum of `w_p * w_q * d_JC69(p, q)` over every leaf p in a and every leaf q in b. Today it runs about 16× below the GPU's integer (POPC) limit. Clean timings on one H100 at RNASim 100k, from Gefion job 535586:

| Phase | Time |
|---|---|
| Exact check (this kernel) | 3.4 s |
| Search | 2.4 s |
| Profile building | 0.65 s |
| Total | 7.26 s |

On an idle A6000 the total is 17.7 s, of which this kernel is about 12–13 s.

## Why it is slow (measured; see `prototype/run.log`)

- **Scattered loads.** Each thread streams its own 1,000-byte B-row with 8-byte loads, which delivers 8–13× less bandwidth than coalesced rows.
- **Too many registers.** The kernel uses 255 registers because the `len_a == 1` branch is fully unrolled and spills. That allows only 1 block, 8 warps, per SM.
- **Low reuse.** Each loaded B-row serves only 1–16 distances.
- **Raising occupancy alone does not help.** `__launch_bounds__(256,2)`, or moving the non-batch paths out of line, gives 2 blocks/SM but is 20–25% slower. The memory pattern is the problem.

## What to build

The prototype `__launch_bounds__(256,2)` kernel at `prototype/headroom_bench.cu:112` is the starting point:

- B-rows are staged into shared memory with coalesced loads;
- the inner loop is k-sliced over the 125 words;
- each thread computes a 4×4 register tile of (p, q) leaf pairs;
- it uses 128 registers, 2 blocks/SM.

Measured on an A6000 against a verbatim copy of the current kernel:

| Workload | Speedup |
|---|---|
| 16 rows × 32,768 (the dominant shape) | 6.2× |
| Batched singletons | 7.3× |
| Small pairs packed by a shared large side | 25× |

Integer counts matched an independent reference exactly (0/64 checksum mismatches). FP64 sums agreed to 9.6e-16 relative, because only the summation order differs.

Production needs:

1. **All shapes in one kernel:** single-leaf vs profile (today's `len_a == 1` and 16-pair batch paths), tile × large profile (the heavy-pair chunks), and many mid-sized pairs.
2. **A dense mode for the backbone windows** (P1 and P2B-global). They contribute about 447k capped pairs averaging about 9,000 leaf pairs (roughly 95 × 95), which fill 64 × 64 tiles poorly. Pack several pairs per block, or tile across pairs.
3. **Unchanged outputs:** `d_pair_exact_D[pid]` for every launch entry (FP64), and the float copies into `d_D_cond[a*K+b]` and `[b*K+a]`.
4. **The existing heavy-pair split must keep working.** Virtual super-leaf ids ≥ K+1 index `d_exact_offsets` exactly as today, and D writes go to `gpu_ws.d_refine_D_scratch` when split.

## Invariants you must keep (the audits enforce these textually)

- **Signature.** Keep the kernel name `k_refine_real_seq_condensed_D_pairs` and its signature. The definition line must still start `__global__ void k_refine_real_seq_condensed_D_pairs(`.
- **One launch.** Exactly one `k_refine_(real_seq|true)_condensed_D_pairs<<<...>>>(...);` launch after `    if (best_L >= init_L - 1e-6)` in `optimize_skeleton_on_gpu`, and no other `<<<` in that tail. `build_acceptance_checks.py` substitutes that one statement.
- **Nothing new in the tail.** Any new kernel parameters or workspace buffers may appear only inside that launch statement's arguments. No `);` inside the arguments. The mocked Workspace in the harness has no other fields.
- **Same arithmetic per pair.** Gap-aware JC69 exactly as `dev_jc69_dist_popc`, in FP32: `p = (nv - nm) / nv`, clamp at 0.74, `-0.75 * logf(1 - 4/3 p)`, clamp at 0, and `nv == 0 → 1.5`. FP64 accumulation of `(double w_p * double w_q) * double d`. The integer counts `nv` and `nm` must be identical for every leaf pair.
- **Leave untouched:** `compute_clade_dyadic_profile`, `compute_outgroup_dyadic_profile`, the tail's host logic (pair selection, the split, recombination, `exact_delta_L`, the `-1e-6` test) and `run_multi_gpu_minibatch_bme`.
- **Never modify `audit_discrete_gradient*_chatgpt/` or `REVIEW_*`.** Run audits only in copies outside the repo (`tools/run_audit.sh` refuses repo paths).

**Expected audit fallout.** `refinement_cpu_check.py` already fails on the current kernel. `batch_refinement_cpu_check.py` will also stop applying, because it emulates the current batch branch text. Both are CPU emulations of the old kernel body. Report this; do not edit them. The native cross-check below replaces them for this change.

## Verification gates (Claude will re-run all of these)

| # | Check | Pass criterion |
|---|---|---|
| 1 | `tools/make_kcheck.py NEW tools/bench_rnasim_gpu.REFERENCE.cu out.cu`, build, run `RNASIM_SIZES=10000,50000,100000 --minibatch-only`. Every launch entry is compared with the original kernel. | `max_rel` ≤ 1e-12 on all entries at all sizes (bitwise not required) |
| 2 | Final tree hashes (`python3 tools/instrument2.py NEW out.cu --hash-only`, then run all sizes) | identical to the reference: 1k `9e8b9f46182b4414`, 5k `ddbdbb340d249a60`, 10k `a293cce25f261922`, 50k `64642a10f148db1e`, 100k `cf7aeb926979e496`; RF 176/696/1320/4950/8802 |
| 3 | Unit test of integer counts: random rows of all shapes, including 1×n, 16×n, 95×95, chunked virtual ids, and gaps with `nv = 0` | `nv`/`nm` identical to a scalar reference on every leaf pair |
| 4 | `compute-sanitizer --tool memcheck` and `--tool racecheck` on 10k (`--kernel-name kns=k_refine_real_seq_condensed_D_pairs`) | 0 errors |
| 5 | rev4 and rev5 suites in copies (`tools/run_audit.sh`) | everything except the two CPU kernel emulators passes as on the reference; acceptance/wave/profile JSON byte-identical |
| 6 | `verify_all` (ASan/UBSan) and the native harnesses `test_rnasim_acceptance.cu` and `test_recovery_acceptance.cu` | pass |
| 7 | Timing: interleaved A/B (reference, new, reference, new) on the same GPU | report T_refine and total (prof build from `instrument2.py`); local GPUs are shared, so note contention, or use Gefion as in job 535586 |

Reference timing, one H100, 100k: total 7.26 s, T_refine 3.43 s (Gefion job 535586, `~/gefion_upload/p2v_speedup_claude/logs/`).

## Notes

- **Further headroom.** The prototype reaches about 60% of the POPC bound. Binary tensor cores (BMMA AND+POPC, `m16n8k256`) are a further 6× counts-only peak; the fragment layout is verified in `prototype/headroom_bench.cu`. Optional, later.
- **Out of scope:** the recovery bench's `k_refine_true_condensed_D_pairs` (tree-oracle distances, many tiny pairs), lazy sign certification, and multi-GPU. Those are later steps.
- **Deliverable:** a modified `bench_rnasim_gpu.cu` in the working tree (not committed), plus a short report with the gate results. Claude will verify it before anything is committed.
