# Gemini Implementation & Verification Report: `k_refine_real_seq_condensed_D_pairs`

- **Handoff Spec**: `handoff_refine_kernel/HANDOFF_REFINE_KERNEL.md`
- **Target File Modified**: `bench_rnasim_gpu.cu` (lines 258–665)
- **Status**: All 6 strict invariants preserved; all 7 verification gates passed.

---

## 1. What Was Built & How the 4 Prototype Issues Were Resolved

`k_refine_real_seq_condensed_D_pairs` in `bench_rnasim_gpu.cu` was replaced with a shared-memory staged, k-sliced, 4×4 register-tiled kernel (`BM = 64`, `BN = 64`, `NSL = 5` slices of `SW = 25` `uint64_t` words across all 5 bitplanes, padded strides `PA = PB = 65`).

### Resolution of Prototype (`k_refine_v1.cuh`) Issues
1. **All-gap leaf pairs (`n_valid == 0`)**:
   - Replaced the reciprocal trick (`1.0f / (float)max(nv, 1)`) with the exact branch:
     ```cuda
     float d_jc = 1.5f;
     if (n_valid > 0) {
         float p = (float)(n_valid - n_match) / (float)n_valid;
         if (p >= 0.74f) p = 0.74f;
         d_jc = -0.75f * logf(1.0f - (4.0f / 3.0f) * p);
         if (d_jc < 0.0f) d_jc = 0.0f;
     }
     ```
2. **`__launch_bounds__(256, 2)` placement for regex extractors**:
   - Placed on the same line as `__global__` and the function name:
     ```cuda
     __global__ __launch_bounds__(256, 2) void k_refine_real_seq_condensed_D_pairs(
     ```
     so `^(?:__global__|__device__)[^\n]*\bk_refine_real_seq_condensed_D_pairs\(` in `make_kcheck.py` and `instrument2.py` captures `__launch_bounds__(256, 2)` automatically.
3. **Register pressure, zero spills, and adaptive `RI`**:
   - Dispatched specialized macro paths `NK_RUN_ATILE_16()` (`RI = 1`) and `NK_RUN_ATILE_64(2/3/4)` (`RI = 2, 3, 4`) based on active `tile_a <= 16, 32, 48, 64`.
   - In the `W != 25` scalar fallback, inlined a `#pragma unroll 1` loop over `0 .. W - 1` using `sx[c * W + w]` instead of calling `dev_jc69_dist_popc` (which hardcoded `25` and used `#pragma unroll`, causing an 8-byte stack frame on `sm_86`).
   - Verified with `nvcc -O3 -std=c++17 -Xptxas -v` on both `-arch=sm_86` and `-arch=sm_90`:
     ```text
     ptxas info : Compiling entry function '_Z35k_refine_real_seq_condensed_D_pairsiiiPKiS0_S0_S0_PKfPKmPfPd' for 'sm_86'
     ptxas info : Function properties for _Z35k_refine_real_seq_condensed_D_pairsiiiPKiS0_S0_S0_PKfPKmPfPd
         0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
     ptxas info : Used 128 registers, used 1 barriers, 31792 bytes smem, 432 bytes cmem[0]

     ptxas info : Compiling entry function '_Z35k_refine_real_seq_condensed_D_pairsiiiPKiS0_S0_S0_PKfPKmPfPd' for 'sm_90'
     ptxas info : Function properties for _Z35k_refine_real_seq_condensed_D_pairsiiiPKiS0_S0_S0_PKfPKmPfPd
         0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
     ptxas info : Used 128 registers, used 1 barriers, 31792 bytes smem
     ```
4. **Multi-pair packing for singleton/small `A` profiles (`min(len_a, len_b) = 1`)**:
   - Within deterministic 64-PID aligned windows (`ws = pid & ~63`), all threads load the window's `(a, b, off_a, len_a, off_b, len_b)` metadata into shared memory and normalize `len_a <= len_b` per pair when `len_a > len_b`.
   - Consecutive pairs sharing the same `b` with `0 < len_a <= len_b` are greedily packed into the lead block (`my_idx == lead_idx`) up to `cap_a`, while non-lead blocks in the pack exit cleanly (`if (my_idx != lead_idx) return;`).
   - When `len_b <= 512` (the `P2Bg` interior-clade regime), `cap_a = 128` (or `64` when `n_pairs < 2048`), packing up to 64 pairs across two 64-row A-tiles.
   - When `len_b > 512` (the `P2A` outside-profile regime), `cap_a = 16` (for `n_pairs <= 65536`), so 16 singletons pack into one `RI = 1` (`16 × 64`) tile without starving SMs. *(Note: `n_pairs` in `P2A` launches is `~2,000–8,000` because it includes the `P2Bg` pairs in the same launch, so a threshold of `n_pairs <= 1024` would falsely fall back to `cap_a = 64` and compress 300 singletons into just 5 blocks.)*
   - Each row `r` in a packed tile reduces across `tx = 0..15` via `__shfl_down_sync`, multiplies by `w_p` once (`s_row_sum[r] = s_wa[r] * acc`), and accumulates into its owning pair's `s_pair_sum[pair_idx]`, preserving exact per-pair arithmetic.

---

## 2. Verification Gate Results (Gates 1–7)

### Gate 1: Per-Call Numerical Accuracy (`make_kcheck.py`)
Built and ran `make_kcheck.py` comparing every `d_pair_exact_D` entry against `bench_rnasim_gpu.REFERENCE.cu` across all 5 benchmark sizes (`--minibatch-only`):

```text
[kcheck] N=1000   launches=0   entries=0      bitwise_mismatches=0      max_abs=0.000e+00 max_rel=0.000e+00
[kcheck] N=5000   launches=8   entries=3633   bitwise_mismatches=1224   max_abs=3.331e-16 max_rel=7.339e-16
[kcheck] N=10000  launches=18  entries=12964  bitwise_mismatches=3606   max_abs=1.887e-15 max_rel=3.300e-15
[kcheck] N=50000  launches=86  entries=371717 bitwise_mismatches=41604  max_abs=1.354e-14 max_rel=3.403e-14
[kcheck] N=100000 launches=169 entries=817514 bitwise_mismatches=154994 max_abs=1.166e-14 max_rel=4.658e-14
```
All relative differences are `<= 4.658e-14` (`> 20×` below the `1e-12` gate threshold).

### Gate 2: End-to-End Tree Hash & RF Identity (`instrument2.py --hash-only`)
All 5 sizes produce bit-identical final tree hashes, RF distances, and $\Delta L$ values:

| `N` | Tree Hash | Minibatch RF | `dL` | Match vs Reference |
| ---: | :---: | :---: | :---: | :---: |
| `1,000` | `9e8b9f46182b4414` | `176` | `-0.18866794928908348` | Exact |
| `5,000` | `ddbdbb340d249a60` | `696` | `-0.89929753949532054` | Exact |
| `10,000` | `a293cce25f261922` | `1320` | `-1.6302217664546101` | Exact |
| `50,000` | `64642a10f148db1e` | `4950` | `-6.3301613909025152` | Exact |
| `100,000` | `cf7aeb926979e496` | `8802` | `-11.347810639633703` | Exact |

### Gate 3: Standalone Integer-Count & Diverse-Shape Unit Test
Tested on random 1,596-bp packed sequences (`W = 25`, including explicit all-gap `n_valid == 0` sequence pairs, zero-weight entries `w == 0.0f`, empty profiles `len == 0`, transposed shapes `len_a > len_b`, and virtual super-leaf IDs `>= K + 1` across `1×1`, `1×64`, `1×200`, `16×64`, `16×150`, `32×150`, `48×150`, `64×150`, and `95×95` tiles):
- **Integer `(n_valid, n_match)` mismatches**: `0`
- **Diverse pair shapes (68 pairs)**: `max_rel = 5.033e-16`, `max_D_cond_abs = 0.000e+00`

### Gate 4: `compute-sanitizer` (`memcheck` & `racecheck` at `N = 10,000`)
- `compute-sanitizer --tool memcheck`: `ERROR SUMMARY: 0 errors`
- `compute-sanitizer --tool racecheck`: `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`

### Gate 5: ChatGPT `rev4` and `rev5` Audit Suites (Run in `/tmp/` Copies)
Copied `audit_discrete_gradient_rev4_chatgpt` and `audit_discrete_gradient_rev5_chatgpt` to `/tmp/`, updated `snapshot/bench_rnasim_gpu.cu` and `late_snapshot/bench_rnasim_gpu.cu`, rebuilt `rna_profile_gpu`, `recovery_profile_gpu`, and `rna_profile_gpu_late`, and ran all audit scripts (`profile_checks.py`, `acceptance_checks.py`, `wave_checks.py`, `wave_fix_checks.py`, `representative_check.py`, `followup_boundary.py`, `followup_api.py`):
- **`rev4`**: `0` diffs on `acceptance_checks.json`, `wave_checks.json`, `profile_checks.json`
- **`rev5`**: `0` diffs on `acceptance_checks.json`, `wave_fix_checks.json`, `profile_checks.json`; all boundary and API follow-ups passed (`bad_accepts_found = 0`, `bad_rejects_found = 0`, `mismatches = 0`).

### Gate 6: Native Verification & Acceptance Suites
- `verify_all` (compiled with `-fsanitize=address,undefined`): all 9 tests passed (`exit 0`).
- `test_rnasim_acceptance` & `test_recovery_acceptance`: all interior-clade and outside-profile exact acceptance checks and balanced-clade counterexample rejection checks passed (`exit 0`).

### Gate 7: Interleaved A/B Timing on Idle GPU 0 (`NVIDIA RTX A6000`)

| Run | `N` | `T_prof` (s) | `T_search` (s) | `T_refine` (s) | Total Minibatch (s) | Tree Hash |
| :--- | ---: | :---: | :---: | :---: | :---: | :---: |
| **NEW 1** | `50,000` | `0.17` | `2.18` | **`1.90`** (`-25.5%`) | **`4.72`** (`-0.66s`) | `64642a10f148db1e` |
| **REF 1** | `50,000` | `0.17` | `2.19` | `2.55` | `5.38` | `64642a10f148db1e` |
| **NEW 2** | `50,000` | `0.17` | `2.23` | **`1.95`** (`-23.8%`) | **`4.81`** (`-0.61s`) | `64642a10f148db1e` |
| **REF 2** | `50,000` | `0.17` | `2.22` | `2.56` | `5.42` | `64642a10f148db1e` |
| **NEW 1** | `100,000` | `0.65` | `3.46` | **`7.94`** (`-39.3%`, `1.65×`) | **`12.97`** (`-5.15s`) | `cf7aeb926979e496` |
| **REF 1** | `100,000` | `0.65` | `3.49` | `13.07` | `18.12` | `cf7aeb926979e496` |
| **NEW 2** | `100,000` | `0.67` | `3.52` | **`8.05`** (`-38.4%`, `1.62×`) | **`13.17`** (`-4.97s`) | `cf7aeb926979e496` |
| **REF 2** | `100,000` | `0.65` | `3.51` | `13.06` | `18.14` | `cf7aeb926979e496` |

---

## 3. Assessment: Is the Method Done, or Is There One More Speedup Available?

### A. For Paper Readiness & Methodological Completeness: **Done**
- **100% Correctness & Audit Coverage**: All five rounds of ChatGPT review issues—exact uncapped verification of both interior clades and the outside profile, rejection of the balanced-clade counterexample, wave-free sequential minibatch consistency, and near-machine-epsilon (`4.66e-14`) agreement across all `817,514` refinement entries—pass every gate and audit suite with zero diffs.
- **Headline Benchmark**: At `N = 100,000` on 1× RTX A6000, total minibatch runtime is **`~13.0s`** (`RF = 8,802`, $\Delta L = -11.35$), which is **~60× faster** than VeryFastTree (`772.5s`, `RF = 8,857`) and **~14× faster** than `VFT-InitNJ` (`179.0s`, `RF = 12,375`).

### B. Where the `12.97s` at `N = 100,000` Currently Goes
1. **`T_prof` (`0.65s`, 5%)**: CPU exact/capped dyadic profile construction.
2. **`T_search` (`3.46s`, 27%)**: GPU condensed-$D$ build + NNI/SPR discrete-gradient search across 352 windows.
3. **Host/tree-update overhead (`0.92s`, 7%)**: Tree re-encoding and window setup.
4. **`T_refine` (`7.94s`, 61%)**, which breaks down into three distinct regimes across the 169 launches (`1.994 × 10^{10}` total leaf pairs):
   - **Regime A — Large balanced interior launches (`tail 1` with `447,295` pairs / `4.03e9` leaf pairs, and `tail 167` with `223,962` pairs / `2.01e9` leaf pairs = 30.3% of all leaf pairs)**:
     Takes **only `0.66s` combined** (`0.44s` + `0.22s`), achieving **`9.15 × 10^9` leaf-pairs/s** (surpassing the `6.84 × 10^9` leaf-pairs/s synthetic `headroom_bench.cu` figure thanks to `cap_a = 128` multi-pair packing).
   - **Regime B — Heavy split launches (`17` launches where `split_pairs == true`, `7.90e9` leaf pairs = 39.6% of all leaf pairs)**:
     Takes **`~1.8s` combined** (`~4.4 × 10^9` leaf-pairs/s).
   - **Regime C — `150` unsplit outside-profile launches (`split_pairs == false`, `max(len_b) ≈ 98,000`, `6.00e9` leaf pairs = 30.1% of all leaf pairs)**:
     Takes **`~5.5s` combined (`~36 ms` per launch — 70% of `T_refine`!)**, running at only **`1.1 × 10^9` leaf-pairs/s**.

---

## 4. Optional Next Step: Squeezing Out the Remaining `~4–5s` in Regime C

### Why Regime C (`150` unsplit outside-profile launches) takes `5.5s`:
In each of those `150` launches, there are only `~300–600` small interior clades (`len_a = 1..31`, mostly singletons `len_a = 1`) paired against the `~98,000`-leaf outside profile (`b = K`, which is **`98 MB`** of packed sequences—**16× larger** than the RTX A6000's `6 MB` L2 cache):
- The host splitter in `optimize_skeleton_on_gpu` (`bench_rnasim_gpu.cu` lines 2222–2268) currently checks:
  ```cpp
  if (ls >= 32 && (double)ls * (double)ll > 2.0 * chunk_leaf_pairs)
  ```
  and only splits the **smaller** clade `ls = min(lu, lv)`. Because `ls < 32` for singletons/small clades, `split_pairs` does not trigger.
- Packing `16` singletons per block yields only `~20–38` active blocks on an `84`-SM GPU (`< 25%` wave occupancy), each running `RI = 1` (`16` A-rows per `sB` load instead of `64` A-rows per `sB` load).
- Worse, as those `20–38` blocks drift apart along `q0 = 0 .. 98,000`, each block misses in the `6 MB` L2 cache and re-streams `98 MB` from DRAM.

### Two Clean Ways to Fix Regime C (~25–35 Lines of Code, ~90–95% Confidence)

1. **Option 1 — Kernel-Internal B-Partitioning Across Already-Launched Pack Blocks (`0` Host Changes)**:
   - When `pack_m` singletons (up to `64` pairs sharing `b = K`) are packed into a 64-PID window, the host **already launches all `pack_m` blocks**, and currently `pack_m - 1` of those blocks execute `if (my_idx != lead_idx) return;` and exit idle.
   - Instead of exiting idle, `G` blocks in the pack (`G = min(pack_m, 16)`) can partition `q0 = 0 .. len_b0` into `G` contiguous B-chunks while running the full `64 × 64` (`RI = 4`) tile (`4×` more work per `sB` load and `16×` more active blocks saturating all `84` SMs).
   - Each of the `G` blocks writes its `pack_m` chunk partial sums to a static device scratch slot, and the last finishing block of the pack (`atomicAdd` arrival counter reaching `G - 1`) sums the `G` partials in deterministic `0 .. G - 1` order into `d_pair_exact_D` and `d_D_cond`.
   - Because this lives entirely inside `bench_rnasim_gpu.cu` with zero host changes, `make_kcheck.py` and `instrument2.py` work out of the box.

2. **Option 2 — Host-Side B-Chunking in `optimize_skeleton_on_gpu` (~25 Lines of Host C++)**:
   - Extend the existing `refine_offsets` / `launch_a` / `launch_b` / `chunk_owner` block in `optimize_skeleton_on_gpu` so that when `ll > 8192` (the outside profile `b = K`), `b = K` is split once into `~12–16` virtual super-leaf chunks of `~6,000–8,000` leaves (`6–8 MB` each, fitting inside L2 cache), and pairs against `b = K` are expanded across those virtual B-chunks and summed back via `chunk_owner` in deterministic order.
