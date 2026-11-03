# Gemini Implementation & Verification Report: `k_refine_real_seq_condensed_D_pairs` (Phase 1 + Phase 2 Complete)

- **Handoff Spec**: `handoff_refine_kernel/HANDOFF_REFINE_KERNEL.md`
- **Target File Modified**: `bench_rnasim_gpu.cu`
- **Status**: **Phase 1 & Phase 2 (Option 2: Host-Side `B`-Chunking in `optimize_skeleton_on_gpu`) Complete & Verified Across All Gates.**
  - **`N = 100,000` `T_refine`**: **`2.83s`** (down from `13.07s` Reference and `7.94s` Phase 1 — **`4.62×` speedup** over Reference, saving **`10.24s`**).
  - **`N = 100,000` Total Minibatch Runtime**: **`7.91s`** (down from `18.12s` Reference and `13.06s` Phase 1 — **`2.29×` end-to-end speedup**).
  - **`N = 50,000` `T_refine`**: **`0.67s`** (down from `2.55s` Reference and `1.90s` Phase 1 — **`3.81×` speedup** over Reference).
  - **`N = 50,000` Total Minibatch Runtime**: **`3.53s`** (down from `5.38s` Reference and `4.77s` Phase 1).

---

## 1. Phase 1: Kernel Replacement & Resolution of Prototype Issues

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
   - Within deterministic 64-PID aligned windows (`ws = pid & ~63`), all threads load the window's `(a, b, off_a, len_a, off_b, len_b)` metadata into shared memory.
   - Consecutive pairs sharing the same `b` with `0 < len_a <= len_b` are greedily packed into the lead block (`my_idx == lead_idx`) up to `cap_a`, while non-lead blocks in the pack exit cleanly (`if (my_idx != lead_idx) return;`).
   - When `len_b <= 512` (the `P2Bg` interior-clade regime), `cap_a = 128` (or `64` when `n_pairs < 2048`), packing up to 64 pairs across two 64-row A-tiles.
   - When `len_b > 512`, `cap_a` scales adaptively with the total launch size (`64` when `n_pairs >= 4096`, `32` when `n_pairs >= 2048`, `16` otherwise), so large split launches pack full `64 × 64` (`RI = 4`) tiles while smaller launches preserve block count across SMs.
   - Each row `r` in a packed tile reduces across `tx = 0..15` via `__shfl_down_sync`, multiplies by `w_p` once (`s_row_sum[r] = s_wa[r] * acc`), and accumulates into its owning pair's `s_pair_sum[pair_idx]`.

---

## 2. Phase 2 (Option 2, per Claude's Recommendation): Host-Side `B`-Chunking in `optimize_skeleton_on_gpu`

### Why Phase 2 Was Needed
In Phase 1 (`T_refine = 7.94s` at `N = 100,000`), `150` out of `169` refinement launches were Pass 2A outside-profile windows (`len_b ≈ 98,000`, `~300–600` small/singleton interior clades `len_a = 1..64`). Because `ls < 32` (and `ls * ll <= 2 * 4,194,304`), the existing `heavy_pid` splitter in `optimize_skeleton_on_gpu` did not split these launches, leaving only `~20–38` active blocks scanning a `98,000`-leaf profile (`98 MB` of sequence data, `16×` larger than the GPU's `6 MB` L2 cache) at `RI = 1`. That accounted for `~5.5s` (`70%` of `T_refine`).

### Option 2 Implementation (`bench_rnasim_gpu.cu` lines 2212–2313)
Following Claude's recommendation to extend the host-side heavy-pair split in `optimize_skeleton_on_gpu`:
1. **Balanced 64-Leaf-Aligned L2 Chunks of Large `B` Profiles**:
   - Whenever a pair's larger profile `l` has `ll > 6144` (`b_chunk_target = 6144` leaves) and `(double)ls * (double)b_rows <= 2.0 * chunk_leaf_pairs`, `l` is partitioned once in `refine_offsets` into `c <= 16` balanced chunks of multiple-of-64 leaves:
     ```cpp
     int64_t b_c = std::min((int64_t)16, (ll + b_chunk_target - 1) / b_chunk_target);
     int64_t b_rows = (((ll + b_c - 1) / b_c) + 63) & ~(int64_t)63;
     int64_t c = (ll + b_rows - 1) / b_rows;
     ```
   - At `N = 100,000` (`ll ≈ 98,000`), `l` is cut into `c = 16` virtual super-leaves `base + 0 .. base + 15` of `6,144` leaves (`96` B-tiles, `~6.1 MB` of packed sequences, fitting inside the `6 MB` L2 cache). This consumes only `c + 1 = 17` virtual CSR slots in `refine_offsets` (out of `max_virtual = 4 * K`).
   - At `N = 50,000` (`ll ≈ 48,000`), `l` is cut into `c = 8` chunks of `6,016` leaves.
2. **Chunk-Major Pair Expansion (`for j = 1..c-1: for pid in b_split_pids[l]`)**:
   - Chunk `0` (`base + 0`) replaces `l` in-place in `launch_a[pid]` / `launch_b[pid]`.
   - Chunks `j = 1 .. c - 1` are appended in **chunk-major order** (`for j = 1 .. c - 1`, iterating over `pid : b_split_pids[l]`).
   - Because all pairs for a given chunk `j` are contiguous in `launch_a, launch_b` and share the identical virtual super-leaf `base + j`, the kernel's 64-pair window packer packs up to `cap_a = 64` singletons (`RI = 4`, full `64 × 64` register-tiled evaluation) within each chunk `j`, while all `16` chunks execute concurrently across all SMs (`~100–160` active blocks) and share each chunk's `6,144` leaves in L2 cache.
3. **Zero Kernel Atomics & Deterministic Host Reduction**:
   - No global scratch arrays, atomic arrival counters, or `__threadfence()` calls are needed in the kernel.
   - The existing deterministic host reduction loop:
     ```cpp
     for (int32_t t = n_pairs; t < n_launch; ++t) h_pair_exact_D[chunk_owner[t - n_pairs]] += h_launch_exact_D[t];
     ```
     accumulates chunks `j = 0, 1, ..., c - 1` in fixed order for every `pid`, and `make_kcheck.py` and `instrument2.py` (including `--xcheck`) work out of the box without modification.

---

## 3. Verification Gate Results (Phase 1 + Phase 2 Option 2)

### Gate 1: Per-Call Numerical Accuracy (`make_kcheck.py`)
Ran `make_kcheck.py` comparing every `d_pair_exact_D` entry (`1,808,393` launch entries at `N = 100,000`) against `bench_rnasim_gpu.REFERENCE.cu` across all 5 benchmark sizes (`--minibatch-only`):

```text
[kcheck] N=1000   launches=0   entries=0       bitwise_mismatches=0      max_abs=0.000e+00 max_rel=0.000e+00
[kcheck] N=5000   launches=8   entries=3633    bitwise_mismatches=1224   max_abs=3.331e-16 max_rel=7.339e-16
[kcheck] N=10000  launches=18  entries=18122   bitwise_mismatches=4173   max_abs=7.772e-16 max_rel=2.491e-15
[kcheck] N=50000  launches=86  entries=611540  bitwise_mismatches=88419  max_abs=3.442e-15 max_rel=1.887e-14
[kcheck] N=100000 launches=169 entries=1808393 bitwise_mismatches=307890 max_abs=9.881e-15 max_rel=4.234e-14
```
All relative differences are `<= 4.234e-14` (`> 23×` below the `1e-12` gate threshold).

### Gate 2: End-to-End Tree Hash & RF Identity (`instrument2.py`)
All 5 sizes produce bit-identical final tree hashes, RF distances, and $\Delta L$ values:

| `N` | Tree Hash | Minibatch RF | `dL` | Match vs Reference |
| ---: | :---: | :---: | :---: | :---: |
| `1,000` | `9e8b9f46182b4414` | `176` | `-0.18866794928908348` | Exact |
| `5,000` | `ddbdbb340d249a60` | `696` | `-0.89929753949532054` | Exact |
| `10,000` | `a293cce25f261922` | `1320` | `-1.6302217664546101` | Exact |
| `50,000` | `64642a10f148db1e` | `4950` | `-6.3301613909025152` | Exact |
| `100,000` | `cf7aeb926979e496` | `8802` | `-11.347810639633703` | Exact |

### Gate 3: Split-vs-Unsplit & CPU Reference Cross-Check (`instrument2.py --xcheck`) + Standalone Unit Test
- **`instrument2.py --xcheck`** (`N = 1,000, 5,000, 10,000`):
  ```text
  [xcheck] N=5000  windows=8  pairs=3633  split_pairs=0    gpu_split_vs_unsplit: max_rel=0.000e+00 max_abs=0.000e+00 unsplit_max_rel=0.000e+00 | cpu_ref: pairs=162  split=0    max_rel_split=0.000e+00 max_rel_unsplit=3.967e-08
  [xcheck] N=10000 windows=18 pairs=12964 split_pairs=5158 gpu_split_vs_unsplit: max_rel=1.941e-15 max_abs=1.110e-15 unsplit_max_rel=0.000e+00 | cpu_ref: pairs=5326 split=5158 max_rel_split=7.667e-08 max_rel_unsplit=6.563e-08
  ```
  Every one of the `5,158` split pairs at `N = 10,000` matches the unsplit GPU launch to `max_rel = 1.941e-15` and matches the 32-thread OpenMP CPU `long double` reference to `7.667e-08` (FP32 JC69 log precision).
- **Standalone integer-count & diverse-shape test**: `0` integer `(n_valid, n_match)` mismatches; `max_rel = 5.033e-16` across all 68 synthetic pair shapes (`1×1` through `95×95`, including all-gap `n_valid == 0`, zero weights, empty profiles, transposed `len_a > len_b`, and virtual super-leaf IDs `>= K + 1`).

### Gate 4: `compute-sanitizer` (`memcheck` & `racecheck` at `N = 10,000` with `16/18` Split Launches)
- `compute-sanitizer --tool memcheck`: `ERROR SUMMARY: 0 errors`
- `compute-sanitizer --tool racecheck`: `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`

### Gate 5 & Gate 6: ChatGPT `rev4`/`rev5` Audit Suites & Native Verification
- **`rev4` & `rev5` audit suites (in `/tmp/` copies)**: `0` diffs on `acceptance_checks.json`, `wave_checks.json`, `wave_fix_checks.json`, and `profile_checks.json`; `0` diffs on all RNASim RF distances and $\Delta L$ values (`1k..100k`).
- **`verify_all` (ASan/UBSan)**: all 9 tests passed (`exit 0`).

### Gate 7: Interleaved A/B Benchmark Timing on Idle GPU 0 (`NVIDIA RTX A6000`)

#### Progression Summary (`N = 50,000` and `N = 100,000` on Idle GPU 0)

| Version | `50k` `T_refine` (s) | `50k` Total (s) | `100k` `T_refine` (s) | `100k` Total (s) | `100k` `T_refine` Speedup | `100k` Total Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Reference (`bench_rnasim_gpu.REFERENCE.cu`)** | `2.55s` | `5.38s` | `13.07s` | `18.12s` | `1.00×` | `1.00×` |
| **Phase 1 (Kernel Only)** | `1.90s` | `4.72s` | `7.94s` | `12.97s` | `1.65×` | `1.40×` |
| **Phase 2 (Kernel + Host `B`-Chunking, Run 1)** | **`0.67s`** | **`3.53s`** | **`2.83s`** | **`7.91s`** | **`4.62×`** | **`2.29×`** |
| **Phase 2 (Kernel + Host `B`-Chunking, Run 2)** | **`0.70s`** | **`3.58s`** | **`2.87s`** | **`8.04s`** | **`4.55×`** | **`2.25×`** |

#### All 5 Benchmark Sizes (`Phase 2` on Idle GPU 0)

| `N` | Windows | Accepted | Tails (`split_tails`) | Launch Pairs | `T_prof` (s) | `T_search` (s) | `T_refine` (s) | Total Minibatch (s) | Tree Hash |
| ---: | ---: | ---: | ---: | ---: | :---: | :---: | :---: | :---: | :---: |
| `1,000` | `1` | `1` | `0 (0)` | `0` | `0.00` | `0.01` | **`0.00`** | **`0.01`** | `9e8b9f46182b4414` |
| `5,000` | `19` | `8` | `8 (0)` | `3,633` | `0.00` | `0.14` | **`0.01`** | **`0.20`** | `ddbdbb340d249a60` |
| `10,000` | `38` | `18` | `18 (16)` | `18,122` | `0.01` | `0.31` | **`0.06`** | **`0.46`** | `a293cce25f261922` |
| `50,000` | `181` | `83` | `86 (84)` | `611,540` | `0.17` | `2.18` | **`0.67`** | **`3.53`** | `64642a10f148db1e` |
| `100,000` | `352` | `167` | `169 (167)` | `1,808,393` | `0.68` | `3.45` | **`2.83`** | **`7.91`** | `cf7aeb926979e496` |

---

## 4. Final Breakdown of the `7.91s` Runtime at `N = 100,000`

With Phase 1 + Phase 2 complete, `T_refine` at `N = 100,000` evaluates `1.994 × 10^{10}` exact leaf pairs in **`2.83s`** — an effective end-to-end sustained throughput of **`7.05 × 10^9` leaf-pairs/s** across all 169 launches (matching the `6.84 × 10^9` leaf-pairs/s synthetic roofline from `headroom_bench.cu`!).

The `7.91s` total minibatch runtime at `N = 100,000` is now balanced across:
- **`T_search` (`3.45s`, 44%)**: Condensed-$D$ build + NNI/SPR discrete-gradient search across 352 windows.
- **`T_refine` (`2.83s`, 36%)**: Exact uncapped full-objective verification across 169 accepted windows (`7.05 × 10^9` leaf-pairs/s).
- **Host tree-summary & window management (`0.95s`, 12%)**: Tree re-encoding and topological distance updates.
- **`T_prof` (`0.68s`, 8%)**: Capped and exact dyadic profile construction.

---

## 5. Gefion 1× NVIDIA H100 (`sm_90`) Results & 10-Million-Taxon Scaling

We also compiled `-arch=sm_90` binaries and ran both benchmarks on **Gefion (`dgx158`, NVIDIA H100 80GB HBM3)** via the guarded [`gefion_sbatch.sh`](file:///home/sjbhatt/phylo2vec_discrete_gradient/gefion_sbatch.sh) workflow (Slurm job `535847`).

### 5.1 Empirical RNASim Minibatch (`1,000` to `100,000` Taxa on 1× H100)

| Variant (`N = 100,000` on 1× H100) | `T_prof` | `T_search` | `T_refine` | Total Minibatch | `T_refine` Speedup | End-to-End Speedup | Tree Hash |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original Unsplit Baseline (`hash_base`)** | — | — | `~38.0s` | `41.78s` | `1.00×` | `1.00×` | `cf7aeb926979e496` |
| **Host Heavy-Pair Split Only (`prof_new`)** | `0.64s` | `2.41s` | `3.43s` | `7.24s` | `11.08×` | `5.77×` | `cf7aeb926979e496` |
| **Phase 2 (Kernel + Host `B`-Chunking, `prof_phase2_sm90`)** | **`0.65s`** | **`2.41s`** | **`2.12s`** | **`5.96s`** | **`17.92×`** (`1.62×` vs `prof_new`) | **`7.01×`** (`1.21×` vs `prof_new`) | `cf7aeb926979e496` |

- **`make_kcheck.py` on H100 (`kcheck_phase2_sm90`)**:
  - `N = 10,000`: `max_rel = 4.016e-15`
  - `N = 50,000`: `max_rel = 1.975e-14` (`Total Minibatch = 2.61s`, down from `4.58s` unsplit and `2.96s` `prof_new`)
  - `N = 100,000`: `max_rel = 4.658e-14` across all `1,808,393` launch entries (`Total Minibatch = 5.96s`, down from `41.78s` unsplit and `7.24s` `prof_new`)

### 5.2 Clade-Condensed Minibatch BME Recovery up to `N = 10,000,000` Taxa (1× H100 vs 1× RTX A6000)

| Taxa (`N`) | Window $N_{\text{eff}}$ | Init $\to$ Final $H(v)$ | Init $\to$ Final RF | $H(v)$ Recovery % | RF Split Recov % | $\Delta L$ | Time (`sm_86` A6000) | Time (`sm_90` 1× H100) |
| ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `10,000` | `629` | `766 -> 11` | `2,650 -> 14` | `98.56%` | `99.47%` | `-2853.36` | `0.66 s` | **`0.56 s`** |
| `50,000` | `692` | `780 -> 50` | `2,457 -> 70` | `93.59%` | `97.15%` | `-2844.04` | `0.97 s` | **`0.77 s`** |
| `100,000` | `808` | `778 -> 16` | `2,609 -> 21` | `97.94%` | `99.20%` | `-2855.55` | `1.30 s` | **`0.95 s`** |
| `500,000` | `1,310` | `773 -> 29` | `2,599 -> 44` | `96.25%` | `98.31%` | `-2876.57` | `7.42 s` | **`3.19 s`** |
| `1,000,000` | `1,346` | `778 -> 54` | `2,816 -> 85` | `93.06%` | `96.98%` | `-3058.54` | `15.56 s` | **`4.90 s`** |
| `5,000,000` | `1,282` | `778 -> 69` | `2,595 -> 100` | `91.13%` | `96.15%` | `-2892.28` | — | **`14.76 s`** |
| **`10,000,000`** | **`1,123`** | **`780 -> 32`** | **`2,524 -> 54`** | **`95.90%`** | **`97.86%`** | **`-2872.88`** | — | **`29.84 s`** |

