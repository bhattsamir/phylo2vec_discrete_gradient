# Phylo2Vec Discrete Gradient & BME Tree Optimizer

High-performance $O(N^2)$ Balanced Minimum Evolution (BME) discrete gradient computation and topological tree optimization in Phylo2Vec vector space.

## Repository Contents

- **[phylo2vec_fastme.c](phylo2vec_fastme.c)** & **[libphylo2vec_fastme.dylib](libphylo2vec_fastme.dylib)**  
  Core C99 library (with overflow-safe `size_t` dynamic workspace allocation up to $N = 100{,}000$) implementing:
  - `calculate_discrete_gradient_c`: Lower-triangular ($0 \le j < i$) discrete gradient (shape `(N-1, N-2)`).
  - `calculate_discrete_gradient_full_c`: Full ($0 \le j \le 2i$) discrete gradient (shape `(N-1, 2N-3)`), with boundary guard `2 * i - 1 > i` at row $i = 1$.
  - `relabel_from_rooted_tree` & `relabel_cherry_first_c`: Top-down cherry-first taxon relabeling ($v^{\text{new}}_i \le i$) with clade-flip duality (`flip_mode = 0` or `1`).
  - `encode_rooted_to_v_c` & `encode_unrooted_to_v_c`: Exact inverse bijection from rooted or unrooted trees back to $v \in \mathcal{V}$ ($0 \le v_i \le 2i$).
  - `optimize_phylo2vec_bme_c`: End-to-end Phylo2Vec BME tree optimizer with full $0 \le j \le 2i$ gradients, node-disjoint multi-coordinate ($K$-SPR) batch updates, $O(N)$ cherry-first gauge relabeling, and 2-SPR basin-hopping kicks.
  - `optimize_fastme_bme_c`: In-memory C99 implementation of FastME's full unrooted `BMEAveragesTable` + `bmeSPR` hill-climber (Desper & Gascuel 2002; Lefort et al. 2015).

- **[discrete_gradient.py](discrete_gradient.py)**  
  Unified Python / NumPy / Numba / `ctypes` wrapper around universal (`arm64` + `x86_64`) `libphylo2vec_fastme.dylib`, with input validation and path-compressed Union-Find ancestry decoding.

- **[verify_all.c](verify_all.c)** (compiled to `verify_all`)  
  Standalone mathematical verification and regression harness testing:
  - Test 0: Fault-injection self-tests (`+1.0` and `NaN` trap verification) and $N=6$ boundary counterexamples (`v=[0,0,0,0,0]` and `v=[0,1,1,1,1]`).
  - Test 1: Full $0 \le j \le 2i$ discrete gradient across 36,000 coordinate perturbations against brute-force $O(N^3)$ Pauplin evaluations.
  - Test 2: Cherry-first gauge relabeling under `flip_mode = 0` and `1` with bijective permutation verification.
  - Test 3: Exact rooted (`memcmp`) and unrooted (canonical 64-bit split bipartition isomorphism) inverse bijection recovery.
  - Test 4: Large-$N$ ($N \in \{1024, 1025, 1026\}$) dynamic buffer boundary verification under UndefinedBehaviorSanitizer.

- **[bench_optimizers.c](bench_optimizers.c)** (compiled to `bench_optimizers`)  
  Head-to-head C99 benchmark linked directly against `phylo2vec_fastme.c`, comparing `optimize_fastme_bme_c` vs. `optimize_phylo2vec_bme_c` across $N = 20 \dots 200$ on both noiseless ($\sigma = 0.00$) and log-normal noisy ($\sigma = 0.15$) phylogenetic distance matrices with `Win/Tie/Loss`, `mean±SD`, and `median[IQR]` reporting.

- **[compare_with_fastme_cli.py](compare_with_fastme_cli.py)**  
  Python script comparing `optimize_phylo2vec_bme` and `optimize_fastme_bme` with independent Newick round-trip parsing and Pauplin verification, and invoking external `fastme -i D.phy -u init.nwk -s` if installed on `PATH`.

- **[bench_parallel_gpu.m](bench_parallel_gpu.m)** (compiled to `bench_parallel_gpu`)  
  Large-$N$ ($N = 100 \dots 4{,}000$) parallel CPU (10-Core Grand Central Dispatch), Column-Tiled Low-Memory CPU, and Apple Silicon Metal GPU benchmark implementing:
  - Lockstep Column-Parallel `subdist` matrix DP (100% coalesced SIMT loads/stores, zero per-level kernel launches, $16 \times 16$ threadgroup tiled transpose).
  - Stackless Ancestor-Walk Top-Down + Symmetrization pass ($O(N \log N)$ expected work on balanced trees, $O(1)$ stack space).
  - Heavy-First $\lceil \log_2(2N) \rceil$-Slot Register-Stack SPR traversal + $O(1)$ incremental destination mapping.
  - Column-Tiled Low-Memory `subdist` ($4\times$ `subdist` memory reduction via `subdist_int_leaf`) + $O(N)$ Autonomous Row Reconstruction (`compute_autonomous_row_f64`).
  - Zero-copy Hybrid GPU (`subdist`) + 10-Core CPU (`SPR scan`) pipeline on Apple Silicon Unified Memory.

- **[phylo2vec_cuda_8xh100.cu](phylo2vec_cuda_8xh100.cu)**  
  Multi-GPU CUDA (`sm_90` / `sm_100`) implementation for 1x to 8x NVIDIA Hopper H100 SXM5 and RTX PRO 6000 Blackwell nodes featuring lockstep column-parallel kernels, $16 \times 16$ shared-memory transpose (`k_transpose_leaf_block`), race-free SIMT destination indexing (`next_k_same_lmin`), dynamic shared-memory `edge_delta` caching (`k_spr_rows_smem`) with automatic global-memory fallback (`k_spr_rows_gmem`), fused top-1 row reduction (`k_spr_rows_top1`), column-tiled autonomous row kernels (`k_tiled_leaf_cols`, `k_spr_rows_tiled`), and a barrier-accurate host SIMT emulator.

- **[paper/phylo2vec_bme_paper.tex](paper/phylo2vec_bme_paper.tex)** & **[paper/phylo2vec_bme_paper.pdf](paper/phylo2vec_bme_paper.pdf)**  
  LaTeX manuscript presenting all mathematical theorems, proofs, algorithms, multi-core/GPU/column-tiling architectures, empirical benchmarks, and analytical multi-GPU scaling projections.

---

## Quick Start

### 1. Run Mathematical & Sanitizer Verification Suite
```bash
/usr/bin/clang -O2 -g -fsanitize=undefined -fno-sanitize-recover=all verify_all.c phylo2vec_fastme.c -o verify_all
./verify_all
```

### 2. Run Barrier-Accurate CUDA SIMT Emulation & Kernel Verification
```bash
/usr/bin/clang -O3 -c phylo2vec_fastme.c -o phylo2vec_fastme.o
/usr/bin/clang++ -O3 -std=c++17 -x c++ phylo2vec_cuda_8xh100.cu -x none phylo2vec_fastme.o -o phylo2vec_cuda_verify
./phylo2vec_cuda_verify
```

### 3. Run End-to-End Optimizer Benchmarks (Phylo2Vec vs. FastME)
```bash
/usr/bin/clang -O3 -march=native bench_optimizers.c phylo2vec_fastme.c -o bench_optimizers
./bench_optimizers
```

### 4. Run Large-N Multi-Core CPU & Metal GPU Benchmarks ($N = 100 \dots 4{,}000$)
```bash
/usr/bin/clang -O3 -march=native -fobjc-arc -framework Metal -framework Foundation bench_parallel_gpu.m phylo2vec_fastme.c -o bench_parallel_gpu
./bench_parallel_gpu
```

### 5. Recompile Universal Shared Library & Use from Python
```bash
/usr/bin/clang -O3 -arch arm64 -arch x86_64 -shared -fPIC phylo2vec_fastme.c -o libphylo2vec_fastme.dylib
python3 compare_with_fastme_cli.py
```

```python
import numpy as np
from discrete_gradient import DiscreteGradient

dg = DiscreteGradient()

# 1. Compute full 0 <= j <= 2i discrete gradient on a cherry-first vector v (v_i <= i)
delta_full = dg.calculate_discrete_gradient_full(v, D)

# 2. Canonicalize any general Phylo2Vec vector (0 <= v_i <= 2i) to cherry-first (v_new[i] <= i)
v_cf, perm = dg.relabel_cherry_first(v_general, flip_mode=0)

# 3. Optimize a tree end-to-end under the BME criterion
v_opt, bme_len, n_evals, elapsed_ms = dg.optimize_phylo2vec_bme(v_init, D)
```
