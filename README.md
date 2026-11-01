# Phylo2Vec Discrete Gradient & BME Tree Optimizer

High-performance $O(N^2)$ Balanced Minimum Evolution (BME) discrete gradient computation and topological tree optimization in Phylo2Vec vector space.

## Repository Contents

- **[phylo2vec_fastme.c](phylo2vec_fastme.c)** & **[libphylo2vec_fastme.dylib](libphylo2vec_fastme.dylib)**  
  Core C99 library implementing:
  - `calculate_discrete_gradient_c`: Lower-triangular ($0 \le j < i$) discrete gradient (shape `(N-1, N-2)`).
  - `calculate_discrete_gradient_full_c`: Full ($0 \le j \le 2i$) discrete gradient (shape `(N-1, 2N-3)`).
  - `relabel_cherry_first_c`: $O(N)$ top-down cherry-first taxon relabeling ($v^{\text{new}}_i \le i$) with clade-flip duality (`flip_mode = 0` or `1`).
  - `encode_rooted_to_v_c` & `encode_unrooted_to_v_c`: Exact $O(N)$ inverse bijection from rooted or unrooted trees back to $v \in \mathcal{V}$ ($0 \le v_i \le 2i$).
  - `optimize_phylo2vec_bme_c`: End-to-end Phylo2Vec BME tree optimizer with full $0 \le j \le 2i$ gradients, node-disjoint multi-coordinate ($K$-SPR) batch updates, $O(N)$ cherry-first gauge relabeling, and 2-SPR basin-hopping kicks.
  - `optimize_fastme_bme_c`: In-memory C99 implementation of FastME's full unrooted `BMEAveragesTable` + `bmeSPR` hill-climber (Desper & Gascuel 2002; Lefort et al. 2015).

- **[discrete_gradient.py](discrete_gradient.py)**  
  Unified Python / NumPy / Numba / `ctypes` wrapper around `libphylo2vec_fastme.dylib`.

- **[verify_all.c](verify_all.c)** (compiled to `verify_all`)  
  Standalone mathematical verification suite testing the full $0 \le j \le 2i$ discrete gradient, $O(N)$ cherry-first relabeling, and exact inverse bijection against brute-force $O(N^3)$ Pauplin path-length evaluations.

- **[bench_optimizers.c](bench_optimizers.c)** (compiled to `bench_optimizers`)  
  Head-to-head C99 benchmark comparing `optimize_fastme_bme_c` vs. `optimize_phylo2vec_bme_c` across $N = 20 \dots 200$ on both noiseless ($\sigma = 0.00$) and noisy ($\sigma = 0.15$) phylogenetic distance matrices.

- **[compare_with_fastme_cli.py](compare_with_fastme_cli.py)**  
  Python script comparing `optimize_phylo2vec_bme` and `optimize_fastme_bme` and, if the external `fastme` CLI binary is installed on `PATH`, invoking `fastme -i D.phy -u init.nwk -s` on exported PHYLIP/Newick files.

- **[bench_parallel_gpu.m](bench_parallel_gpu.m)** (compiled to `bench_parallel_gpu`)  
  Large-$N$ ($N = 100 \dots 4{,}000$) parallel CPU (10-Core Grand Central Dispatch), Column-Tiled Low-Memory CPU, and Apple Silicon Metal GPU benchmark implementing:
  - Lockstep Column-Parallel `subdist` matrix DP (100% coalesced SIMT loads/stores, zero per-level kernel launches).
  - $O(N \log N)$ Stackless Ancestor-Walk Top-Down + Symmetrization pass.
  - Heavy-First $\lceil \log_2(2N) \rceil$-Slot Register-Stack SPR traversal + $O(1)$ incremental destination mapping.
  - Column-Tiled Low-Memory `subdist` ($4\times$ memory reduction via `subdist_int_leaf`) + $O(N)$ Autonomous Row Reconstruction (`compute_autonomous_row_f64`).
  - Zero-copy Hybrid GPU (`subdist`) + 10-Core CPU (`SPR scan`) pipeline on Apple Silicon Unified Memory.

- **[phylo2vec_cuda_8xh100.cu](phylo2vec_cuda_8xh100.cu)**  
  Multi-GPU CUDA (`sm_90` / `sm_100`) implementation for 1x to 8x NVIDIA Hopper H100 SXM5 and RTX PRO 6000 Blackwell nodes featuring lockstep column-parallel kernels, 228 KB shared-memory `edge_delta` caching (`k_spr_rows_smem`), and column-tiled autonomous row kernels (`k_tiled_leaf_cols`, `dev_compute_autonomous_row`).

- **[paper/phylo2vec_bme_paper.tex](paper/phylo2vec_bme_paper.tex)** & **[paper/phylo2vec_bme_paper.pdf](paper/phylo2vec_bme_paper.pdf)**  
  13-page LaTeX manuscript presenting all mathematical theorems, proofs, algorithms, multi-core/GPU/column-tiling architectures, and empirical benchmarks.

---

## Quick Start

### 1. Run Mathematical Verification Suite
```bash
/usr/bin/clang -O3 -march=native verify_all.c phylo2vec_fastme.c -o verify_all
./verify_all
```

### 2. Run End-to-End Optimizer Benchmarks (Phylo2Vec vs. FastME)
```bash
/usr/bin/clang -O3 -march=native -ffast-math bench_optimizers.c -o bench_optimizers
./bench_optimizers
```

### 3. Run Large-N Multi-Core CPU & Metal GPU Benchmarks ($N = 100 \dots 4{,}000$)
```bash
/usr/bin/clang -O3 -march=native -ffast-math -framework Metal -framework Foundation bench_parallel_gpu.m phylo2vec_fastme.c -o bench_parallel_gpu
./bench_parallel_gpu
```

### 4. Recompile Shared Library & Use from Python
```bash
/usr/bin/clang -O3 -march=native -ffast-math -shared -fPIC phylo2vec_fastme.c -o libphylo2vec_fastme.dylib
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

