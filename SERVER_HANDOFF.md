# Server Handoff Guide: Phylo2Vec $O(N^2)$ BME Discrete Gradient & GPU Suite

**Repository**: `https://github.com/bhattsamir/phylo2vec_discrete_gradient.git`  
**Branch**: `main`

---

## 1. Current State Summary (What Is Done & Verified)

All mathematical formulations, C99 CPU kernels, Apple M5 Metal shaders, NVIDIA CUDA C++ kernels, Python/`ctypes` bindings, and the 15-page LaTeX paper (`paper/phylo2vec_bme_paper.tex` & `.pdf`) are complete and verified:

1. **Core C99 Library (`phylo2vec_fastme.c`)**:
   - `calculate_discrete_gradient_c`: Lower-triangular ($0 \le j < i$) discrete gradient of shape `(N-1, N-2)`.
   - `calculate_discrete_gradient_full_c`: Full ($0 \le j \le 2i$) discrete gradient of shape `(N-1, 2N-3)` with boundary guard `2*i - 1 > i` at row $i = 1$.
   - `relabel_cherry_first_c`: $O(N)$ top-down cherry-first gauge relabeling ($0 \le v^{\text{cf}}_i \le i$) with clade-flip duality (`flip_mode = 0` or `1`).
   - `encode_rooted_to_v_c` & `encode_unrooted_to_v_c`: Exact $O(N)$ inverse bijections from rooted or unrooted trees back to $v \in \mathcal{V}$ ($0 \le v_i \le 2i$).
   - `optimize_phylo2vec_bme_c`: 4-stage Phylo2Vec BME tree optimizer with node-disjoint multi-coordinate ($K$-SPR) batch updates, dual-mode gauge rotation, 2-SPR basin hopping, and scale-relative convergence tolerance $\varepsilon_{\text{tol}} = 10^{-12}(|L| + 1)$.
   - `optimize_fastme_bme_c`: Reference C99 implementation of FastME's full unrooted `BMEAveragesTable` + `bmeSPR` topological hill-climber.
   - **Low-level speedups already implemented**:
     - Branchless $O(N)$ cherry-first component-root tree builder (eliminates `pairs` array and Union-Find path compression).
     - Heavy-first 32-slot register-stack DFS (`run_half_tree_dfs`, depth $\le \lfloor \log_2(2N-2) \rfloor + 2$).
     - Race-free 3-edge incident child-endpoint zeroing of `edge_delta`.
     - $O(1)$ incremental 1D `dest_row[N]` update per row (eliminates 2D `destinations[N-1][N]` table).
     - Leaf-order 1D `up[N]` fast BME length evaluator (`evaluate_bme_length_fast`).

2. **Exhaustive & Adversarial Verification (`verify_all.c`)**:
   - Passes all 7 test suites under `-fsanitize=undefined,bounds -fno-sanitize-recover=all`:
     - **Test 1**: $36{,}000$ full $0 \le j \le 2i$ gradient entries ($N=20$) vs. brute-force $O(N^3)$ Pauplin + `+1.0`/`NaN` fault injection (`max_err = 3.553e-14`).
     - **Test 2**: $N=6$ repeated-minimum-leaf (`v=[0,0,0,0,0]`) and $i=1$ root-overlap (`v=[0,1,1,1,1]`) counterexamples (`max_err = 0.000e+00`).
     - **Test 3**: Dual-mode cherry-first gauge relabeling across 200 general trees ($N=25$, `max_err = 2.665e-14`).
     - **Test 4**: $1{,}000 / 1{,}000$ exact rooted (`memcmp`) and unrooted (canonical 64-bit split bipartition) inverse bijections ($N=40$).
     - **Test 5**: Dynamic workspace allocation & optimizer scaling across $N \in \{1024, 1025, 1026\}$.
     - **Test 6**: Exhaustive $(2N-3)!!$ enumeration of **all $11{,}460$ rooted binary trees** for $N = 4, 5, 6, 7$ ($15 + 105 + 945 + 10{,}395$) and **all $870$ cherry-first trees** ($28{,}488$ gradient entries, `max_err = 2.776e-15`).
     - **Test 7**: Adversarial topologies (left/right caterpillars, balanced trees) & degenerate distance matrices (zero, constant, ultrametric, $[10^{-12}, 10^5]$ range, `max_rel_err = 1.255e-13`).

3. **CUDA Multi-GPU Pipeline (`phylo2vec_cuda_8xh100.cu`)**:
   - Contains 4 Stage-3 kernel paths + Stage-2 column-parallel `subdist` kernels:
     1. `k_spr_rows_smem`: Dynamic shared-memory `s_edge_delta` kernel (for $(2N-2)\times 4\text{B} \le \text{max\_smem\_optin}$).
     2. `k_spr_rows_gmem`: Global-memory L2-cached fallback kernel (supports arbitrary $N$).
     3. `k_spr_rows_top1`: Fused 128-thread parallel shared-memory argmin reduction kernel outputting `(best_delta[i], best_j[i])` in $O(N)$ memory.
     4. `k_tiled_leaf_cols` + `k_spr_rows_tiled`: Low-memory column-tiled pipeline (Theorem 4.4) storing only `subdist_int_leaf` of shape $(N-1)\times N$ ($\frac{1}{4}$ the size of `subdist`) and reconstructing the 3 active rows (`tree_root`, `other0`, `other1`) autonomously on the fly.
   - Verified on host SIMT emulator under **FORWARD**, **REVERSE**, and **RANDOMIZED FUZZ** (Fisher-Yates thread-order shuffle per `__syncthreads()` barrier) with `max_err <= 1.244e-06`.

---

## 2. Commands to Run on the Linux GPU Server

### Step 1: Clone & Run CPU Verification + Optimizer Benchmarks
```bash
git clone https://github.com/bhattsamir/phylo2vec_discrete_gradient.git
cd phylo2vec_discrete_gradient

# 1. Compile and run strict mathematical verification suite
gcc -O2 -march=native verify_all.c phylo2vec_fastme.c -lm -o verify_all
./verify_all

# 2. Compile and run head-to-head Phylo2Vec vs. FastME-SPR benchmark (N = 20..200)
gcc -O3 -march=native -ffast-math bench_optimizers.c phylo2vec_fastme.c -lm -o bench_optimizers
./bench_optimizers
```

### Step 2: Compile & Benchmark on NVIDIA GPU (`nvcc`)
Detect your GPU architecture with `nvidia-smi --query-gpu=name,compute_cap --format=csv` and compile `phylo2vec_cuda_8xh100.cu`:

- **On NVIDIA RTX A6000 (Ampere, 48 GB GDDR6, `sm_86`)**:
  ```bash
  nvcc -O3 -arch=sm_86 phylo2vec_cuda_8xh100.cu phylo2vec_fastme.c -o phylo2vec_cuda_bench
  ./phylo2vec_cuda_bench
  ```
- **On NVIDIA RTX 6000 Ada (`sm_89`) or RTX PRO 6000 Blackwell (`sm_100`)**:
  ```bash
  nvcc -O3 -arch=sm_89 phylo2vec_cuda_8xh100.cu phylo2vec_fastme.c -o phylo2vec_cuda_bench
  ./phylo2vec_cuda_bench
  ```
- **On NVIDIA H100 SXM5 / PCIe (Hopper, 80 GB HBM3, `sm_90`)**:
  ```bash
  nvcc -O3 -arch=sm_90 phylo2vec_cuda_8xh100.cu phylo2vec_fastme.c -o phylo2vec_cuda_bench
  ./phylo2vec_cuda_bench
  ```

When compiled with `nvcc` (`__CUDACC__` defined), `./phylo2vec_cuda_bench` automatically:
1. Detects all installed CUDA GPUs and prints their VRAM, SM count, and shared memory capacity.
2. Runs the $N=6$ repeated-minimum-leaf and root-overlap regression tests on the GPU.
3. Benchmarks and validates all 4 GPU kernel paths (`subdist`, `k_spr_rows_smem`, `k_spr_rows_gmem`, `k_spr_rows_top1`, and `k_spr_rows_tiled`) against the FP64 CPU reference across $N \in \{100, 250, 500, 1000, 2000, 4000\}$.

### Step 3: Run Python Wrapper & External `fastme` CLI Comparison
`discrete_gradient.py` automatically detects Linux vs. macOS and builds `libphylo2vec_fastme.so` on first import:
```bash
# Optional: install official FastME CLI binary if not already on PATH
# conda install -c bioconda fastme

python3 compare_with_fastme_cli.py
```

---

## 3. Next Tasks on the Server (For Agent / User)

1. **Collect Real GPU Timings on RTX A6000 / H100 ($N = 100 \dots 4{,}000$)**:
   - Run `./phylo2vec_cuda_bench` and record the measured `GPU sub (ms)`, `SPR smem (ms)`, and `Tiled (ms)` columns.
2. **Large-$N$ Scaling Sweep ($N = 10{,}000, 20{,}000, 40{,}000$)**:
   - In `phylo2vec_cuda_8xh100.cu` (`main()`), extend `sizes[]` to test $N = 10{,}000, 20{,}000, 40{,}000$ using the column-tiled + `k_spr_rows_top1` pipeline ( skipping the $O(N^2)$ CPU reference check at $N \ge 10{,}000$ or comparing against a single spot-checked row).
   - Note: At $N = 40{,}000$, `D` + `subdist_int_leaf` in FP32 uses only $8 N^2 = 12.8\text{ GB}$ of VRAM, fitting comfortably inside a single 48 GB RTX A6000 or 80 GB H100.
3. **Update Table 4 in `paper/phylo2vec_bme_paper.tex`**:
   - Replace/augment the analytical multi-GPU projections in Section 6.4 (`tab:h100_scaling`) with the empirical RTX A6000 / H100 measurements and recompile `paper/phylo2vec_bme_paper.pdf`.
