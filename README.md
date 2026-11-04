# Phylo2Vec Discrete Gradient & BME Tree Optimizer

High-performance $O(N^2)$ Balanced Minimum Evolution (BME) discrete gradient computation, hierarchical minibatch condensation, and GPU-accelerated topological tree optimization in Phylo2Vec vector space ($N = 10 \dots 10{,}000{,}000$ taxa).

---

## Directory Structure

```text
phylo2vec_bme/
├── include/
│   └── phylo2vec_eigen_dist.h          # Header for bitplane DNA alignment & Eigen GTR-MAP distance engine
├── src/
│   ├── phylo2vec_fastme.c              # Core C99 Phylo2Vec discrete gradient, bijections, FastME & minibatch BME
│   ├── phylo2vec_eigen_dist.c          # CPU bitplane DNA parser, empirical GTR+Gamma fitter & log-det distances
│   └── phylo2vec_eigen_dist_cuda.cu    # CUDA warp-tiled popc bitplane distance kernel & GPU tree builder
├── python/
│   ├── __init__.py
│   ├── discrete_gradient.py            # Python / NumPy / ctypes / Numba interface to Phylo2Vec BME
│   ├── eigen_distance.py               # Python wrapper for the CUDA/CPU Eigen GTR-MAP distance engine
│   └── compare_with_fastme_cli.py      # End-to-end comparison against FastME CLI & Pauplin verification
├── benchmarks/
│   ├── bench_optimizers.c              # CPU head-to-head benchmark: Phylo2Vec vs. FastME (N = 20..200)
│   ├── bench_minibatch_recovery.cu     # GPU Hierarchical Minibatch BME benchmark up to N = 10,000,000
│   ├── bench_rnasim_gpu.cu             # Empirical RNASim 1,596-bp alignment GPU benchmark (N = 1,000..100,000)
│   ├── phylo2vec_cuda_8xh100.cu        # Multi-GPU (1x-8x H100) exact O(N^2) column-parallel & tiled solver
│   ├── bench_parallel_gpu.m            # Apple Silicon Metal GPU & Grand Central Dispatch CPU benchmark
│   └── slurm/                          # SLURM batch scripts for 8x NVIDIA H100 (Gefion)
├── tests/
│   ├── verify_all.c                    # Mathematical & sanitizer test suite (exhaustive N<=7, ASan/UBSan)
│   ├── verify_eigen_dist.cu            # Verification suite for bitplane DNA & Eigen GTR-MAP distances
│   └── verify_spr_coverage.py          # Brute-force SPR neighborhood coverage & local optimality checks
├── scripts/
│   └── rnasim/                         # RNASim dataset preparation & VeryFastTree / FastTree baseline scripts
└── paper/
    ├── phylo2vec_bme_paper.tex         # Main manuscript
    ├── phylo2vec_bme_appendix.tex      # Mathematical proofs & supplementary tables
    ├── phylo2vec_bme_references.tex    # Bibliography
    └── phylo2vec_bme_paper.pdf         # Compiled PDF
```

---

## Quick Start

### 1. Build Core Library & Run Verification Suites
```bash
make verify
make verify_asan
```

### 2. Build All CUDA GPU Targets (Linux / NVIDIA GPUs)
```bash
# Default CUDA_ARCH=sm_86 (override with CUDA_ARCH=sm_90 on NVIDIA Hopper H100)
make cuda CUDA_ARCH=sm_86
```
This compiles into `build/`:
- `build/verify_eigen_dist`: Bitplane DNA & Eigen GTR-MAP verification suite
- `build/bench_minibatch_recovery`: Hierarchical Minibatch BME benchmark ($N = 10{,}000 \dots 10{,}000{,}000$)
- `build/bench_rnasim_gpu`: Empirical RNASim $1{,}596$-bp sequence benchmark ($N = 1{,}000 \dots 100{,}000$)
- `build/phylo2vec_cuda_8xh100`: Multi-GPU exact $O(N^2)$ BME optimizer

### 3. Run CPU Optimizer Benchmark (Phylo2Vec vs. FastME)
```bash
make bench_cpu
```

### 4. Python Interface
```python
import sys
sys.path.insert(0, "python")
from discrete_gradient import DiscreteGradient

dg = DiscreteGradient()

# 1. Compute full 0 <= j <= 2i discrete gradient on a cherry-first vector v (v_i <= i)
delta_full = dg.calculate_discrete_gradient_full(v, D)

# 2. Canonicalize any general Phylo2Vec vector (0 <= v_i <= 2i) to cherry-first (v_new[i] <= i)
v_cf, perm = dg.relabel_cherry_first(v_general, flip_mode=0)

# 3. Optimize a tree end-to-end under the BME criterion
v_opt, bme_len, n_evals, elapsed_ms = dg.optimize_phylo2vec_bme(v_init, D)
```

### 5. Compile Paper PDF
```bash
make paper
```

