# Review artifacts

See [the review](../REVIEW_discrete_gradient_chatgpt.md) for findings and interpretation. Everything here is an audit artifact. No production source changes are proposed or installed by these scripts.

The snapshot contains the source actually tested. source_manifest.json records its hashes and the repository HEAD. initial_git_status.txt records pre-existing edits. source_recheck.json records the comparison with the live sources after testing.

Run from this directory. The commands below rebuild only audit copies and can overwrite the recorded audit outputs. In particular, race counts and uninitialized-memory values can change between runs. The report quotes the captured run. The deterministic workspace-poisoning example and the representative-profile counterexamples do not rely on a particular allocator or scheduler.

## CPU build and reference tests

    cd /home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_chatgpt
    REVIEW_PY=/home/sjbhatt/anaconda3/envs/rcpms-jax/bin/python
    export PYTHONDONTWRITEBYTECODE=1
    export OPENBLAS_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1
    gcc -O2 -c snapshot/phylo2vec_fastme.c -o core.o
    gcc -O2 -shared -fPIC snapshot/phylo2vec_fastme.c -lm -o snapshot/libphylo2vec_fastme.so
    gcc -O2 -g -fsanitize=address,undefined -fno-sanitize-recover=all snapshot/verify_all.c snapshot/phylo2vec_fastme.c -lm -o verify_all_asan
    ASAN_OPTIONS=detect_leaks=0 ./verify_all_asan > verify_all_asan.out 2>&1
    "$REVIEW_PY" -u snapshot/verify_spr_coverage.py > verify_spr_coverage.out 2>&1
    "$REVIEW_PY" -u api_checks.py > api_checks.out 2>&1
    "$REVIEW_PY" minimal_general_check.py > minimal_general_check.out 2>&1
    gcc -O2 general_poison.c -lm -o general_poison
    ./general_poison > general_poison.out

api_checks.py tests all ordered/general vectors at N = 4, 5, 6, using graph-distance Pauplin scoring, then normalization, fractional-input handling and two concurrent CPU calls. The decoder comes from the source snapshot. Its inverse and topology behavior are also exercised by the existing exhaustive tests. The score is independently calculated from the decoded graph.

## Condensation examples and scheduler helpers

    "$REVIEW_PY" extract_host_helpers.py
    g++ -O2 -std=c++17 representative_check.cpp -o representative_check
    ./representative_check > representative_check.out
    "$REVIEW_PY" representative_check.py > representative_check_math.out
    "$REVIEW_PY" representative_additive_check.py > representative_additive_check.out

extract_host_helpers.py extracts unmodified host functions from the RNASim snapshot. The C++ test calls its actual representative helper. The Python scripts independently score the full and condensed trees. The additive script also runs the first metric example, so its text output contains both results.

The scheduler checks are the no-argument mode of rnasim_checks, built in the CUDA section below. That mode executes only host code:

    ./rnasim_checks > rnasim_host_checks.out

It measures extraction and exposed nodes without optimizing or loading a large dataset. It tests component Pass 2A and the global skeleton separately. It does not claim to prove complete proposal connectivity.

## Diagnostic boundary and validator tests

    "$REVIEW_PY" boundary_checks.py > boundary_checks.out 2>&1
    "$REVIEW_PY" validation_checks.py > validation_checks.out 2>&1
    "$REVIEW_PY" mac_host_checks.py > mac_host_checks.out 2>&1

boundary_checks.py creates core_iteration_cap_1.c from the snapshot, changing only the P2V optimizer loop cap from 500 to 1. This forces the terminal path without pretending that the unmodified optimizer naturally reaches 500 on that input.

validation_checks.py creates a CUDA-source diagnostic copy that injects NaN top-1 gains and -999 indices immediately before validation. It runs with the host SIMT adapter. It also compiles isolated copies of the overflowing index expressions under UBSan, without allocating large matrices. UBSan messages in its output are expected evidence.

mac_host_checks.py copies only the CPU part of bench_parallel_gpu.m, substitutes serial dispatch calls and C++ lambda syntax for the platform dispatch syntax, then compares full/tiled gradients with the C reference. This does not test native Metal or native dispatch concurrency.

## CUDA tests

Check GPU availability before running. The recorded runs used only GPU 0.

    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
    REVIEW_NVCC=/usr/local/cuda-12.8/bin/nvcc
    g++ -O2 -std=c++17 -x c++ snapshot/phylo2vec_cuda_8xh100.cu -x none core.o -o cuda_host_verify
    ./cuda_host_verify > cuda_host_verify.out 2>&1
    "$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp cuda_checks.cu core.o -o cuda_checks
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 ./cuda_checks > cuda_checks.out 2>&1
    "$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp cuda_top1_checks.cu core.o -o cuda_top1_checks
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 ./cuda_top1_checks > cuda_top1_checks.out 2>&1
    "$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp rnasim_checks.cu -o rnasim_checks
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 ./rnasim_checks gpu > rnasim_gpu_checks.out 2>&1
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 ./rnasim_checks gpu-safe > rnasim_gpu_safe_checks.out 2>&1

cuda_checks tests the main full/fallback/tiled gradient implementations. cuda_top1_checks independently checks selected coordinates as well as gains for the main full, tiled and sharded top-1 entry points, each with one GPU. It does not exercise peer transfers.

rnasim_checks includes the snapshot benchmark under a renamed main and uses its actual metadata, kernels and SPR application. Graph-distance scoring supplies an independent objective check. Its gpu-safe mode replaces only the internal-distance kernel with an audit-local own-column-only version. The production source remains unchanged.

General input checks, race tests, boundary diagnostics and validator fault injection deliberately demonstrate defects. Their successful completion is evidence that the harness ran, not an assertion that the reviewed implementation passed those checks.
