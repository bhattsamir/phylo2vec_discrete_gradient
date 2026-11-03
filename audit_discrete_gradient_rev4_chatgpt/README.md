# Review 4 artifacts

Source snapshots and independent checks for [the fourth review](../REVIEW_discrete_gradient_rev4_chatgpt.md). No production source was edited. `source_manifest.json` records hashes and `changes.diff` compares with review 3.

Run from this directory:

```bash
REVIEW_PY=/home/sjbhatt/anaconda3/envs/rcpms-jax/bin/python
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$REVIEW_PY" build_profile_library.py
"$REVIEW_PY" profile_checks.py > profile_checks.out 2>&1
"$REVIEW_PY" build_acceptance_checks.py
"$REVIEW_PY" acceptance_checks.py > acceptance_checks.out 2>&1
"$REVIEW_PY" find_wave_case.py
"$REVIEW_PY" build_wave_checks.py
"$REVIEW_PY" wave_checks.py > wave_checks.out 2>&1
"$REVIEW_PY" build_wave_driver.py
"$REVIEW_PY" wave_driver_check.py > wave_driver_check.out 2>&1
gcc -O2 -pthread -shared -fPIC snapshot/phylo2vec_fastme.c -lm -o snapshot/libphylo2vec_fastme.so
"$REVIEW_PY" followup_api.py > followup_api.out 2>&1
"$REVIEW_PY" followup_boundary.py > followup_boundary.out 2>&1
gcc -O2 -g -pthread -fsanitize=address,undefined -fno-sanitize-recover=all snapshot/verify_all.c snapshot/phylo2vec_fastme.c -lm -o verify_all_asan
ASAN_OPTIONS=detect_leaks=1 ./verify_all_asan > verify_all_asan.out 2>&1
```

`profile_checks.py` retains the previous raw surrogate counterexamples. Their sign disagreement is expected to persist because proposals remain approximate. `acceptance_checks.py` tests the new acceptance layer separately.

`build_acceptance_checks.py` extracts the real host acceptance tails, skeleton-distance functions and relabelling function. GPU allocation/copy operations are replaced with host storage/copies. A refinement launch is replaced by lookup into an independently supplied exact condensed matrix. Pair selection, weight differences, acceptance, return values and topology restoration run as written in the source. This is a CPU host-logic test, not GPU kernel validation.

`wave_checks.py` uses the actual scheduler and outside-profile helpers, plus the current sequence-distance function compiled for CPU execution. It checks prescribed valid proposals against independent graph scoring. The sequence construction uses seed 9019, finding the stored case at trial 2530. `wave_sequence_case.json` records all patterns/counts and the initial mathematical distances. `wave_checks.json` records the source function's FP32 distances used for reported results.

`build_wave_driver.py` extracts the actual wave controller and its helpers. It substitutes a controlled exact-local-acceptance proposal callback for the GPU optimizer. The controller itself is unchanged. `wave_driver_check.py` runs this controller with one worker and independently scores the entire expanded tree before and after. It does not establish which proposals an unconstrained native optimizer would choose.

Both CUDA sources were compiled, but GPU tests were not run because GPU 0 was occupied:

```bash
gcc -O2 -pthread -c snapshot/phylo2vec_fastme.c -o core.o
REVIEW_NVCC=/usr/local/cuda-12.8/bin/nvcc
"$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 recovery_profile_gpu.cu core.o -lpthread -o recovery_profile_gpu
"$REVIEW_PY" build_rna_gpu_check.py
"$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp rna_profile_gpu.cu -o rna_profile_gpu
```

The copied CUDA harnesses exercise the old raw-profile kernel path, not the new acceptance guard. They are useful compile checks here, and must not be described as native acceptance tests. The copied legacy `api_checks.py` and `representative_check.py` supply independent helper definitions to newer tests. Run the documented followup/profile tests instead of treating their old expectations as current failures.
