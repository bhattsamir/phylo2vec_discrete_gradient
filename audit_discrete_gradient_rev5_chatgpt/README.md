# Review 5 artifacts

Source snapshots and independent checks for [review 5](../REVIEW_discrete_gradient_rev5_chatgpt.md). Production files are read-only. `source_manifest.json` records hashes and `changes.diff` compares with review 4.

Run from this directory:

```bash
REVIEW_PY=/home/sjbhatt/anaconda3/envs/rcpms-jax/bin/python
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$REVIEW_PY" build_profile_library.py
"$REVIEW_PY" build_acceptance_checks.py
"$REVIEW_PY" acceptance_checks.py > acceptance_checks.out 2>&1
"$REVIEW_PY" build_wave_checks.py
"$REVIEW_PY" build_wave_driver.py
"$REVIEW_PY" wave_fix_checks.py > wave_fix_checks.out 2>&1
"$REVIEW_PY" refinement_cpu_check.py > refinement_cpu_check.out 2>&1
gcc -O2 -pthread -shared -fPIC snapshot/phylo2vec_fastme.c -lm -o snapshot/libphylo2vec_fastme.so
"$REVIEW_PY" followup_api.py > followup_api.out 2>&1
"$REVIEW_PY" followup_boundary.py > followup_boundary.out 2>&1
gcc -O2 -g -pthread -fsanitize=address,undefined -fno-sanitize-recover=all snapshot/verify_all.c snapshot/phylo2vec_fastme.c -lm -o verify_all_asan
ASAN_OPTIONS=detect_leaks=1 ./verify_all_asan > verify_all_asan.out 2>&1
```

The acceptance harness retains the production host acceptance calculation but replaces GPU transfers/refinement with host copies and independently supplied full-profile distances. It is not a native GPU test.

The wave harness extracts the production controller unchanged and substitutes a controlled exact-local-acceptance callback for GPU search. The callback proposes the same two swaps as review 4. It scores each against the outside profile supplied by the controller. Independent graph traversal scores the complete expanded tree. `wave_fix_checks.py` varies sibling traversal order and configured GPU count, checking exactly one accepted proposal and exact agreement of reported and actual BME changes.

The copied old `wave_driver_check.py` retains the previous failure expectation. It was run once against the initial unchanged harness interface, producing `original_wave_replay.out`, whose final assertion fails because the fix rejects the second move. The harness was then extended with trace output and a configured-GPU-count argument. Use `wave_fix_checks.py` for the current interface and passing regression.

`wave_checks.py` retains the independent mathematical example and source scheduler checks. Its joint-score increase describes what happens if both prescribed changes are applied. It does not mean the fixed controller applies both.

`refinement_cpu_check.py` emulates the current refinement kernel's independent per-thread accumulations and barrier-separated reduction on the CPU. Cooperative shared loading is represented by serial loading before use. It compares against a separate scalar pair-sum loop using the source distance function. Native GPU execution and GPU race detection are not covered.

The original sequence pattern/count construction is preserved in `wave_sequence_case.json`. To regenerate it, run `find_wave_case.py` using seed 9019. Helper files copied from prior reviews are not standalone current-failure tests unless listed above.

CUDA compile checks, performed without executing on either busy GPU:

```bash
gcc -O2 -pthread -c snapshot/phylo2vec_fastme.c -o core.o
REVIEW_NVCC=/usr/local/cuda-12.8/bin/nvcc
"$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 recovery_profile_gpu.cu core.o -lpthread -o recovery_profile_gpu
"$REVIEW_PY" build_rna_gpu_check.py
"$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp rna_profile_gpu.cu -o rna_profile_gpu
```

These CUDA harnesses include the complete current benchmark sources. They were compiled only. Their old raw-profile kernel tests are not tests of the new acceptance guard.

A late CUDA-only update is captured in `late_snapshot/bench_rnasim_gpu.cu`, with hashes in `late_source_manifest.json` and diff in `late_changes.diff`. The five tested host functions remain byte-identical, as recorded by `source_recheck.json`. Its new batch branch was checked separately:

```bash
"$REVIEW_PY" batch_refinement_cpu_check.py > batch_refinement_cpu_check.out 2>&1
"$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp rna_profile_gpu_late.cu -o rna_profile_gpu_late
```

The late-branch emulator omits scalar fallback execution and leaves its outputs NaN. It verifies exactly which pairs the batch branch should handle, then compares those outputs against independent scalar pair sums. The unchanged fallback was checked by `refinement_cpu_check.py`. Neither emulation is a native GPU runtime test.
