# Weighted-profile recheck artifacts

See [the report](../REVIEW_discrete_gradient_rev3_chatgpt.md). All source copies and diagnostic builds here are isolated from the implementation. source_manifest.json records captured source hashes. changes.diff compares this snapshot with the preceding review.

Run from this directory. Commands overwrite only audit outputs. The copied legacy representative_check.py and representative_additive_check.py retain the old mathematical examples. The current-profile recheck is profile_checks.py, which calls the newly extracted source helpers.

    cd /home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_rev3_chatgpt
    REVIEW_PY=/home/sjbhatt/anaconda3/envs/rcpms-jax/bin/python
    export OPENBLAS_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1
    export PYTHONDONTWRITEBYTECODE=1
    "$REVIEW_PY" build_profile_library.py
    "$REVIEW_PY" profile_checks.py > profile_checks.out 2>&1
    gcc -O2 -pthread -c snapshot/phylo2vec_fastme.c -o core.o
    gcc -O2 -pthread -shared -fPIC snapshot/phylo2vec_fastme.c -lm -o snapshot/libphylo2vec_fastme.so
    gcc -O2 -g -pthread -fsanitize=address,undefined -fno-sanitize-recover=all snapshot/verify_all.c snapshot/phylo2vec_fastme.c -lm -o verify_all_asan
    ASAN_OPTIONS=detect_leaks=1 ./verify_all_asan > verify_all_asan.out 2>&1
    "$REVIEW_PY" followup_api.py > followup_api.out 2>&1
    "$REVIEW_PY" followup_boundary.py > followup_boundary.out 2>&1

build_profile_library.py extracts the actual clade/outside builders and host weighted-distance helper without changing their bodies. profile_checks.py checks those via ctypes. Small-tree reference scores use graph distances. The large additive-distance example uses an independent linear-time split-metric Pauplin recursion, checked against graph scoring on the small cases.

GPU checks, after confirming GPU 0 is available:

    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
    REVIEW_NVCC=/usr/local/cuda-12.8/bin/nvcc
    "$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 recovery_profile_gpu.cu core.o -lpthread -o recovery_profile_gpu
    CUDA_VISIBLE_DEVICES=0 ./recovery_profile_gpu > recovery_profile_gpu.out 2>&1
    "$REVIEW_PY" build_rna_gpu_check.py
    "$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp rna_profile_gpu.cu -o rna_profile_gpu
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 ./rna_profile_gpu > rna_profile_gpu.out 2>&1

The GPU harnesses include the complete current benchmark source under renamed main functions and call the actual weighted-distance kernels. They do not run the benchmark main programs. The recovery harness initializes the source HLD oracle and supplies root distances for the explicit adversarial split metric. The sequence harness generates small synthetic packed sequence inputs and independently evaluates weighted pair distances on the CPU.

The large counterexample demonstrates the capped profile's objective error. It does not assert that a complete scheduler run produces that partition or accepts that specific move.
