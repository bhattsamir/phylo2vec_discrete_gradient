# Follow-up review artifacts

See [the follow-up report](../REVIEW_discrete_gradient_followup_chatgpt.md). The snapshot records the current source tested. Earlier harnesses were copied from the first audit, preserving the original evidence there. Use the followup_* drivers below for changed behavior. Some copied legacy drivers deliberately expect the old failures and are not the current pass/fail tests.

All commands below write only inside this audit directory. Rerunning overwrites recorded audit outputs. No production source is edited.

    cd /home/sjbhatt/phylo2vec_discrete_gradient/audit_discrete_gradient_followup_chatgpt
    REVIEW_PY=/home/sjbhatt/anaconda3/envs/rcpms-jax/bin/python
    export PYTHONDONTWRITEBYTECODE=1
    export OPENBLAS_NUM_THREADS=1
    export NUMBA_NUM_THREADS=1
    gcc -O2 -pthread -c snapshot/phylo2vec_fastme.c -o core.o
    gcc -O2 -pthread -shared -fPIC snapshot/phylo2vec_fastme.c -lm -o snapshot/libphylo2vec_fastme.so
    gcc -O2 -g -pthread -fsanitize=address,undefined -fno-sanitize-recover=all snapshot/verify_all.c snapshot/phylo2vec_fastme.c -lm -o verify_all_asan
    ASAN_OPTIONS=detect_leaks=1 ./verify_all_asan > verify_all_asan.out 2>&1
    "$REVIEW_PY" followup_api.py > followup_api.out 2>&1
    "$REVIEW_PY" followup_boundary.py > followup_boundary.out 2>&1
    "$REVIEW_PY" followup_validation.py > followup_validation.out 2>&1
    "$REVIEW_PY" snapshot/verify_spr_coverage.py > verify_spr_coverage.out 2>&1
    gcc -O2 -pthread general_poison.c -lm -o general_poison
    ./general_poison > general_poison.out

followup_boundary.py changes the iteration limit and its terminal-hop guard only in an audit-generated library, allowing the original cap failure and additional exits to be exercised cheaply. followup_validation.py injects three independent faults immediately before the CUDA validator in an audit copy, then checks the corrected index arithmetic under UBSan. It uses the source's host SIMT implementation.

Representative profiles and unchanged numerical counterexamples:

    "$REVIEW_PY" extract_host_helpers.py
    g++ -O2 -std=c++17 representative_check.cpp -o representative_check
    ./representative_check > representative_check.out
    "$REVIEW_PY" representative_additive_check.py > representative_additive_check.out
    "$REVIEW_PY" small_clade_check.py > small_clade_check.out 2>&1

small_clade_check.py extracts and calls the actual current representative helpers for both widths. The additive-distance script independently scores the full and condensed trees. It also executes the original 14-leaf example before the 17-leaf example.

GPU checks, after confirming GPU 0 is available:

    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
    REVIEW_NVCC=/usr/local/cuda-12.8/bin/nvcc
    "$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp rnasim_checks.cu -o rnasim_checks
    ./rnasim_checks > rnasim_host_checks.out
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 ./rnasim_checks gpu > rnasim_gpu_checks.out 2>&1
    "$REVIEW_NVCC" -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp cuda_checks.cu core.o -lpthread -o cuda_checks
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 ./cuda_checks > cuda_checks.out 2>&1
    g++ -O2 -std=c++17 -x c++ snapshot/phylo2vec_cuda_8xh100.cu -x none core.o -lpthread -o cuda_host_verify
    ./cuda_host_verify > cuda_host_verify.out 2>&1

The plain gpu mode exercises the current production kernel. The inherited gpu-safe diagnostic mode is unnecessary for this recheck because the current kernel already uses the corrected ownership rule. No multi-GPU communication or native Metal execution was tested.
