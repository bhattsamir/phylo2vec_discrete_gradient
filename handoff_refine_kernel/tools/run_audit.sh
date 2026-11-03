#!/bin/bash
# Runs an audit suite inside a SCRATCH copy of audit_discrete_gradient_<rev>_chatgpt.
# usage: run_audit.sh DIR rev4|rev5
D=$1; R=$2
D=$(realpath "$D"); case "$D" in /home/sjbhatt/phylo2vec_discrete_gradient|/home/sjbhatt/phylo2vec_discrete_gradient/*) echo "refusing: run audits only in a copy outside the repo"; exit 2;; esac
cd "$D" || exit 1
PY=/home/sjbhatt/anaconda3/envs/rcpms-jax/bin/python
NVCC=/usr/local/cuda-12.8/bin/nvcc
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
step() { local out=$1; shift; "$@" > "$out" 2>&1; local rc=$?; printf "%-34s rc=%d\n" "$out" $rc; }
step _build_profile_library.log    $PY build_profile_library.py
step profile_checks.out            $PY profile_checks.py
step _build_acceptance.log         $PY build_acceptance_checks.py
step acceptance_checks.out         $PY acceptance_checks.py
[ $R = rev4 ] && step _find_wave_case.log $PY find_wave_case.py
step _build_wave_checks.log        $PY build_wave_checks.py
step wave_checks.out               $PY wave_checks.py
step _build_wave_driver.log        $PY build_wave_driver.py
if [ $R = rev5 ]; then
  step wave_fix_checks.out           $PY wave_fix_checks.py
  step refinement_cpu_check.out      $PY refinement_cpu_check.py
  step batch_refinement_cpu_check.out $PY batch_refinement_cpu_check.py
else
  step wave_driver_check.out         $PY wave_driver_check.py
fi
step _libfastme.log                gcc -O2 -pthread -shared -fPIC snapshot/phylo2vec_fastme.c -lm -o snapshot/libphylo2vec_fastme.so
step followup_api.out              $PY followup_api.py
step followup_boundary.out         $PY followup_boundary.py
step _asan_build.log               gcc -O2 -g -pthread -fsanitize=address,undefined -fno-sanitize-recover=all snapshot/verify_all.c snapshot/phylo2vec_fastme.c -lm -o verify_all_asan
ASAN_OPTIONS=detect_leaks=1 step verify_all_asan.out ./verify_all_asan
step _core_o.log                   gcc -O2 -pthread -c snapshot/phylo2vec_fastme.c -o core.o
step _nvcc_recovery.log            $NVCC -O2 -lineinfo -arch=sm_86 recovery_profile_gpu.cu core.o -lpthread -o recovery_profile_gpu
step _build_rna_gpu_check.log      $PY build_rna_gpu_check.py
step _nvcc_rna.log                 $NVCC -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp rna_profile_gpu.cu -o rna_profile_gpu
[ -f rna_profile_gpu_late.cu ] && step _nvcc_rna_late.log $NVCC -O2 -lineinfo -arch=sm_86 -Xcompiler -fopenmp rna_profile_gpu_late.cu -o rna_profile_gpu_late
echo DONE
