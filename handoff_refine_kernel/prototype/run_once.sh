#!/bin/bash
# Single run of the headroom microbenchmark, <= 60 s wall in total.
# Picks the local GPU with the lowest memory use; records contention before/after and a 100 ms clock trace.
L=/tmp/claude-1001/-home-sjbhatt-phylo2vec-discrete-gradient/705be631-2b5e-4441-8b09-a77b1e6fee66/scratchpad/speedup/recover/headroom
cd "$L" || exit 1
date -Is > run_meta.txt
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,clocks.sm,power.draw --format=csv > pre_smi.txt
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv >> pre_smi.txt
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1)
echo "selected GPU $GPU" >> run_meta.txt
nvidia-smi -i "$GPU" --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,utilization.gpu,temperature.gpu --format=csv -lms 100 > smi_trace.csv 2>&1 &
SMI=$!
nvidia-smi pmon -i "$GPU" -s u -d 1 > pmon.txt 2>&1 &
PMON=$!
T0=$(date +%s.%N)
CUDA_VISIBLE_DEVICES=$GPU CUDA_MODULE_LOADING=EAGER timeout 58 ./headroom_bench > run.log 2>&1
echo "exit=$?" >> run.log
T1=$(date +%s.%N)
kill $SMI $PMON 2>/dev/null
echo "wall_seconds $(echo "$T1 - $T0" | bc)" >> run_meta.txt
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,clocks.sm,power.draw --format=csv > post_smi.txt
