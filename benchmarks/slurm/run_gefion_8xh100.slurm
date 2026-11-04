#!/bin/bash
#SBATCH --account=iu_0148
#SBATCH --partition=defq
#SBATCH --reservation=iu_0148
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=%x_%j.slurm.out
# Phylo2Vec 8x NVIDIA H100 (80GB SXM5) Multi-GPU & Extreme-N Scaling Benchmark on Gefion dgx158.
# Submit ONLY via guarded script from the bundle directory on the Gefion login node:
#   RESERVATION=iu_0148 bash ./gefion_sbatch.sh --job-name=p2v_8xh100 run_gefion_8xh100.slurm

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}" || { echo "FATAL: cd to SLURM_SUBMIT_DIR failed"; exit 2; }
set -euo pipefail

mkdir -p "$SLURM_SUBMIT_DIR/logs"
BIN="$SLURM_SUBMIT_DIR/phylo2vec_cuda_bench_sm90"
[ -x "$BIN" ] || { echo "FATAL: Compiled binary not found or not executable at $BIN" >&2; exit 2; }

echo "=== JOB=p2v_8xh100 HOST=$(hostname) SLURM_JOB_ID=${SLURM_JOB_ID:-local} GPUS=${CUDA_VISIBLE_DEVICES:-unset} START=$(date -Is) ==="

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN command: $BIN"
  exit 0
fi

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | awk 'NF>0{c++} END{print c+0}')
[ "$NGPU" -ge 8 ] || { echo "FATAL: Expected 8 visible H100 GPUs, found $NGPU" >&2; exit 4; }

"$BIN" 2>&1 | tee "$SLURM_SUBMIT_DIR/logs/p2v_8xh100_${SLURM_JOB_ID:-local}.log"
echo "=== JOB=p2v_8xh100 COMPLETE $(date -Is) ==="

