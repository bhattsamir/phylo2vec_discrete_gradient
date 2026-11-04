#!/bin/bash
# gefion_sbatch.sh: a guarded sbatch for a shared Gefion DGX node. Use it for EVERY submission to the node.
#
# Why: when jobs on the node together allocate its last CPUs, Slurm's CPU binding fails
# ("task_g_set_affinity: Invalid argument", "Slurmd could not execve job", exit 0:52) and Slurm DRAINS the node
# ("batch job complete failure"). A drained node blocks everyone's GPU work until DCAI support resumes it.
# A burst of jobs that all fail at once (wrong working dir, missing module, bad path) can drain it the same way.
#
# Usage, on a Gefion login node, from the folder you want as the job's working directory:
#   bash gefion_sbatch.sh [--allow-array] <sbatch options> job.slurm [script args]
# Settings (environment variables):
#   NODE         node to protect                      (default dgx158)
#   CAP          max CPUs allocated on NODE in total  (default 200, of 216; above ~208 binding fails)
#   RESERVATION  if set, CPUs of PENDING jobs in this reservation also count towards CAP
#   WAIT         seconds to wait before the post-submit check (default 60)
# Refuses (exit 2) unless:
#   - NODE is not DRAIN/DOWN/FAIL,
#   - the job states its CPUs (--cpus-per-task, or --cpus-per-gpu with --gpus) and is not --exclusive,
#   - allocated CPUs on NODE (+ pending in RESERVATION) + this job's CPUs <= CAP,
#   - an array has a %N concurrency limit and is submitted with --allow-array (i.e. after ONE task ran cleanly),
#   - sbatch --test-only accepts it.
# Then submits, waits WAIT s, and if NODE is draining cancels this job's pending tasks (exit 3).
# Exit codes: 0 submitted and node healthy, 2 refused (nothing submitted), 3 submitted but node draining.
set -uo pipefail
NODE=${NODE:-dgx158}
CAP=${CAP:-200}
WAIT=${WAIT:-60}
ALLOW_ARRAY=0
if [ "${1:-}" = "--allow-array" ]; then ALLOW_ARRAY=1; shift; fi
[ $# -ge 1 ] || { echo "usage: gefion_sbatch.sh [--allow-array] <sbatch options> job.slurm [script args]"; exit 2; }
die() { echo "REFUSED: $*" >&2; exit 2; }

state=$(scontrol show node "$NODE" | grep -o 'State=[A-Z+_]*')
[ -n "$state" ] || die "cannot read the state of $NODE (scontrol show node $NODE)"
case "$state" in *DRAIN*|*DOWN*|*FAIL*) die "$NODE is $state; ask DCAI support to resume it first" ;; esac

# This job's request: command-line options win over #SBATCH lines in the script (as in sbatch itself)
args="$*"
script=""; for a in "$@"; do case "$a" in *.slurm|*.sh|*.sbatch) script="$a"; break ;; esac; done
[ -z "$script" ] || [ -f "$script" ] || die "job script $script not found"
opt() {   # opt LONG [SHORT] -> value from the command line, else from the script header ("-c 8" needs the space)
    local pat v
    if [ -n "${2:-}" ]; then pat="(--$1[= ]|-$2 )"; else pat="(--$1[= ])"; fi
    v=$(printf ' %s\n' "$args" | grep -oE -- " $pat[^ ]+" | head -1 | sed -E "s/^ $pat//")
    [ -z "$v" ] && [ -n "$script" ] && v=$(grep -E "^#SBATCH +$pat" "$script" | head -1 | sed -E "s/^#SBATCH +$pat//" | awk '{print $1}')
    echo "$v"
}
flag() {  # flag LONG -> 1 if given on the command line or in the script header
    printf '%s\n' "$args" | grep -qE -- "--$1( |=|$)" && { echo 1; return; }
    [ -n "$script" ] && grep -qE "^#SBATCH +--$1( |=|$)" "$script" && { echo 1; return; }
    echo 0
}
[ "$(flag exclusive)" = 0 ] || die "--exclusive takes every CPU of the node, which is exactly what drains it"
cpt=$(opt cpus-per-task c)
cpg=$(opt cpus-per-gpu)
gpus=$(opt gpus G); gpus=${gpus##*:}
ntasks=$(opt ntasks n); ntasks=${ntasks:-1}
if [ -n "$cpt" ]; then per=$((cpt * ntasks))
elif [ -n "$cpg" ] && [ -n "$gpus" ]; then per=$((cpg * gpus))
else die "state the job's CPUs explicitly with --cpus-per-task (or --cpus-per-gpu with --gpus)"; fi
arr=$(opt array a)
conc=1
if [ -n "$arr" ] && ! echo "$arr" | grep -qE '^[0-9]+$'; then   # a single index (the "one task first" step) is always fine
    [ $ALLOW_ARRAY = 1 ] || die "array job: submit ONE task first (--array=<one index>), check it runs and writes output, then rerun with --allow-array"
    case "$arr" in
        *%*) conc=${arr##*%} ;;
        *)   die "array '$arr' has no %N concurrency limit" ;;
    esac
fi
need=$((per * conc))
alloc=$(scontrol show node "$NODE" | grep -oE 'AllocTRES=[^ ]*' | grep -oE 'cpu=[0-9]+' | cut -d= -f2); alloc=${alloc:-0}
pend=0
if [ -n "${RESERVATION:-}" ]; then
    pend=$(squeue -h -t PENDING --reservation="$RESERVATION" -o '%C' | awk '{s+=$1} END {print s+0}')
fi
echo "$NODE $state | allocated CPUs $alloc | pending in reservation ${RESERVATION:-<not counted>} $pend | this job up to $need (CPUs per task-set $per x concurrency $conc) | cap $CAP | gpus ${gpus:-0}"
[ $((alloc + pend + need)) -le "$CAP" ] || die "$alloc + $pend + $need CPUs > cap $CAP (the last CPUs of $NODE fail CPU binding and drain the node)"

sbatch --test-only "$@" 2>&1 | tail -1 | grep -q "to start at" || die "sbatch --test-only did not accept the job: $(sbatch --test-only "$@" 2>&1 | tail -1)"
out=$(sbatch --parsable "$@") || die "sbatch failed: $out"
jid=${out%%;*}
echo "submitted $jid; checking $NODE in $WAIT s"
sleep "$WAIT"
state=$(scontrol show node "$NODE" | grep -o 'State=[A-Z+_]*')
echo "$NODE $state"
sacct -j "$jid" -X -n -o JobID%18,State%12,Elapsed,ExitCode | head -8
case "$state" in *DRAIN*)
    echo "WARNING: $NODE is draining: cancelling the pending tasks of $jid. Tell DCAI support (support@dcai.dk)." >&2
    scancel -t PENDING "$jid"; exit 3 ;; esac
if sacct -j "$jid" -n -o State | grep -qE 'FAILED|NODE_FAIL'; then
    echo "WARNING: tasks of $jid already FAILED. Check the log, and cancel the rest (scancel $jid) before more fail." >&2
fi
exit 0
