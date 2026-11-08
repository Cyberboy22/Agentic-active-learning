#!/bin/bash
#SBATCH --job-name=aal-screen
#SBATCH --array=0-23%8
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/aal-%A_%a.out
#
# One array task per (target, seed) pair.
#
# The point of this file is that the checkpoint/resume claim in the README is
# testable rather than aspirational. Each task runs a bounded slice of a
# campaign with --stop-after-rounds and exits; the next task along resumes from
# the checkpoint and continues. A campaign longer than the queue's wall-time
# limit therefore completes across several jobs and produces exactly the result
# an uninterrupted run would have -- there is a test asserting that equality.
#
#   mkdir -p logs && sbatch slurm/array_campaign.sbatch

set -euo pipefail

TARGETS=(egfr cdk2 cmet)
SEEDS=(0 1 2 3 4 5 6 7)

T=${TARGETS[$((SLURM_ARRAY_TASK_ID / ${#SEEDS[@]}))]}
S=${SEEDS[$((SLURM_ARRAY_TASK_ID % ${#SEEDS[@]}))]}
OUT="runs/${T}_seed${S}"

module load Python/3.11 2>/dev/null || true
source "${VENV:-.venv}/bin/activate" 2>/dev/null || true

echo "task ${SLURM_ARRAY_TASK_ID}: target=${T} seed=${S} -> ${OUT}"
mkdir -p "${OUT}"

# Resume automatically when a checkpoint is already there.
RESUME=""
[ -f "${OUT}/checkpoint.json" ] && RESUME="--resume"

srun python run_screening.py \
    --target "${T}" \
    --planner cascade \
    --budget 400 \
    --seed "${S}" \
    --out "${OUT}" \
    --stop-after-rounds 6 \
    ${RESUME}

echo "done; resubmit this array to continue any campaign that has budget left"
