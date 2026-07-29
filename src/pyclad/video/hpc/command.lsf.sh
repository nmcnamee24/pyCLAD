#!/usr/bin/env bash
#BSUB -J pyclad-command
#BSUB -q h100
#BSUB -gpu "num=1:gmodel=NVIDIAH100PCIe"
#BSUB -n 8
#BSUB -R "rusage[mem=8192]"
#BSUB -W 48:00
#BSUB -o pyvad_hpc/logs/command.%J.out
#BSUB -e pyvad_hpc/logs/command.%J.err

set -euo pipefail

HPC_ROOT="${PYVAD_HPC_ROOT:-${HOME}/pyvad_hpc}"
RUN_ID="${PYCLAD_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_DIR="${HPC_ROOT}/results/${RUN_ID}"
COMMIT_FILE="${HPC_ROOT}/code/PYCLAD_COMMIT_SHA"

test -f "${COMMIT_FILE}"
test -d "${HPC_ROOT}/data/command_ucf_crime/UCF-Crime"
test -x "${HPC_ROOT}/env/bin/python"
mkdir -p "${RESULT_DIR}"

export PYCLAD_COMMIT_SHA
PYCLAD_COMMIT_SHA="$(tr -d '[:space:]' < "${COMMIT_FILE}")"
export PYTHONHASHSEED="${PYCLAD_SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME="${HPC_ROOT}/cache/torch"

source "${HPC_ROOT}/env/bin/activate"
nvidia-smi > "${RESULT_DIR}/command.nvidia-smi.txt"
python -m pip freeze > "${RESULT_DIR}/command.environment.txt"
python -m pyclad.video command \
  --data-root "${HPC_ROOT}/data/command_ucf_crime/UCF-Crime" \
  --strategy cumulative \
  --concepts Abuse,Arrest,Arson,Assault,Burglary,Explosion,Fighting,RoadAccidents,Robbery,Shooting,Shoplifting,Stealing,Vandalism \
  --videos-per-class 0 \
  --test-normal-videos 0 \
  --test-anomaly-videos 0 \
  --epochs 10 \
  --batch-size 256 \
  --hidden-dim 128 \
  --embedding-dim 128 \
  --memory-size 64 \
  --device cuda \
  --seed "${PYCLAD_SEED:-42}" \
  --output-json "${RESULT_DIR}/command.json"
