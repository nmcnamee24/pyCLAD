#!/usr/bin/env bash
#BSUB -J pyclad-nola-score
#BSUB -q normal
#BSUB -n 8
#BSUB -R "rusage[mem=8192]"
#BSUB -W 48:00
#BSUB -o pyvad_hpc/logs/nola-score.%J.out
#BSUB -e pyvad_hpc/logs/nola-score.%J.err

set -euo pipefail

HPC_ROOT="${PYVAD_HPC_ROOT:-${HOME}/pyvad_hpc}"
RUN_ID="${PYCLAD_RUN_ID:?Set PYCLAD_RUN_ID before submitting NOLA scoring}"
MANIFEST="${HPC_ROOT}/jobs/nola_test_ids.txt"
RESULT_DIR="${HPC_ROOT}/results/${RUN_ID}"
PROCESSED_ROOT="${HPC_ROOT}/data/nola/processed"
COMMIT_FILE="${HPC_ROOT}/code/PYCLAD_COMMIT_SHA"

test -f "${MANIFEST}"
test -f "${COMMIT_FILE}"
test -x "${HPC_ROOT}/env/bin/python"
mkdir -p "${RESULT_DIR}/nola-cache-validation"

export PYCLAD_COMMIT_SHA
PYCLAD_COMMIT_SHA="$(tr -d '[:space:]' < "${COMMIT_FILE}")"
export PYTHONHASHSEED="${PYCLAD_SEED:-42}"
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-8}"
export MKL_NUM_THREADS="${LSB_DJOB_NUMPROC:-8}"

source "${HPC_ROOT}/env/bin/activate"
while IFS= read -r video_id; do
  test -n "${video_id}"
  python -m pyclad.video.hpc.validate_nola_cache \
    "${PROCESSED_ROOT}/${video_id}" \
    --expected-frame-stride 1 \
    > "${RESULT_DIR}/nola-cache-validation/${video_id}.json"
done < "${MANIFEST}"

python -m pip freeze > "${RESULT_DIR}/nola.environment.txt"
python -m pyclad.video nola \
  --data-root "${HPC_ROOT}/data/nola/NOLA" \
  --processed-test-root "${PROCESSED_ROOT}" \
  --ground-truth "${HPC_ROOT}/data/nola/gt.txt" \
  --strategy cumulative \
  --stages M-Train,Train0,Train1,Train2,Train3,Train4,Train5,Train6,Train7,Train8,Train9 \
  --frame-stride 30 \
  --videos-per-stage 0 \
  --frames-per-video 0 \
  --neighbors 5 \
  --odit \
  --drift 7 \
  --seed "${PYCLAD_SEED:-42}" \
  --output-json "${RESULT_DIR}/nola.json"
