#!/usr/bin/env bash
#BSUB -J "nola-train-repair[1-3]%3"
#BSUB -q normal
#BSUB -gpu "num=1:gmodel=NVIDIAL40"
#BSUB -n 2
#BSUB -R "rusage[mem=8192]"
#BSUB -W 48:00
#BSUB -o pyvad_hpc/logs/nola-train-repair.%J.%I.out
#BSUB -e pyvad_hpc/logs/nola-train-repair.%J.%I.err

set -euo pipefail

HPC_ROOT="${PYVAD_HPC_ROOT:-${HOME}/pyvad_hpc}"
RUN_ID="${PYCLAD_RUN_ID:?Set PYCLAD_RUN_ID before submitting the array}"
MANIFEST="${HPC_ROOT}/code/src/pyclad/video/hpc/nola_train_cache_repairs.txt"
RESULT_DIR="${HPC_ROOT}/results/${RUN_ID}"
PROCESSED_ROOT="${HPC_ROOT}/data/nola/processed-train-paper"
STAGING_ROOT="${HPC_ROOT}/staging/nola-train"
COMMIT_FILE="${HPC_ROOT}/code/PYCLAD_COMMIT_SHA"

test -n "${LSB_JOBINDEX:-}"
test -f "${MANIFEST}"
test -f "${COMMIT_FILE}"
test -x "${HPC_ROOT}/env/bin/python"
record="$(sed -n "${LSB_JOBINDEX}p" "${MANIFEST}")"
test -n "${record}"
stage="${record%%/*}"
video_id="${record#*/}"
test -n "${stage}"
test -n "${video_id}"
source_root="${HPC_ROOT}/data/nola/NOLA/Train/${stage}"
source_video="${source_root}/${video_id}/video.mp4"
final_dir="${PROCESSED_ROOT}/${stage}/${video_id}"
array_stage_root="${STAGING_ROOT}/${LSB_JOBID}.${LSB_JOBINDEX}/${stage}"
stage_dir="${array_stage_root}/${video_id}"

test -f "${source_video}"
mkdir -p "${RESULT_DIR}/nola-train-repair" "${PROCESSED_ROOT}/${stage}" "${STAGING_ROOT}"

export PYCLAD_COMMIT_SHA
PYCLAD_COMMIT_SHA="$(tr -d '[:space:]' < "${COMMIT_FILE}")"
export PYTHONHASHSEED="${PYCLAD_SEED:-42}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME="${HPC_ROOT}/cache/torch"
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-2}"
export MKL_NUM_THREADS="${LSB_DJOB_NUMPROC:-2}"
export OPENBLAS_NUM_THREADS="${LSB_DJOB_NUMPROC:-2}"
export NUMEXPR_NUM_THREADS="${LSB_DJOB_NUMPROC:-2}"

set +eu
source /etc/profile
set -eu
type module >/dev/null 2>&1
module load gcc12/12.2.0
module load cuda12.3/toolkit/12.3.2
source "${HPC_ROOT}/env/bin/activate"
if python -m pyclad.video.hpc.validate_nola_cache \
  "${final_dir}" \
  --expected-frame-stride 1 \
  --expected-detector native-darknet-yolov4-csp \
  --expected-tracker deep-sort-realtime \
  > "${RESULT_DIR}/nola-train-repair/${stage}.${video_id}.validation.json" 2>/dev/null; then
  echo "Validated cache already exists for ${record}; skipping."
  exit 0
fi

if test -e "${stage_dir}"; then
  echo "Staging path already exists: ${stage_dir}" >&2
  exit 1
fi

python -m pyclad.video nola-preprocess \
  --data-root "${HPC_ROOT}/data/nola/NOLA" \
  --source-root "${source_root}" \
  --output-root "${array_stage_root}" \
  --video-ids "${video_id}" \
  --frame-stride 1 \
  --confidence-threshold 0.25 \
  --detector darknet \
  --darknet-binary "${HPC_ROOT}/tools/darknet/darknet" \
  --darknet-source-commit 59596d7880f6504768df41d6daa586f5cb2b932f \
  --darknet-data "${HPC_ROOT}/data/nola/darknet/coco.data" \
  --darknet-config "${HPC_ROOT}/data/nola/darknet/yolov4-csp.cfg" \
  --darknet-weights "${HPC_ROOT}/data/nola/darknet/yolov4-csp.weights" \
  --darknet-weights-sha256 019496affba568f7439e54797a1772657bb01126b707fbd93407c0b20c20dca1 \
  --darknet-names "${HPC_ROOT}/data/nola/darknet/coco.names" \
  --nms-threshold 0.45 \
  --tracker deepsort \
  --tracker-max-age 30 \
  --tracker-n-init 3 \
  --tracker-max-cosine-distance 0.2 \
  --tracker-nn-budget 100 \
  --tracker-embedder mobilenet \
  --device cuda \
  --seed "${PYCLAD_SEED:-42}" \
  --output-json "${RESULT_DIR}/nola-train-repair/${stage}.${video_id}.json"

python -m pyclad.video.hpc.validate_nola_cache \
  "${stage_dir}" \
  --expected-frame-stride 1 \
  --expected-detector native-darknet-yolov4-csp \
  --expected-tracker deep-sort-realtime \
  > "${RESULT_DIR}/nola-train-repair/${stage}.${video_id}.validation.json"

if test -e "${final_dir}"; then
  replaced="${STAGING_ROOT}/replaced-${stage}-${video_id}.${LSB_JOBID}.${LSB_JOBINDEX}"
  test ! -e "${replaced}"
  mv "${final_dir}" "${replaced}"
fi
mv "${stage_dir}" "${final_dir}"
nvidia-smi > "${RESULT_DIR}/nola-train-repair/${stage}.${video_id}.nvidia-smi.txt"
