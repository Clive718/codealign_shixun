#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_PYTHON="/opt/conda/envs/env_A/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" tts/evaluate_cot_code.py \
  --model_path "${MODEL_PATH:-dpo/outputs/qwen15_code_full_dpo}" \
  --input_file "${INPUT_FILE:-/root/siton-tmp/assignment_A/mbpp/sanitized/test-00000-of-00001.parquet}" \
  --output_dir "${OUTPUT_DIR:-tts/outputs/cot_code_eval}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-768}" \
  "$@"
