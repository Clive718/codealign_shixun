#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_DIR="${SOURCE_DIR:-py-dpo-v0.1}"
OUTPUT_DIR="${OUTPUT_DIR:-dpo/data}"

SEED="${SEED:-42}"
TEST_RATIO="${TEST_RATIO:-0.05}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MIN_RESPONSE_LEN="${MIN_RESPONSE_LEN:-20}"
PREVIEW_SIZE="${PREVIEW_SIZE:-10}"

cd "${PROJECT_ROOT}"

echo "[INFO] Project root       : ${PROJECT_ROOT}"
echo "[INFO] Python bin         : ${PYTHON_BIN}"
echo "[INFO] Source dir         : ${SOURCE_DIR}"
echo "[INFO] Output dir         : ${OUTPUT_DIR}"
echo "[INFO] Seed               : ${SEED}"
echo "[INFO] Test ratio         : ${TEST_RATIO}"
echo "[INFO] Max train samples  : ${MAX_TRAIN_SAMPLES}"
echo "[INFO] Min response len   : ${MIN_RESPONSE_LEN}"
echo "[INFO] Preview size       : ${PREVIEW_SIZE}"

"${PYTHON_BIN}" "dpo/scripts/prepare_dpo_data.py" \
  --source_dir "${SOURCE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --test_ratio "${TEST_RATIO}" \
  --seed "${SEED}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES}" \
  --min_response_len "${MIN_RESPONSE_LEN}" \
  --preview_size "${PREVIEW_SIZE}"

echo "[INFO] Done."
echo "[INFO] Generated files:"
echo "  - ${OUTPUT_DIR}/code_dpo_train.json"
echo "  - ${OUTPUT_DIR}/code_dpo_test.json"
echo "  - ${OUTPUT_DIR}/dataset_info.json"
echo "  - ${OUTPUT_DIR}/sample_preview.json"