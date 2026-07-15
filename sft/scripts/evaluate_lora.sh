#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${SFT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" sft/scripts/evaluate_code_predictions.py \
  --predictions "${PREDICTIONS:-sft/outputs/qwen3_code_lora_predict_epo500/generated_predictions.jsonl}" \
  --mbpp_dir "${MBPP_DIR:-/root/siton-tmp/assignment_A/mbpp}" \
  --config "${MBPP_CONFIG:-sanitized}" \
  --split "${MBPP_SPLIT:-test}" \
  --output_dir "${EVAL_DIR:-sft/outputs/eval_mbpp_lora_epo500}" \
  --limit "${LIMIT:-0}" \
  --test_timeout "${TEST_TIMEOUT:-5.0}" \
  --memory_mb "${MEMORY_MB:-1024}"