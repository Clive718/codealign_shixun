#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

"${SFT_DIR}/scripts/prepare_data.sh"
"${SFT_DIR}/scripts/train_qlora.sh"
"${SFT_DIR}/scripts/predict_qlora.sh"
"${SFT_DIR}/scripts/evaluate_qlora.sh"
