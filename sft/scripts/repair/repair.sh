#!/bin/bash

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

echo "project_root= ${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0

echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES}"

python_bin=/opt/conda/envs/env_A/bin/python

echo "============================================================"
echo "Step1: Prepare Repair Dataset"
echo "============================================================"

${python_bin} \
${PROJECT_ROOT}/sft/scripts/prepare_code_repair_data.py

echo ""
echo "============================================================"
echo "Step2: Split Dataset"
echo "============================================================"

${python_bin} \
${PROJECT_ROOT}/sft/scripts/split_repair_dataset.py

echo ""
echo "============================================================"
echo "Step3: Train Repair Model"
echo "============================================================"

${python_bin} -m llamafactory.cli train \
${PROJECT_ROOT}/sft/configs/qwen3_code_repair_lora.yaml

echo ""
echo "============================================================"
echo "Step4: Generate Repaired Code"
echo "============================================================"

${python_bin} -m llamafactory.cli train \
${PROJECT_ROOT}/sft/configs/qwen3_code_repair_predict.yaml

echo ""
echo "============================================================"
echo "Step5: Execute Repaired Code"
echo "============================================================"

${python_bin} \
${PROJECT_ROOT}/sft/scripts/repair_execute.py

echo ""
echo "============================================================"
echo "Step6: Evaluate Repair Performance"
echo "============================================================"

${python_bin} \
${PROJECT_ROOT}/sft/scripts/repair_eval.py

echo ""
echo "============================================================"
echo "Repair Pipeline Finished"
echo "============================================================"