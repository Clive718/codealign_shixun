#!/bin/bash

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

echo "project_root= ${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

cd ${PROJECT_ROOT}

python -m llamafactory.cli train \
    sft/configs/qwen3_code_repair_lora.yaml