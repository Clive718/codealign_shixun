#!/bin/bash

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)

echo "project_root= ${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES=0

echo "CUDA_VISIBLE_DEVICES= ${CUDA_VISIBLE_DEVICES}"

python - << EOF
import torch
print("torch.cuda.is_available()=", torch.cuda.is_available())
print("visible_device_count=", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0=", torch.cuda.get_device_name(0))
EOF

cd ${PROJECT_ROOT}

python sft/scripts/evaluate_repair.py