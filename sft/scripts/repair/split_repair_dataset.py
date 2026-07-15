import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_FILE = (
    PROJECT_ROOT
    / "sft"
    / "data"
    / "code_repair_train.json"
)

TRAIN_SAVE_PATH = (
    PROJECT_ROOT
    / "sft"
    / "data"
    / "code_repair_train_split.json"
)

VALID_SAVE_PATH = (
    PROJECT_ROOT
    / "sft"
    / "data"
    / "code_repair_valid.json"
)

TEST_SAVE_PATH = (
    PROJECT_ROOT
    / "sft"
    / "data"
    / "code_repair_test.json"
)

# 固定随机种子，保证实验可复现
random.seed(42)

with open(DATA_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# 打乱数据
random.shuffle(data)

# 80%训练，10%验证，10%测试
train_end = int(len(data) * 0.8)
valid_end = int(len(data) * 0.9)

train_data = data[:train_end]
valid_data = data[train_end:valid_end]
test_data = data[valid_end:]

# 保存训练集
with open(
    TRAIN_SAVE_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        train_data,
        f,
        indent=2,
        ensure_ascii=False
    )

# 保存验证集
with open(
    VALID_SAVE_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        valid_data,
        f,
        indent=2,
        ensure_ascii=False
    )

# 保存测试集
with open(
    TEST_SAVE_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        test_data,
        f,
        indent=2,
        ensure_ascii=False
    )

print("=" * 50)
print(f"Total Samples : {len(data)}")
print(f"Train Samples : {len(train_data)}")
print(f"Valid Samples : {len(valid_data)}")
print(f"Test Samples  : {len(test_data)}")
print("=" * 50)

print(f"Train saved to : {TRAIN_SAVE_PATH}")
print(f"Valid saved to : {VALID_SAVE_PATH}")
print(f"Test saved to  : {TEST_SAVE_PATH}")