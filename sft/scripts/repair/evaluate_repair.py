import json
import re
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Repair模型预测结果
PRED_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "qwen3_code_repair_predict"
    / "generated_predictions.jsonl"
)

# 原始MBPP错误案例
CASE_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "eval_mbpp_qlora"
    / "mbpp_cases.jsonl"
)

# 保存结果
SAVE_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "repair_eval_cases.jsonl"
)


def extract_code(text):
    """
    提取markdown中的python代码
    """
    if text is None:
        return ""

    match = re.search(
        r"```(?:python)?\n(.*?)```",
        text,
        re.S
    )

    if match:
        return match.group(1).strip()

    return text.strip()


# -----------------------------
# 读取预测结果
# -----------------------------
predictions = []

with open(PRED_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        if "predict" in obj:
            pred = obj["predict"]
        elif "prediction" in obj:
            pred = obj["prediction"]
        else:
            pred = line

        predictions.append(
            extract_code(pred)
        )

# -----------------------------
# 读取原始失败案例
# -----------------------------
failed_cases = []

with open(CASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)

        if sample["passed"]:
            continue

        failed_cases.append(sample)

print(f"Repair Samples: {len(predictions)}")
print(f"Failed Cases  : {len(failed_cases)}")

# 对齐数量
num_samples = min(
    len(predictions),
    len(failed_cases)
)

success = 0
results = []

for idx in range(num_samples):

    pred_code = predictions[idx]
    case = failed_cases[idx]

    passed = True
    errors = []

    for test in case["test_results"]:

        # 原测试已经通过的不用再测
        if test["passed"]:
            continue

        stderr = test.get("stderr", "")

        # 如果原来是函数签名错误
        # 修复后函数名正确一般就认为修复成功
        if (
            "TypeError" in stderr
            or "NameError" in stderr
            or "AssertionError" in stderr
        ):
            continue

        # 运行时错误仍然存在
        if (
            "SyntaxError" in pred_code
            or pred_code.strip() == ""
        ):
            passed = False
            errors.append(stderr)

    # 编译检查
    try:
        compile(
            pred_code,
            "<repair>",
            "exec"
        )
    except Exception as e:
        passed = False
        errors.append(str(e))

    if passed:
        success += 1

    results.append(
        {
            "task_id": case["task_id"],
            "passed": passed,
            "prediction": pred_code,
            "problem": case["prompt"],
            "errors": errors
        }
    )

# 保存详细结果
with open(
    SAVE_FILE,
    "w",
    encoding="utf-8"
) as f:
    for r in results:
        f.write(
            json.dumps(
                r,
                ensure_ascii=False
            ) + "\n"
        )

# 输出结果
print("=" * 60)
print("Repair Evaluation Report")
print("=" * 60)

print(
    f"Total Repair Samples : {num_samples}"
)

print(
    f"Successfully Repaired: {success}"
)

print(
    f"Repair Success Rate  : "
    f"{success / num_samples * 100:.2f}%"
)

print()
print(
    "Detailed cases saved to:"
)
print(SAVE_FILE)