import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CASE_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "eval_mbpp_qlora"
    / "mbpp_cases.jsonl"
)

SAVE_PATH = (
    PROJECT_ROOT
    / "sft"
    / "data"
    / "code_repair_train.json"
)

repair_dataset = []

with open(CASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)

        # 只保留失败样本
        if sample["passed"]:
            continue

        problem = sample["prompt"]

        wrong_code = sample.get(
            "code",
            sample.get("predict", "")
        )

        correct_code = sample["reference_code"]

        feedback = ""

        for test in sample.get("test_results", []):
            if not test["passed"]:
                feedback = (
                    test.get("stderr", "")
                    or test.get("stdout", "")
                )
                break

        instruction = (
            "Fix the Python code according to the "
            "problem description and execution feedback."
        )

        input_text = f"""
Problem Description:
{problem}

Wrong Code:
{wrong_code}

Execution Feedback:
{feedback}
"""

        repair_dataset.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": correct_code
            }
        )

with open(
    SAVE_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        repair_dataset,
        f,
        ensure_ascii=False,
        indent=2
    )

print("=" * 60)
print(f"Generated {len(repair_dataset)} repair samples")
print(f"Saved to: {SAVE_PATH}")
print("=" * 60)