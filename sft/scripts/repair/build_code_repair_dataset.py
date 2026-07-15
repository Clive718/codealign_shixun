import json
from pathlib import Path

CASE_FILE = Path(
    "sft/outputs/eval_mbpp_qlora/mbpp_cases.jsonl"
)

SAVE_PATH = Path(
    "sft/data/code_repair_train.json"
)

dataset = []

with open(CASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)

        # 只保留失败样本
        if sample["passed"]:
            continue

        prompt = sample["prompt"]

        wrong_code = sample.get(
            "code",
            sample.get("predict", "")
        )

        reference = sample["reference_code"]

        error_msg = ""

        for t in sample["test_results"]:
            if not t["passed"]:
                error_msg += t.get("stderr", "")
                break

        instruction = (
            "Fix the Python code according to "
            "the problem description and execution feedback."
        )

        input_text = f"""
Problem:
{prompt}

Wrong Code:
{wrong_code}

Execution Feedback:
{error_msg}
"""

        dataset.append(
            {
                "instruction": instruction,
                "input": input_text,
                "output": reference
            }
        )

with open(
    SAVE_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        dataset,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"Generated {len(dataset)} repair samples."
)
print(
    f"Saved to {SAVE_PATH}"
)