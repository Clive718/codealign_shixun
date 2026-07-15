import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = (
    PROJECT_ROOT
    / "sft"
    / "data"
    / "code_repair_test.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "repair_predictions.jsonl"
)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for sample in data:
        item = {
            "instruction": sample["instruction"],
            "input": sample["input"],
            "label": sample["output"]
        }

        f.write(
            json.dumps(
                item,
                ensure_ascii=False
            )
            + "\n"
        )

print("=" * 60)
print("Repair prediction file generated.")
print(f"Samples : {len(data)}")
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 60)