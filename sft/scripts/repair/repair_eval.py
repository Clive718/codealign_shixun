import json
from pathlib import Path
from difflib import SequenceMatcher

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PRED_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "qwen3_code_repair_predict"
    / "generated_predictions.jsonl"
)

total = 0
success = 0
similarities = []

with open(PRED_FILE, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)

        pred = sample["predict"]
        label = sample["label"]

        total += 1

        sim = SequenceMatcher(
            None,
            pred,
            label
        ).ratio()

        similarities.append(sim)

        if sim > 0.7:
            success += 1

avg_similarity = (
    sum(similarities)
    / len(similarities)
)

print("=" * 60)
print("Repair Evaluation Report")
print("=" * 60)

print(
    f"Total Repair Samples : {total}"
)

print(
    f"Successfully Repaired: {success}"
)

print(
    f"Repair Success Rate  : "
    f"{success/total*100:.2f}%"
)

print(
    f"Average Similarity   : "
    f"{avg_similarity*100:.2f}%"
)

print("=" * 60)