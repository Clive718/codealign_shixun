import json
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PRED_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "qwen3_code_repair_predict"
    / "generated_predictions.jsonl"
)

SAVE_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "repair_execute_results.jsonl"
)

results = []

with open(PRED_FILE, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        sample = json.loads(line)

        code = sample["predict"]

        success = True
        error_message = ""

        try:
            with tempfile.NamedTemporaryFile(
                    suffix=".py",
                    delete=False,
                    mode="w",
                    encoding="utf-8"
            ) as tmp:

                tmp.write(code)
                tmp_path = tmp.name

            subprocess.run(
                ["python3", tmp_path],
                timeout=5,
                capture_output=True,
                text=True,
                check=True
            )

        except Exception as e:
            success = False
            error_message = str(e)

        result = {
            "id": idx,
            "success": success,
            "error": error_message
        }

        results.append(result)

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
            )
            + "\n"
        )

print("=" * 60)
print(f"Total Samples : {len(results)}")
print(
    f"Executable    : "
    f"{sum(x['success'] for x in results)}"
)
print(f"Saved to : {SAVE_FILE}")
print("=" * 60)