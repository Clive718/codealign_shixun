import json
import difflib
from pathlib import Path


# ==========================================================
# Path Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CASE_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "eval_mbpp_qlora"
    / "mbpp_cases.jsonl"
)

PREDICT_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "qwen3_code_repair_predict"
    / "generated_predictions.jsonl"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "code_repair_report.txt"
)


# ==========================================================
# Load Original Failed Cases
# ==========================================================

failed_cases = []

with open(CASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)

        if sample["passed"]:
            continue

        failed_cases.append(sample)


# ==========================================================
# Load Repair Predictions
# ==========================================================

repair_predictions = []

with open(PREDICT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        repair_predictions.append(json.loads(line))


print(f"Failed Cases      : {len(failed_cases)}")
print(f"Repair Predictions: {len(repair_predictions)}")


# ==========================================================
# Generate Report
# ==========================================================

success_count = 0
similarities = []

with open(REPORT_FILE, "w", encoding="utf-8") as report:

    report.write("=" * 80 + "\n")
    report.write("CODE REPAIR REPORT\n")
    report.write("=" * 80 + "\n\n")

    total_cases = min(
        len(failed_cases),
        len(repair_predictions)
    )

    for idx in range(total_cases):

        original = failed_cases[idx]
        repaired = repair_predictions[idx]

        task_id = original["task_id"]

        problem = original["prompt"]

        wrong_code = original.get(
            "code",
            original.get("predict", "")
        )

        repaired_code = repaired.get(
            "predict",
            ""
        )

        reference_code = original.get(
            "reference_code",
            ""
        )

        # --------------------------------------------------
        # collect first error message
        # --------------------------------------------------

        error_message = ""

        for test in original.get(
                "test_results",
                []
        ):
            if not test["passed"]:
                error_message = test.get(
                    "stderr",
                    ""
                )
                break

        # --------------------------------------------------
        # similarity
        # --------------------------------------------------

        similarity = difflib.SequenceMatcher(
            None,
            repaired_code.strip(),
            reference_code.strip()
        ).ratio()

        similarities.append(similarity)

        repair_success = similarity >= 0.8

        if repair_success:
            success_count += 1

        # --------------------------------------------------
        # write report
        # --------------------------------------------------

        report.write("\n")
        report.write("=" * 80 + "\n")
        report.write(f"Repair Case {idx + 1}\n")
        report.write("=" * 80 + "\n\n")

        report.write(f"Task ID: {task_id}\n\n")

        report.write(
            "Problem Description\n"
        )
        report.write("-" * 80 + "\n")
        report.write(problem + "\n\n")

        report.write(
            "Wrong Code\n"
        )
        report.write("-" * 80 + "\n")
        report.write(wrong_code + "\n\n")

        report.write(
            "Execution Feedback\n"
        )
        report.write("-" * 80 + "\n")
        report.write(error_message + "\n\n")

        report.write(
            "Repaired Code\n"
        )
        report.write("-" * 80 + "\n")
        report.write(repaired_code + "\n\n")

        report.write(
            "Reference Code\n"
        )
        report.write("-" * 80 + "\n")
        report.write(reference_code + "\n\n")

        report.write(
            f"Repair Success : {repair_success}\n"
        )

        report.write(
            f"Similarity     : "
            f"{similarity:.4f}\n"
        )

        report.write("\n")


    # ======================================================
    # Summary
    # ======================================================

    average_similarity = (
        sum(similarities)
        / len(similarities)
        if similarities else 0
    )

    repair_rate = (
        success_count
        / total_cases
        * 100
        if total_cases > 0
        else 0
    )

    report.write("\n")
    report.write("=" * 80 + "\n")
    report.write("SUMMARY\n")
    report.write("=" * 80 + "\n")

    report.write(
        f"Total Repair Samples : "
        f"{total_cases}\n"
    )

    report.write(
        f"Successfully Repaired: "
        f"{success_count}\n"
    )

    report.write(
        f"Repair Success Rate  : "
        f"{repair_rate:.2f}%\n"
    )

    report.write(
        f"Average Similarity   : "
        f"{average_similarity:.4f}\n"
    )


print("\n")
print("=" * 60)
print("Repair Report Generated")
print("=" * 60)
print(
    f"Total Samples : {total_cases}"
)
print(
    f"Repair Success: {success_count}"
)
print(
    f"Repair Rate   : {repair_rate:.2f}%"
)
print(
    f"Average Similarity: "
    f"{average_similarity:.4f}"
)
print()
print(
    f"Saved to:\n{REPORT_FILE}"
)