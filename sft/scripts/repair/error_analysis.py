import json
from pathlib import Path
from collections import defaultdict

# ==========================
# Path Config
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

CASE_FILE = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "eval_mbpp_lora"
    / "mbpp_cases.jsonl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "sft"
    / "outputs"
    / "error_analysis_lora_report.txt"
)

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

report_file = open(REPORT_PATH, "w", encoding="utf-8")


def log(msg=""):
    print(msg)
    report_file.write(str(msg) + "\n")


# ==========================
# Statistics Containers
# ==========================
error_counter = defaultdict(int)
error_ids = defaultdict(list)
examples = {}

total_samples = 0


# ==========================
# Error Classification
# ==========================
def classify_error(sample):

    if sample["passed"]:
        return "Passed"

    code = sample.get("code", "")
    stderr = ""

    for t in sample.get("test_results", []):
        stderr += t.get("stderr", "")

    # ---------------- Syntax Error ----------------
    if not sample.get("syntax_ok", True):
        return "Syntax Error"

    if "SyntaxError" in stderr:
        return "Syntax Error"

    # ---------------- Function Signature Error ----------------
    signature_keywords = [
        "positional argument",
        "unexpected keyword argument",
        "missing 1 required positional argument",
        "takes 0 positional arguments",
        "takes 1 positional argument",
        "takes 2 positional arguments",
        "got an unexpected keyword argument",
    ]

    for keyword in signature_keywords:
        if keyword in stderr:
            return "Function Signature Error"

    # ---------------- IO Format Error ----------------
    if "input()" in code:
        return "Input/Output Format Error"

    if "print(" in code and "return" not in code:
        return "Input/Output Format Error"

    # ---------------- Runtime Error ----------------
    runtime_keywords = [
        "TypeError",  #类型错误
        "ValueError",   #值错误
        "IndexError",   #索引越界错误
        "KeyError",     #键错误
        "ZeroDivisionError",   #除零错误
        "AttributeError",   #属性错误
        "NameError",    #名称错误
        "RecursionError",   #递归深度错误
        "MemoryError",     #内存错误
        "OverflowError",   #数值溢出错误
    ]

    for keyword in runtime_keywords:
        if keyword in stderr:
            return "Runtime Error"

    # ---------------- Timeout Error ----------------
    timeout_keywords = [
        "Timeout",
        "Time Limit Exceeded",
        "timed out",
    ]

    for keyword in timeout_keywords:
        if keyword in stderr:
            return "Runtime Timeout"

    # ---------------- Boundary Condition Error ----------------
    if (
        sample["passed_tests"] > 0
        and sample["passed_tests"] < sample["total_tests"]
    ):
        return "Boundary Condition Error"

    # ---------------- Others ----------------
    return "Failed Test Cases"


# ==========================
# Read Cases
# ==========================
with open(CASE_FILE, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)

        total_samples += 1

        error_type = classify_error(sample)

        error_counter[error_type] += 1
        error_ids[error_type].append(sample["task_id"])

        if error_type not in examples:
            examples[error_type] = sample


# ==========================
# Report Header
# ==========================
log("=" * 70)
log("MBPP ERROR ANALYSIS REPORT")
log("=" * 70)

log(f"Total Samples : {total_samples}")
log()

order = [
    "Passed",
    "Syntax Error",
    "Function Signature Error",
    "Input/Output Format Error",
    "Runtime Error",
    "Runtime Timeout",
    "Boundary Condition Error",
    "Failed Test Cases",
]

for err in order:
    cnt = error_counter[err]

    if total_samples == 0:
        ratio = 0
    else:
        ratio = cnt / total_samples * 100

    log(
        f"{err:<30}"
        f"{cnt:>4} cases"
        f"{ratio:>10.2f}%"
    )

    if cnt > 0:
        ids = sorted(error_ids[err])
        log(f"Task IDs: {ids}")

# ==========================
# Example Cases
# ==========================
log()
log("=" * 70)
log("TYPICAL ERROR EXAMPLES")
log("=" * 70)

for err in order:

    if err == "Passed":
        continue

    if err not in examples:
        continue

    sample = examples[err]

    log()
    log(f"[{err}]")
    log("-" * 60)

    log("Task ID :")
    log(sample["task_id"])

    log("\nProblem :")
    log(sample.get("prompt", ""))

    log("\nPrediction :")
    log(sample.get("code", ""))

    if sample.get("test_results"):
        log("\nExample Error Message :")

        for result in sample["test_results"]:
            stderr = result.get("stderr", "")
            if stderr.strip():
                log(stderr[:1000])
                break

# ==========================
# Summary
# ==========================
log()
log("=" * 70)
log("SUMMARY")
log("=" * 70)

for err in order:
    if err == "Passed":
        continue

    cnt = error_counter[err]

    if cnt > 0:
        log(
            f"{err}: "
            f"{cnt} samples "
            f"({cnt / total_samples * 100:.2f}%)"
        )

log()
log(f"Analysis report saved to:")
log(REPORT_PATH)

report_file.close()