#!/usr/bin/env python3
"""Evaluate a model on Python code tasks with CoT, Self-Consistency, Best-of-N, and Reflexion."""


from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


try:
    from cot_code_example import build_chat_messages, extract_final_code, normalize_text
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from cot_code_example import build_chat_messages, extract_final_code, normalize_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PROJECT_ROOT.parent
DEFAULT_MODEL = PROJECT_ROOT / "dpo" / "outputs" / "qwen15_code_full_dpo"
DEFAULT_INPUT = WORK_ROOT / "mbpp" / "sanitized" / "test-00000-of-00001.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tts" / "outputs" / "cot_code_eval"

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+(?:\.\d+)?|==|!=|<=|>=|[-+*/%]=?|[(){}\[\].,:]")
FUNC_NAME_PATTERNS = [
    re.compile(r"assert\s+([A-Za-z_][A-Za-z_0-9]*)\s*\("),
    re.compile(r"Write a python function\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(", re.IGNORECASE),
    re.compile(r"Write a function\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(", re.IGNORECASE),
    re.compile(r"def\s+([A-Za-z_][A-Za-z_0-9]*)\s*\("),
]
DEF_RE = re.compile(r"def\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(")
THIRD_PARTY_IMPORT_RE = re.compile(
    r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z_0-9]*)",
    re.MULTILINE,
)

ALLOWED_IMPORT_ROOTS = {
    "math",
    "re",
    "itertools",
    "collections",
    "functools",
    "heapq",
    "bisect",
    "string",
    "typing",
    "fractions",
    "decimal",
    "statistics",
    "operator",
    "random",
    "datetime",
    "sys",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--input_file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--test_timeout", type=float, default=5.0)
    parser.add_argument("--memory_mb", type=int, default=1024)
    parser.add_argument("--include_examples", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")

    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument(
        "--selection_mode",
        choices=["single", "best_of_n", "self_consistency"],
        default="single",
    )

    parser.add_argument("--enable_reflection", action="store_true")
    parser.add_argument("--reflection_rounds", type=int, default=1)
    return parser.parse_args()


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value)]
    return []


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    except Exception:
        try:
            import pandas as pd
            return pd.read_parquet(path).to_dict("records")
        except Exception as exc:
            raise RuntimeError(f"Failed to read parquet file: {path}") from exc


def load_rows(path: Path) -> list[Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else [data]
    if path.suffix == ".parquet":
        return read_parquet(path)
    return [block.strip() for block in path.read_text(encoding="utf-8").split("\n\n") if block.strip()]


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = normalize_text(row.get(key))
        if value:
            return value
    return ""


def extract_expected_function_name(task_text: str, reference_code: str = "") -> str:
    for pattern in FUNC_NAME_PATTERNS:
        match = pattern.search(task_text)
        if match:
            return match.group(1)
    match = DEF_RE.search(reference_code or "")
    if match:
        return match.group(1)
    return ""


def add_task_constraints(task: str) -> str:
    task = normalize_text(task)
    expected_name = extract_expected_function_name(task)
    extra = [
        "Important constraints:",
        "- Preserve the exact function name required by the task.",
        "- Use only Python standard library.",
        "- Do not use third-party packages.",
        "- Do not print examples or extra explanations in the final code.",
        "- Return only the smallest correct executable solution.",
    ]
    if expected_name:
        extra.append(f"- Required function name: {expected_name}")
    return task + "\n\n" + "\n".join(extra)


def row_to_task(row: Any) -> str:
    if isinstance(row, str):
        base_task = normalize_text(row)
        return add_task_constraints(base_task)
    if not isinstance(row, dict):
        base_task = normalize_text(row)
        return add_task_constraints(base_task)

    prompt = first_text(row, ("prompt", "instruction", "question", "task", "text"))
    input_text = first_text(row, ("input", "query"))
    if prompt and input_text:
        task = f"{prompt}\n\n{input_text}"
    else:
        task = prompt or input_text

    tests = listify(row.get("test_list"))
    if tests:
        task = f"{task}\n\nYour code should pass these tests:\n" + "\n".join(tests)
    return add_task_constraints(task)


def row_to_reference(row: Any) -> str:
    if not isinstance(row, dict):
        return ""
    return first_text(row, ("output", "chosen", "answer", "reference", "label", "code"))


def row_to_tests(row: Any) -> tuple[str, list[str]]:
    if not isinstance(row, dict):
        return "", []
    setup_parts = []
    setup_parts.extend(listify(row.get("test_imports")))
    setup = first_text(row, ("test_setup_code", "setup_code"))
    if setup:
        setup_parts.append(setup)
    tests = listify(row.get("test_list"))
    tests.extend(listify(row.get("challenge_test_list")))
    return "\n".join(setup_parts), tests


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def code_tokens(code: str) -> list[str]:
    return TOKEN_RE.findall(code)


def token_f1(prediction: str, reference: str) -> float | None:
    if not reference:
        return None
    pred_tokens = code_tokens(prediction)
    ref_tokens = code_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    overlap = sum((pred_counter & ref_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def format_chat(
    tokenizer: Any,
    task: str,
    include_examples: bool = False,
    enable_thinking: bool = False,
) -> str:
    messages = build_chat_messages(task, include_examples=include_examples)
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    system = messages[0]["content"]
    user = messages[1]["content"]
    return f"{system}\n\nUser:\n{user}\n\nAssistant:\n"


def model_input_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def build_generation_kwargs(args: argparse.Namespace, tokenizer: Any) -> dict[str, Any]:
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    return generation_kwargs


def generate_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.model_path.exists():
        raise FileNotFoundError(f"Missing model directory: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    outputs: list[str] = []
    generation_kwargs = build_generation_kwargs(args, tokenizer)

    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        prompts = [
            format_chat(
                tokenizer,
                row["task"],
                include_examples=args.include_examples,
                enable_thinking=args.enable_thinking,
            )
            for row in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_tokens,
        ).to(model_input_device(model))

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **generation_kwargs)

        response_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        outputs.extend(normalize_text(text) for text in tokenizer.batch_decode(response_ids, skip_special_tokens=True))
        print(f"generated {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def generate_n_outputs_for_task(row: dict[str, Any], args: argparse.Namespace, n: int) -> list[str]:
    if n <= 1:
        return generate_outputs([row], args)

    sample_args = argparse.Namespace(**vars(args))
    if sample_args.temperature <= 0:
        sample_args.temperature = 0.6

    repeated_rows = [row for _ in range(n)]
    return generate_outputs(repeated_rows, sample_args)


def limit_resources(memory_mb: int, timeout: float) -> None:
    try:
        import resource
        memory_bytes = memory_mb * 1024 * 1024
        cpu_seconds = max(1, int(timeout) + 1)
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    except Exception:
        return


def run_one_assert(code: str, setup: str, test: str, args: argparse.Namespace) -> dict[str, Any]:
    runner = "\n\n".join(part for part in ("import faulthandler\nfaulthandler.enable()", setup, code, test) if part.strip())
    with tempfile.TemporaryDirectory(prefix="cot_code_eval_") as tmpdir:
        path = Path(tmpdir) / "candidate_test.py"
        path.write_text(runner + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["HOME"] = tmpdir
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
                timeout=args.test_timeout,
                preexec_fn=lambda: limit_resources(args.memory_mb, args.test_timeout) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "error_type": "timeout", "stderr": str(exc)}
    return {
        "passed": result.returncode == 0,
        "error_type": "" if result.returncode == 0 else "runtime_error",
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-2000:],
    }


def remove_disallowed_imports(code: str) -> str:
    cleaned_lines = []
    for line in code.splitlines():
        match = THIRD_PARTY_IMPORT_RE.match(line)
        if not match:
            cleaned_lines.append(line)
            continue
        root = match.group(1).split(".")[0]
        if root in ALLOWED_IMPORT_ROOTS:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def get_defined_functions(code: str) -> list[str]:
    return DEF_RE.findall(code)


def add_function_wrapper_if_needed(code: str, expected_name: str) -> str:
    if not code.strip() or not expected_name:
        return code
    defined = get_defined_functions(code)
    if expected_name in defined:
        return code
    if not defined:
        return code
    first_defined = defined[0]
    if first_defined == expected_name:
        return code
    wrapper = f"\n\n\ndef {expected_name}(*args, **kwargs):\n    return {first_defined}(*args, **kwargs)\n"
    return code.rstrip() + wrapper


def cleanup_generated_code(code: str, expected_name: str) -> str:
    code = normalize_text(code)
    code = remove_disallowed_imports(code)
    code = add_function_wrapper_if_needed(code, expected_name)
    return normalize_text(code)


def build_reflection_task(row: dict[str, Any], code: str, error_message: str) -> str:
    expected_name = row.get("expected_function_name", "")
    reflection_parts = [
        "You previously wrote code for this Python task, but it failed.",
        "",
        "Original task:",
        row["task"],
        "",
        "Previous final code:",
        "```python",
        code,
        "```",
        "",
        "Execution feedback:",
        error_message[:1200],
        "",
        "Please fix the code.",
        "- Keep the exact required function name.",
        "- Use only Python standard library.",
        "- Output the answer in the same format: Reasoning, Key steps, Final code.",
    ]
    if expected_name:
        reflection_parts.append(f"- Required function name: {expected_name}")
    return "\n".join(reflection_parts).strip()


def pick_primary_error(test_results: list[dict[str, Any]]) -> str:
    for item in test_results:
        stderr = normalize_text(item.get("stderr", ""))
        if stderr:
            return stderr
    return "Unknown runtime failure."


def score_case_once(row: dict[str, Any], response: str, args: argparse.Namespace) -> dict[str, Any]:
    expected_name = row.get("expected_function_name", "")
    code = cleanup_generated_code(extract_final_code(response), expected_name)
    reference_code = extract_final_code(row["reference"])
    setup = row["test_setup"]
    tests = row["tests"]
    test_results = [run_one_assert(code, setup, test, args) for test in tests]
    passed_tests = sum(1 for item in test_results if item["passed"])
    exact_match = code == reference_code if reference_code else None
    defined_functions = get_defined_functions(code)
    signature_mismatch = bool(expected_name) and expected_name not in defined_functions

    return {
        "index": row["index"],
        "task": row["task"],
        "reference": row["reference"],
        "response": response,
        "final_code": code,
        "reference_code": reference_code,
        "expected_function_name": expected_name,
        "defined_functions": defined_functions,
        "signature_mismatch": signature_mismatch,
        "syntax_ok": syntax_ok(code),
        "exact_match": exact_match,
        "token_f1": token_f1(code, reference_code),
        "passed": passed_tests == len(tests) and len(tests) > 0,
        "passed_tests": passed_tests,
        "total_tests": len(tests),
        "test_results": test_results,
        "used_reflection": False,
        "reflection_rounds_used": 0,
    }


def candidate_score(case: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        1 if case["syntax_ok"] else 0,
        0 if case["signature_mismatch"] else 1,
        case["passed_tests"],
        1 if case["final_code"].strip() else 0,
    )


def choose_best_of_n(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No candidates to choose from.")
    ranked = sorted(candidates, key=candidate_score, reverse=True)
    return ranked[0]


def canonicalize_code_for_vote(code: str) -> str:
    code = normalize_text(code)
    lines = [line.rstrip() for line in code.splitlines()]
    compact = "\n".join(line for line in lines if line.strip())
    return compact


def choose_self_consistent(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], float]:
    if not candidates:
        raise ValueError("No candidates to choose from.")

    buckets: dict[str, list[dict[str, Any]]] = {}
    for cand in candidates:
        key = canonicalize_code_for_vote(cand["final_code"])
        buckets.setdefault(key, []).append(cand)

    ranked_groups = sorted(
        buckets.values(),
        key=lambda group: (len(group), candidate_score(choose_best_of_n(group))),
        reverse=True,
    )
    best_group = ranked_groups[0]
    chosen = choose_best_of_n(best_group)
    consistency = len(best_group) / len(candidates)
    return chosen, consistency


def run_reflection_rounds(row: dict[str, Any], case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if case["passed"]:
        return case
    if not args.enable_reflection:
        return case

    best_case = case
    current_case = case

    for round_idx in range(args.reflection_rounds):
        error_message = pick_primary_error(current_case["test_results"])
        reflection_task = build_reflection_task(row, current_case["final_code"], error_message)
        retry_rows = [{"task": reflection_task}]
        retry_responses = generate_outputs(retry_rows, args)
        retry_case = score_case_once(row, retry_responses[0], args)
        retry_case["used_reflection"] = True
        retry_case["reflection_rounds_used"] = round_idx + 1
        retry_case["reflection_error_message"] = error_message

        if candidate_score(retry_case) >= candidate_score(best_case):
            best_case = retry_case

        current_case = retry_case
        if best_case["passed"]:
            break

    return best_case


def evaluate_one_row_with_strategy(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.selection_mode == "single" or args.num_samples <= 1:
        responses = generate_outputs([row], args)
        case = score_case_once(row, responses[0], args)
        case = run_reflection_rounds(row, case, args)
        case["selection_mode"] = "single"
        case["num_samples"] = 1
        case["self_consistency_rate"] = None
        case["num_candidates"] = 1
        return case

    responses = generate_n_outputs_for_task(row, args, args.num_samples)
    candidates = [score_case_once(row, response, args) for response in responses]

    if args.selection_mode == "best_of_n":
        chosen = choose_best_of_n(candidates)
        chosen = run_reflection_rounds(row, chosen, args)
        chosen["selection_mode"] = "best_of_n"
        chosen["num_samples"] = args.num_samples
        chosen["self_consistency_rate"] = None
        chosen["num_candidates"] = len(candidates)
        chosen["candidate_summaries"] = [
            {
                "syntax_ok": c["syntax_ok"],
                "signature_mismatch": c["signature_mismatch"],
                "passed_tests": c["passed_tests"],
                "total_tests": c["total_tests"],
            }
            for c in candidates
        ]
        return chosen

    if args.selection_mode == "self_consistency":
        chosen, consistency = choose_self_consistent(candidates)
        chosen = run_reflection_rounds(row, chosen, args)
        chosen["selection_mode"] = "self_consistency"
        chosen["num_samples"] = args.num_samples
        chosen["self_consistency_rate"] = consistency
        chosen["num_candidates"] = len(candidates)
        chosen["candidate_summaries"] = [
            {
                "syntax_ok": c["syntax_ok"],
                "signature_mismatch": c["signature_mismatch"],
                "passed_tests": c["passed_tests"],
                "total_tests": c["total_tests"],
            }
            for c in candidates
        ]
        return chosen

    raise ValueError(f"Unsupported selection mode: {args.selection_mode}")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(cases: list[dict[str, Any]], args: argparse.Namespace, elapsed_seconds: float) -> dict[str, Any]:
    total = len(cases)
    syntax_pass = sum(1 for case in cases if case["syntax_ok"])
    with_tests = [case for case in cases if case["total_tests"] > 0]
    with_exact = [case for case in cases if case["exact_match"] is not None]
    f1_values = [case["token_f1"] for case in cases if case["token_f1"] is not None]
    passed_tasks = sum(1 for case in with_tests if case["passed"])
    total_tests = sum(case["total_tests"] for case in with_tests)
    passed_tests = sum(case["passed_tests"] for case in with_tests)
    signature_mismatches = sum(1 for case in cases if case["signature_mismatch"])
    used_reflection = sum(1 for case in cases if case.get("used_reflection"))
    consistency_values = [case["self_consistency_rate"] for case in cases if case.get("self_consistency_rate") is not None]

    return {
        "model_path": str(args.model_path),
        "input_file": str(args.input_file),
        "total": total,
        "syntax_pass_rate": syntax_pass / total if total else 0.0,
        "pass_at_1": passed_tasks / len(with_tests) if with_tests else None,
        "avg_test_pass_rate": passed_tests / total_tests if total_tests else None,
        "passed_tasks": passed_tasks if with_tests else None,
        "total_tests": total_tests if with_tests else None,
        "passed_tests": passed_tests if with_tests else None,
        "exact_match": sum(1 for case in with_exact if case["exact_match"]) / len(with_exact) if with_exact else None,
        "avg_code_token_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "signature_mismatch_rate": signature_mismatches / total if total else 0.0,
        "reflection_used_count": used_reflection,
        "include_examples": args.include_examples,
        "enable_reflection": args.enable_reflection,
        "reflection_rounds": args.reflection_rounds,
        "enable_thinking": args.enable_thinking,
        "num_samples": args.num_samples,
        "selection_mode": args.selection_mode,
        "avg_self_consistency_rate": sum(consistency_values) / len(consistency_values) if consistency_values else None,
        "inference_seconds": elapsed_seconds,
        "avg_seconds_per_task": elapsed_seconds / total if total else None,
        "prompt": "CoT prompt with strict function-name preservation, standard-library-only constraint, multi-sample selection, and multi-round reflection.",
    }


def main() -> None:
    args = parse_args()
    raw_rows = load_rows(args.input_file)

    processed_rows = []
    for index, row in enumerate(raw_rows):
        test_setup, tests = row_to_tests(row)
        reference = normalize_text(row_to_reference(row))
        task = row_to_task(row)
        expected_name = extract_expected_function_name(task, reference)
        processed_rows.append(
            {
                "index": index,
                "task": task,
                "reference": reference,
                "test_setup": test_setup,
                "tests": tests,
                "expected_function_name": expected_name,
                "raw": row,
            }
        )

    rows = [row for row in processed_rows if row["task"]]
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No valid tasks found in {args.input_file}")

    start_time = time.time()
    cases = []
    for i, row in enumerate(rows, start=1):
        case = evaluate_one_row_with_strategy(row, args)
        cases.append(case)
        print(f"finished {i}/{len(rows)}", flush=True)

    elapsed_seconds = time.time() - start_time
    metrics = summarize(cases, args, elapsed_seconds)

    save_json(args.output_dir / "cot_code_metrics.json", metrics)
    save_jsonl(args.output_dir / "cot_code_cases.jsonl", cases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote metrics to {args.output_dir / 'cot_code_metrics.json'}")
    print(f"Wrote cases to {args.output_dir / 'cot_code_cases.jsonl'}")


if __name__ == "__main__":
    main()