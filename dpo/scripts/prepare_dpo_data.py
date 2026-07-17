#!/usr/bin/env python3
"""Prepare Python-code DPO data for LLaMA-Factory."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "py-dpo-v0.1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dpo" / "data"
DEFAULT_PROMPT_TEMPLATE = (
    "Complete the following Python coding task. Return a correct, readable "
    "Python solution and include brief reasoning when helpful.\n\nTask:\n{prompt}"
)


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    except Exception:
        try:
            import pyarrow.parquet as pq
            return pq.read_table(path).to_pylist()
        except Exception:
            try:
                from datasets import load_dataset
                dataset = load_dataset("parquet", data_files=str(path), split="train")
                return [dict(row) for row in dataset]
            except Exception as datasets_error:
                raise RuntimeError(
                    "Failed to read parquet. Install pandas, pyarrow, or datasets with parquet support "
                    "in the active environment."
                ) from datasets_error


def find_parquet_files(source_dir: Path) -> list[Path]:
    candidates = sorted(source_dir.glob("*.parquet"))
    data_dir = source_dir / "data"
    if data_dir.exists():
        candidates.extend(sorted(data_dir.glob("*.parquet")))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {source_dir}")
    return candidates


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def build_instruction(prompt: str, prompt_template: str) -> str:
    return prompt_template.format(prompt=prompt).strip()


def text_len(text: str) -> int:
    return len(text.strip())


def is_valid_raw_row(row: dict[str, Any], min_response_len: int) -> bool:
    prompt = first_text(row, ("prompt", "instruction", "question"))
    chosen = first_text(row, ("chosen", "accepted", "preferred", "output"))
    rejected = first_text(row, ("rejected", "reject", "dispreferred"))

    if not prompt or not chosen or not rejected:
        return False
    if text_len(chosen) < min_response_len or text_len(rejected) < min_response_len:
        return False
    return True


def convert_train_row(row: dict[str, Any], prompt_template: str) -> dict[str, str] | None:
    prompt = first_text(row, ("prompt", "instruction", "question"))
    chosen = first_text(row, ("chosen", "accepted", "preferred", "output"))
    rejected = first_text(row, ("rejected", "reject", "dispreferred"))
    input_text = first_text(row, ("input", "query"))

    if not prompt or not chosen or not rejected:
        return None

    return {
        "instruction": build_instruction(prompt, prompt_template),
        "input": input_text,
        "chosen": chosen,
        "rejected": rejected,
    }


def convert_test_row(row: dict[str, Any], prompt_template: str) -> dict[str, str] | None:
    prompt = first_text(row, ("prompt", "instruction", "question"))
    output = first_text(row, ("chosen", "accepted", "preferred", "output"))
    input_text = first_text(row, ("input", "query"))

    if not prompt or not output:
        return None

    return {
        "instruction": build_instruction(prompt, prompt_template),
        "input": input_text,
        "output": output,
    }


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    unique_rows = []
    for row in rows:
        key = tuple(row.get(field, "") for field in ("instruction", "input", "chosen", "rejected", "output"))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def compute_avg_lengths(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "avg_prompt_len": 0.0,
            "avg_chosen_len": 0.0,
            "avg_rejected_len": 0.0,
        }

    prompts = [text_len(first_text(row, ("prompt", "instruction", "question"))) for row in rows]
    chosens = [text_len(first_text(row, ("chosen", "accepted", "preferred", "output"))) for row in rows]
    rejecteds = [text_len(first_text(row, ("rejected", "reject", "dispreferred"))) for row in rows]

    return {
        "avg_prompt_len": round(sum(prompts) / len(prompts), 2),
        "avg_chosen_len": round(sum(chosens) / len(chosens), 2),
        "avg_rejected_len": round(sum(rejecteds) / len(rejecteds), 2),
    }


def split_by_ratio(rows: list[dict[str, Any]], test_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test_ratio must be between 0 and 1.")

    test_count = max(1, int(math.floor(len(rows) * test_ratio)))
    if test_count >= len(rows):
        test_count = len(rows) - 1

    test_rows = rows[:test_count]
    train_rows = rows[test_count:]
    return train_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Ratio of valid samples used as test set.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 means use all remaining train rows.")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    parser.add_argument("--min_response_len", type=int, default=0, help="Minimum length for chosen/rejected responses.")
    parser.add_argument("--preview_size", type=int, default=10)
    args = parser.parse_args()

    parquet_files = find_parquet_files(args.source_dir)
    raw_rows: list[dict[str, Any]] = []
    for parquet_file in parquet_files:
        raw_rows.extend(read_parquet(parquet_file))

    rng = random.Random(args.seed)
    shuffled_rows = raw_rows[:]
    rng.shuffle(shuffled_rows)

    valid_raw_rows = [row for row in shuffled_rows if is_valid_raw_row(row, args.min_response_len)]
    filtered_out_rows = len(shuffled_rows) - len(valid_raw_rows)

    if len(valid_raw_rows) < 2:
        raise RuntimeError("Not enough valid rows after cleaning/filtering to create train/test splits.")

    raw_train_rows, raw_test_rows = split_by_ratio(valid_raw_rows, args.test_ratio)

    if args.max_train_samples > 0:
        raw_train_rows = raw_train_rows[: args.max_train_samples]

    train_rows = deduplicate(
        [row for row in (convert_train_row(raw, args.prompt_template) for raw in raw_train_rows) if row is not None]
    )
    test_rows = deduplicate(
        [row for row in (convert_test_row(raw, args.prompt_template) for raw in raw_test_rows) if row is not None]
    )

    if not train_rows:
        raise RuntimeError("No valid DPO train rows were converted.")
    if not test_rows:
        raise RuntimeError("No valid code test rows were converted.")

    preview_rows = train_rows[: args.preview_size]

    dataset_info = {
        "code_dpo_train": {
            "file_name": "code_dpo_train.json",
            "ranking": True,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        },
        "code_dpo_test": {
            "file_name": "code_dpo_test.json",
        },
    }

    stats = compute_avg_lengths(valid_raw_rows)

    save_json(args.output_dir / "code_dpo_train.json", train_rows)
    save_json(args.output_dir / "code_dpo_test.json", test_rows)
    save_json(args.output_dir / "dataset_info.json", dataset_info)
    save_json(args.output_dir / "sample_preview.json", preview_rows)

    print(f"Read {len(raw_rows)} raw rows from {len(parquet_files)} parquet file(s).")
    print(f"Kept {len(valid_raw_rows)} valid rows after cleaning/filtering.")
    print(f"Filtered out {filtered_out_rows} invalid rows.")
    print(
        f"Average lengths - prompt: {stats['avg_prompt_len']}, "
        f"chosen: {stats['avg_chosen_len']}, rejected: {stats['avg_rejected_len']}"
    )
    print(f"Split ratio -> train: {1 - args.test_ratio:.2%}, test: {args.test_ratio:.2%}")
    print(f"Wrote train: {len(train_rows)} -> {args.output_dir / 'code_dpo_train.json'}")
    print(f"Wrote test : {len(test_rows)} -> {args.output_dir / 'code_dpo_test.json'}")
    print(f"Wrote preview -> {args.output_dir / 'sample_preview.json'}")
    print(f"Wrote registry -> {args.output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()