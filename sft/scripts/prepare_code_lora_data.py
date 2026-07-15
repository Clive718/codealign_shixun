#!/usr/bin/env python3
"""Prepare python-code Alpaca data for LLaMA-Factory SFT."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "python_code_instructions_18k_alpaca"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sft" / "data"


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    except Exception as pandas_error:
        try:
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pylist()
        except Exception as arrow_error:
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
    data_dir = source_dir / "data"
    candidates = sorted(data_dir.glob("*.parquet")) if data_dir.exists() else []
    if not candidates:
        candidates = sorted(source_dir.glob("*.parquet"))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {source_dir}")
    return candidates


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def convert_row(row: dict[str, Any]) -> dict[str, str] | None:
    instruction = clean_text(row.get("instruction"))
    input_text = clean_text(row.get("input"))
    output = clean_text(row.get("output"))

    if not instruction:
        instruction = clean_text(row.get("prompt"))
    if not output:
        return None

    if not instruction and input_text:
        instruction, input_text = input_text, ""
    if not instruction:
        return None

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
    }


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    unique_rows = []
    for row in rows:
        key = (row["instruction"], row["input"], row["output"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def compute_statistics(rows: list[dict[str, str]], original_count: int) -> dict[str, Any]:
    """计算数据集的统计信息"""
    if not rows:
        return {}

    # 提取各字段长度
    instruction_lens = [len(row["instruction"]) for row in rows]
    input_lens = [len(row["input"]) for row in rows]
    output_lens = [len(row["output"]) for row in rows]

    # 计算空值数量
    empty_instruction = sum(1 for row in rows if not row["instruction"])
    empty_input = sum(1 for row in rows if not row["input"])
    empty_output = sum(1 for row in rows if not row["output"])

    total = len(rows)

    def get_length_stats(lengths: list[int]) -> dict[str, Any]:
        """计算长度的统计指标"""
        sorted_lens = sorted(lengths)
        return {
            "mean": sum(lengths) / len(lengths) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "median": sorted_lens[len(sorted_lens) // 2] if sorted_lens else 0,
            "p25": sorted_lens[len(sorted_lens) // 4] if sorted_lens else 0,
            "p75": sorted_lens[len(sorted_lens) * 3 // 4] if sorted_lens else 0,
        }

    return {
        "total_samples": total,
        "original_samples": original_count,
        "duplicate_count": original_count - total,
        "duplicate_rate": (original_count - total) / original_count if original_count > 0 else 0,
        "instruction_stats": {
            **get_length_stats(instruction_lens),
            "empty_count": empty_instruction,
            "empty_rate": empty_instruction / total if total > 0 else 0,
        },
        "input_stats": {
            **get_length_stats(input_lens),
            "empty_count": empty_input,
            "empty_rate": empty_input / total if total > 0 else 0,
        },
        "output_stats": {
            **get_length_stats(output_lens),
            "empty_count": empty_output,
            "empty_rate": empty_output / total if total > 0 else 0,
        },
    }


def filter_rows(
    rows: list[dict[str, str]],
    min_instruction_len: int = 0,
    max_instruction_len: int = 0,
    min_output_len: int = 0,
    max_output_len: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """过滤不符合质量要求的样本，返回 (合格样本列表, 被过滤样本列表)。

    被过滤样本附带 reason 字段，说明过滤原因，方便后续写入 bad_cases.json。
    """
    passed: list[dict[str, str]] = []
    bad_cases: list[dict[str, Any]] = []

    for row in rows:
        reasons: list[str] = []

        inst_len = len(row["instruction"])
        out_len = len(row["output"])

        if min_instruction_len > 0 and inst_len < min_instruction_len:
            reasons.append(f"instruction 过短: {inst_len} < {min_instruction_len}")
        if max_instruction_len > 0 and inst_len > max_instruction_len:
            reasons.append(f"instruction 过长: {inst_len} > {max_instruction_len}")
        if min_output_len > 0 and out_len < min_output_len:
            reasons.append(f"output 过短: {out_len} < {min_output_len}")
        if max_output_len > 0 and out_len > max_output_len:
            reasons.append(f"output 过长: {out_len} > {max_output_len}")

        if reasons:
            bad_cases.append({**row, "filter_reasons": reasons})
        else:
            passed.append(row)

    return passed, bad_cases


def split_rows(
    rows: list[dict[str, str]], train_ratio: float, valid_ratio: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1.")
    if not 0 <= valid_ratio < 1:
        raise ValueError("--valid_ratio must be between 0 and 1.")
    if train_ratio + valid_ratio >= 1:
        raise ValueError("--train_ratio + --valid_ratio must be less than 1.")

    shuffled = rows[:]
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)

    train_rows = shuffled[:train_end]
    valid_rows = shuffled[train_end:valid_end]
    test_rows = shuffled[valid_end:]
    if not train_rows or not valid_rows or not test_rows:
        raise ValueError(
            f"Split produced an empty subset: train={len(train_rows)}, valid={len(valid_rows)}, test={len(test_rows)}"
        )
    return train_rows, valid_rows, test_rows

def convert_mbpp_split(split):
    split_path = (
        PROJECT_ROOT
        / "mbpp"
        / "sanitized"
        / f"{split}-00000-of-00001.parquet"
    )

    rows = read_parquet(split_path)

    dataset = []

    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        code = str(row.get("code", "")).strip()

        if not prompt or not code:
            continue

        dataset.append(
            {
                "instruction":
                    "Write a Python function according to the following description. "
                    "Return only executable Python code without explanation.\n\n"
                    + prompt,
                "input": "",
                "output": code,
                "task_id": int(row["task_id"])
            }
        )

    return dataset


def save_json(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all examples.")
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--valid_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--min_output_len",
        type=int,
        default=10,
        help="output 最短字符数"
    )

    parser.add_argument(
        "--max_output_len",
        type=int,
        default=4096,
        help="output 最长字符数"
    )

    parser.add_argument(
        "--min_instruction_len",
        type=int,
        default=5,
        help="instruction 最短字符数"
    )

    parser.add_argument(
        "--max_instruction_len",
        type=int,
        default=0,
        help="instruction 最长字符数"
    )

    parser.add_argument(
        "--remove_duplicates",
        type=lambda x: x.lower() != "false",
        default=True,
        help="是否去重"
    )

    parser.add_argument(
        "--preview_count",
        type=int,
        default=10,
        help="预览样本数量"
    )

    args = parser.parse_args()

    ###########################################################
    # Code Alpaca 数据处理
    ###########################################################

    parquet_files = find_parquet_files(args.source_dir)

    raw_rows: list[dict[str, Any]] = []

    for parquet_file in parquet_files:
        raw_rows.extend(read_parquet(parquet_file))

    converted = [
        row
        for row in (convert_row(raw) for raw in raw_rows)
        if row is not None
    ]

    converted_before_dedup = len(converted)

    if args.remove_duplicates:
        converted = deduplicate(converted)

    converted_after_dedup = len(converted)

    converted, bad_cases = filter_rows(
        converted,
        min_instruction_len=args.min_instruction_len,
        max_instruction_len=args.max_instruction_len,
        min_output_len=args.min_output_len,
        max_output_len=args.max_output_len,
    )

    statistics = compute_statistics(
        converted,
        converted_after_dedup
    )

    if args.limit > 0:
        converted = converted[:args.limit]

    if len(converted) < 3:
        raise ValueError(
            f"Need at least 3 valid examples, got {len(converted)}"
        )

    train_rows, valid_rows, test_rows = split_rows(
        converted,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed
    )

    ###########################################################
    # 保存 Code Alpaca
    ###########################################################

    save_json(
        args.output_dir / "code_sft_train.json",
        train_rows
    )

    save_json(
        args.output_dir / "code_sft_valid.json",
        valid_rows
    )

    save_json(
        args.output_dir / "code_sft_test.json",
        test_rows
    )

    ###########################################################
    # MBPP 数据处理
    ###########################################################

    mbpp_dir = PROJECT_ROOT / "mbpp" / "sanitized"

    mbpp_datasets = {}

    for split in ["train", "validation", "test"]:

        parquet_path = (
            mbpp_dir /
            f"{split}-00000-of-00001.parquet"
        )

        if not parquet_path.exists():
            print(f"Skip MBPP {split}: file not found")
            continue

        rows = read_parquet(parquet_path)

        dataset = []

        for row in rows:

            prompt = str(
                row.get("prompt", "")
            ).strip()

            code = str(
                row.get("code", "")
            ).strip()

            match = re.search(
                r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                code
            )

            func_name = (
                match.group(1)
                if match
                else "solution"
            )

            if not prompt or not code:
                continue

            dataset.append(
            {
                "instruction":
                "You are an expert Python programmer.\n"
                f"The function name must be `{func_name}`.\n"
                "Write ONLY executable Python code.\n"
                "Do not include explanations.\n"
                "Do not include markdown.\n"
                "Do not include comments.\n\n"
                + prompt,
                    "input": "",
                    "output": code,
                    "task_id":
                        int(row["task_id"])
                        if row.get("task_id") is not None
                        else len(dataset)
                }
            )

        mbpp_datasets[split] = dataset

        save_json(
            args.output_dir /
            f"mbpp_sanitized_{split}.json",
            dataset
        )

        print(
            f"Wrote MBPP {split}: "
            f"{len(dataset)} samples"
        )

    ###########################################################
    # dataset_info.json
    ###########################################################

    dataset_info = {
        "code_sft_train": {
            "file_name": "code_sft_train.json"
        },
        "code_sft_valid": {
            "file_name": "code_sft_valid.json"
        },
        "code_sft_test": {
            "file_name": "code_sft_test.json"
        }
    }

    if "train" in mbpp_datasets:
        dataset_info["mbpp_sanitized_train"] = {
            "file_name": "mbpp_sanitized_train.json"
        }

    if "validation" in mbpp_datasets:
        dataset_info["mbpp_sanitized_validation"] = {
            "file_name": "mbpp_sanitized_validation.json"
        }

    if "test" in mbpp_datasets:
        dataset_info["mbpp_sanitized_test"] = {
            "file_name": "mbpp_sanitized_test.json"
        }

    save_json(
        args.output_dir / "dataset_info.json",
        dataset_info
    )

    ###########################################################
    # 保存统计文件
    ###########################################################

    save_json(
        args.output_dir / "data_statistics.json",
        statistics
    )

    if args.preview_count > 0:
        preview_samples = converted[:args.preview_count]

        preview_output = [
            {
                "index": i,
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
                "instruction_len": len(row["instruction"]),
                "input_len": len(row["input"]),
                "output_len": len(row["output"]),
            }
            for i, row in enumerate(preview_samples)
        ]

        save_json(
            args.output_dir / "sample_preview.json",
            preview_output
        )

    if bad_cases:
        save_json(
            args.output_dir / "bad_cases.json",
            bad_cases
        )

    ###########################################################
    # 输出统计信息
    ###########################################################

    print()
    print("========== 数据集生成完成 ==========")

    print(f"Code Train : {len(train_rows)}")
    print(f"Code Valid : {len(valid_rows)}")
    print(f"Code Test  : {len(test_rows)}")

    if "train" in mbpp_datasets:
        print(
            f"MBPP Train : "
            f"{len(mbpp_datasets['train'])}"
        )

    if "validation" in mbpp_datasets:
        print(
            f"MBPP Valid : "
            f"{len(mbpp_datasets['validation'])}"
        )

    if "test" in mbpp_datasets:
        print(
            f"MBPP Test  : "
            f"{len(mbpp_datasets['test'])}"
        )

    print("===================================")

if __name__ == "__main__":
    main()
