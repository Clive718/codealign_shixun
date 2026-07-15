#!/usr/bin/env python3
import json
from pathlib import Path

src = Path("dpo/data/code_dpo_train.json")
out = Path("dpo/data/code_ppo_prompt_train.json")
reg = Path("dpo/data/dataset_info.json")

data = json.loads(src.read_text(encoding="utf-8"))
ppo_data = []

for item in data:
    instruction = (item.get("instruction") or "").strip()
    input_text = (item.get("input") or "").strip()
    ppo_data.append({
        "instruction": instruction,
        "input": input_text,
        "output": ""
    })

out.write_text(json.dumps(ppo_data, ensure_ascii=False, indent=2), encoding="utf-8")

registry = json.loads(reg.read_text(encoding="utf-8")) if reg.exists() else {}
registry["code_ppo_prompt_train"] = {
    "file_name": "code_ppo_prompt_train.json"
}
reg.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Wrote {len(ppo_data)} PPO prompt samples to {out}")
print(f"Updated registry: {reg}")