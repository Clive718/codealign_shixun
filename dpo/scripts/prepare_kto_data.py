#!/usr/bin/env python3
import json
from pathlib import Path

SRC = Path("dpo/data/code_dpo_train.json")
OUT = Path("dpo/data/code_kto_train.json")
REG = Path("dpo/data/dataset_info.json")

data = json.loads(SRC.read_text(encoding="utf-8"))
kto_data = []

for item in data:
    instruction = (item.get("instruction") or "").strip()
    user_text = instruction
    if item.get("input"):
        user_text = f"{instruction}\n\n{item['input'].strip()}"

    chosen = (item.get("chosen") or "").strip()
    rejected = (item.get("rejected") or "").strip()

    if chosen:
        kto_data.append({
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": chosen}
            ],
            "kto_tag": True
        })

    if rejected:
        kto_data.append({
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": rejected}
            ],
            "kto_tag": False
        })

OUT.write_text(json.dumps(kto_data, ensure_ascii=False, indent=2), encoding="utf-8")

registry = json.loads(REG.read_text(encoding="utf-8")) if REG.exists() else {}
registry["code_kto_train"] = {
    "file_name": "code_kto_train.json",
    "formatting": "sharegpt",
    "columns": {
        "messages": "conversations",
        "kto_tag": "kto_tag"
    }
}
REG.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Wrote {len(kto_data)} KTO samples to {OUT}")
print(f"Updated registry: {REG}")