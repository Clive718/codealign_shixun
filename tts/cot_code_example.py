"""CoT prompt helpers for Python code generation tasks."""


from __future__ import annotations

import re

#提取最终代码
FINAL_CODE_RE = re.compile(
    r"(?:final python code|final code|python code)\s*:?\s*```(?:python|py)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

#规定模型身份和输出格式
system_prompt = """
You are a careful Python programming assistant.

You must follow all of these rules:
1. Preserve the exact required function name from the task.
2. Preserve the required parameters if they are given.
3. Use only Python standard library. Do not use third-party packages.
4. Write the smallest correct executable solution.
5. Do not include example usage, prints, or extra tests in Final code.
6. Final code must contain only Python code.

Answer in exactly this structure:

Reasoning:
- Briefly explain the idea in 1-2 short bullet points.

Key steps:
1. List 2-4 short implementation steps.

Final code:
```python
# complete executable Python code here
```

Only the code inside Final code will be evaluated.
"""

#给模型提供示范样例（Few-shot）
cot_examples = """
Task:
Write a Python function `two_sum(nums, target)` that returns the indices of the two numbers that add up to the target value. If no such pair exists, return an empty list.

Important requirements:
- Exact function name: two_sum
- Use only Python standard library.
- Output only the final executable code in Final code.

Reasoning:
- Use a hash map to store seen values and their indices.
- For each number, check whether its complement has already appeared.

Key steps:
1. Create an empty dictionary to store seen numbers.
2. Iterate through the list with indices.
3. Compute the complement for each value.
4. Return the matching pair if found, otherwise return an empty list.

Final code:
```python
from typing import List

def two_sum(nums: List[int], target: int) -> List[int]:
    seen = {{}}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

---

Task:
Write a Python function `is_palindrome(s)` that checks whether a string is a palindrome, ignoring case and non-alphanumeric characters.

Important requirements:
- Exact function name: is_palindrome
- Use only Python standard library.
- Output only the final executable code in Final code.

Reasoning:
- Filter out non-alphanumeric characters and lowercase the rest.
- Compare the cleaned string with its reverse.

Key steps:
1. Remove all non-alphanumeric characters.
2. Convert the cleaned string to lowercase.
3. Compare it with its reversed version.

Final code:
```python
import re

def is_palindrome(s: str) -> bool:
    filtered = re.sub(r"[^A-Za-z0-9]", "", s).lower()
    return filtered == filtered[::-1]
```

---
"""

#任务模板
cot_task_prompt = """
Task:
{task_description}

Important requirements:
- Preserve the exact function name required by the task.
- Use only Python standard library.
- Do not invent extra helper APIs unless needed.
- Do not output example calls, print statements, or extra explanation inside Final code.

Reasoning:
"""


cot_example_prompt = cot_examples + "\n" + cot_task_prompt

#统一管理提示词组件
prompt_template = {
    "system_prompt": system_prompt,
    "cot_prompt": cot_task_prompt,
    "cot_prompt_with_examples": cot_example_prompt,
    "final_instruction": (
        "Continue with Reasoning, Key steps, and Final code. "
        "The Final code must be one complete executable Python solution only."
    ),
}

#生成最终文本prompt
def build_prompt(task_description: str, include_examples: bool = False) -> str:
    template = cot_example_prompt if include_examples else cot_task_prompt
    return template.format(task_description=task_description).strip()

#生成聊天信息格式
def build_chat_messages(task_description: str, include_examples: bool = False) -> list[dict[str, str]]:
    """Build Qwen-style chat messages with the CoT code prompt."""
    return [
        {"role": "system", "content": system_prompt.strip()},
        {
            "role": "user",
            "content": f"{build_prompt(task_description, include_examples=include_examples)}\n\n{prompt_template['final_instruction']}",
        },
    ]

#清洗文本
def normalize_text(text: object) -> str:
    lines = [line.rstrip() for line in str(text or "").strip().splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)

#从模型回答里提取最终代码
def extract_final_code(response: object) -> str:
    """Extract the code block intended for evaluation from a CoT response."""
    text = str(response or "")
    matches = FINAL_CODE_RE.findall(text)
    if matches:
        return normalize_text(matches[-1])

    fenced = FENCED_CODE_RE.findall(text)
    if fenced:
        return normalize_text(fenced[-1])

    lowered = text.lower()
    marker = lowered.rfind("final code")
    if marker >= 0:
        text = text[marker:].split(":", 1)[-1]
    return normalize_text(text)

#本地测试入口
if __name__ == "__main__":
    sample_task = "Write a function find_Max_Num(arr) to return the largest number formed by the digits in arr."
    print(build_prompt(sample_task, include_examples=True))