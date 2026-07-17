# A3 偏好数据构造模块 — 个人 README

> **姓名**：徐本钰
> **学号**：20236496
> **小组 / 项目**：CodeAlign —— 基于大模型微调与偏好对齐的 Python 代码生成系统

---

## 1. 模块概述

### 1.1 模块名称

`A3 — 偏好数据构造（Preference Data Construction）`

### 1.2 模块说明

本模块位于 CodeAlign 系统流水线的 **第三阶段**，处于 SFT 微调（A2）之后、DPO 对齐训练（A4）之前，是连接监督学习和偏好学习的**关键桥梁**。

它从 HuggingFace 上的 `jondurbin/py-dpo-v0.1` 数据集读取原始的 prompt/chosen/rejected（问题/好答案/差答案）三元组，经过字段归一化、缺失过滤、长度过滤、去重、随机打乱、比例划分、格式转换等一系列处理，最终输出 LLaMA-Factory 框架可以直接加载的 DPO 训练/测试数据。

**如果没有本模块**，A4 的 DPO 训练将面临数据格式错误、训练不稳定、评估不可靠等问题，整个偏好对齐流水线将断裂。

```text
[输入] py-dpo-v0.1 (Parquet, 9400+ 条)
  → 字段兼容归一化 (prompt/instruction/question 等)
  → 缺失值过滤 + 回答长度过滤
  → 精确去重
  → 随机打乱 (seed=42)
  → 训练/测试按比例划分 (默认 95%/5%)
  → 格式转换 (DPO Ranking 格式)
  → dataset_info.json 注册
[输出] code_dpo_train.json + code_dpo_test.json + dataset_info.json + sample_preview.json
```

### 1.3 完成情况概览

| 类型 | 完成情况 |
|---|---|
| 基础要求 | ✅ 完成：从原始 Parquet 读取 → 清洗 → 过滤 → 划分 → 转换 → 输出 LLaMA-Factory 兼容格式 |
| 进阶要求 | ✅ 完成 5 项（详见第 5 节） |
| 可独立运行的演示 | ✅ `bash dpo/scripts/prepare_data.sh` |
| 与团队系统集成情况 | ✅ 输出写入 `dpo/data/` 目录，通过 `dataset_info.json` 被 A4 DPO 训练模块自动识别加载 |

---

## 2. 环境、模型与数据依赖

### 2.1 运行环境

| 项目 | 要求 |
|---|---|
| Python 版本 | 3.11（与 LLaMA-Factory 兼容） |
| 必要依赖 | pandas ≥ 2.0 或 pyarrow ≥ 12.0 或 `datasets`（三选一，自动兜底） |
| 是否需要模型 | 不需要 |
| 是否需要 GPU | 不需要（纯 CPU 数据处理） |
| 是否需要外部数据集 | 需要：`jondurbin/py-dpo-v0.1`（Parquet 格式） |

### 2.2 模型依赖

本模块为纯数据处理模块，不依赖任何模型。

### 2.3 数据集或样例数据依赖

| 数据或文件 | 来源 | 项目内相对路径 | 用途 |
|---|---|---|---|
| `py-dpo-v0.1` Parquet 文件 | HuggingFace [jondurbin/py-dpo-v0.1](https://huggingface.co/datasets/jondurbin/py-dpo-v0.1) | `py-dpo-v0.1/` | 原始 prompt/chosen/rejected 三元组 |

### 2.4 安装步骤

```bash
# 创建并激活 conda 环境（项目统一环境）
conda create -n assignment_A python=3.11
conda activate assignment_A

# 安装依赖（三选一即可，推荐 pandas）
pip install pandas>=2.0

# 或
pip install pyarrow>=12.0

# 或
pip install datasets
```

个人模块可以脱离完整系统独立运行，最小依赖仅为 `pandas`。

---

## 3. 文件结构与接口边界

### 3.1 文件结构

只列出与本模块直接相关的文件：

```text
项目根目录/
├── py-dpo-v0.1/                        # 原始 Parquet 数据目录（需自行下载）
│   └── py-dpo.parquet
├── dpo/
│   ├── scripts/
│   │   ├── prepare_data.sh             # Bash 包装脚本（参数化启动入口）
│   │   └── prepare_dpo_data.py         # Python 核心处理脚本（数据处理引擎）
│   └── data/                           # 输出目录（运行后自动生成）
│       ├── code_dpo_train.json         # DPO 训练集（ranking 格式）
│       ├── code_dpo_test.json          # 测试集（标准格式，用于 MBPP 评测）
│       ├── dataset_info.json           # LLaMA-Factory 数据集注册文件
│       └── sample_preview.json         # 前 10 条训练样本预览（人工质量检查）
```

### 3.2 接口边界

| 类型 | 来源 / 去向 | 数据格式 | 说明 |
|---|---|---|---|
| 输入 | `py-dpo-v0.1/*.parquet` | Parquet（列式存储） | 原始 prompt/chosen/rejected 三元组，由用户自行下载到该目录 |
| 输出 (训练集) | → A4 DPO 训练模块 | JSON (DPO Ranking) | `{instruction, input, chosen, rejected}`，通过 `dataset_info.json` 注册 |
| 输出 (测试集) | → MBPP 评测 | JSON | `{instruction, input, output}`，output 取 chosen 作为参考答案 |
| 输出 (注册文件) | → LLaMA-Factory | JSON | 标记 `ranking: true`，框架自动识别为 DPO pair-wise 数据 |
| 输出 (预览文件) | → 人工抽检 | JSON | 前 10 条转换后样本，用于人工质量检查 |

---

## 4. 基础要求实现与演示

### 4.1 基础功能说明

基础功能实现了完整的 DPO 偏好数据构造流水线：从原始 Parquet 文件读取数据，经过清洗、过滤、去重、打乱、划分、格式转换，最终输出 LLaMA-Factory 兼容的训练/测试数据及注册文件。

对应课程/项目要求中的 **A3 偏好数据构造** 模块。

### 4.2 基础功能实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `prepare_data.sh` | Bash 包装脚本，将所有参数通过环境变量暴露，提供一个统一的启动入口 |
| `prepare_dpo_data.py` | Python 核心处理脚本，包含全部数据处理逻辑 |
| `find_parquet_files()` | 扫描源目录，搜集所有 `*.parquet` 文件（含 `data/` 子目录） |
| `read_parquet()` | 读取 Parquet 文件，三路回退兜底（Pandas → PyArrow → datasets） |
| `first_text()` / `clean_text()` | 字段兼容归一化：多 key 候选 + 去空 + 去 None |
| `is_valid_raw_row()` | 判断原始行是否有效：缺失字段过滤 + 回答长度过滤 |
| `split_by_ratio()` | 按比例（默认 95%/5%）划分训练集和测试集 |
| `convert_train_row()` / `convert_test_row()` | 分别转为 DPO Ranking 格式和标准评测格式 |
| `deduplicate()` | 基于 5 个字段的精确匹配去重 |
| `compute_avg_lengths()` | 统计平均 prompt/chosen/rejected 长度 |
| `save_json()` | 以美观格式（UTF-8, indent=2）写出 JSON 文件 |
| `main()` | 串联全流程，解析命令行参数，控制整体执行顺序 |

完整处理流程：

```text
[Parquet 文件] → find_parquet_files → read_parquet
  → random.shuffle (seed=42)
  → is_valid_raw_row (缺失过滤 + 长度过滤)
  → split_by_ratio (test_ratio=0.05)
  → convert_train_row / convert_test_row
  → deduplicate
  → save_json (4 个输出文件)
```

### 4.3 基础功能输入格式与样例

| 字段 / 输入文件 | 类型 / 格式 | 是否必需 | 说明 |
|---|---|---|---|
| `prompt` | 字符串 | 是（或其别名 instruction / question） | 编程问题描述 |
| `chosen` | 字符串 | 是（或其别名 accepted / preferred / output） | 人工标注的好回答 |
| `rejected` | 字符串 | 是（或其别名 reject / dispreferred） | 人工标注的差回答 |
| `input` / `query` | 字符串 | 否 | 额外输入上下文 |
| Parquet 文件 | `.parquet` | 是 | 存放于 `--source_dir` 下 |

样例输入（来自 HuggingFace 原始数据）：

```python
{
    "prompt": "Write a Python function to compute the factorial of a number.",
    "chosen": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
    "rejected": "def factorial(n):\n    # TODO\n    pass",
    "input": ""
}
```

### 4.4 基础功能演示命令

```bash
# 基础运行（全量数据，9400+ 条）
# 会在 dpo/data/ 下生成 4 个文件
bash dpo/scripts/prepare_data.sh

# 小样本调试（快速验证）
MAX_TRAIN_SAMPLES=200 bash dpo/scripts/prepare_data.sh

# 自定义参数
SEED=123 TEST_RATIO=0.1 MIN_RESPONSE_LEN=50 bash dpo/scripts/prepare_data.sh
```

命令运行后应该观察到的现象：

- 终端输出参数配置确认（项目路径、seed、test_ratio 等）
- 终端打印读取的原始行数、清洗后保留的行数、过滤掉的行数
- 终端打印平均 prompt/chosen/rejected 长度
- 终端打印训练集和测试集的最终条数
- 终端显示 4 个输出文件的路径
- `dpo/data/` 目录下生成 `code_dpo_train.json`、`code_dpo_test.json`、`dataset_info.json`、`sample_preview.json`

### 4.5 基础功能输出格式

| 输出文件 / 返回字段 | 格式 | 说明 |
|---|---|---|
| `code_dpo_train.json` | JSON (DPO Ranking) | 每条含 `{instruction, input, chosen, rejected}` |
| `code_dpo_test.json` | JSON | 每条含 `{instruction, input, output}` |
| `dataset_info.json` | JSON | LLaMA-Factory 注册文件，标记 `ranking: true` |
| `sample_preview.json` | JSON | 前 10 条训练样本，供人工质量抽检 |

训练集输出样例：

```json
{
  "instruction": "Complete the following Python coding task. Return a correct, readable Python solution and include brief reasoning when helpful.\n\nTask:\nWrite a Python function to compute the factorial of a number.",
  "input": "",
  "chosen": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
  "rejected": "def factorial(n):\n    # TODO\n    pass"
}
```

### 4.6 基础功能结果截图

```text
[待补充：运行 prepare_data.sh 的终端输出截图]
[待补充：code_dpo_train.json 文件内容示例截图]
```

---

## 5. 进阶要求实现与演示

### 5.1 选择的进阶要求

| 进阶要求 | 是否完成 | 对应文件 / 函数 | 简要说明 |
|---|---|---|---|
| ① 增强数据统计功能 | ✅ 完成 | `compute_avg_lengths()` | 输出平均 prompt/chosen/rejected 长度 |
| ② 增加质量过滤逻辑 | ✅ 完成 | `is_valid_raw_row()` + `--min_response_len` | 过滤回答过短的样本（默认 20 字符） |
| ③ 按比例划分数据集 | ✅ 完成 | `split_by_ratio()` + `--test_ratio` | 支持 `--test_ratio` 参数（如 0.05 表示 5% 测试），替代固定数量划分 |
| ④ 生成样本预览文件 | ✅ 完成 | `--preview_size` 参数 + `sample_preview.json` | 自动保存前 N 条转换后样本 |
| ⑤ 数据去重增强 | ✅ 完成 | `deduplicate()` | 基于 instruction/input/chosen/rejected/output 五字段精确匹配去重 |

### 5.2 进阶功能 ①：增强数据统计功能

#### 功能说明

在数据处理完成后，自动计算并输出全量有效样本的平均 prompt 长度、chosen 长度、rejected 长度。通过对比 chosen 和 rejected 的平均长度，可以快速判断数据质量——如果两者长度非常接近，说明偏好区分不够明显，需要警惕。

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `compute_avg_lengths()` | 遍历所有有效行，分别计算三组文本的平均长度（四舍五入保留两位小数） |

#### 输出示例

```text
Average lengths - prompt: 58.32, chosen: 142.67, rejected: 45.21
```

### 5.3 进阶功能 ②：增加质量过滤逻辑

#### 功能说明

通过 `--min_response_len` 参数（默认 20）过滤掉 chosen 或 rejected 回答过短的样本。太短的代码回答（如 `"Yes"`、`"No"`、`"pass"`）对 DPO 训练没有区分价值，属于应去除的噪声。

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `is_valid_raw_row()` | 检查 `text_len(chosen) >= min_response_len` 且 `text_len(rejected) >= min_response_len` |

#### 演示命令

```bash
# 过滤掉回答短于 100 字符的样本
MIN_RESPONSE_LEN=100 bash dpo/scripts/prepare_data.sh
```

### 5.4 进阶功能 ③：按比例划分数据集

#### 功能说明

支持 `--test_ratio` 参数（默认 `0.05`，即 5%），取代 proposal 中最初设计的固定 500 条测试集划分方式。按比例划分的好处是：如果源数据扩充了，比例自动适配，不需要改参数。

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `split_by_ratio()` | `test_count = max(1, floor(len * test_ratio))`，确保至少 1 条测试数据且不会全部分给测试集 |

#### 演示命令

```bash
# 10% 作为测试集
TEST_RATIO=0.1 bash dpo/scripts/prepare_data.sh
```

### 5.5 进阶功能 ④：生成样本预览文件

#### 功能说明

自动输出 `sample_preview.json`，包含前 N 条（默认 10 条）转换后的训练样本。该文件供人工质量检查使用，无需打开数百 MB 的大文件即可快速验证格式和内容的正确性。

#### 演示命令

```bash
# 预览前 20 条
PREVIEW_SIZE=20 bash dpo/scripts/prepare_data.sh
```

#### 输出格式

输出至 `dpo/data/sample_preview.json`，为 `code_dpo_train.json` 的前 N 条切片，格式完全一致。

### 5.6 进阶功能 ⑤：数据去重增强

#### 功能说明

在格式转换后，基于 `instruction` / `input` / `chosen` / `rejected` / `output` 五个字段的精确匹配进行去重。使用 Python `set` 哈希表实现 O(1) 查重，毫秒级完成。训练集和测试集各自独立去重。

#### 实现路径

| 文件 / 函数 / 脚本 | 作用 |
|---|---|
| `deduplicate()` | 遍历列表，以五字段元组为 key，用 `set` 记录已见 key，首次出现保留，后续跳过 |

#### 关键代码

```python
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
```

---

## 6. 与团队系统的集成说明

本模块为 **A4 DPO 对齐训练** 模块提供数据供给。

**集成方式**：文件共享 + 注册表

```
A3 输出:                              A4 读取:
─────────────────────                ─────────────────────
dpo/data/                            dpo/configs/qwen15_code_full_dpo.yaml
├── code_dpo_train.json  ────────→    dataset: code_dpo_train  ──→  DPO Trainer
├── code_dpo_test.json   ────────→    (MBPP 评测用)
└── dataset_info.json    ────────→    LLaMA-Factory 自动识别
```

**调用过程**：
1. 先运行 A3 脚本生成数据文件
2. A4 的 LLaMA-Factory 训练配置（YAML）中设置 `dataset: code_dpo_train`
3. LLaMA-Factory 启动时自动读取 `dataset_info.json`，将 `code_dpo_train` 映射到 `dpo/data/code_dpo_train.json`
4. 框架自动识别 `ranking: true`，使用 DPO pair-wise 损失函数进行训练

**接口约定**：
- 数据格式：JSON，UTF-8 编码
- 文件命名：`code_dpo_train.json` / `code_dpo_test.json`
- 注册机制：通过 `dataset_info.json` 注册到 LLaMA-Factory
- 质量保证：提供 `sample_preview.json` 人工检查 + 日志统计输出
- 错误处理：脚本返回非零退出码，输出错误信息；A4 检测到数据缺失时提前报错

---

## 7. 已知问题与后续改进

| 问题 | 当前原因 | 后续改进 |
|---|---|---|
| 去重为精确匹配，无法识别语义重复（如变量名不同但逻辑相同） | 精确去重基于字段字符串元组比较，不涉及语义理解 | 引入 sentence-transformers 或 CodeBERT + 余弦相似度，实现语义去重 |
| 仅输出统计文本，无可视化图表 | `compute_avg_lengths()` 仅计算平均值并打印 | 增加 matplotlib 或 seaborn 绘制长度分布直方图，自动保存为 PNG |
| 无中间结果缓存，每次运行重新处理全部数据 | 没有引入缓存机制 | 增加检查点（checkpoint）功能，如果输出文件已存在且参数未变则跳过处理 |
| 无 LLM 自动质量评估 | 目前仅依赖长度和缺失值等统计指标 | 引入 LLM-as-Judge，对 chosen/rejected 的代码正确性、可读性进行自动打分 |
<｜end▁of▁thinking｜>

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="shell_command">
<｜｜DSML｜｜parameter name="command" string="true">@'
