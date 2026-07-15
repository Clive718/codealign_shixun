#!/usr/bin/env python3
from __future__ import annotations

import html
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.etree import ElementTree as ET


ROOT = Path("/data/yekaiyang/zjx/20260617/assignment_A")
TEMPLATE = ROOT / "上课文档.docx"
OUTPUT = ROOT / "2026实训A方向上课文档.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def esc(text: object) -> str:
    return html.escape(str(text), quote=False)


def get_template_sectpr() -> str:
    with ZipFile(TEMPLATE) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": W_NS}
    sect = root.find(".//w:body/w:sectPr", ns)
    if sect is None:
        return (
            '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
            '<w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" '
            'w:header="851" w:footer="992" w:gutter="0"/></w:sectPr>'
        )
    return ET.tostring(sect, encoding="unicode")


def p(text: object = "", style: str | None = None, bold: bool = False, size: int | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    rpr_parts = []
    if bold:
        rpr_parts.append("<w:b/>")
    if size:
        rpr_parts.append(f'<w:sz w:val="{size}"/>')
    rpr = f"<w:rPr>{''.join(rpr_parts)}</w:rPr>" if rpr_parts else ""
    return f"<w:p>{ppr}<w:r>{rpr}<w:t>{esc(text)}</w:t></w:r></w:p>"


def br_paragraph(text: str, style: str | None = None, bold: bool = False) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
    parts = []
    for i, line in enumerate(str(text).splitlines()):
        if i:
            parts.append("<w:br/>")
        parts.append(f'<w:t xml:space="preserve">{esc(line)}</w:t>')
    return f"<w:p>{ppr}<w:r>{rpr}{''.join(parts)}</w:r></w:p>"


def code_block(code: str) -> str:
    paras = []
    for line in str(code).splitlines() or [""]:
        paras.append(
            '<w:p><w:pPr><w:shd w:val="clear" w:color="auto" w:fill="F2F2F2"/>'
            '<w:spacing w:before="0" w:after="0"/></w:pPr>'
            '<w:r><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/>'
            '<w:sz w:val="18"/></w:rPr>'
            f'<w:t xml:space="preserve">{esc(line)}</w:t></w:r></w:p>'
        )
    return "".join(paras)


def bullet(items: list[str]) -> str:
    return "".join(p(f"• {item}") for item in items)


def numbered(items: list[str]) -> str:
    return "".join(p(f"{i}. {item}") for i, item in enumerate(items, start=1))


def table(rows: list[list[str]], widths: list[int] | None = None) -> str:
    if not rows:
        return ""
    cols = max(len(r) for r in rows)
    widths = widths or [9000 // cols] * cols
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths[:cols])
    trs = []
    for ridx, row in enumerate(rows):
        cells = []
        for cidx in range(cols):
            text = row[cidx] if cidx < len(row) else ""
            fill = "D9EAF7" if ridx == 0 else "FFFFFF"
            bold = "<w:b/>" if ridx == 0 else ""
            cells.append(
                '<w:tc><w:tcPr>'
                f'<w:tcW w:w="{widths[cidx if cidx < len(widths) else -1]}" w:type="dxa"/>'
                f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>'
                "</w:tcPr>"
                f'<w:p><w:r><w:rPr>{bold}</w:rPr><w:t>{esc(text)}</w:t></w:r></w:p>'
                "</w:tc>"
            )
        trs.append(f"<w:tr>{''.join(cells)}</w:tr>")
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="A6A6A6"/>'
        f"</w:tblBorders></w:tblPr><w:tblGrid>{grid}</w:tblGrid>{''.join(trs)}</w:tbl>"
    )


def placeholder(text: str) -> str:
    return (
        '<w:tbl><w:tblPr><w:tblW w:w="0" w:type="auto"/>'
        '<w:tblBorders><w:top w:val="dashed" w:sz="8" w:space="0" w:color="4472C4"/>'
        '<w:left w:val="dashed" w:sz="8" w:space="0" w:color="4472C4"/>'
        '<w:bottom w:val="dashed" w:sz="8" w:space="0" w:color="4472C4"/>'
        '<w:right w:val="dashed" w:sz="8" w:space="0" w:color="4472C4"/>'
        "</w:tblBorders></w:tblPr>"
        '<w:tblGrid><w:gridCol w:w="9000"/></w:tblGrid>'
        '<w:tr><w:trPr><w:trHeight w:val="1800"/></w:trPr>'
        '<w:tc><w:tcPr><w:tcW w:w="9000" w:type="dxa"/>'
        '<w:shd w:val="clear" w:color="auto" w:fill="EAF2F8"/>'
        '<w:vAlign w:val="center"/></w:tcPr>'
        f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:rPr><w:b/></w:rPr><w:t>{esc(text)}</w:t></w:r></w:p>'
        "</w:tc></w:tr></w:tbl>"
    )


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def build_body() -> str:
    parts: list[str] = []
    add = parts.append

    add(p("2026实训A方向上课文档", "Title"))
    add(p("推理 / 强化学习 / 对齐：偏好数据构造、DPO 对齐训练与推理增强策略", "Subtitle"))
    add(p("基于 LLaMA-Factory 和本地 Qwen1.5-0.5B-Chat，完成 Python 代码任务的偏好数据构造、DPO 对齐训练、MBPP 执行评测和 CoT 推理增强。"))
    add(p("适用目录：/data/yekaiyang/zjx/20260617/assignment_A"))
    add(p("配套材料：2026实训A方向.pptx、dpo 代码目录、tts 代码目录。"))

    add(p("目录", "Heading1"))
    add(bullet([
        "一、课程总体要求与个人模块要求",
        "二、环境配置与项目目录",
        "三、A1：推理指令数据构造",
        "四、A2：SFT 微调",
        "五、A3：偏好数据构造",
        "六、A4：DPO 对齐训练",
        "七、A5：推理增强策略",
        "八、验收材料与常见问题",
    ]))
    add(page_break())

    add(p("一、课程总体要求与个人模块要求", "Heading1"))
    add(p("本方向围绕“推理 / 强化学习 / 对齐”展开，要求学生以小组形式完成一个完整系统，同时每位同学必须负责一个能够独立运行、独立演示、独立讲解的核心模块。"))
    add(p("1.1 总体要求", "Heading2"))
    add(bullet([
        "每位同学必须认领至少一个核心模块，可以两位同学共同负责一个模块。",
        "每个核心模块必须能够独立运行、独立演示、独立讲解。",
        "每组建议由 4–5 名同学组成，并形成一个完整系统。",
        "组内模块之间必须存在明确的接口关系和协作关系。",
        "最终验收时既看小组系统，也严格检查个人模块。",
        "评测、错误分析、训练前后对比、运行日志等属于全组公共要求，不建议单独作为个人核心模块。",
    ]))
    add(placeholder("图片占位：插入 PPT 第 1 页“总体要求”截图"))
    add(p("1.2 个人模块要求", "Heading2"))
    add(table([
        ["要求", "说明"],
        ["独立输入输出", "明确模块输入和输出，例如输入 prompt，输出模型回答；输入任务描述，输出调用轨迹。"],
        ["独立运行", "必须有自己的脚本、Notebook、CLI 命令或 Web Demo。"],
        ["独立演示", "验收时本人需要亲自运行、讲解并展示结果，可使用 vibe coding 制作演示系统。"],
        ["独立代码验收", "需要提交个人负责部分的代码、README、配置文件、运行日志、截图或视频。"],
        ["可集成", "需要提供函数、API、JSON 文件、命令行或中间文件接口给组内其他模块调用。"],
        ["有实验或对比", "至少包含 baseline、训练前后对比、不同参数对比、错误分析或案例分析。"],
    ], [2200, 6800]))
    add(placeholder("图片占位：插入 PPT 第 2 页“个人模块要求”截图"))

    add(p("二、环境配置与项目目录", "Heading1"))
    add(p("2.1 基础环境", "Heading2"))
    add(p("本项目默认使用已经配置好的 assignment_A Conda 环境。所有命令建议在项目根目录执行。"))
    add(code_block("conda activate assignment_A\ncd /data/yekaiyang/zjx/20260617/assignment_A"))
    add(p("检查 GPU："))
    add(code_block("python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())\""))
    add(p("如遇 torchvision 或 torchaudio 版本不匹配，需要保证 torch / torchvision / torchaudio 与 CUDA 版本一致。例如当前环境 torch 为 2.5.1+cu124 时，建议使用 torchvision==0.20.1、torchaudio==2.5.1。"))
    add(p("2.2 目录说明", "Heading2"))
    add(table([
        ["目录或文件", "说明"],
        ["Qwen1.5-0.5B-Chat", "本地基础模型。"],
        ["py-dpo-v0.1/py-dpo.parquet", "Python 代码任务偏好数据原始文件。"],
        ["dpo", "偏好数据构造、DPO 训练、MBPP 评测相关代码和输出。"],
        ["tts", "推理增强、CoT prompt、CoT 评测相关代码。"],
        ["LlamaFactory", "训练框架源码。"],
        ["2026实训A方向.pptx", "课程 PPT。"],
    ], [3200, 5800]))

    add(page_break())
    add(p("三、A1：推理指令数据构造", "Heading1"))
    add(p("3.1 模块目标", "Heading2"))
    add(p("A1 模块负责把原始 Python 代码指令数据整理为 LLaMA-Factory 可训练的 instruction / input / output 格式，作为后续 SFT 微调的基础数据。它的核心作用是把“原始任务文本”变成“模型可学习的监督样本”。"))
    add(p("3.2 输入数据与目标", "Heading2"))
    add(table([
        ["项目", "内容"],
        ["原始数据", "python_code_instructions_18k_alpaca 中的 parquet 文件"],
        ["任务类型", "Python 代码指令生成"],
        ["目标格式", "instruction / input / output"],
        ["主要用途", "供 SFT 训练以及后续推理验证"],
        ["关键脚本", "sft/scripts/prepare_code_sft_data.py"],
    ], [2400, 6600]))
    add(p("每条样本都需要保留清晰的任务描述、可选输入和标准答案；这样在训练时模型才能真正学习“给定问题 → 正确代码”的映射关系。"))
    add(placeholder("图片占位：插入 PPT 第 7 页“A1 原始数据与处理目标”截图"))

    add(p("3.3 数据处理流程", "Heading2"))
    add(numbered([
        "读取原始 parquet 文件，并统一尝试 pandas / pyarrow / datasets 三种读取方式。",
        "提取 instruction、input、output 等字段；若缺少 instruction 或 output，则将 prompt 兜底为 instruction。",
        "清洗样本，删除空值、重复样本以及格式异常的记录。",
        "按 90% / 5% / 5% 的比例划分 train / valid / test，并保证每个集合都至少有样本。",
        "将处理结果保存为 code_sft_train.json、code_sft_valid.json、code_sft_test.json，并生成 dataset_info.json。",
        "输出读取样本数、保留样本数和最终保存路径，方便后续训练与复现。",
    ]))
    add(placeholder("图片占位：插入 PPT 第 8 页“A1 数据清洗与划分流程”截图"))

    add(p("3.4 运行命令", "Heading2"))
    add(p("执行数据准备："))
    add(code_block("bash sft/scripts/prepare_data.sh"))
    add(p("快速小样本调试："))
    add(code_block("LIMIT=1000 bash sft/scripts/prepare_data.sh"))
    add(p("也可以直接运行脚本并设置参数："))
    add(code_block("python3 sft/scripts/prepare_code_sft_data.py \\\n  --source_dir python_code_instructions_18k_alpaca \\\n  --output_dir sft/data \\\n  --train_ratio 0.90 \\\n  --valid_ratio 0.05 \\\n  --seed 42"))

    add(p("3.5 输出文件", "Heading2"))
    add(table([
        ["文件", "说明"],
        ["sft/data/code_sft_train.json", "用于 SFT 的训练集。"],
        ["sft/data/code_sft_valid.json", "用于监控训练效果的验证集。"],
        ["sft/data/code_sft_test.json", "用于最终推理评测的测试集。"],
        ["sft/data/dataset_info.json", "LLaMA-Factory 数据注册信息。"],
    ], [3000, 6000]))
    add(placeholder("图片占位：插入 PPT 第 9 页左侧“训练集示例”截图"))
    add(placeholder("图片占位：插入 PPT 第 9 页中间“验证集示例”截图"))
    add(placeholder("图片占位：插入 PPT 第 9 页右侧“数据目录结果”截图"))

    add(page_break())
    add(p("四、A2：SFT 微调", "Heading1"))
    add(p("4.1 模块目标", "Heading2"))
    add(p("A2 模块利用 A1 生成的监督数据，在本地 Qwen1.5-0.5B-Chat 上进行全量 SFT。目标是让模型学会根据指令生成更准确、更符合 Python 代码风格的回答。"))
    add(p("4.2 模型与配置", "Heading2"))
    add(table([
        ["项目", "内容"],
        ["基础模型", "./Qwen1.5-0.5B-Chat"],
        ["训练框架", "LLaMA-Factory"],
        ["训练数据", "sft/data/code_sft_train.json"],
        ["验证数据", "sft/data/code_sft_valid.json"],
        ["训练配置", "sft/configs/qwen15_code_full_sft.yaml"],
        ["输出目录", "sft/outputs/qwen15_code_full_sft"],
    ], [2400, 6600]))
    add(placeholder("图片占位：插入 PPT 第 10 页“模型配置与训练数据”截图"))

    add(p("4.3 训练流程", "Heading2"))
    add(numbered([
        "切换到项目根目录，并设置环境变量（如 GPU_ID、LLAMA_FACTORY_DIR）。",
        "加载本地基础模型和训练配置文件，启动 LLaMA-Factory 的训练命令。",
        "在训练过程中实时观察 loss、学习率和 GPU 显存占用情况。",
        "训练结束后保存 checkpoints、tokenizer、配置和训练日志。",
        "使用保存的 checkpoint 对测试集进行批量生成。",
        "再用 MBPP 执行评测脚本评估生成代码的正确率。",
    ]))
    add(placeholder("图片占位：插入 PPT 第 11 页“A2 训练流程示意图”截图"))

    add(p("4.4 运行命令", "Heading2"))
    add(p("启动训练："))
    add(code_block("bash sft/scripts/train.sh"))
    add(p("指定 GPU："))
    add(code_block("GPU_ID=0 bash sft/scripts/train.sh"))
    add(p("批量预测测试集："))
    add(code_block("MODEL_PATH=/data/yekaiyang/zjx/20260617/assignment_A/sft/outputs/qwen15_code_full_sft/checkpoint-600 \\\n bash sft/scripts/predict_full.sh"))
    add(p("执行评测："))
    add(code_block("bash sft/scripts/evaluate_full.sh"))
    add(p("单条推理："))
    add(code_block("bash sft/scripts/infer_full.sh --prompt \"Write a Python function to check whether a string is a palindrome.\""))
    add(p("一键完整流程："))
    add(code_block("bash sft/scripts/run_all.sh"))

    add(p("4.5 输出结果", "Heading2"))
    add(table([
        ["输出", "说明"],
        ["sft/outputs/qwen15_code_full_sft", "训练得到的模型目录。"],
        ["sft/outputs/qwen15_code_full_predict/generated_predictions.jsonl", "测试集生成结果。"],
        ["sft/outputs/eval_mbpp/mbpp_metrics.json", "MBPP 执行评测结果。"],
        ["sft/outputs/infer_results.jsonl", "单条或者批量推理结果。"],
    ], [3400, 5600]))
    add(placeholder("图片占位：插入 PPT 第 12 页左侧“训练终端输出”截图"))
    add(placeholder("图片占位：插入 PPT 第 12 页中间“预测结果文件”截图"))
    add(placeholder("图片占位：插入 PPT 第 12 页右侧“评测指标结果”截图"))

    add(p("4.6 训练中的关键注意点", "Heading2"))
    add(bullet([
        "建议先用小样本验证数据处理是否正确，再正式启动大规模训练。",
        "若显存不足，可适当减小 batch size 或缩短 cutoff_len。",
        "若要用 bf16，需同时把配置中的 fp16 和 bf16 参数调整为对应设置。",
        "评测结果需要关注 pass_at_1、syntax_pass_rate、avg_test_pass_rate 三类指标。",
    ]))
    add(placeholder("图片占位：插入 PPT 第 13 页“A2 关键注意点与结果说明”截图"))

    add(page_break())
    add(p("五、A3：偏好数据构造", "Heading1"))
    add(p("3.1 模块目标", "Heading2"))
    add(p("A3 模块负责构造 DPO 或其他偏好学习算法所需的 chosen / rejected 数据，是后续对齐训练的关键输入。数据质量会直接影响 DPO 训练稳定性和模型输出质量。"))
    add(p("3.2 原始数据说明", "Heading2"))
    add(table([
        ["项目", "内容"],
        ["数据集地址", "https://huggingface.co/datasets/jondurbin/py-dpo-v0.1"],
        ["本地路径", "py-dpo-v0.1/py-dpo.parquet"],
        ["数据类型", "Apache Parquet"],
        ["任务方向", "Python coding DPO 偏好数据"],
        ["核心字段", "prompt、chosen、rejected、id"],
    ], [2400, 6600]))
    add(p("原始数据是典型的 DPO 三元组：同一个 prompt 下，chosen 是更优代码回答，rejected 是较差代码回答。DPO 训练不需要人工分数，而是学习 chosen 相对 rejected 更优的偏好关系。"))
    add(code_block('{\n  "id": "样本 id",\n  "prompt": "Python 编程任务描述",\n  "chosen": "更优的 Python 代码回答",\n  "rejected": "较差的 Python 代码回答"\n}'))
    add(placeholder("图片占位：插入 PPT 第 4 页“原始数据下载地址与数据内容”截图"))

    add(p("3.3 本节实训基础要求（可以在当前代码的引导下完全实现）", "Heading2"))
    add(numbered([
        "读取原始 py-dpo.parquet 数据文件，提取其中的 prompt、chosen、rejected 等字段。",
        "对原始样本进行清洗，去除缺失 prompt、chosen 或 rejected 的无效样本。",
        "将原始偏好数据转换为 LLaMA-Factory 支持的 DPO ranking 数据格式。",
        "按照随机种子划分训练集和测试集，并分别保存为 code_dpo_train.json 和 code_dpo_test.json。",
        "自动生成 dataset_info.json，完成数据集在 LLaMA-Factory 中的注册。",
        "输出数据处理日志，包括读取样本数、训练集数量、测试集数量和输出文件路径。",
    ]))
    add(placeholder("图片占位：插入 PPT 第 5 页“A3 基础要求”截图"))

    add(p("3.4 处理流程", "Heading2"))
    add(numbered([
        "读取原始 parquet 文件：py-dpo-v0.1/py-dpo.parquet。",
        "使用随机种子打乱数据，默认 seed=42。",
        "清洗无效样本，保留 prompt、chosen、rejected 都非空的数据。",
        "默认抽取 TEST_SIZE=500 条作为测试集，其余样本作为训练集。",
        "训练集转换为 instruction / input / chosen / rejected 格式。",
        "测试集转换为 instruction / input / output 格式，output 使用原始 chosen。",
        "写出 dataset_info.json，将 code_dpo_train 注册为 ranking 数据集。",
    ]))
    add(code_block('训练集格式：\n{\n  "instruction": "Complete the following Python coding task. ...",\n  "input": "",\n  "chosen": "better Python solution",\n  "rejected": "worse Python solution"\n}\n\n测试集格式：\n{\n  "instruction": "Complete the following Python coding task. ...",\n  "input": "",\n  "output": "reference Python solution"\n}'))
    add(placeholder("图片占位：插入 PPT 第 6 页“A3 处理流程”截图"))

    add(p("3.5 代码运行命令", "Heading2"))
    add(p("运行数据准备："))
    add(code_block("bash dpo/scripts/prepare_data.sh"))
    add(p("小样本调试："))
    add(code_block("MAX_TRAIN_SAMPLES=200 TEST_SIZE=50 bash dpo/scripts/prepare_data.sh"))
    add(p("自定义随机种子："))
    add(code_block("SEED=123 bash dpo/scripts/prepare_data.sh"))
    add(p("自定义原始数据目录和输出目录："))
    add(code_block("SOURCE_DIR=py-dpo-v0.1 OUTPUT_DIR=dpo/data bash dpo/scripts/prepare_data.sh"))
    add(p("核心 Python 脚本也可以直接运行："))
    add(code_block("python3 dpo/scripts/prepare_dpo_data.py \\\n  --source_dir py-dpo-v0.1 \\\n  --output_dir dpo/data \\\n  --test_size 500 \\\n  --seed 42 \\\n  --max_train_samples 0"))

    add(p("3.6 代码运行图例", "Heading2"))
    add(placeholder("图片占位：插入 PPT 第 8 页左侧“数据预处理输出”截图"))
    add(placeholder("图片占位：插入 PPT 第 8 页中间“保存的文件”截图"))
    add(placeholder("图片占位：插入 PPT 第 8 页右侧“训练和测试集文件示例”截图"))
    add(p("当前项目已生成的数据规模：训练集 8924 条，测试集 485 条，总计 9409 条有效样本。"))

    add(p("3.7 输出文件", "Heading2"))
    add(table([
        ["文件", "说明"],
        ["dpo/data/code_dpo_train.json", "DPO 训练集，包含 instruction / input / chosen / rejected。"],
        ["dpo/data/code_dpo_test.json", "生成测试集，包含 instruction / input / output。"],
        ["dpo/data/dataset_info.json", "LLaMA-Factory 数据注册表。"],
    ], [3600, 5400]))

    add(p("3.8 本节实训进阶要求", "Heading2"))
    add(numbered([
        "修改 prepare_dpo_data.py，增加数据统计功能，例如输出平均 prompt 长度、chosen 长度、rejected 长度。",
        "增加数据质量过滤逻辑，例如过滤掉 chosen 或 rejected 太短的样本。",
        "增加命令行参数 --min_response_len，用于控制最短回答长度。",
        "将训练集和测试集划分方式从固定数量改为按比例划分，例如 95% 训练、5% 测试。",
        "增加 sample_preview.json，保存前 10 条转换后样本，方便人工检查。",
    ]))

    add(page_break())
    add(p("六、A4：DPO 对齐训练", "Heading1"))
    add(p("4.1 模块目标", "Heading2"))
    add(p("A4 模块使用 A3 生成的偏好数据进行 DPO 对齐训练，使模型在 Python 代码任务上更倾向于生成 chosen 风格的高质量答案。"))
    add(p("4.2 模型与训练数据", "Heading2"))
    add(table([
        ["项目", "内容"],
        ["基础模型", "assignment_A/Qwen1.5-0.5B-Chat"],
        ["训练框架", "LLaMA-Factory"],
        ["训练数据", "dpo/data/code_dpo_train.json"],
        ["训练配置", "dpo/configs/qwen15_code_full_dpo.yaml"],
        ["训练输出", "dpo/outputs/qwen15_code_full_dpo"],
    ], [2400, 6600]))
    add(placeholder("图片占位：插入 PPT 第 10 页“模型与环境配置 / 训练数据示例 / 注册信息”截图"))

    add(p("4.3 本节实训基础要求（可以在当前代码的引导下完全实现）", "Heading2"))
    add(numbered([
        "基于已经构造好的 code_dpo_train.json，配置 DPO 训练所需的数据路径和数据集名称。",
        "加载本地基础模型 Qwen1.5-0.5B-Chat，并将其同时作为训练初始模型和参考模型。",
        "使用 LLaMA-Factory 启动 DPO 训练流程，完成 chosen / rejected 偏好对齐训练。",
        "配置训练过程中的关键参数，包括训练轮数、学习率、batch size、梯度累积和最大输入长度。",
        "支持通过环境变量指定 GPU、Python 解释器和训练配置文件。",
        "将训练后的模型、tokenizer、训练状态和日志保存到指定输出目录。",
        "使用给出的 MBPP 评估脚本或公开 benchmark 评估模型性能。",
    ]))
    add(placeholder("图片占位：插入 PPT 第 11 页“A4 基础要求”截图"))

    add(p("4.4 关键配置说明", "Heading2"))
    add(table([
        ["参数", "值", "说明"],
        ["model_name_or_path", "./Qwen1.5-0.5B-Chat", "policy 初始模型。"],
        ["stage", "dpo", "使用 DPO 训练阶段。"],
        ["finetuning_type", "full", "全参数微调。"],
        ["pref_beta", "0.1", "DPO 偏好约束强度。"],
        ["pref_loss", "sigmoid", "标准 DPO 损失。"],
        ["ref_model", "./Qwen1.5-0.5B-Chat", "reference model，与 policy 同一起点。"],
        ["dataset", "code_dpo_train", "A3 生成并注册的数据集。"],
        ["cutoff_len", "2048", "最大输入长度。"],
        ["per_device_train_batch_size", "1", "单卡 batch size。"],
        ["gradient_accumulation_steps", "8", "梯度累积步数。"],
        ["num_train_epochs", "3.0", "训练轮数。"],
        ["plot_loss", "true", "训练结束后保存 loss 曲线。"],
    ], [2600, 2200, 4200]))
    add(p("注意：DPO 会同时加载 policy model 和 reference model，显存占用通常高于 SFT。如遇 OOM，可降低 batch size、减少 cutoff_len，或改用 LoRA / QLoRA。"))

    add(p("4.5 代码运行命令", "Heading2"))
    add(p("单卡训练："))
    add(code_block("GPU_ID=0 bash dpo/scripts/train.sh"))
    add(p("指定配置文件："))
    add(code_block("CONFIG=dpo/configs/qwen15_code_full_dpo.yaml GPU_ID=0 bash dpo/scripts/train.sh"))
    add(p("八卡 torchrun 训练示例："))
    add(code_block("export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\nexport PYTHONPATH=/data/yekaiyang/zjx/20260617/assignment_A/LlamaFactory/src:$PYTHONPATH\nexport HF_DATASETS_OFFLINE=1\nexport TRANSFORMERS_OFFLINE=1\nexport WANDB_DISABLED=true\nexport TOKENIZERS_PARALLELISM=false\n\ntorchrun --nproc_per_node=8 \\\n  -m llamafactory.cli train \\\n  dpo/configs/qwen15_code_full_dpo.yaml"))
    add(p("评估 base 模型："))
    add(code_block("BATCH_SIZE=32 bash dpo/scripts/run_mbpp_base_eval.sh"))
    add(p("评估 DPO 模型："))
    add(code_block("BATCH_SIZE=32 bash dpo/scripts/run_mbpp_dpo_final_eval.sh"))
    add(p("并行评估 base 与 DPO 并生成对比指标："))
    add(code_block("BASE_GPUS=0 DPO_GPUS=1 BATCH_SIZE=16 bash dpo/scripts/run_mbpp_base_dpo_eval.sh"))
    add(placeholder("图片占位：插入 PPT 第 12 页“A4 代码运行命令”截图"))

    add(p("4.6 代码运行图例", "Heading2"))
    add(placeholder("图片占位：插入 PPT 第 13 页左侧“训练时终端输出”截图"))
    add(placeholder("图片占位：插入 PPT 第 13 页中间“损失曲线图”截图，可对应 dpo/outputs/qwen15_code_full_dpo/training_loss.png"))
    add(placeholder("图片占位：插入 PPT 第 13 页右侧“保存目录结果”截图"))
    add(placeholder("图片占位：插入 PPT 第 14 页左侧“Base 测试结果和指标”截图"))
    add(placeholder("图片占位：插入 PPT 第 14 页右侧“DPO 测试结果和指标”截图"))

    add(p("4.7 训练输出与示例结果", "Heading2"))
    add(table([
        ["输出文件", "说明"],
        ["dpo/outputs/qwen15_code_full_dpo/model.safetensors", "训练后的模型权重。"],
        ["dpo/outputs/qwen15_code_full_dpo/training_loss.png", "LLaMA-Factory 自动绘制的训练 loss 曲线。"],
        ["dpo/outputs/qwen15_code_full_dpo/training_rewards_accuracies.png", "DPO 奖励准确率曲线。"],
        ["dpo/outputs/qwen15_code_full_dpo/trainer_state.json", "训练过程日志，包括 loss、reward、learning_rate 等。"],
        ["dpo/outputs/qwen15_code_full_dpo/checkpoint-*", "训练中间 checkpoint。"],
        ["dpo/outputs/mbpp_eval_base/mbpp_metrics.json", "base 模型 MBPP 执行评测结果。"],
        ["dpo/outputs/mbpp_eval_dpo_final/mbpp_metrics.json", "DPO 模型 MBPP 执行评测结果。"],
    ], [4100, 4900]))
    add(p("当前已完成的一次训练结果示例："))
    add(table([
        ["指标", "数值"],
        ["epoch", "3.0"],
        ["global_step", "420"],
        ["train_loss", "0.0643"],
        ["train_runtime", "1272.64 秒"],
        ["train_samples_per_second", "21.037"],
    ], [3500, 5500]))
    add(p("当前 MBPP sanitized test 执行评测结果示例："))
    add(table([
        ["模型", "num_tasks", "pass_at_1", "syntax_pass_rate", "avg_test_pass_rate", "passed_tasks"],
        ["Base Qwen1.5-0.5B-Chat", "257", "0.0039", "0.7938", "0.0090", "1"],
        ["DPO qwen15_code_full_dpo", "257", "0.0156", "0.7121", "0.0361", "4"],
    ], [2400, 1300, 1300, 1700, 1700, 1300]))
    add(p("说明：MBPP 评测会将生成代码写入临时文件，在子进程中运行 assert 测试，并设置超时与基础资源限制。因此它是执行型 benchmark，不只是文本相似度评估。"))

    add(p("4.8 本节实训进阶要求", "Heading2"))
    add(numbered([
        "将当前 full DPO 改为 LoRA DPO：把 finetuning_type 改为 lora，并增加 lora_rank、lora_alpha、lora_dropout、lora_target 等参数。",
        "对比 full finetuning 与 LoRA finetuning 的显存占用、训练速度、loss 曲线、MBPP pass_at_1 和样例质量。",
        "尝试 DPO 类损失函数改进，例如 pref_loss: hinge、ipo、orpo、simpo，并对比训练稳定性和最终评测结果。",
        "使用 SimPO 或 ORPO 作为 DPO 改进算法进行实验，分析是否减少 reference model 依赖、是否降低显存占用。",
        "学习并尝试 KTO 偏好优化方法，完成数据格式适配并运行 stage: kto 的训练配置。",
        "尝试 GRPO 或 PPO 等强化学习式对齐算法，设计代码任务 reward，例如语法正确、通过单元测试、函数名匹配、输出格式正确等。",
        "对比 Base、Full DPO、LoRA DPO、SimPO/ORPO、GRPO/PPO 等模型在训练成本和 MBPP 执行评测上的效果差异。",
    ]))
    add(placeholder("图片占位：插入 PPT 第 15 页“A4 进阶要求”截图"))

    add(page_break())
    add(p("七、A5：推理增强策略", "Heading1"))
    add(p("5.1 模块目标", "Heading2"))
    add(p("A5 模块不继续训练模型，而是在推理阶段通过 prompt 设计、CoT、采样、验证器重排序和 test-time scaling 方法提升模型输出质量。它适合与 A4 模块形成“训练增强 + 推理增强”的对比。"))

    add(p("5.2 本节实训基础要求（可以在当前代码的引导下完全实现）", "Heading2"))
    add(numbered([
        "分别加载训练前的 base 模型和训练后的 DPO 模型，对同一批代码任务进行生成推理。",
        "使用相同测试集、相同生成参数，对 base 模型和 DPO 模型输出进行公平对比。",
        "使用 CoT few-shot prompt 增强推理过程，引导模型输出 Reasoning、Key steps 和 Final code。",
        "从模型回答中提取 Final code 代码块，并进行语法检查和 MBPP assert 执行测试。",
        "将推理结果保存为 metrics 和 cases 文件，方便后续展示和错误分析。",
    ]))
    add(placeholder("图片占位：插入 PPT 第 16 页“A5 基础要求”截图"))

    add(p("5.3 CoT Prompt 设计", "Heading2"))
    add(p("当前 CoT 模板位于 tts/cot_code_example.py。系统提示要求模型严格按照 Reasoning、Key steps、Final code 三段输出，评测时只提取 Final code 中的 Python 代码。"))
    add(code_block("Reasoning:\n- Briefly explain the idea.\n\nKey steps:\n1. List the implementation steps.\n\nFinal code:\n```python\n# complete executable Python code here\n```"))

    add(p("5.4 代码运行命令", "Heading2"))
    add(p("查看 CoT prompt 示例："))
    add(code_block("python3 tts/cot_code_example.py"))
    add(p("评估 base 模型的普通 MBPP 结果："))
    add(code_block("BATCH_SIZE=32 bash dpo/scripts/run_mbpp_base_eval.sh"))
    add(p("评估 DPO 模型的普通 MBPP 结果："))
    add(code_block("BATCH_SIZE=32 bash dpo/scripts/run_mbpp_dpo_final_eval.sh"))
    add(p("评估 CoT 增强后的模型："))
    add(code_block("MODEL_PATH=Qwen1.5-0.5B-Chat \\\nBATCH_SIZE=32 \\\nOUTPUT_DIR=tts/outputs/cot_code_eval_base \\\nbash tts/run_cot_code_eval.sh"))
    add(p("评估 DPO 模型 + CoT："))
    add(code_block("MODEL_PATH=dpo/outputs/qwen15_code_full_dpo \\\nBATCH_SIZE=32 \\\nOUTPUT_DIR=tts/outputs/cot_code_eval_dpo \\\nbash tts/run_cot_code_eval.sh"))
    add(p("小样本快速调试："))
    add(code_block("MODEL_PATH=dpo/outputs/qwen15_code_full_dpo \\\nBATCH_SIZE=4 \\\nOUTPUT_DIR=tts/outputs/cot_code_eval_debug \\\nbash tts/run_cot_code_eval.sh --limit 20"))
    add(placeholder("图片占位：插入 PPT 第 17 页“A5 代码运行命令”截图"))

    add(p("5.5 处理流程", "Heading2"))
    add(numbered([
        "读取输入任务，默认使用 ../mbpp/sanitized/test-00000-of-00001.parquet。",
        "将每道题转换为包含题目描述和测试用例的 task 文本。",
        "调用 cot_code_example.py 中的 build_chat_messages 构造 CoT prompt。",
        "加载指定模型，按 batch 生成回答。",
        "从回答中提取 Final code 代码块。",
        "使用 ast.parse 做语法检查。",
        "将代码和 MBPP 测试用例写入临时 Python 文件，在子进程中执行。",
        "统计 syntax_pass_rate、pass_at_1、avg_test_pass_rate、exact_match 和 avg_code_token_f1。",
    ]))
    add(p("5.6 输出文件", "Heading2"))
    add(table([
        ["文件", "说明"],
        ["tts/outputs/cot_code_eval/cot_code_metrics.json", "CoT 评测指标。"],
        ["tts/outputs/cot_code_eval/cot_code_cases.jsonl", "每道题的任务、回答、代码、测试结果。"],
        ["tts/outputs/cot_code_eval_base", "base 模型 + CoT 输出目录示例。"],
        ["tts/outputs/cot_code_eval_dpo", "DPO 模型 + CoT 输出目录示例。"],
    ], [4100, 4900]))
    add(placeholder("图片占位：插入 PPT 第 18 页“加入 CoT 后 base 测试结果和指标”截图"))

    add(p("5.7 本节实训进阶要求：Test-Time Scaling", "Heading2"))
    add(p("Test-Time Scaling 的核心思想是：训练结束后不再更新模型参数，而是在推理阶段投入更多计算量，通过多候选采样、搜索、验证、执行反馈和反思修正来选择更优答案。"))
    add(numbered([
        "实现 Self-Consistency 多路径采样：同一道题采样多个推理路径，提取最终代码或执行结果，选择最一致的答案。",
        "实现 Best-of-N 生成与重排序：同一道题生成 N 个候选代码，根据语法、测试通过数、代码格式、verifier 分数选择最优答案。",
        "增加执行验证过滤：对候选代码运行 MBPP assert 测试，优先选择通过更多测试的候选答案。",
        "学习并实现 CodeT 思路：让模型同时生成候选代码和测试用例，通过生成测试筛选代码。",
        "实现 Self-Verification 自验证：模型先生成代码，再检查自己的代码是否满足题目要求，并给出 verification_score。",
        "实现 Reflexion 反思修正机制：根据语法错误或测试失败信息生成反思，再带着反思重新生成代码。",
        "实现 Tree of Thoughts 搜索式推理：生成多个中间思路，对路径进行评分、扩展和剪枝，最后选择得分最高的代码。",
        "实现 Verifier / Reward Model 重排序：输入题目、候选代码、执行结果，输出 correct_score，对多个候选进行排序。",
        "系统对比 greedy decoding、CoT、Self-Consistency、Best-of-N、execution filtering、Reflexion、Tree of Thoughts 的效果与成本。",
    ]))
    add(table([
        ["策略", "需要修改的代码", "主要指标"],
        ["Self-Consistency", "tts/evaluate_cot_code.py 增加 num_samples", "多数一致率、pass_at_1、耗时"],
        ["Best-of-N", "增加候选生成与规则打分函数", "best pass_at_1、生成成本"],
        ["Execution Filtering", "复用 run_one_assert", "passed_tests、avg_test_pass_rate"],
        ["Reflexion", "增加多轮 refine prompt", "修正前后通过率"],
        ["Tree of Thoughts", "增加 tree_width / tree_depth 搜索", "搜索成本、最终 pass rate"],
        ["Verifier Rerank", "增加规则版或模型版 verifier", "排序前后 pass_at_1"],
    ], [2200, 4000, 2800]))
    add(placeholder("图片占位：插入 PPT 第 19 页“A5 进阶要求”截图"))

    add(page_break())
    add(p("八、验收材料与常见问题", "Heading1"))
    add(p("6.1 建议提交材料", "Heading2"))
    add(table([
        ["材料", "说明"],
        ["代码目录", "个人负责模块的 scripts、configs、README。"],
        ["运行命令", "能复现实验的完整命令，包括环境变量。"],
        ["运行日志", "数据处理日志、训练日志、评测日志。"],
        ["模型或输出", "训练模型目录、metrics.json、cases.jsonl、对比报告。"],
        ["截图或视频", "PPT 截图、终端截图、loss 曲线、评测指标截图。"],
        ["实验分析", "baseline vs 改进方法、错误案例、成本分析。"],
    ], [2600, 6400]))
    add(p("6.2 常见问题", "Heading2"))
    add(table([
        ["问题", "处理建议"],
        ["torchvision::nms does not exist", "torch 与 torchvision 版本不匹配，torch 2.5.1+cu124 建议 torchvision==0.20.1。"],
        ["libcudart.so.13 not found", "torchaudio 版本过新，torch 2.5.1 建议 torchaudio==2.5.1。"],
        ["DPO 训练 OOM", "降低 batch size / cutoff_len，使用 LoRA，或增加 GPU 数。"],
        ["训练脚本不会自动 benchmark", "训练完成后单独运行 run_mbpp_base_eval.sh / run_mbpp_dpo_final_eval.sh。"],
        ["静态指标和执行评测混淆", "test_code_dpo.py 是静态规则指标；mbpp_eval_dpo.py 和 evaluate_cot_code.py 会执行测试。"],
        ["CoT 输出无法提取代码", "检查是否包含 Final code 标记和 Python 代码块。"],
    ], [3200, 5800]))
    add(p("6.3 课堂总结", "Heading2"))
    add(p("本课程 A 方向从数据、训练、推理三个层次组织实训：A3 负责把 Python 偏好数据构造成 DPO ranking 格式；A4 使用 LLaMA-Factory 完成 DPO 对齐训练并用 MBPP 执行评测验证效果；A5 在不改模型参数的情况下，通过 CoT 和 test-time scaling 策略进一步提升推理阶段表现。"))
    add(p("完整基础流程命令："))
    add(code_block("cd /data/yekaiyang/zjx/20260617/assignment_A\n\nbash dpo/scripts/prepare_data.sh\nGPU_ID=0 bash dpo/scripts/train.sh\nBATCH_SIZE=32 bash dpo/scripts/run_mbpp_base_eval.sh\nBATCH_SIZE=32 bash dpo/scripts/run_mbpp_dpo_final_eval.sh\nMODEL_PATH=dpo/outputs/qwen15_code_full_dpo BATCH_SIZE=32 bash tts/run_cot_code_eval.sh"))

    return "".join(parts)


def write_docx() -> None:
    sectpr = get_template_sectpr()
    body = build_body() + sectpr
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        f'xmlns:w="{W_NS}" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:w15="http://schemas.microsoft.com/office/word/2012/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/word/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 w15 wp14">'
        f"<w:body>{body}</w:body></w:document>"
    )

    tmp = OUTPUT.with_suffix(".tmp.docx")
    shutil.copyfile(TEMPLATE, tmp)
    with ZipFile(TEMPLATE, "r") as zin, ZipFile(OUTPUT, "w", ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename == "word/document.xml":
                zout.writestr(item, document_xml.encode("utf-8"))
            elif item.filename == "docProps/core.xml":
                zout.writestr(item, zin.read(item.filename))
            else:
                zout.writestr(item, zin.read(item.filename))
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    write_docx()
    print(OUTPUT)
