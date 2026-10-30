"""Vision-OPD 探针实验的公共逻辑：样本读取、prompt 构造、答案抽取。

三种条件：
  bbox      带红框全图（images/），prompt 原样保留红框提示句
  original  原图（original_images/），删掉红框提示句
  no_image  无图，同样删掉红框提示句

original vs no_image 因此只有「有无图像」一个变量，文字完全一致。
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "Vision-OPD-6K")
PROBE_DIR = os.path.join(ROOT, "probe")
OUT_DIR = os.path.join(PROBE_DIR, "out")
RESULT_DIR = os.path.join(PROBE_DIR, "results")

CONDITIONS = ["no_image", "original", "bbox"]
MODELS = {
    "4B": os.path.join(ROOT, "models", "Qwen3.5-4B"),
    "9B": os.path.join(ROOT, "models", "Qwen3.5-9B"),
}

REMOVE_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)

# 红框提示句只在 bbox 条件下保留
KEEP_HINT = {"bbox": True, "original": False, "no_image": False}
USE_IMAGE = {"bbox": "images", "original": "original_images", "no_image": None}


# --------------------------------------------------------------------------
# 样本
# --------------------------------------------------------------------------
def load_train_jsonl():
    path = os.path.join(DATA_DIR, "train.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_samples():
    """读取已抽样的样本（由 sample.py 生成）。"""
    path = os.path.join(PROBE_DIR, "samples.jsonl")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# prompt 构造
# --------------------------------------------------------------------------
def clean_problem(problem: str, keep_hint: bool) -> str:
    """去掉 <image> 占位符（图像作为独立 content part 传入）；
    keep_hint=False 时连同红框提示句一起删掉，并压缩多余空行。"""
    t = problem.replace("<image>", "")
    if not keep_hint:
        t = t.replace(REMOVE_HINT, "")
    # 压缩 3 个以上连续换行，并去掉首尾空白
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def build_prompt(item: dict, condition: str) -> tuple:
    """返回 (prompt 文本, 图像绝对路径 or None)。"""
    assert condition in CONDITIONS, condition
    text = clean_problem(item["problem"], KEEP_HINT[condition])
    key = USE_IMAGE[condition]
    if key is None:
        return text, None
    rel = item[key][0]
    return text, os.path.join(DATA_DIR, rel)


def build_from_sample(sample: dict, condition: str) -> tuple:
    """由 samples.jsonl 的一条样本构造 (prompt 文本, 图像绝对路径 or None)。"""
    assert condition in CONDITIONS, condition
    text = clean_problem(sample["problem"], KEEP_HINT[condition])
    key = {"bbox": "img_bbox", "original": "img_original", "no_image": None}[condition]
    return text, (sample[key] if key else None)


def build_messages(prompt: str, image_path):
    """构造 chat messages。图像以 PIL 对象嵌入 content（供 chat template 生成
    vision token），实际像素另行交给推理引擎。

    返回 (messages, pil_image_or_None)。
    """
    if image_path is None:
        return [{"role": "user", "content": [{"type": "text", "text": prompt}]}], None

    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages, img


# --------------------------------------------------------------------------
# 答案抽取（适配 thinking 输出，不需要 judge 模型）
# --------------------------------------------------------------------------
def extract_answer(resp: str):
    """从模型回复中抽取 A-D 选项字母。返回 (letter, method)。"""
    if not resp:
        return None, "empty"

    # 1) 显式 <answer>X</answer>
    m = re.findall(r"<answer>\s*\(?\s*([A-Da-d])\b", resp)
    if m:
        return m[-1].upper(), "tag"

    # 2) 只看最后一个 </think> 之后的文本（think 段落里的字母是干扰）
    seg = resp.split("</think>")[-1] if "</think>" in resp else resp

    # 3) 显式 "Answer: X" / "答案：X"
    m = re.search(r"(?:answer|答案)\s*[:：\-]\s*\(?\s*([A-Da-d])\b", seg, re.I)
    if m:
        return m.group(1).upper(), "explicit"

    # 4) 尾部 80 字符内的独立字母（模型通常在最后给出结论）
    m = re.findall(r"\b([A-D])\b", seg.strip()[-80:])
    if m:
        return m[-1], "tail"

    # 5) 兜底：截断输出、或没走 think 分支时
    m = re.findall(r"\b([A-D])\b", seg)
    if m:
        return m[-1], "fallback"
    return None, "none"


# --------------------------------------------------------------------------
# 结果落盘
# --------------------------------------------------------------------------
def out_path(model_key: str, condition: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, f"{model_key}__{condition}.jsonl")


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
