#!/usr/bin/env python3
"""单模型 × 单条件推理。

用法:
    python probe/run_infer.py --model-key 4B --condition bbox
    python probe/run_infer.py --model-key 4B --condition bbox --limit 20   # 冒烟

产出 probe/out/{model_key}__{condition}.jsonl，逐样本一行，含原始回复。
支持断点续跑：已完成的 idx 会跳过。
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

MAX_NEW_TOKENS = 4096  # 三条件统一的 thinking 预算（无图时模型会失控空转，见 report.md）
CHUNK = 64


def build_inputs(processor, samples, condition):
    """构建 (vllm 输入, 元信息) 列表。"""
    inputs, metas = [], []
    for s in samples:
        prompt_text, image_path = common.build_from_sample(s, condition)
        messages, pil = common.build_messages(prompt_text, image_path)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        item = {"prompt": text}
        if pil is not None:
            item["multi_modal_data"] = {"image": pil}
        inputs.append(item)
        metas.append(
            {
                "idx": s["idx"],
                "gold_answer": s["gold_answer"],
                "prompt": prompt_text,
                "has_image": pil is not None,
            }
        )
    return inputs, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", required=True, choices=list(common.MODELS))
    ap.add_argument("--condition", required=True, choices=common.CONDITIONS)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=16384)
    args = ap.parse_args()

    model_path = common.MODELS[args.model_key]
    if not os.path.isdir(model_path):
        sys.exit(f"✗ 模型目录不存在: {model_path}")

    samples = common.load_samples()
    if args.limit:
        samples = samples[: args.limit]

    out_path = common.out_path(args.model_key, args.condition)
    done = {r["idx"] for r in common.read_jsonl(out_path)}
    todo = [s for s in samples if s["idx"] not in done]
    print(f"[{args.model_key} / {args.condition}] 共 {len(samples)} 条, 已完成 {len(done)}, 待跑 {len(todo)}")
    if not todo:
        print("无需推理")
        return

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    print("加载 processor ...")
    processor = AutoProcessor.from_pretrained(model_path)

    print(f"加载 vLLM: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=args.gpu_util,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0, max_tokens=MAX_NEW_TOKENS)

    fout = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    for start in range(0, len(todo), CHUNK):
        batch = todo[start : start + CHUNK]
        inputs, metas = build_inputs(processor, batch, args.condition)
        outputs = llm.generate(inputs, sampling)
        for meta, out in zip(metas, outputs):
            resp = out.outputs[0].text if out.outputs else ""
            finish = out.outputs[0].finish_reason if out.outputs else None
            letter, method = common.extract_answer(resp)
            fout.write(
                json.dumps(
                    {
                        "model": args.model_key,
                        "condition": args.condition,
                        "idx": meta["idx"],
                        "gold_answer": meta["gold_answer"],
                        "has_image": meta["has_image"],
                        "prompt": meta["prompt"],
                        "raw_response": resp,
                        "pred_answer": letter,
                        "extract_method": method,
                        "n_tokens": len(out.outputs[0].token_ids) if out.outputs else 0,
                        "finish_reason": finish,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        fout.flush()
        el = time.time() - t0
        n = min(start + CHUNK, len(todo))
        print(f"  {n}/{len(todo)}  已用 {el/60:.1f} 分钟", flush=True)

    fout.close()
    print(f"✓ 写入 {out_path}")


if __name__ == "__main__":
    main()
