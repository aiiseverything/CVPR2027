"""Check the OPD objective, all configs, and exact student/teacher input equality on CPU."""
import asyncio
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from opd_config import HERE, ROOT, REPO, CURATION, VIEWS, SETTINGS, atomic_json, compose_config, prepare_model_views, validate_config, digest
sys.path[:0] = [str(ROOT), str(REPO), str(HERE)]


def check(rows):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoProcessor
    from verl.utils import hf_processor
    from verl.trainer.ppo.core_algos import compute_self_distillation_loss
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from matched_inputs import MatchedImageDataset
    from omegaconf import OmegaConf
    torch.set_num_threads(1)
    prepare_model_views()
    configs = {t:compose_config(t) for t in SETTINGS["teachers"]}
    for teacher, cfg in configs.items():
        validate_config(cfg)
        (HERE / f"{teacher}.preview.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    student = hf_processor(str(VIEWS["student"]), use_fast=True)
    teachers = {t:AutoProcessor.from_pretrained(VIEWS[t], local_files_only=True) for t in SETTINGS["teachers"]}
    for p in teachers.values():
        assert p.tokenizer.get_vocab() == student.tokenizer.get_vocab()
        assert p.image_processor.size == student.image_processor.size
    cfg = configs["perception_v1"]
    dataset = object.__new__(MatchedImageDataset)
    dataset.prompt_key, dataset.image_key, dataset.video_key = "prompt", "images", "videos"
    dataset.processor = student
    trainer = object.__new__(RayPPOTrainer)
    trainer.config, trainer.processor, trainer.tokenizer = cfg, student, student.tokenizer
    trainer.tokenizer.truncation_side = "right"
    checks = []
    for row in rows:
        images = [{"path": p, "min_pixels": 4096, "max_pixels": 786432} for p in row["images"]]
        item = {"prompt": [{"role":"user", "content":"<image>"+row["prompt_text"]}], "images": images}
        raw = dataset._build_messages(deepcopy(item))
        original_images, videos = asyncio.run(MatchedImageDataset.process_vision_info(raw,16,cfg.data))
        assert videos is None
        prompt = student.apply_chat_template(raw, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        x = student(text=[prompt], images=original_images, return_tensors="pt")
        assert x["input_ids"].shape[1] == row["prompt_tokens"] <= 8192
        if "teachers" in row:
            actual_prompt_hash = digest(x["input_ids"][0].tolist())
            for rec in row["teachers"].values():
                assert actual_prompt_hash == rec["prompt_token_ids_sha256"], "Training prompt IDs differ from actual curation requests"
        if "prompt" in row:
            assert prompt == row["prompt"]
        teacher_messages = trainer._prepare_teacher_messages(raw, row["images"])
        response = torch.tensor([123,456], dtype=torch.long)
        teacher_inputs = trainer._build_teacher_prompt_inputs(teacher_messages,response,torch.ones(2,dtype=torch.long),8192)
        ids, attention, positions, start, mm = teacher_inputs
        assert int(start) == x["input_ids"].shape[1]
        assert torch.equal(ids[:start], x["input_ids"][0]) and torch.equal(ids[start:],response)
        assert torch.equal(mm["pixel_values"], x["pixel_values"])
        assert torch.equal(mm["image_grid_thw"], x["image_grid_thw"])
        for name, p in teachers.items():
            assert p.apply_chat_template(raw,tokenize=False,add_generation_prompt=True,enable_thinking=True) == prompt
            tx = p(text=[prompt],images=original_images,return_tensors="pt")
            for key in ["input_ids", "pixel_values", "image_grid_thw"]:
                assert torch.equal(x[key],tx[key]), (row["id"],name,key)
        checks.append({"id": row["id"], "prompt_tokens": int(start), "pixel_values_identical": True,
                       "prompt_ids_identical": True, "matches_recorded_vllm_prompt_ids": "teachers" in row,
                       "unchanged_original_image_size": list(original_images[0].size)})
    # This checks the actual repository loss, including stop-gradient and masking.
    student_lp = torch.tensor([[-.3,-1.2,-.5],[-.7,-.2,-.9]],requires_grad=True)
    teacher_lp = torch.tensor([[-.1,-1.7,-.6],[-.4,-.6,-.2]],requires_grad=True)
    mask = torch.tensor([[1.,1.,0.],[1.,0.,0.]])
    sd = SimpleNamespace(full_logit_distillation=False,alpha=1.0,is_clip=None)
    loss, metrics = compute_self_distillation_loss(student_lp,teacher_lp,mask,sd,loss_agg_mode=SETTINGS["loss_aggregation"])
    loss.backward()
    expected = (student_lp.detach()-teacher_lp.detach())*mask/mask.sum(dim=-1,keepdim=True)/mask.shape[0]
    assert torch.allclose(student_lp.grad,expected)
    assert teacher_lp.grad is None
    assert metrics["self_distillation/num_distill_tokens"] == 3
    # Assert that the trainer selects reward-free OPD and does not update a teacher.
    assert trainer._use_reward_free_teacher_vopd()
    result = {"status":"passed", "gpu_smoke_performed":False, "settings":SETTINGS,
              "checked_rows":checks, "sampled_token_gradient_matches_formula":True,
              "teacher_no_gradient":True, "reward_free_path":True,
              "versions":{p:importlib.metadata.version(p) for p in ["torch","transformers","vllm","ray","flash-attn","flash-linear-attention"]}}
    atomic_json(HERE / "cpu_validation.json",result)
    return result


if __name__ == "__main__":
    # Candidate images only exercise CPU preprocessing; they are not a training subset.
    path = CURATION / "candidates.jsonl"
    if not path.exists():
        path = CURATION / "candidates.jsonl.partial"
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) == 3:
                break
    print(json.dumps(check(rows),ensure_ascii=False,indent=2))
