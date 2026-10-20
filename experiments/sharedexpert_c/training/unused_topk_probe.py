#!/usr/bin/env python3
"""Independent PPO equivalence and warm-Adam long-sequence memory probe."""
from __future__ import annotations
import argparse
import ast
import asyncio
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
TRAINING = Path(__file__).resolve().parent
EXPERIMENT = TRAINING.parent
ROOT = EXPERIMENT.parents[1]
REPO = ROOT / "pre-lab/repo"
sys.path[:0] = [str(ROOT), str(REPO)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--execute", action="store_true")
    p.add_argument("--output", type=Path, default=EXPERIMENT / "logs/unused_topk_probe")
    p.add_argument("--gpus", default="3,4,5,6")
    p.add_argument("--host-limit-gib", type=float, default=54)
    args = p.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(EXPERIMENT / "logs"):
        p.error("Probe artifacts must stay inside experiment/logs")
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def host():
    root = Path("/sys/fs/cgroup")
    stat = {k: int(v) for k, v in (line.split() for line in (root / "memory.stat").read_text().splitlines())}
    current = int((root / "memory.current").read_text())
    events = {k: int(v) for k, v in (line.split() for line in (root / "memory.events").read_text().splitlines())}
    return {"current_bytes": current, "working_set_bytes": current - stat["inactive_file"],
            "anon_bytes": stat["anon"], "events": events}


async def long_input(args, cfg):
    import torch
    import pyarrow.parquet as pq
    from verl.utils import hf_processor
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, DictConfigWrap
    from experiments.sharedexpert_c.training.agent_loop import SharedExpertSingleTurn
    processor = hf_processor(cfg.actor_rollout_ref.model.path, local_files_only=True)
    tok = processor.tokenizer
    rows = pq.read_table(cfg.data.train_files[0]).slice(0, 30).to_pylist()
    generations = [json.loads(x) for x in (args.output / "formal_step1_rollouts.jsonl").read_text().splitlines()]
    candidates = sorted((r for r in generations if r["response_tokens"] == 8192),
                        key=lambda r: r["prompt_tokens"], reverse=True)
    for gen in candidates:
        ids = tok.encode(gen["output"], add_special_tokens=False)
        if len(ids) != 8192:
            continue
        matches = [r for r in rows if gen["audit_source"] == "sharedexpert_" + r["extra_info"]["source"]
                   and r["prompt"][0]["content"].replace("<image>", "").strip() in gen["input"]]
        if len(matches) != 1:
            continue
        row = matches[0]
        break
    else:
        raise RuntimeError("No exact 8192-token saved response and unambiguous original input")
    class Server:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=ids, log_probs=None, routed_experts=None)
    loop = SharedExpertSingleTurn(trainer_config=DictConfigWrap(cfg), server_manager=Server(),
        tokenizer=tok, processor=processor, dataset_cls=RLHFDataset, dataset_config=cfg.data)
    stub = SimpleNamespace(prompt_key="prompt", image_key="images", video_key="videos", processor=processor)
    messages = RLHFDataset._build_messages(stub, copy.deepcopy(row))
    output = await loop.run({}, raw_prompt=messages)
    assert len(output.prompt_ids) == gen["prompt_tokens"]
    input_ids = torch.tensor([output.prompt_ids + ids])
    attention = torch.ones_like(input_ids)
    helper = SimpleNamespace(processor=processor, tokenizer=tok)
    mmi = AgentLoopWorker._compute_multi_modal_inputs(helper, output, input_ids)
    pos = AgentLoopWorker._compute_position_ids(helper, input_ids, attention, mmi)
    batch = {"input_ids": input_ids, "attention_mask": attention, "position_ids": pos,
             "responses": torch.tensor([ids]), "response_mask": torch.ones((1, len(ids)), dtype=torch.long),
             "multi_modal_inputs": [mmi]}
    provenance = {"source": row["extra_info"]["source"], "source_id": row["extra_info"]["source_id"],
        "response_tokens": len(ids), "prompt_tokens": len(output.prompt_ids),
        "response_text_sha256": hashlib.sha256(gen["output"].encode()).hexdigest(),
        "saved_rollout_sha256": sha(args.output / "formal_step1_rollouts.jsonl"),
        "response_retokenized_from_saved_text": True, "retokenized_length_exact": True,
        "image_sha256": {x["path"]: sha(x["path"]) for x in row["images"]},
        "construction": "Actual dataset messages, SharedExpertSingleTurn and AgentLoopWorker multimodal/mRoPE methods"}
    return batch, provenance


def prepare(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from experiments.sharedexpert_c.training.launch import make_overrides
    torch.set_num_threads(1)
    protocol = json.loads((TRAINING / "config/common.json").read_text())
    teacher = json.loads((TRAINING / "config/perception.json").read_text())
    old = OmegaConf.load(args.output / "resolved_before.yaml")
    with initialize_config_dir(config_dir=str(REPO / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=make_overrides(protocol, teacher, 4,
            EXPERIMENT / "runs/perception", protocol["steps"], protocol["batch_size"]))
    differences = []
    def compare(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                compare(a[k], b[k], path + "." + k if path else k)
        elif a != b:
            differences.append({"path": path, "old": a, "new": b})
    compare(OmegaConf.to_container(old, resolve=True), OmegaConf.to_container(cfg, resolve=True))
    assert differences == [
        {"path": "actor_rollout_ref.actor.self_distillation.full_logit_distillation", "old": True, "new": False},
        {"path": "actor_rollout_ref.actor.self_distillation.distillation_topk", "old": 100, "new": None}], differences
    expected_differences = copy.deepcopy(differences)
    old_function = next(node for node in ast.parse((args.output / "launch_before.py").read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == "make_overrides")
    old_namespace = {"json": json, "WORKSPACE": ROOT, "TRAINING": TRAINING, "EXPERIMENT": EXPERIMENT}
    exec(compile(ast.Module(body=[old_function], type_ignores=[]), "captured_old_launch", "exec"), old_namespace)
    teacher_differences = {}
    for name in ("perception", "reasoning", "generalist"):
        t = json.loads((TRAINING / f"config/{name}.json").read_text())
        with initialize_config_dir(config_dir=str(REPO / "verl/trainer/config"), version_base=None):
            old_cfg = compose(config_name="ppo_trainer", overrides=old_namespace["make_overrides"](
                protocol,t,4,EXPERIMENT/"runs"/name,protocol["steps"],protocol["batch_size"]))
            new_cfg = compose(config_name="ppo_trainer", overrides=make_overrides(
                protocol,t,4,EXPERIMENT/"runs"/name,protocol["steps"],protocol["batch_size"]))
        differences = []
        compare(OmegaConf.to_container(old_cfg,resolve=True),OmegaConf.to_container(new_cfg,resolve=True))
        assert differences == expected_differences, (name,differences)
        teacher_differences[name] = copy.deepcopy(differences)
        (args.output/f"resolved_after_{name}.yaml").write_text(OmegaConf.to_yaml(new_cfg,resolve=True))
    (args.output / "resolved_after.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    short_path = EXPERIMENT / "logs/packing_probe/inputs.pt"
    short = torch.load(short_path, map_location="cpu", weights_only=False)
    long, provenance = asyncio.run(long_input(args, cfg))
    torch.save({"short": short["batch"], "long": long}, args.output / "inputs.pt")
    files = [Path(__file__), TRAINING / "launch.py", TRAINING / "run.py", TRAINING / "agent_loop.py",
             TRAINING / "reward_manager.py", EXPERIMENT / "rewards.py", REPO / "verl/workers/actor/dp_actor.py",
             REPO / "verl/workers/fsdp_workers.py", REPO / "verl/utils/fsdp_utils.py"]
    report = {"status": "cpu_prepared", "effective_config_differences": expected_differences,
        "three_teacher_config_differences": teacher_differences,
        "old_launch_sha256": sha(args.output / "launch_before.py"), "new_launch_sha256": sha(TRAINING / "launch.py"),
        "old_protocol_sha256": sha(args.output / "protocol_before.json"),
        "old_config_sha256": sha(args.output / "resolved_before.yaml"),
        "new_config_sha256": sha(args.output / "resolved_after.yaml"),
        "inputs_sha256": sha(args.output / "inputs.pt"), "source_sha256": {str(p): sha(p) for p in files},
        "short_input_source_sha256": sha(short_path), "short_provenance": short["provenance"],
        "long_provenance": provenance, "new_rollout_generated": False,
        "short_input_provenance_note": "Reconstructed controlled inputs only. DocVQA has an appended EOS: 250 tokens versus saved 249. Historical source predates final reward revision; reward is not evaluated here.",
        "scientific_parameters_changed": False}
    save(args.output / "cpu_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def supervise(args):
    import pynvml
    pynvml.nvmlInit()
    gpu_ids = [int(x) for x in args.gpus.split(",")]
    assert len(gpu_ids) == 4 and len(set(gpu_ids)) == 4
    handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_ids}
    assert all(not pynvml.nvmlDeviceGetComputeRunningProcesses(h) for h in handles.values()), "GPUs occupied"
    start = host()
    assert start["working_set_bytes"] < 20 * 2**30, start
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus)
    cmd = [str(EXPERIMENT / ".venv-train/bin/torchrun"), "--standalone", "--nproc-per-node=4",
           str(Path(__file__).resolve()), "--execute", "--output", str(args.output)]
    summary = {"status": "running", "command": cmd, "gpu_ids": gpu_ids, "host_before": start,
               "peak_working_set_bytes": start["working_set_bytes"], "peak_current_bytes": start["current_bytes"],
               "gpu_peak_used_bytes": {str(i): 0 for i in gpu_ids}, "guard_limit_gib": args.host_limit_gib}
    with (args.output / "gpu_probe.log").open("w", buffering=1) as log, (args.output / "resources.jsonl").open("w", buffering=1) as samples:
        child = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        summary["torchrun_pid"] = child.pid
        save(args.output / "supervisor.json", summary)
        try:
            while child.poll() is None:
                sample = host()
                sample["time"] = time.time()
                sample["gpu_used_bytes"] = {str(i): pynvml.nvmlDeviceGetMemoryInfo(h).used for i, h in handles.items()}
                samples.write(json.dumps(sample) + "\n")
                for k in ("working_set_bytes", "current_bytes"):
                    summary["peak_" + k] = max(summary["peak_" + k], sample[k])
                for k, v in sample["gpu_used_bytes"].items():
                    summary["gpu_peak_used_bytes"][k] = max(summary["gpu_peak_used_bytes"][k], v)
                if sample["working_set_bytes"] > args.host_limit_gib * 2**30:
                    summary["guard_triggered"] = True
                    raise RuntimeError("Probe working-set guard")
                time.sleep(.5)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            summary.update(status="complete" if child.returncode == 0 else "failed", returncode=child.returncode,
                           host_after=host())
            summary["memory_events_delta"] = {k: summary["host_after"]["events"][k] - v for k, v in start["events"].items()}
            save(args.output / "supervisor.json", summary)
    raise SystemExit(child.returncode)


def execute(args):
    # FA2 exposes a deterministic-backward control. It is used only for the
    # short numerical comparison and restored before production-path pressure.
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    import numpy as np
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoModelForImageTextToText
    from torch.distributed.device_mesh import init_device_mesh
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict, get_shard_placement_fn, MixedPrecisionPolicy
    from verl.utils.model import update_model_config
    from verl.utils import hf_tokenizer
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.config.optimizer import build_optimizer
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs
    from verl import DataProto
    import functools
    import inspect
    import flash_attn.flash_attn_interface as fa_interface
    rank, local_rank, world = [int(os.environ[x]) for x in ("RANK", "LOCAL_RANK", "WORLD_SIZE")]
    assert world == 4
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    set_ulysses_sequence_parallel_group(None)
    fa_phase = "short_deterministic"
    fa_calls = {}
    for fa_class in (fa_interface.FlashAttnFunc, fa_interface.FlashAttnVarlenFunc):
        original_fa_forward = fa_class.forward
        signature = inspect.signature(original_fa_forward)
        def wrap_fa(original, signature, label):
            @functools.wraps(original)
            def tracked(*a, **kw):
                values = signature.bind(*a, **kw)
                values.apply_defaults()
                q = values.arguments["q"]
                deterministic = values.arguments["deterministic"]
                key = f"{fa_phase}:{label}:head{q.shape[-1]}:causal{values.arguments['causal']}:deterministic{deterministic}"
                fa_calls[key] = fa_calls.get(key,0)+1
                assert deterministic == (fa_phase == "short_deterministic"), key
                return original(*a, **kw)
            return tracked
        fa_class.forward = staticmethod(wrap_fa(original_fa_forward, signature, fa_class.__name__))
    report = json.loads((args.output / "cpu_report.json").read_text())
    for p, expected in report["source_sha256"].items():
        assert sha(p) == expected, p
    assert sha(args.output / "inputs.pt") == report["inputs_sha256"]
    assert sha(args.output / "resolved_after.yaml") == report["new_config_sha256"]
    cfg = OmegaConf.load(args.output / "resolved_after.yaml")
    # Probe mini=2 has two independent micro=1 calls, including accumulated grads.
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 2
    actor_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    torch.manual_seed(20261019)
    torch.cuda.manual_seed_all(20261019)
    tok = hf_tokenizer(cfg.actor_rollout_ref.model.path, local_files_only=True)
    mc = AutoConfig.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True, attn_implementation="flash_attention_2")
    update_model_config(mc, override_config_kwargs={"bos_token_id": tok.bos_token_id,
        "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id})
    model = AutoModelForImageTextToText.from_pretrained(cfg.actor_rollout_ref.model.path, config=mc,
        torch_dtype=torch.float32, attn_implementation="flash_attention_2", device_map={"": device}, local_files_only=True)
    apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False,
                      use_prefix_grouper=False, use_tiled_mlp=False)
    model.to(torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert not [(n, m.p) for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.p]
    assert not mc.text_config.attention_dropout
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))
    full = model.state_dict()
    apply_fsdp2(model, {"mesh": mesh, "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32, cast_forward_inputs=True), "offload_policy": None,
        "reshard_after_forward": actor_cfg.fsdp_config.reshard_after_forward,
        "shard_placement_fn": get_shard_placement_fn(world)}, actor_cfg.fsdp_config)
    fsdp2_load_full_state_dict(model, full, mesh, None)
    del full
    torch.cuda.empty_cache()
    model.train()
    actor = DataParallelPPOActor(actor_cfg, model, actor_optimizer=None)
    payload = torch.load(args.output / "inputs.pt", map_location="cpu", weights_only=False)
    short, long = payload["short"], payload["long"]
    del payload
    def select(batch, idx):
        return {k: [v[idx]] if k == "multi_modal_inputs" else v[idx:idx+1].to(device) for k, v in batch.items()}
    def local(t):
        return t.to_local() if hasattr(t, "to_local") else t
    def grad_bytes():
        return sum(local(p.grad).numel()*local(p.grad).element_size() for p in model.parameters() if p.grad is not None)
    def differences(a, b):
        d = a.double() - b.double()
        return {"max_abs": float(d.abs().max()), "rms": float(d.square().mean().sqrt())}
    oldps = []
    with torch.no_grad():
        for i in range(2):
            oldps.append(actor._forward_micro_batch(select(short, i), 1.0)["log_probs"].detach())
    def short_pass(topk):
        model.zero_grad(set_to_none=True)
        logps, losses, metrics = [], [], []
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for i in range(2):
            micro = select(short, i)
            out = actor._forward_micro_batch(micro, 1.0, calculate_entropy=False,
                                            return_all_logps=False, distill_topk=topk)
            lp = out["log_probs"]
            # Identical nonzero controlled PPO advantages and IS weights in both modes.
            adv = torch.ones_like(lp) * (1 if i == 0 else -1)
            loss, met = get_policy_loss_fn("vanilla")(old_log_prob=oldps[i], log_prob=lp,
                advantages=adv, response_mask=micro["response_mask"], loss_agg_mode=actor_cfg.loss_agg_mode,
                config=actor_cfg, rollout_is_weights=torch.ones_like(lp))
            corr = compute_rollout_corr_metrics_from_logprobs(lp, oldps[i], micro["response_mask"])
            (loss / 2).backward()
            logps.append(lp.detach().float().cpu())
            losses.append(float(loss.detach()))
            metrics.append({k: float(v) for k, v in dict(met, **corr).items()})
            del out, lp, loss
        torch.cuda.synchronize()
        return torch.cat(logps), {"losses": losses, "metrics": metrics, "seconds": time.perf_counter()-started}
    def gradient_difference(reference):
        sums = torch.zeros((2, 3), dtype=torch.float64, device=device)
        maxima = torch.zeros(2, dtype=torch.float64, device=device)
        missing = []
        per_parameter = []
        for name, p in model.named_parameters():
            if p.grad is None and name not in reference:
                continue
            if p.grad is None or name not in reference:
                missing.append(name)
                continue
            a, b = local(p.grad).detach().reshape(-1), reference[name].reshape(-1)
            psums = torch.zeros(2,dtype=torch.float64,device=device)
            pmax = torch.zeros((),dtype=torch.float64,device=device)
            groups = [0, 1] if ".visual." in name else [0]
            for start in range(0, a.numel(), 2**20):
                x, y = a[start:start+2**20].double(), b[start:start+2**20].double()
                d = x-y
                psums += torch.stack((d.square().sum(),y.square().sum()))
                pmax = torch.maximum(pmax,d.abs().max())
                for g in groups:
                    sums[g] += torch.stack((d.square().sum(), y.square().sum(), x.square().sum()))
                    maxima[g] = torch.maximum(maxima[g], d.abs().max())
            per_parameter.append({"name":name,"rank_local_max_abs":float(pmax),
                                  "rank_local_relative_l2":float((psums[0]/psums[1].clamp_min(1e-300)).sqrt())})
        dist.all_reduce(sums)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        return {**{label: {"max_abs": float(maxima[i]),
            "relative_l2": float((sums[i,0]/sums[i,1].clamp_min(1e-300)).sqrt()),
            "reference_l2": float(sums[i,1].sqrt()), "actual_l2": float(sums[i,2].sqrt()),
            "finite": bool(torch.isfinite(sums[i]).all())} for i,label in enumerate(("full","vision"))},
            "missing_parameter_gradients": missing,
            "rank0_top_difference_parameters":sorted(per_parameter,key=lambda x:x["rank_local_max_abs"],reverse=True)[:12]}
    report.update(status="running", passed=False, world_size=world, short_comparison={}, long_updates=[],
        deterministic_validation="Only short comparison sets FLASH_ATTENTION_DETERMINISTIC=1; actual FA2 Function.forward arguments are checked. Long optimizer test restores production False.",
        limits=["Controlled saved inputs; no new rollouts or scientific training steps",
                "Single-sequence microbatch 1 only; four ranks replicate inputs for equivalence",
                "Long probe mini 2 tests Adam and accumulated gradients, not full 30-prompt Ray aggregation",
                "Does not initialize rollout engines; final formal long-run stability remains separately monitored"])
    short_pass(100)  # warmup reference kernels
    reference = None
    ref_lp = None
    for mode, topk in (("default_top100",100),("disabled",None)):
        report["short_comparison"][mode] = []
        for repeat in range(3):
            lp, result = short_pass(topk)
            if reference is None:
                reference = {n: local(p.grad).detach().clone() for n,p in model.named_parameters() if p.grad is not None}
                ref_lp = lp
                report["reference_gradient_bytes_per_rank"] = sum(x.numel()*x.element_size() for x in reference.values())
            result.update(gradient=gradient_difference(reference), logprob_vs_reference=differences(lp,ref_lp))
            report["short_comparison"][mode].append(result)
            if rank == 0:
                report["flash_attention_calls"] = dict(fa_calls)
                save(args.output / "report.json",report)
                print(json.dumps({"short": mode,"repeat":repeat,**result}),flush=True)
    del reference, ref_lp, oldps
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    reference_runs = report["short_comparison"]["default_top100"]
    changed_runs = report["short_comparison"]["disabled"]
    comparison = []
    for part in ("full","vision"):
        for metric in ("max_abs","relative_l2"):
            noise = max(r["gradient"][part][metric] for r in reference_runs[1:])
            observed = max(r["gradient"][part][metric] for r in changed_runs)
            comparison.append({"part":part,"metric":metric,"repeat_noise":noise,"changed_difference":observed,"passed":observed<=noise})
    report["gradient_noise_comparison"] = comparison
    short_passed = (all(x["passed"] for x in comparison)
        and all(not r["gradient"]["missing_parameter_gradients"] and all(
            r["gradient"][group]["finite"] and r["gradient"][group]["reference_l2"]>0
            for group in ("full","vision")) for r in reference_runs+changed_runs)
        and all(r["logprob_vs_reference"]["max_abs"] == 0 for r in changed_runs)
        and all(r["losses"] == reference_runs[0]["losses"] and r["metrics"] == reference_runs[0]["metrics"] for r in changed_runs))
    report["short_equivalence_passed"] = short_passed
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "0"
    from verl.models.transformers import qwen2_vl
    qwen2_vl._flash_deterministic_enabled = False
    fa_phase = "long_production_nondeterministic"
    optimizer = build_optimizer(model.parameters(),actor_cfg.optim)
    actor.actor_optimizer = optimizer
    batch = {k: v.repeat(2,*([1]*(v.ndim-1))) for k,v in long.items() if torch.is_tensor(v)}
    batch["old_log_probs"] = torch.zeros_like(batch["responses"],dtype=torch.float32)
    batch["advantages"] = torch.ones_like(batch["old_log_probs"])
    batch["rollout_is_weights"] = torch.ones_like(batch["old_log_probs"])
    mm = np.empty(2,dtype=object)
    mm[:] = [long["multi_modal_inputs"][0]]*2
    data = DataProto.from_dict(tensors=batch,non_tensors={"multi_modal_inputs":mm},
                              meta_info={"temperature":1.0,"pad_token_id":tok.pad_token_id})
    observed_calls = []
    original_forward = actor._forward_micro_batch
    def tracked_forward(*a,**kw):
        observed_calls.append({"return_all_logps":kw.get("return_all_logps"),"distill_topk":kw.get("distill_topk"),
                              "calculate_entropy":kw.get("calculate_entropy"),"grad_bytes_before":grad_bytes(),
                              "parameters_with_grad_before":sum(p.grad is not None for p in model.parameters())})
        assert kw.get("return_all_logps") is False and kw.get("distill_topk") is None
        return original_forward(*a,**kw)
    actor._forward_micro_batch = tracked_forward
    def adam_state():
        out = {"parameters_with_state":0,"state_bytes_per_rank":0,"all_states_fp32_cuda":True,"steps":[]}
        for p,state in optimizer.state.items():
            if not state:
                continue
            out["parameters_with_state"] += 1
            out["steps"].append(float(state["step"]))
            for key in ("exp_avg","exp_avg_sq"):
                t = local(state[key])
                out["state_bytes_per_rank"] += t.numel()*t.element_size()
                out["all_states_fp32_cuda"] &= t.dtype==torch.float32 and t.device.type=="cuda"
        return out
    for update in (1,2):
        lr = 0.0 if update == 1 else 1e-7
        for group in optimizer.param_groups:
            group["lr"] = lr
        before = adam_state()
        if update==2:
            assert before["parameters_with_state"]>0 and before["all_states_fp32_cuda"]
        start_call = len(observed_calls)
        dist.barrier()
        torch.cuda.synchronize()
        started=time.perf_counter()
        metrics=actor.update_policy(data)
        torch.cuda.synchronize()
        values={k:float(v) for k,v in metrics.items()}
        assert all(np.isfinite(v) for v in values.values()),values
        assert values["actor/grad_norm"]>0
        calls=observed_calls[start_call:]
        assert len(calls)==2 and calls[1]["grad_bytes_before"]>0,calls
        after = adam_state()
        assert after["parameters_with_state"] == calls[1]["parameters_with_grad_before"]
        if update==2:
            assert before["parameters_with_state"] == calls[1]["parameters_with_grad_before"]
        result={"update":update,"learning_rate":lr,"seconds":time.perf_counter()-started,"metrics":values,
                "adam_before":before,"adam_after":after,"forward_calls":calls}
        report["long_updates"].append(result)
        if rank==0:
            report["flash_attention_calls"] = dict(fa_calls)
            save(args.output/"report.json",report)
            print(json.dumps({"long_update":result}),flush=True)
    report.update(status="complete",passed=short_passed,actual_8192_response=True,full_model_finite_gradients=True,
                  long_pressure_passed=True,flash_attention_calls=dict(fa_calls))
    if rank==0:
        save(args.output/"report.json",report)
        print("UNUSED_TOPK_PROBE_COMPLETE",flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args=arguments()
    if args.prepare:
        prepare(args)
    elif args.supervise:
        supervise(args)
    else:
        execute(args)
