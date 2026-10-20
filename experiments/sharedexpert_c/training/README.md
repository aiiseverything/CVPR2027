# Three-teacher RL execution

This directory prepares the first stage of SharedExpert option C: three independent
Qwen3.5-4B teachers, followed by common evaluation. It does not perform distillation.
Every production teacher starts from `pre-lab/models/Qwen3.5-4B` and uses 300 optimizer
steps, 30 prompts per step, 8 responses per prompt: **9,000 prompt exposures and
72,000 sampled responses per teacher**. The original 12,000-row prepared files stay
unchanged; `shuffle=False` consumes only the first 9,000 rows. This user-approved
batch reduction is a 25% exposure-budget change from the old batch-40 protocol.
Repeated visits to a small dataset count as exposures, not new unique examples.

**Current snapshot (2026-10-20 12:58:50 Asia/Shanghai): formal run restarted from
step 0 after independent fix validation.** Driver 1842956 is in preflight, monitor
1809563 continues, and all three teachers are 0/300 in this snapshot. The launcher
explicitly sets `actor_rollout_ref.actor.self_distillation.full_logit_distillation=false`
and `distillation_topk=null`, disabling unused forward work under vanilla GRPO.
Batch 30, group 8, target 300, loss, LR, length limits, KL/entropy, sampling temperature,
fused kernels and production FA2 settings remain unchanged. The prior successful
step 1 and step-2 CUDA OOM remain archived; there was no checkpoint, and that step is
not added to this new run. Probe updates are also not formal training steps.

`config/common.json` freezes the shared protocol. Source-specific files identify the
prepared parquet and prescribed mixture. Source ratios are enforced by the data
pipeline's ordered 200-row blocks; at batch 30, every 600 rows / 20 steps covers
three complete blocks with exact ratios. The run consumes 45 blocks; an individual
30-row batch need not have exact ratios. The dataloader must not reshuffle.
Generalist: VQAv2 40%, document/chart 10%, reasoning 20%, perception 20%, IF-format 10%.
The 40% category uses its VQA branch; there is no caption objective in this run.

The loss is ordinary clipped GRPO (`ppo_trainer`, `loss_mode=vanilla`), not Vision-OPD.
Full parameters, including the vision tower, train in FSDP2; no LoRA. Learning rate
1e-6, 10-step warmup then constant, one PPO epoch, minibatch 30 prompts, clip [0.2,0.2],
dual clip 3, gradient clip 1, weight decay 0, group-standardized advantage and
sequence-mean token-mean aggregation. KL-in-reward, KL loss, entropy regularization
and entropy calculation are all explicitly off. PPO's diagnostic approximate KL
may still be logged; it is not a regularization loss or reference-model call.

The container has a 64 GiB host-memory limit. Tied initial weights stream to each
rank's GPU before sharding; fp32 master weights use bf16 forward computation.
Parameter, optimizer and FSDP policy CPU offload are disabled. Rollout reserves
40% of each selected GPU. These resource settings are identical for all teachers.
Actor updates and old-logprob recomputation use one sequence per GPU microbatch
with dynamic batching disabled. Padding is removed only within that single
sequence: the pinned Qwen3.5 linear-attention forward does not consume packed
sequence boundaries, so concatenating different samples would mix recurrent
states. Gradient accumulation preserves the prescribed global batch of 30×8
responses, or 60 responses per rank on four GPUs.
This reduces batching parallelism; training logs record the resulting step time.

All-correct/all-wrong groups remain in the rollout budget and have zero group
advantage. No rejection/resampling is used. Rollout correction uses token importance
weights capped at 2 to account for vLLM/transformers numerical differences; it is
not a KL penalty. Sampling uses temperature 1, top-p 1, top-k -1, thinking enabled.
Rollout replicas use seeds 20261019 through 20261022 (base seed plus replica rank),
recorded in each run protocol; actor/model initialization keeps the base seed.
No forced answer or injected completion is used. The parent's rule reward requires
a valid final boxed/JSON answer and assigns zero to truncated responses.

Prompt limit is 2048 **including expanded image tokens and chat template**; response
limit is 8192 and total context 10240. `agent_loop.py` trims only ordinary user text
when necessary, preserving image spans and chat boundaries. Images default to
configured min/max of 4096/786432 pixels. For multiple images, the configured maximum
is min(786432, 1572864 // image count). These input limits precede a second HF
processor resize: its installed default minimum is 65536 pixels, so small images can
be enlarged again. The CPU header audit in `../logs/visual_batch30_budget_audit.json`
uses both installed resize functions; it is not a runtime RSS or GPU-memory test.
Images/templates alone exceeding the actual processor-expanded prompt limit cause
an explicit error. FP32 image transport remains unchanged; the BF16 transport probe
was neither run nor enabled.

## Installation and checks

Run from the workspace root:

```bash
bash experiments/sharedexpert_c/training/setup_train_env.sh
python3 experiments/sharedexpert_c/training/launch.py perception --gpus 3,4,5,6 --dry-run
experiments/sharedexpert_c/.venv-train/bin/python experiments/sharedexpert_c/training/launch.py perception --gpus 3,4,5,6 --check
```

The dedicated `.venv-train` leaves the pre-existing inference environment untouched.
Core versions follow the checked-out fork: torch 2.10.0, transformers 5.5.0, vLLM
0.18.0. The repository freeze intentionally exceeds vLLM 0.18's `transformers<5`
metadata bound: installation reproduces that exact combination in two stages and
records the one known metadata exception in `logs/dependency-metadata.json`.
All other dependency conflicts fail the setup. FlashAttention is built or obtains
its upstream ABI-matched wheel. For the pinned torch2.10/Python3.10 combination,
FA2.8.3 is compiled with ninja and FLASH_ATTN_CUDA_ARCHS=80 for L20 (sm89).
The task-specific build uses upstream disable switches for dropout, ALiBi,
softcap, uneven head dimensions and local attention; Qwen uses none of these
features (attention head dimensions are 64 and 256). Backward remains enabled.
Only BF16 kernels for these two dimensions are compiled; the API explicitly
rejects FP16 and all other head dimensions. Causal/noncausal, variable-length,
backward and split-KV paths are retained and numerically checked against SDPA.
The same switches apply to the C++ API so unsupported calls fail explicitly.
Source tarball, patch and wheel hashes are recorded in `logs/flash-attention-build.json`.
The pinned upstream tarball and resulting wheel are retained in `build_sources/`,
and `setup_train_env.sh` reproduces the specialization from that source.
The freeze
file and import check are recorded under `experiments/sharedexpert_c/logs/`.
CPU import/configuration checks alone do not establish GPU training compatibility.
NumPy uses 2.2.6 rather than the repository freeze's 1.26.4: vLLM's required OpenCV
4.13 requires NumPy 2, and the vLLM-pinned numba accepts NumPy below 2.3.

## Smoke, training, resume and export

Only run after the selected GPU processes have been released by their owner:

```bash
experiments/sharedexpert_c/.venv-train/bin/python experiments/sharedexpert_c/training/launch.py perception --gpus 3,4,5,6 --smoke
experiments/sharedexpert_c/.venv-train/bin/python experiments/sharedexpert_c/training/launch.py perception --gpus 3,4,5,6 --execute
```

Smoke uses four prompts and one step, retaining the exact 2048/8192/group-8 limits;
its output is separate and is never resumed into the 300-step scientific run. An
optional `--smoke-data PATH` can use a small real, prepared training parquet.
Production executes one teacher at a time. Select `reasoning` or `generalist` for
the other teachers. The launcher refuses busy GPUs and never kills another process.
It creates an isolated local Ray cluster and never joins an existing one.

The batch-30 transition preserves the completed four-prompt GPU smoke unchanged.
`../logs/batch30_transition_validation.json` independently verifies on CPU that its
effective overrides and resolved configuration are identical; GPU numerical code,
reward, repository patch, environment and kernels are unchanged. The root gate also
passes one positive and three negative controls in
`../logs/batch30_preparation_gate_validation.json`. This does not claim a new GPU
smoke run or a complete batch-30 GPU step. Driver 1826689 started at 2026-10-20
11:56:08 Asia/Shanghai; the first complete formal step was still pending in this
snapshot. The old batch-40 run is preserved in
`../runs/archive_batch40_pre_step_20261020_1147/perception`; it completed no formal
step and is not resumed into this changed protocol.

The first complete formal batch-30 perception step passed at **2026-10-20 12:22:04
Asia/Shanghai**, independently recorded in `../runs/archive_batch30_step2_cuda_oom_20261020_1234/first_step_validation.json`.
It contains 240 responses, 12 nonuniform-reward groups, mean reward 0.636603415,
pg loss -2.309531e-5, gradient norm 1.65841317 and 105 finite formal scalars;
39 truncated responses all receive zero reward. TensorBoard and monitor CSV agree.
The first warmup step logs `actor/lr=0`: the full gradient/optimizer flow completed,
without a measured weight-change claim. Step time was 983.745 seconds, including
695.419 generation, 63.533 old-logprob and 223.598 actor update seconds. Five-second
resource samples over startup through the first scalar found container working-set
/current peaks of 53.44862/57.27291 GiB, no new OOM/kill and no monitor alerts.
At that first-step snapshot perception was 1/300 and step 2 was continuing, while
reasoning/generalist were 0/300. Step 2 subsequently failed with CUDA OOM, as stated
above. The first-step receipt remains historical evidence, not later-step stability.

The unused-top-k transition is validated in
`../logs/unused_topk_transition_validation.json`; launcher SHA256 is
`94a6b409eff04ba9c771ad7fec3011599b59bb834eaf93f82b54660a8a0a22e6`.
The initial nondeterministic probe failed its unchanged gradient criterion and is
preserved in `../logs/unused_topk_probe_nondeterministic_failed_20261020_1252`.
The successful second probe enabled FA2 deterministic backward **only for its short
controlled comparison**, verified actual head-64/256 kernel arguments, and ran three
repeats per mode: logprob/PPO/IS, full/vision gradient differences and repeat noise
were all zero. It restored production deterministic=False for two long updates using
432 prompt and 8192 response tokens, mini=2 / micro=1. Before update 2, each rank
held 9,078,531,072 bytes of FP32 CUDA Adam state for 723 parameter tensors; before
its second microbatch, 4,539,265,536 bytes of accumulated gradients were present.
Update-2 LR was 1e-7 and gradient norm 4.60671139, finite. GPU/host working-set sampled
peaks were 40.84833/23.385 GiB; supervisor exited 0 without guard or OOM. This probe
omits Ray/vLLM and the full 30-prompt aggregation, so it does not establish a formal
second step or 300-step stability. One positive and four negative gate controls pass
in `../logs/unused_topk_gate_validation.json`; actual restart identity is recorded in
`../runtime/unused_topk_fix_launch.json`.

Checkpoints (weights, optimizer, RNG/extra and dataloader state) save every 25 steps
and at step 300. The latest two are retained. Repeating `--execute` resumes the same
run; a changed data/reward checksum or changed resolved config is rejected.
Completed step-300 runs only export, and do not accidentally take step 301.

Outputs:

- `runs/<teacher>/training.log` and `tensorboard/`: training diagnostics.
- `runs/<teacher>/protocol.json`, `resolved_config.yaml`, `preflight.json`: provenance.
- `runs/<teacher>/repository.patch`: the exact local fork changes, with its SHA256
  stored in the protocol. They honor disabled entropy and allow recomputing the
  actor's old log probabilities for rollout correction instead of reusing vLLM's.
- `runs/<teacher>/checkpoints/global_step_300/actor/`: resumable final FSDP shards.
- `runs/<teacher>/hf/global_step_300/`: merged HF checkpoint for the evaluation pipeline.

The merge runs on CPU in a fresh directory and leaves FSDP shards intact. An
`sharedexpert_export_complete.json` marker is written only after checking the export.
`--merge-only` can retry an export without starting training. No evaluation score is
claimed by this launcher; use the common evaluation pipeline after export.

Primary references checked for model/framework behavior:
[Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B),
[vLLM 0.18 model support](https://docs.vllm.ai/en/v0.18.0/models/supported_models/),
[verl configuration documentation](https://verl.readthedocs.io/en/latest/examples/config.html).
