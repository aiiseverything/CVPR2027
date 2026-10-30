"""Use VERL's reward transport for exact hard-teacher token log probabilities."""

from copy import deepcopy

import torch
from verl.workers.reward_manager.abstract import AbstractRewardManager

# Registers the estimator in the driver. model.external_lib registers it in each
# actor worker independently; Python imports in one Ray process do not propagate.
from experiments.sharedexpert_c.distillation import verl_plugin  # noqa: F401
from experiments.sharedexpert_c.distillation.routing import route_record


class HardTeacherRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine=0, compute_score=None,
                 reward_fn_key="data_source", mopd_config=None):
        if mopd_config is None:
            raise ValueError("reward_model.reward_kwargs.mopd_config is required")
        self.config = deepcopy(dict(mopd_config))
        self._scorer = None

    def _get_scorer(self):
        if self._scorer is None:
            import ray
            from experiments.sharedexpert_c.distillation.teacher_scorer import FrozenTeacherScorer

            if not ray.is_initialized():
                raise RuntimeError("Initialize the isolated training Ray cluster before scoring")
            backend = self.config.get("teacher_backend", {})
            device = backend.get("device", "cuda")
            if device not in ("cuda", "cpu"):
                raise ValueError("Teacher device must be 'cuda' or 'cpu'; Ray selects the GPU")
            if device == "cuda" and ray.available_resources().get("GPU", 0) < 1:
                raise RuntimeError(
                    "No dedicated Ray GPU remains for teacher scoring; reserve one GPU "
                    "outside trainer.n_gpus_per_node or select the CPU teacher backend"
                )
            remote_cls = ray.remote(
                num_gpus=1 if device == "cuda" else 0,
                num_cpus=int(backend.get("num_cpus", 2)),
                max_restarts=0,
                max_task_retries=0,
            )(FrozenTeacherScorer)
            self._scorer = remote_cls.remote(self.config)
        return self._scorer

    @staticmethod
    def _cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().contiguous()
        if isinstance(value, dict):
            return {key: HardTeacherRewardManager._cpu(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(HardTeacherRewardManager._cpu(item) for item in value)
        return value

    def __call__(self, data, return_dict=True):
        import ray

        required = ("input_ids", "attention_mask", "position_ids", "responses", "response_mask")
        missing = set(required) - set(data.batch.keys())
        if missing:
            raise ValueError(f"Rollout batch missing exact teacher inputs: {sorted(missing)}")
        targets = torch.zeros_like(data.batch["responses"], dtype=torch.float32, device="cpu")
        routed = []
        groups = {}
        for index in range(len(data)):
            record = {key: value[index] for key, value in data.non_tensor_batch.items()}
            teacher = route_record(record, self.config)
            routed.append(teacher)
            sample = {key: self._cpu(data.batch[key][index]) for key in required}
            sample["multi_modal_inputs"] = self._cpu(record.get("multi_modal_inputs") or {})
            groups.setdefault(teacher, []).append((index, sample))

        scorer = self._get_scorer()
        backend = self.config.get("teacher_backend", {})
        rpc_batch_size = int(backend.get("rpc_batch_size", 8))
        if rpc_batch_size < 1:
            raise ValueError("Teacher rpc_batch_size must be positive")
        # Grouping by teacher avoids reloading its checkpoint for every sequence.
        # Send only a small number of visual tensors through the object store.
        for teacher, entries in groups.items():
            for start in range(0, len(entries), rpc_batch_size):
                chunk = entries[start:start + rpc_batch_size]
                result = ray.get(scorer.score.remote(teacher, [sample for _, sample in chunk]))
                if len(result) != len(chunk):
                    raise RuntimeError("Teacher scorer returned the wrong number of sequences")
                for (index, sample), log_probs in zip(chunk, result):
                    if log_probs.shape != targets[index].shape:
                        raise RuntimeError("Teacher scorer response-length mismatch")
                    valid = sample["response_mask"].bool()
                    if not torch.isfinite(log_probs[valid]).all() or (log_probs[valid] > 1e-4).any():
                        raise RuntimeError("Teacher scorer returned invalid normalized log probabilities")
                    targets[index] = torch.where(valid, log_probs.detach().cpu(), 0)

        if return_dict:
            return {"reward_tensor": targets, "reward_extra_info": {"mopd_teacher": routed}}
        return targets
