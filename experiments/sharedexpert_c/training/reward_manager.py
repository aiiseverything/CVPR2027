"""Outcome rewards with thinking delimiters retained and explicit truncation.

The local Qwen tokenizer preserves thinking delimiters even with special-token
skipping. We explicitly retain all markers and pass an EOS-based truncation
flag, so reward verification can inspect the final-answer format unambiguously.
"""

from collections import defaultdict

import torch
from verl.workers.reward_manager.abstract import AbstractRewardManager


class SharedExpertRewardManager(AbstractRewardManager):
    def __init__(self, tokenizer, num_examine=0, compute_score=None, reward_fn_key="data_source"):
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        # Local model text_config uses endoftext, while its tokenizer advertises
        # im_end. Both are valid Qwen terminal markers; do not mislabel one.
        self.eos_ids = {tokenizer.eos_token_id}
        for marker in ("<|im_end|>", "<|endoftext|>"):
            token_id = tokenizer.convert_tokens_to_ids(marker)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                self.eos_ids.add(token_id)
        self.eos_ids.discard(None)
        if compute_score is None:
            raise ValueError("A rule-based compute_score is required")

    def __call__(self, data, return_dict=True):
        rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        metrics = defaultdict(list)
        for i in range(len(data)):
            item = data[i]
            prompt_length = item.batch["prompts"].shape[-1]
            length = int(item.batch["attention_mask"][prompt_length:].sum().item())
            response = item.batch["responses"][:length]
            ended = bool(length and int(response[-1].item()) in self.eos_ids)
            extra = dict(item.non_tensor_batch.get("extra_info") or {})
            extra["truncated"] = not ended
            prompt_tail = self.tokenizer.decode(item.batch["prompts"][-64:], skip_special_tokens=False)
            extra["require_think_close"] = prompt_tail.rfind("<think>") > prompt_tail.rfind("</think>")
            score = self.compute_score(
                data_source=item.non_tensor_batch[self.reward_fn_key],
                solution_str=self.tokenizer.decode(response, skip_special_tokens=False),
                ground_truth=item.non_tensor_batch["reward_model"]["ground_truth"],
                extra_info=extra,
            )
            if length:
                rewards[i, length - 1] = float(score["score"])
            for key, value in score.items():
                metrics[key].append(value)
            metrics["audit_source"].append(str(item.non_tensor_batch[self.reward_fn_key]))
            metrics["group_uid"].append(str(item.non_tensor_batch.get("uid", "")))
            metrics["response_tokens"].append(length)
            metrics["prompt_tokens"].append(int(item.batch["attention_mask"][:prompt_length].sum().item()))
            metrics["prompt_truncated"].append(int(item.non_tensor_batch.get("prompt_truncated", 0)))
            metrics["original_prompt_tokens"].append(int(item.non_tensor_batch.get("original_prompt_tokens", 0)))
        if return_dict:
            return {"reward_tensor": rewards, "reward_extra_info": dict(metrics)}
        return rewards
