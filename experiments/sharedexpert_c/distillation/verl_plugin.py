"""Sampled-token, hard-teacher reverse KL extensions for the pinned VERL fork.

The reward/advantage fields transport frozen teacher log probabilities; they are
not outcome rewards or GRPO advantages. For a fixed student prefix and samples
from the old policy, the gradient of

    stopgrad(exp(log p - log p_old) * (log p - log q)) * log p

has expectation grad KL(p || q). The omitted +1 is a zero-mean score-function
baseline. State-distribution derivatives are intentionally excluded. Rollout
correction weights account separately for the sampling-engine discrepancy;
capping those weights, if configured, is an approximation. The numerical loss
itself is a score-function surrogate, not a nonnegative KL measurement.
"""

import torch


def sampled_reverse_kl_terms(old_log_prob, log_prob, target_log_prob,
                             response_mask, rollout_is_weights=None):
    """Return masked per-token surrogate terms, with only student gradients."""
    if not (old_log_prob.shape == log_prob.shape == target_log_prob.shape == response_mask.shape):
        raise ValueError("Student, old policy, teacher and response mask shapes must agree")
    if log_prob.ndim != 2:
        raise ValueError("Expected [batch, response_length] log probabilities")
    if not torch.all((response_mask == 0) | (response_mask == 1)):
        raise ValueError("Response mask must be binary")
    valid = response_mask.bool()
    for name, value in (("student", log_prob), ("old policy", old_log_prob),
                        ("teacher", target_log_prob)):
        if not torch.isfinite(value[valid]).all():
            raise ValueError(f"Nonfinite {name} log probability on a response token")
    if (target_log_prob[valid] > 1e-4).any():
        raise ValueError("Teacher targets must be normalized log probabilities, not rewards")

    # Sanitize before multiplication: NaN * 0 remains NaN on padded tokens.
    student = torch.where(valid, log_prob, 0)
    old = torch.where(valid, old_log_prob.detach(), 0)
    teacher = torch.where(valid, target_log_prob.detach(), 0)
    ratio = (student.detach() - old).exp()
    if not torch.isfinite(ratio[valid]).all():
        raise ValueError("Importance ratio overflow; refresh the on-policy rollout")
    coefficient = ratio * (student.detach() - teacher)
    if rollout_is_weights is not None:
        if rollout_is_weights.shape != log_prob.shape:
            raise ValueError("Rollout importance weights must match log probability shape")
        weights = torch.where(valid, rollout_is_weights.detach(), 0)
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("Rollout importance weights must be finite and nonnegative")
        coefficient = coefficient * weights
    return coefficient.detach() * student


def teacher_targets(token_level_rewards, response_mask, **kwargs):
    """Preserve token-local teacher targets without centering or return sums."""
    if token_level_rewards.shape != response_mask.shape:
        raise ValueError("Teacher target and response mask shapes differ")
    valid = response_mask.bool()
    if not torch.isfinite(token_level_rewards[valid]).all():
        raise ValueError("Teacher targets contain nonfinite response log probabilities")
    targets = torch.where(valid, token_level_rewards.detach(), 0)
    return targets, targets.clone()


def reverse_kl_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                           loss_agg_mode="seq-mean-token-mean", config=None,
                           rollout_is_weights=None):
    from verl.trainer.ppo.core_algos import agg_loss

    terms = sampled_reverse_kl_terms(old_log_prob, log_prob, advantages,
                                    response_mask, rollout_is_weights)
    valid = response_mask.bool()
    if valid.any():
        global_info = getattr(config, "global_batch_info", {}) if config is not None else {}
        loss = agg_loss(terms, response_mask, loss_agg_mode, **global_info)
        sampled_log_ratio = (log_prob.detach() - advantages.detach())[valid].mean().item()
        teacher_mean = advantages.detach()[valid].mean().item()
    else:
        loss = terms.sum()  # Differentiable zero even for an empty local microbatch.
        sampled_log_ratio = teacher_mean = 0.0
    return loss, {
        "mopd/sampled_student_teacher_log_ratio": sampled_log_ratio,
        "mopd/teacher_log_prob_mean": teacher_mean,
        "mopd/response_tokens": float(valid.sum().item()),
    }


def register():
    from verl.trainer.ppo.core_algos import register_adv_est, register_policy_loss

    register_adv_est("mopd_targets")(teacher_targets)
    register_policy_loss("mopd_reverse_kl")(reverse_kl_policy_loss)


register()
