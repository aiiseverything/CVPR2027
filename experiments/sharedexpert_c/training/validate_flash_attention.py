#!/usr/bin/env python3
"""Numerically check the task-specific FA build before any scientific training."""

import json
from pathlib import Path

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache


def main():
    torch.manual_seed(20261019)
    cases = []
    for dim in (64, 256):
        for causal in (False, True):
            tensors = [torch.randn(2, 32, 4, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                       for _ in range(3)]
            reference = [x.detach().float().requires_grad_() for x in tensors]
            output = flash_attn_func(*tensors, dropout_p=0.0, causal=causal)
            expected = torch.nn.functional.scaled_dot_product_attention(
                *[x.transpose(1, 2) for x in reference], is_causal=causal
            ).transpose(1, 2)
            gradient = torch.randn_like(output)
            output.backward(gradient)
            expected.backward(gradient.float())
            error = (output.float() - expected).abs().max().item()
            grad_error = max((x.grad.float() - y.grad).abs().max().item()
                             for x, y in zip(tensors, reference))
            assert torch.isfinite(output).all() and all(torch.isfinite(x.grad).all() for x in tensors)
            assert error < 0.04 and grad_error < 0.06, (dim, causal, error, grad_error)
            packed = [x.detach()[0].contiguous().requires_grad_() for x in tensors]
            cumulative = torch.tensor([0, 13, 32], device="cuda", dtype=torch.int32)
            variable = flash_attn_varlen_func(*packed, cumulative, cumulative, 19, 19,
                                             dropout_p=0.0, causal=causal)
            variable.sum().backward()
            assert torch.isfinite(variable).all() and all(torch.isfinite(x.grad).all() for x in packed)
            cases.append({"head_dimension": dim, "causal": causal, "max_forward_error": error,
                          "max_gradient_error": grad_error, "variable_length_backward_finite": True})
        query = torch.randn(1, 4, 4, dim, device="cuda", dtype=torch.bfloat16)
        key = torch.randn(1, 64, 4, dim, device="cuda", dtype=torch.bfloat16)
        value = torch.randn_like(key)
        split = flash_attn_with_kvcache(query, key, value, num_splits=2, causal=False)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query.float().transpose(1, 2), key.float().transpose(1, 2), value.float().transpose(1, 2)
        ).transpose(1, 2)
        error = (split.float() - expected).abs().max().item()
        assert error < 0.04, (dim, error)
        cases.append({"head_dimension": dim, "split_kv_max_forward_error": error})
    rejected = []
    for name, dtype, dim, kwargs in (
        ("fp16", torch.float16, 64, {}),
        ("head128", torch.bfloat16, 128, {}),
        ("dropout", torch.bfloat16, 64, {"dropout_p": 0.1}),
        ("local_attention", torch.bfloat16, 64, {"window_size": (8, 8)}),
        ("softcap", torch.bfloat16, 64, {"softcap": 1.0}),
        ("alibi", torch.bfloat16, 64, {"alibi_slopes": torch.ones(4, device="cuda", dtype=torch.float32)}),
    ):
        qkv = [torch.randn(1, 32, 4, dim, device="cuda", dtype=dtype) for _ in range(3)]
        try:
            flash_attn_func(*qkv, **kwargs)
        except RuntimeError:
            rejected.append(name)
        else:
            raise AssertionError(f"Unsupported feature {name} did not fail explicitly")
    report = {"cases": cases, "unsupported_rejected": rejected, "passed": True}
    target = Path(__file__).resolve().parent.parent / "logs/flash-attention-validation.json"
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
