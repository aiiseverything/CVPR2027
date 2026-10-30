import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

from experiments.sharedexpert_c.distillation.verl_plugin import (
    sampled_reverse_kl_terms, teacher_targets, reverse_kl_policy_loss,
)


class SampledReverseKLTest(unittest.TestCase):
    def test_expected_gradient_matches_exact_reverse_kl_on_and_off_policy(self):
        for old_logits in (None, torch.tensor([1.0, -1.0, 0.4], dtype=torch.float64)):
            logits = torch.tensor([0.2, -0.4, 1.1], dtype=torch.float64, requires_grad=True)
            teacher_logits = torch.tensor([0.7, -0.2, 0.3], dtype=torch.float64, requires_grad=True)
            student = logits.log_softmax(-1)
            teacher = teacher_logits.log_softmax(-1)
            old = student.detach() if old_logits is None else old_logits.log_softmax(-1)
            # Enumerate every possible sampled action under the old policy.
            terms = sampled_reverse_kl_terms(old[None], student[None], teacher[None],
                                             torch.ones(1, 3))
            surrogate = (terms[0] * old.exp()).sum()
            got = torch.autograd.grad(surrogate, logits, retain_graph=True)[0]
            exact = (student.exp() * (student - teacher.detach())).sum()
            expected = torch.autograd.grad(exact, logits)[0]
            torch.testing.assert_close(got, expected, atol=1e-12, rtol=1e-12)
            self.assertIsNone(torch.autograd.grad(surrogate, teacher_logits, allow_unused=True)[0])

    def test_padding_nan_has_zero_gradient_and_no_nan_loss(self):
        student = torch.tensor([[-0.2, float("nan")]], requires_grad=True)
        old = torch.tensor([[-0.2, float("nan")]])
        teacher = torch.tensor([[-0.7, float("nan")]], requires_grad=True)
        terms = sampled_reverse_kl_terms(old, student, teacher, torch.tensor([[1, 0]]))
        terms.sum().backward()
        self.assertTrue(torch.isfinite(terms).all())
        torch.testing.assert_close(student.grad, torch.tensor([[0.5, 0.0]]))
        self.assertIsNone(teacher.grad)

    def test_rollout_correction_is_detached_and_applied_once(self):
        student = torch.tensor([[-0.2, -0.3]], requires_grad=True)
        target = torch.tensor([[-0.7, -0.9]])
        weights = torch.tensor([[2.0, 0.0]], requires_grad=True)
        terms = sampled_reverse_kl_terms(student.detach(), student, target,
                                         torch.ones_like(student), weights)
        terms.sum().backward()
        torch.testing.assert_close(student.grad, torch.tensor([[1.0, 0.0]]))
        self.assertIsNone(weights.grad)

    def test_target_transport_does_not_center_or_sum_across_tokens(self):
        targets = torch.tensor([[-0.1, -4.0, float("nan")]], requires_grad=True)
        advantages, returns = teacher_targets(targets, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(advantages, torch.tensor([[-0.1, -4.0, 0.0]]))
        torch.testing.assert_close(returns, advantages)
        self.assertFalse(advantages.requires_grad)

    def test_all_masked_loss_is_differentiable_zero(self):
        student = torch.full((1, 2), float("nan"), requires_grad=True)
        loss, metrics = reverse_kl_policy_loss(student.detach(), student,
                                               student.detach(), torch.zeros(1, 2))
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        torch.testing.assert_close(student.grad, torch.zeros_like(student))
        self.assertEqual(metrics["mopd/response_tokens"], 0.0)

    def test_rejects_nonfinite_real_token_or_positive_teacher_score(self):
        for target in (float("nan"), 1.0):
            with self.assertRaises(ValueError):
                sampled_reverse_kl_terms(torch.tensor([[-0.3]]), torch.tensor([[-0.3]]),
                                          torch.tensor([[target]]), torch.ones(1, 1))


class TeacherTransportTest(unittest.TestCase):
    def make_batch(self):
        class Batch:
            def __len__(self):
                return len(self.batch["responses"])
        data = Batch()
        data.batch = {
            "input_ids": torch.tensor([[11, 12, 21, 22], [11, 12, 23, 0], [11, 12, 24, 25]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]]),
            "position_ids": torch.arange(4).repeat(3, 1),
            "responses": torch.tensor([[21, 22], [23, 0], [24, 25]]),
            "response_mask": torch.tensor([[1, 1], [1, 0], [1, 1]]),
        }
        data.non_tensor_batch = {
            "domain": ["vision", "math", "vision"],
            "multi_modal_inputs": [{"pixel_values": torch.tensor([[float(i)]])} for i in range(3)],
        }
        return data

    def test_hard_routing_preserves_exact_tensor_inputs_and_restores_order(self):
        from experiments.sharedexpert_c.distillation.reward_manager import HardTeacherRewardManager
        config = {"domain_key": "domain", "domain_to_teacher": {
            "vision": "perception", "math": "reasoning"},
            "teacher_backend": {"rpc_batch_size": 1}}
        manager = HardTeacherRewardManager(None, mopd_config=config)
        received = []
        def score(teacher, samples):
            received.extend((teacher, sample) for sample in samples)
            return [-sample["responses"].float() / 100 for sample in samples]
        manager._scorer = SimpleNamespace(score=SimpleNamespace(remote=score))
        data = self.make_batch()
        with patch("ray.get", side_effect=lambda value: value):
            output = manager(data)
        torch.testing.assert_close(output["reward_tensor"],
                                   torch.tensor([[-.21, -.22], [-.23, 0], [-.24, -.25]]))
        self.assertEqual(output["reward_extra_info"]["mopd_teacher"],
                         ["perception", "reasoning", "perception"])
        self.assertEqual([teacher for teacher, _ in received],
                         ["perception", "perception", "reasoning"])
        for _, sample in received:
            index = {21: 0, 23: 1, 24: 2}[int(sample["responses"][0])]
            torch.testing.assert_close(sample["input_ids"], data.batch["input_ids"][index])
            torch.testing.assert_close(sample["position_ids"], data.batch["position_ids"][index])
            torch.testing.assert_close(sample["multi_modal_inputs"]["pixel_values"],
                                       data.non_tensor_batch["multi_modal_inputs"][index]["pixel_values"])

    def test_unknown_domain_fails_before_launching_scorer(self):
        from experiments.sharedexpert_c.distillation.reward_manager import HardTeacherRewardManager
        manager = HardTeacherRewardManager(None, mopd_config={"domain_to_teacher": {}})
        with patch.object(manager, "_get_scorer") as factory:
            with self.assertRaises(ValueError):
                manager(self.make_batch())
            factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
