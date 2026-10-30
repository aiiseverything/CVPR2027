import json
from pathlib import Path
import tempfile
import unittest

from experiments.sharedexpert_c.mix_rl.launch import HERE, load_config, make_overrides, compose_config, run_completed


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(HERE / "configs/mix_rl.json")

    def test_real_hydra_config_has_32_prompts_and_256_responses_per_update(self):
        from verl.utils.config import validate_config
        from verl.trainer.ppo.utils import need_reference_policy, need_critic
        cfg = compose_config(make_overrides(self.config, Path("/tmp/mix-rl-test"), 2))
        validate_config(cfg, use_reference_policy=False, use_critic=False)
        self.assertEqual(cfg.trainer.total_training_steps, 200)
        self.assertEqual(cfg.data.train_batch_size, 32)
        self.assertEqual(cfg.actor_rollout_ref.actor.ppo_mini_batch_size, 32)
        self.assertEqual(cfg.actor_rollout_ref.rollout.n, 8)
        self.assertEqual(cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n, 256)
        self.assertEqual(cfg.data.train_max_samples, 6400)
        self.assertEqual(cfg.trainer.total_epochs, 1)
        self.assertFalse(cfg.data.shuffle)
        self.assertEqual(cfg.algorithm.adv_estimator, "grpo")
        self.assertTrue(cfg.algorithm.norm_adv_by_std_in_grpo)
        self.assertEqual(cfg.actor_rollout_ref.actor.policy_loss.loss_mode, "vanilla")
        self.assertEqual(cfg.actor_rollout_ref.actor.ppo_epochs, 1)
        self.assertEqual(cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, 1)
        self.assertEqual(cfg.trainer.n_gpus_per_node, 2)
        self.assertEqual(cfg.ray_kwargs.ray_init.num_gpus, 2)
        self.assertIsNone(cfg.actor_rollout_ref.model.external_lib)
        self.assertFalse(need_reference_policy(cfg))
        self.assertFalse(need_critic(cfg))
        self.assertFalse(cfg.actor_rollout_ref.actor.self_distillation.full_logit_distillation)
        self.assertIsNone(cfg.actor_rollout_ref.actor.self_distillation.distillation_topk)
        self.assertEqual(cfg.reward_manager.name, "SharedExpertRewardManager")
        self.assertFalse(cfg.reward_model.launch_reward_fn_async)
        self.assertFalse(cfg.reward_model.use_reward_loop)
        self.assertTrue(cfg.custom_reward_function.path.endswith("/rewards.py"))

    def test_matches_opd_student_data_and_non_algorithm_settings(self):
        from experiments.sharedexpert_c.distillation.config import load_config as opd_load
        from experiments.sharedexpert_c.distillation.launch import make_overrides as opd_overrides
        from experiments.sharedexpert_c.distillation.launch import compose_config as opd_compose
        from omegaconf import OmegaConf
        opd = opd_compose(opd_overrides(opd_load(self.config["opd_config_path"]), Path("/tmp/opd"), 2))
        mix = compose_config(make_overrides(self.config, Path("/tmp/mix"), 2))
        for path in (
            "data.train_files", "data.train_batch_size", "data.shuffle", "data.seed",
            "data.max_prompt_length", "data.max_response_length", "data.train_max_samples",
            "actor_rollout_ref.model.path", "actor_rollout_ref.model.use_remove_padding",
            "actor_rollout_ref.model.enable_gradient_checkpointing", "actor_rollout_ref.model.lora_rank",
            "actor_rollout_ref.actor.freeze_vision_tower", "actor_rollout_ref.actor.optim.lr",
            "actor_rollout_ref.actor.optim.lr_warmup_steps", "actor_rollout_ref.actor.optim.weight_decay",
            "actor_rollout_ref.actor.loss_agg_mode", "actor_rollout_ref.actor.grad_clip",
            "actor_rollout_ref.actor.use_kl_loss", "actor_rollout_ref.actor.entropy_coeff",
            "actor_rollout_ref.rollout.temperature", "actor_rollout_ref.rollout.top_p",
            "actor_rollout_ref.rollout.top_k", "actor_rollout_ref.rollout.gpu_memory_utilization",
            "actor_rollout_ref.rollout.engine_kwargs.vllm.seed", "trainer.total_training_steps",
        ):
            with self.subTest(path=path):
                self.assertEqual(OmegaConf.select(mix, path), OmegaConf.select(opd, path))

    def test_smoke_keeps_group_eight_and_uses_separate_nonzero_update(self):
        cfg = compose_config(make_overrides(self.config, Path("/tmp/mix-smoke"), 2, True))
        self.assertEqual(cfg.trainer.total_training_steps, 1)
        self.assertEqual(cfg.data.train_batch_size, 4)
        self.assertEqual(cfg.actor_rollout_ref.rollout.n, 8)
        self.assertEqual(cfg.actor_rollout_ref.actor.optim.lr_warmup_steps, 0)
        self.assertEqual(cfg.actor_rollout_ref.rollout.agent.num_workers, 1)
        self.assertEqual(cfg.data.max_response_length, 8192)

    def test_invalid_comparison_budget_is_rejected(self):
        raw = json.loads((HERE / "configs/mix_rl.json").read_text())
        for key, value in (("steps", 100), ("steps", 200.0), ("rollout_n", 1),
                           ("batch_size", 256), ("algorithm", "mopd_targets"),
                           ("learning_rate", float("nan")), ("expected_data_sha256", "")):
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.json"
                path.write_text(json.dumps({**raw, key: value}))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_grpo_groups_are_per_prompt_and_uniform_group_has_zero_advantage(self):
        import numpy as np
        import torch
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
        outcomes = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.] + [1.] * 8)
        rewards = torch.zeros(16, 2)
        rewards[:, -1] = outcomes
        advantages, _ = compute_grpo_outcome_advantage(
            rewards, torch.ones_like(rewards), np.array(["promptA"] * 8 + ["promptB"] * 8),
            norm_adv_by_std_in_grpo=True)
        self.assertTrue(torch.all(advantages[:4] < 0))
        self.assertTrue(torch.all(advantages[4:8] > 0))
        torch.testing.assert_close(advantages[8:], torch.zeros_like(advantages[8:]))

    def test_interrupted_smoke_checkpoint_does_not_report_success(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            self.assertFalse(run_completed(run, 1, smoke=True))
            (run / "checkpoints").mkdir()
            (run / "checkpoints/latest_checkpointed_iteration.txt").write_text("1")
            with self.assertRaisesRegex(RuntimeError, "without a success receipt"):
                run_completed(run, 1, smoke=True)
            self.assertFalse(run_completed(run, 200))
            (run / "smoke_complete.json").write_text('{}')
            self.assertTrue(run_completed(run, 1, smoke=True))


if __name__ == "__main__":
    unittest.main()
