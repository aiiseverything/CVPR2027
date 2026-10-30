import json
from pathlib import Path
import tempfile
import unittest

from experiments.sharedexpert_c.distillation.config import load_config
from experiments.sharedexpert_c.distillation.launch import HERE, make_overrides, compose_config, check_dataset


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(HERE / "configs/domain_opd.json")

    def test_real_hydra_composition_and_plugin_loading(self):
        from verl.utils.config import validate_config
        from verl.trainer.ppo.reward import load_reward_manager
        from verl.trainer.ppo.core_algos import get_policy_loss_fn, get_adv_estimator_fn
        cfg = compose_config(make_overrides(self.config, Path("/tmp/opd-test"), 2))
        validate_config(cfg, use_reference_policy=False, use_critic=False)
        manager = load_reward_manager(cfg, None, 0, **cfg.reward_model.reward_kwargs)
        self.assertIsNone(manager._scorer)
        self.assertEqual(cfg.trainer.total_training_steps, 200)
        self.assertEqual(cfg.data.train_batch_size, 32)
        self.assertEqual(cfg.actor_rollout_ref.rollout.n, 1)
        self.assertEqual(cfg.actor_rollout_ref.actor.ppo_mini_batch_size, 32)
        self.assertEqual(cfg.actor_rollout_ref.actor.ppo_epochs, 1)
        self.assertEqual(cfg.trainer.n_gpus_per_node, 1)
        self.assertEqual(cfg.ray_kwargs.ray_init.num_gpus, 2)
        self.assertEqual(cfg.actor_rollout_ref.rollout.agent.default_agent_loop, "domain_opd_single_turn")
        self.assertFalse(cfg.actor_rollout_ref.actor.self_distillation.full_logit_distillation)
        self.assertIsNone(cfg.actor_rollout_ref.actor.self_distillation.distillation_topk)
        self.assertFalse(cfg.critic.enable)
        self.assertFalse(cfg.reward_model.launch_reward_fn_async)
        self.assertFalse(cfg.actor_rollout_ref.actor.use_kl_loss)
        self.assertFalse(cfg.algorithm.use_kl_in_reward)
        self.assertEqual(cfg.actor_rollout_ref.actor.entropy_coeff, 0)
        self.assertTrue(callable(get_policy_loss_fn("mopd_reverse_kl")))
        self.assertTrue(callable(get_adv_estimator_fn("mopd_targets")))

    def test_smoke_has_nonzero_learning_rate_without_changing_lengths(self):
        cfg = compose_config(make_overrides(self.config, Path("/tmp/opd-smoke"), 2, True))
        self.assertEqual(cfg.trainer.total_training_steps, 1)
        self.assertEqual(cfg.data.train_batch_size, 3)
        self.assertEqual(cfg.actor_rollout_ref.actor.optim.lr_warmup_steps, 0)
        self.assertEqual(cfg.data.max_prompt_length, 2048)
        self.assertEqual(cfg.data.max_response_length, 8192)

    def test_reject_insufficient_or_incompatible_gpu_layout(self):
        for total in (1, 4):
            with self.assertRaises(ValueError):
                make_overrides(self.config, Path("/tmp/opd-test"), total)

    def test_metadata_dictionary_keeps_literal_punctuation(self):
        self.config["domain_to_teacher"] = {"math: geometry, algebra": "reasoning"}
        cfg = compose_config(make_overrides(self.config, Path("/tmp/opd-test"), 2))
        self.assertEqual(cfg.reward_model.reward_kwargs.mopd_config.domain_to_teacher[
            "math: geometry, algebra"], "reasoning")

    def test_dataset_preflight_rejects_missing_domain_and_short_budget(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            pq.write_table(pa.Table.from_pylist([{"prompt": [{"role": "user", "content": "q"}]}]), path)
            self.config["data_path"] = str(path)
            errors = check_dataset(self.config)["errors"]
            self.assertTrue(any("domain" in error for error in errors))
            self.assertTrue(any("6400" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
