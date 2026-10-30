"""CPU-only checks of actual batch chunking and one-server-per-worker routing."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import torch
from verl import DataProto
from seven_gpu_rollout import SevenGPUAgentLoopManager


class SevenGPUContractTests(unittest.TestCase):
    def test_actual_dataproto_splits_63_into_seven_nines_without_reordering(self):
        data = DataProto.from_dict(tensors={"index": torch.arange(63)})
        pieces = data.chunk(7)
        self.assertEqual([len(p) for p in pieces], [9] * 7)
        self.assertTrue(torch.equal(DataProto.concat(pieces).batch["index"], data.batch["index"]))
        with self.assertRaises(AssertionError):
            data.chunk(4)

    def test_each_worker_only_receives_its_own_gpu_server(self):
        class Factory:
            def __init__(self): self.calls = []
            def options(self, **options): self.options_ = options; return self
            def remote(self, config, servers, reward):
                self.calls.append((config, servers, reward, self.options_))
                return len(self.calls) - 1
        with TemporaryDirectory() as folder:
            manager = object.__new__(SevenGPUAgentLoopManager)
            manager.config = NS(data=NS(train_batch_size=63),
                trainer=NS(n_gpus_per_node=7, nnodes=1, default_local_dir=str(Path(folder)/"checkpoints")),
                actor_rollout_ref=NS(rollout=NS(agent=NS(num_workers=7), tensor_model_parallel_size=1,
                    data_parallel_size=1, pipeline_model_parallel_size=1)))
            manager.server_handles = [f"server-{i}" for i in range(7)]
            manager.server_addresses = [f"localhost:{9000+i}" for i in range(7)]
            manager.rollout_replicas = [NS(world_size=1, workers=[NS(__ray_call__=NS(remote=lambda fn, i=i:[i]))]) for i in range(7)]
            manager.agent_loop_workers_class = Factory()
            manager.reward_router_address = None
            with patch("seven_gpu_rollout.ray.get", side_effect=lambda values:values), patch("seven_gpu_rollout.ray.nodes", return_value=[{"NodeID":"1"*56, "Alive":True, "Resources":{"CPU":32}}]):
                manager._init_agent_loop_workers()
            self.assertEqual([c[1] for c in manager.agent_loop_workers_class.calls], [[f"server-{i}"] for i in range(7)])
            mapping = json.loads((Path(folder)/"rollout_worker_gpu_mapping.json").read_text())
            self.assertEqual([r["gpu"] for r in mapping["bindings"]], list(range(7)))
            self.assertEqual([r["samples_per_batch"] for r in mapping["bindings"]], [9]*7)


if __name__ == "__main__":
    unittest.main()
