"""Seven rollout clients, each bound to one of the seven training GPU replicas."""
import json
from pathlib import Path
from uuid import uuid4

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl.experimental.agent_loop.agent_loop import AgentLoopManager


class SevenGPUAgentLoopManager(AgentLoopManager):
    def _init_agent_loop_workers(self):
        cfg = self.config
        rollout = cfg.actor_rollout_ref.rollout
        count = rollout.agent.num_workers
        assert count == cfg.trainer.n_gpus_per_node == 7 and cfg.trainer.nnodes == 1
        assert cfg.data.train_batch_size == 63 and cfg.data.train_batch_size % count == 0
        assert rollout.tensor_model_parallel_size == rollout.data_parallel_size == rollout.pipeline_model_parallel_size == 1
        assert len(self.server_handles) == len(self.rollout_replicas) == count
        assert all(replica.world_size == len(replica.workers) == 1 for replica in self.rollout_replicas)
        # The underlying hybrid replica already shares one GPU with the FSDP
        # training rank. Bind each CPU orchestration client to that one server;
        # do not reserve a second GPU for the client or load-balance across GPUs.
        gpu_ids = ray.get([
            replica.workers[0].__ray_call__.remote(lambda _: ray.get_gpu_ids())
            for replica in self.rollout_replicas
        ])
        assert all(len(ids) == 1 for ids in gpu_ids)
        assigned = [int(float(ids[0])) for ids in gpu_ids]
        assert sorted(assigned) == list(range(7)), assigned
        nodes = [n["NodeID"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU", 0) > 0]
        assert len(nodes) == 1
        self.agent_loop_workers = []
        bindings = []
        for i in range(count):
            worker = self.agent_loop_workers_class.options(
                name=f"mfr_rollout_gpu_{assigned[i]}_{uuid4().hex[:8]}",
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=nodes[0], soft=False),
            ).remote(cfg, [self.server_handles[i]], self.reward_router_address)
            self.agent_loop_workers.append(worker)
            bindings.append({"rollout_worker": i, "training_rank": i, "gpu": assigned[i],
                "server_address": self.server_addresses[i], "samples_per_batch": 9})
        artifact = Path(cfg.trainer.default_local_dir).parent / "rollout_worker_gpu_mapping.json"
        artifact.write_text(json.dumps({"rollout_workers": 7, "training_gpus": 7,
            "global_batch_size": 63, "one_gpu_replica_per_worker": True, "bindings": bindings}, indent=2) + "\n")
        print("MMFineReason OPD seven-GPU rollout mapping:", bindings, flush=True)
