"""CPU contract checks for the job-local reference/teacher memory policy."""
from types import SimpleNamespace
import unittest

from memory_policy import gpu_resident_fsdp2, release_unused_host_memory


class MemoryPolicyTests(unittest.TestCase):
    def test_allocator_cleanup_preserves_live_tensor(self):
        import torch
        tensor = torch.arange(1024, dtype=torch.float32)
        pointer = tensor.data_ptr()
        report = release_unused_host_memory('cpu_contract_test')
        self.assertEqual(tensor.data_ptr(), pointer)
        self.assertTrue(torch.equal(tensor, torch.arange(1024, dtype=torch.float32)))
        self.assertFalse(report['moves_live_tensors'])

    def test_preserves_model_state_and_sharding_but_disables_offload(self):
        calls = []
        model, state, mesh, config, mp, placement, offload = [object() for _ in range(7)]
        def apply(*args):
            calls.append(("apply", args))
        def load(*args, **kwargs):
            calls.append(("load", args, kwargs))
        module = SimpleNamespace(apply_fsdp2=apply, fsdp2_load_full_state_dict=load)
        settings = {"mesh": mesh, "mp_policy": mp, "shard_placement_fn": placement,
                    "reshard_after_forward": True, "offload_policy": offload}
        with gpu_resident_fsdp2(module):
            module.apply_fsdp2(model, settings, config)
            module.fsdp2_load_full_state_dict(model, state, mesh, offload)
        self.assertIs(module.apply_fsdp2, apply)
        self.assertIs(module.fsdp2_load_full_state_dict, load)
        self.assertIs(settings["offload_policy"], offload)
        self.assertEqual(calls[0], ("apply", (model, dict(settings, offload_policy=None), config)))
        self.assertEqual(calls[1], ("load", (model, state, mesh), {"cpu_offload": None}))

    def test_restores_functions_when_loading_fails(self):
        apply, load = object(), object()
        module = SimpleNamespace(apply_fsdp2=apply, fsdp2_load_full_state_dict=load)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with gpu_resident_fsdp2(module):
                raise RuntimeError("test failure")
        self.assertIs(module.apply_fsdp2, apply)
        self.assertIs(module.fsdp2_load_full_state_dict, load)


if __name__ == "__main__":
    unittest.main()
