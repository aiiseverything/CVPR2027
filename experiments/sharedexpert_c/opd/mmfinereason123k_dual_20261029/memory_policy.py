"""Job-local FSDP2 reference/teacher residency under the 64 GiB host limit.

The shared worker unconditionally enables CPU offload for non-actor models.
Keep the same sharded parameters on GPU instead. No model weights, precision,
collectives, optimizer, forward function or distillation objective is changed.
"""
from contextlib import contextmanager
from functools import wraps
import ctypes
import gc
import inspect
import json
import os
from pathlib import Path
import threading


_initialization_lock = threading.RLock()


def release_unused_host_memory(stage):
    """Return freed libc allocations; never move or discard live tensors."""
    def rss():
        return int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    before = rss()
    gc.collect()
    libc = ctypes.CDLL(None)
    trim = getattr(libc, 'malloc_trim', None)
    if trim is not None:
        trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
        trim(0)
    after = rss()
    report = {'stage': stage, 'pid': os.getpid(), 'rss_before_bytes': before, 'rss_after_bytes': after,
              'returned_bytes': max(0, before - after), 'moves_live_tensors': False}
    print('MMFineReason host allocator: ' + json.dumps(report), flush=True)
    return report


@contextmanager
def gpu_resident_fsdp2(worker_module):
    """Override only this worker process's synchronous model construction."""
    with _initialization_lock:
        original_apply = worker_module.apply_fsdp2
        original_load = worker_module.fsdp2_load_full_state_dict

        def apply(model, fsdp_kwargs, config):
            resident_kwargs = dict(fsdp_kwargs, offload_policy=None)
            return original_apply(model, resident_kwargs, config)

        def load(model, full_state, device_mesh=None, cpu_offload=None):
            # No CPU copies of the full checkpoint or of the parameter shards.
            return original_load(model, full_state, device_mesh, cpu_offload=None)

        worker_module.apply_fsdp2 = apply
        worker_module.fsdp2_load_full_state_dict = load
        try:
            yield
        finally:
            worker_module.apply_fsdp2 = original_apply
            worker_module.fsdp2_load_full_state_dict = original_load


def install():
    from verl.workers import fsdp_workers
    cls = fsdp_workers.ActorRolloutRefWorker
    original = cls._build_model_optimizer
    if getattr(original, "_mfr_gpu_resident_teacher", False):
        return
    signature = inspect.signature(original)

    @wraps(original)
    def build(self, *args, **kwargs):
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        role = bound.arguments["role"]
        assert self.config.actor.strategy == "fsdp2"
        assert not self.config.actor.fsdp_config.offload_policy
        assert not self.config.actor.fsdp_config.param_offload
        assert not self.config.actor.fsdp_config.optimizer_offload
        assert not self.config.ref.fsdp_config.param_offload
        assert not self.config.model.enable_activation_offload
        if role in {"ref", "teacher"}:
            assert bound.arguments["optim_config"] is None
            with gpu_resident_fsdp2(fsdp_workers):
                result = original(self, *args, **kwargs)
        else:
            result = original(self, *args, **kwargs)
        model = result[0]
        assert all(p.device.type == "cuda" for p in model.parameters()), role
        release_unused_host_memory('after_' + role + '_initialization')
        if self.rank == 0:
            print(f"MMFineReason OPD: {role} FSDP2 shards remain on GPU; CPU offload disabled", flush=True)
        return result

    build._mfr_gpu_resident_teacher = True
    cls._build_model_optimizer = build

    original_rollout_mode = cls.rollout_mode

    @wraps(original_rollout_mode)
    async def rollout_mode(self, *args, **kwargs):
        result = await original_rollout_mode(self, *args, **kwargs)
        assert not self._is_offload_param and not self._is_offload_optimizer
        assert self.rollout.sleep_level == 2, 'Rollout sleep must discard weights, not CPU-offload them'
        release_unused_host_memory('after_rollout_weight_sync')
        return result

    cls.rollout_mode = rollout_mode


install()
