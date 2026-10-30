"""CPU integration checks for the parallel entry point and vLLM factory patch."""
from contextlib import contextmanager, ExitStack
import copy
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v3_parallel_main_checks", ROOT / "data_pipeline/filter_perception_v3_parallel.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class ParallelEntrypointTests(unittest.TestCase):
    @contextmanager
    def harness(self, arguments, prepare_only=False):
        module = ModuleType("vllm")
        original_llm = Mock(name="original_llm")
        module.LLM = original_llm
        backend = Mock(name="parallel_backend")
        constructor = Mock(return_value=backend)
        observed = {}
        kwargs = {"model": "frozen-model", "tensor_parallel_size": 1, "seed": 123,
                  "dtype": "bfloat16", "max_model_len": 10240, "max_num_seqs": 120,
                  "gpu_memory_utilization": 0.70, "max_num_batched_tokens": 10240,
                  "limit_mm_per_prompt": {"image": 1}, "mm_processor_cache_gb": 0,
                  "mm_processor_kwargs": {"min_pixels": 4096, "max_pixels": 786432},
                  "compilation_config": {"pass_config": {"fuse_allreduce_rms": False}},
                  "kernel_config": {"enable_flashinfer_autotune": False}, "enable_prefix_caching": True}

        def core_main():
            observed.update(argv=list(sys.argv), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                            bytecode=os.environ.get("PYTHONDONTWRITEBYTECODE"),
                            worker_method=os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
                            core_file=DRIVER.core.__file__)
            if not prepare_only:
                from vllm import LLM
                observed["backend"] = LLM(**kwargs)

        core_mock = Mock(side_effect=core_main)
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {}, clear=False))
            os.environ.pop("VLLM_WORKER_MULTIPROC_METHOD", None)
            stack.enter_context(patch.object(sys, "argv", [DRIVER.__file__, *arguments]))
            # None makes any accidental import fail during a CPU-only run.
            stack.enter_context(patch.dict(sys.modules, {"vllm": None if prepare_only else module}))
            stack.enter_context(patch.object(DRIVER, "ParallelLLM", constructor))
            stack.enter_context(patch.object(DRIVER.core, "main", core_mock))
            signals = stack.enter_context(patch.object(DRIVER.signal, "signal"))
            yield SimpleNamespace(module=module, original_llm=original_llm, backend=backend,
                                  constructor=constructor, observed=observed, kwargs=kwargs,
                                  core=core_mock, invoke_core=core_main, signals=signals)

    def test_success_forwards_core_arguments_gpu_identity_and_engine_kwargs(self):
        original_file = DRIVER.core.__file__
        original_hash = DRIVER.core.file_sha(original_file)
        forwarded = ["--pool-dir", "/fixture/pool", "--output-dir", "/fixture/output",
                     "--seed", "20261029", "--target", "3000", "--protocol", "/fixture/protocol.json",
                     "--model", "/fixture/model"]
        with self.harness(["--gpus", "3,7", *forwarded]) as h:
            original_kwargs = copy.deepcopy(h.kwargs)
            DRIVER.main()
            self.assertEqual(h.observed["argv"], [str(DRIVER.HERE / "filter_perception_v3.py"),
                             *forwarded, "--batch-prompts", "256", "--gpu", "3"])
            self.assertEqual(h.observed["gpu"], "3")
            self.assertEqual(h.observed["bytecode"], "1")
            self.assertEqual(h.observed["worker_method"], "spawn")
            self.assertEqual(h.observed["core_file"], original_file)
            self.assertEqual(DRIVER.core.file_sha(original_file), original_hash)
            expected_kwargs = {**original_kwargs, "max_num_seqs": 256, "gpu_memory_utilization": 0.88,
                               "disable_log_stats": False}
            h.constructor.assert_called_once_with([3, 7], Path("/fixture/output"), 256, **expected_kwargs)
            self.assertEqual(h.kwargs, original_kwargs)
            changed = {key for key, value in h.constructor.call_args.kwargs.items()
                       if key not in original_kwargs or value != original_kwargs[key]}
            self.assertEqual(changed, {"max_num_seqs", "gpu_memory_utilization", "disable_log_stats"})
            self.assertIs(h.observed["backend"], h.backend)
            h.backend.close.assert_called_once_with(error=None)
            self.assertIs(h.module.LLM, h.original_llm)
            h.original_llm.assert_not_called()

    def test_explicit_batch_size_is_preserved(self):
        for arguments in (["--batch-prompts", "10"], ["--batch-prompts=10"]):
            with self.subTest(arguments=arguments), self.harness(["--gpus=3,7", *arguments]) as h:
                DRIVER.main()
                self.assertEqual(sum(arg == "--batch-prompts" or arg.startswith("--batch-prompts=")
                                     for arg in h.observed["argv"]), 1)
                self.assertEqual(h.constructor.call_args.args[0], [3, 7])
                self.assertEqual(h.constructor.call_args.args[2], 10)

    def test_capacity_overrides_preserve_all_other_model_and_preprocessing_options(self):
        alternatives = (["--max-num-seqs", "192", "--gpu-memory-utilization", "0.82"],
                        ["--max-num-seqs=192", "--gpu-memory-utilization=0.82"])
        for arguments in alternatives:
            with self.subTest(arguments=arguments), self.harness(arguments) as h:
                original_kwargs = copy.deepcopy(h.kwargs)
                DRIVER.main()
                actual = h.constructor.call_args.kwargs
                self.assertEqual(actual, {**original_kwargs, "max_num_seqs": 192,
                                          "gpu_memory_utilization": 0.82, "disable_log_stats": False})
                self.assertEqual(h.kwargs, original_kwargs)
                self.assertFalse(any(arg.startswith("--max-num-seqs") or arg.startswith("--gpu-memory-utilization")
                                     for arg in h.observed["argv"]))

    def test_capacity_boundaries_rejected_before_core_or_engine(self):
        alternatives = (["--max-num-seqs", "0"], ["--max-num-seqs=-1"],
                        ["--gpu-memory-utilization", "0"], ["--gpu-memory-utilization=-0.1"],
                        ["--gpu-memory-utilization=0.951"], ["--gpu-memory-utilization=1"],
                        ["--gpu-memory-utilization=nan"], ["--gpu-memory-utilization=inf"])
        for arguments in alternatives:
            with self.subTest(arguments=arguments), self.harness(arguments) as h:
                with self.assertRaisesRegex(ValueError, "Invalid screening runtime capacity"):
                    DRIVER.main()
                h.core.assert_not_called()
                h.constructor.assert_not_called()
        with self.harness(["--max-num-seqs=1", "--gpu-memory-utilization=0.95"]) as h:
            DRIVER.main()
            self.assertEqual(h.constructor.call_args.kwargs["max_num_seqs"], 1)
            self.assertEqual(h.constructor.call_args.kwargs["gpu_memory_utilization"], 0.95)

    def test_core_failure_closes_backend_and_restores_module_factory(self):
        with self.harness([]) as h:
            def failing_core():
                h.invoke_core()
                raise RuntimeError("core failed after generation")
            h.core.side_effect = failing_core
            with self.assertRaisesRegex(RuntimeError, "core failed after generation"):
                DRIVER.main()
            h.backend.close.assert_called_once_with(error="RuntimeError: core failed after generation")
            self.assertIs(h.module.LLM, h.original_llm)

    def test_no_engine_created_when_core_finishes_without_inference(self):
        with self.harness([]) as h:
            h.core.side_effect = None
            DRIVER.main()
            h.constructor.assert_not_called()
            h.backend.close.assert_not_called()
            self.assertIs(h.module.LLM, h.original_llm)

    def test_prepare_only_never_imports_vllm_or_creates_workers(self):
        with self.harness(["--gpus", "0,1", "--prepare-only", "--max-num-seqs=128",
                           "--gpu-memory-utilization=0.85"], prepare_only=True) as h:
            DRIVER.main()
            h.core.assert_called_once_with()
            h.constructor.assert_not_called()
            h.backend.close.assert_not_called()
            self.assertIn("--prepare-only", h.observed["argv"])
            self.assertEqual(h.observed["argv"][-2:], ["--gpu", "0"])
            self.assertFalse(any(arg.startswith("--max-num-seqs") or arg.startswith("--gpu-memory-utilization")
                                 for arg in h.observed["argv"]))
            h.signals.assert_not_called()

    def test_single_gpu_recovery_uses_one_replica_and_preserves_core_arguments(self):
        forwarded = ["--pool-dir", "/fixture/pool", "--output-dir", "/fixture/output",
                     "--batch-prompts", "128", "--seed", "20261029"]
        with self.harness(["--gpus", "4", *forwarded]) as h:
            DRIVER.main()
            self.assertEqual(h.observed["argv"], [str(DRIVER.HERE / "filter_perception_v3.py"),
                                                  *forwarded, "--gpu", "4"])
            self.assertEqual(h.observed["gpu"], "4")
            self.assertEqual(h.constructor.call_args.args, ([4], Path("/fixture/output"), 128))
            self.assertEqual(h.constructor.call_args.kwargs["tensor_parallel_size"], 1)
            self.assertEqual(h.constructor.call_args.kwargs["seed"], h.kwargs["seed"])
            self.assertEqual(h.constructor.call_args.kwargs["mm_processor_kwargs"],
                             h.kwargs["mm_processor_kwargs"])
            h.backend.close.assert_called_once_with(error=None)
            self.assertIs(h.module.LLM, h.original_llm)

    def test_invalid_gpu_layout_rejected_before_core_or_engine(self):
        for arguments in (["--gpus", "0,0"], ["--gpus", "0,1,2"], ["--gpu", "1"], ["--gpu=1"]):
            with self.subTest(arguments=arguments), self.harness(arguments) as h:
                with self.assertRaises(ValueError):
                    DRIVER.main()
                h.core.assert_not_called()
                h.constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
