"""CPU-only regressions for independent GPU screening replicas."""
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v3_parallel_regression", ROOT / "data_pipeline/filter_perception_v3_parallel.py"
)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def fixture(size, worker_count=2):
    prompts = [{"prompt_token_ids": [100 + i], "multi_modal_data": {"image": [f"image-{i}"]}}
               for i in range(size)]
    params = [SimpleNamespace(n=8, seed=1000 + i, temperature=1.0, top_p=1.0,
                              top_k=-1, max_tokens=8192) for i in range(size)]
    shards = DRIVER.split_batch(prompts, params, worker_count)
    replies = []
    for gpu, shard in enumerate(shards):
        replies.append({"gpu": gpu, "indices": list(shard["indices"]), "outputs": [
            {"prompt_token_ids": prompts[i]["prompt_token_ids"], "finished": True,
             "outputs": [{"index": j, "token_ids": [i, j],
                          "finish_reason": "length" if j == 7 else "stop"}
                         for j in range(params[i].n)]}
            for i in shard["indices"]]})
    return prompts, params, shards, replies


class BatchPartitionTests(unittest.TestCase):
    def test_empty_single_and_odd_batches_are_disjoint_and_complete(self):
        for size in (0, 1, 5, 25):
            with self.subTest(size=size):
                prompts, params, shards, _ = fixture(size)
                self.assertEqual(len(shards), 2)
                indices = [i for shard in shards for i in shard["indices"]]
                self.assertEqual(sorted(indices), list(range(size)))
                self.assertEqual(len(indices), len(set(indices)))
                for worker, shard in enumerate(shards):
                    self.assertEqual(shard["indices"], list(range(worker, size, 2)))
                    for i, prompt, param in zip(shard["indices"], shard["prompts"],
                                                shard["sampling_params"], strict=True):
                        self.assertIs(prompt, prompts[i])
                        self.assertIs(param, params[i])
                        self.assertEqual(param.n, 8)
                        self.assertEqual(param.seed, 1000 + i)

    def test_partition_and_assembly_do_not_change_prompts_seeds_or_settings(self):
        prompts, params, _, _ = fixture(5)
        before = copy.deepcopy((prompts, params))
        shards = DRIVER.split_batch(prompts, params, 2)
        _, _, _, replies = fixture(5)
        DRIVER.assemble_batch(shards, replies, 5)
        self.assertEqual((prompts, params), before)

    def test_invalid_dimensions_fail(self):
        for prompts, params, workers in (([], [], 0), ([], [], -1), ([{}], [], 2)):
            with self.subTest(workers=workers, prompts=prompts):
                with self.assertRaises(ValueError):
                    DRIVER.split_batch(prompts, params, workers)


class BatchAssemblyTests(unittest.TestCase):
    def test_results_return_in_original_prompt_order_with_real_completion_indices(self):
        for size in (0, 1, 5):
            with self.subTest(size=size):
                _, _, shards, replies = fixture(size)
                for reply in replies:
                    for output in reply["outputs"]:
                        output["outputs"].reverse()
                result = DRIVER.assemble_batch(shards, replies, size)
                self.assertEqual(len(result), size)
                for i, output in enumerate(result):
                    self.assertIsInstance(output, SimpleNamespace)
                    self.assertEqual(output.prompt_token_ids, [100 + i])
                    self.assertEqual([x.index for x in output.outputs], list(reversed(range(8))))
                    self.assertEqual([x.token_ids for x in output.outputs], [[i, j] for j in reversed(range(8))])
                    self.assertEqual(output.outputs[0].finish_reason, "length")

    def test_each_prompt_uses_its_own_expected_completion_count(self):
        _, params, shards, replies = fixture(3)
        params[1].n = 3
        replies[1]["outputs"][0]["outputs"] = replies[1]["outputs"][0]["outputs"][:3]
        outputs = DRIVER.assemble_batch(shards, replies, 3)
        self.assertEqual([len(output.outputs) for output in outputs], [8, 3, 8])

    def test_missing_reply_wrong_indices_and_wrong_cardinality_are_rejected(self):
        mutations = (
            lambda shards, replies: replies.pop(),
            lambda shards, replies: replies[0].update(indices=[2, 0, 4]),
            lambda shards, replies: replies[0]["outputs"].pop(),
            lambda shards, replies: replies[0]["outputs"].append(copy.deepcopy(replies[0]["outputs"][0])),
            lambda shards, replies: shards[0]["sampling_params"].pop(),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                _, _, shards, replies = fixture(5)
                mutate(shards, replies)
                with self.assertRaises(ValueError):
                    DRIVER.assemble_batch(shards, replies, 5)

    def test_duplicate_missing_and_out_of_range_request_indices_are_rejected(self):
        for invalid_index, size in ((0, 5), (-1, 5), (5, 5), (1, 6)):
            with self.subTest(invalid_index=invalid_index, size=size):
                _, _, shards, replies = fixture(5)
                shards[1]["indices"][0] = invalid_index
                replies[1]["indices"][0] = invalid_index
                with self.assertRaises(ValueError):
                    DRIVER.assemble_batch(shards, replies, size)

    def test_incomplete_later_group_cannot_return_an_earlier_partial_batch(self):
        mutations = (
            lambda output: output.update(finished=False),
            lambda output: output.pop("finished"),
            lambda output: output["outputs"].pop(),
            lambda output: output["outputs"][-1].update(index=0),
            lambda output: output["outputs"][-1].update(index=8),
            lambda output: output["outputs"][-1].update(finish_reason=None),
            lambda output: output["outputs"][-1].update(finish_reason="abort"),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                _, _, shards, replies = fixture(5)
                mutate(replies[1]["outputs"][-1])
                before = copy.deepcopy(replies)
                with self.assertRaisesRegex(ValueError, "Incomplete worker generation"):
                    DRIVER.assemble_batch(shards, replies, 5)
                self.assertEqual(replies, before)


class BackendFailureTests(unittest.TestCase):
    def backend(self):
        backend = DRIVER.ParallelLLM.__new__(DRIVER.ParallelLLM)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        backend.output_dir = Path(temporary.name)
        backend.workers = [
            {"gpu": i, "pipe": Mock(), "process": Mock(exitcode=7)} for i in (0, 1)
        ]
        backend.batch_id = 0
        return backend

    def test_generate_dispatches_disjoint_prompts_and_reorders_worker_results(self):
        backend = self.backend()
        prompts, params, _, replies = fixture(5)
        replies = [dict(reply, batch_id=1) for reply in replies]
        backend._collect = Mock(return_value={1: replies[1], 0: replies[0]})
        with redirect_stdout(io.StringIO()):
            result = backend.generate(prompts, params)
        self.assertEqual([x.prompt_token_ids for x in result], [[100 + i] for i in range(5)])
        for i, worker in enumerate(backend.workers):
            worker["pipe"].send.assert_called_once()
            job = worker["pipe"].send.call_args.args[0]
            self.assertEqual(job["indices"], list(range(i, 5, 2)))
            self.assertEqual(job["sampling_params"], params[i::2])
            self.assertEqual(job["batch_id"], 1)
            self.assertTrue(all(p.n == 8 for p in job["sampling_params"]))

    def test_empty_shards_are_not_dispatched(self):
        for size in (0, 1):
            with self.subTest(size=size):
                backend = self.backend()
                prompts, params, _, replies = fixture(size)
                backend._collect = Mock(return_value={0: dict(replies[0], batch_id=1)} if size else {})
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(len(backend.generate(prompts, params)), size)
                backend.workers[1]["pipe"].send.assert_not_called()
                if not size:
                    backend.workers[0]["pipe"].send.assert_not_called()

    def test_failed_worker_never_assembles_or_fabricates_a_batch(self):
        backend = self.backend()
        prompts, params, _, _ = fixture(5)
        backend._collect = Mock(side_effect=RuntimeError("GPU 1 failed"))
        with patch.object(DRIVER, "assemble_batch") as assemble:
            with self.assertRaisesRegex(RuntimeError, "GPU 1 failed"):
                backend.generate(prompts, params)
            assemble.assert_not_called()
        self.assertFalse((backend.output_dir / "throughput.jsonl").exists())

    def test_stale_batch_is_rejected_before_assembly(self):
        backend = self.backend()
        prompts, params, _, replies = fixture(5)
        backend._collect = Mock(return_value={i: dict(reply, batch_id=0) for i, reply in enumerate(replies)})
        with patch.object(DRIVER, "assemble_batch") as assemble:
            with self.assertRaisesRegex(RuntimeError, "Stale GPU worker batch"):
                backend.generate(prompts, params)
            assemble.assert_not_called()
        self.assertFalse((backend.output_dir / "throughput.jsonl").exists())

    def test_throughput_uses_actual_outputs_and_parent_elapsed_time_without_mutation(self):
        backend = self.backend()
        prompts, params, _, replies = fixture(5)
        for gpu, reply in enumerate(replies):
            reply.update(batch_id=1, metrics={"prompts": len(reply["outputs"]),
                "responses": len(reply["outputs"]) * 8,
                "output_tokens": len(reply["outputs"]) * 16,
                "generation_seconds": 5.0 if gpu == 0 else 10.0})
        before = copy.deepcopy((prompts, params, replies))
        backend._collect = Mock(return_value={1: replies[1], 0: replies[0]})
        log = backend.output_dir / "throughput.jsonl"
        previous = '{"batch_id":0,"output_tokens":12}\n'
        log.write_text(previous)
        printed = io.StringIO()
        with patch.object(DRIVER.time, "monotonic", side_effect=[100.0, 115.0]), redirect_stdout(printed):
            outputs = backend.generate(prompts, params)
        records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertTrue(log.read_text().startswith(previous))
        self.assertEqual(len(records), 2)
        metrics = records[1]
        self.assertEqual(metrics["batch_id"], 1)
        self.assertEqual(metrics["parent_pid"], os.getpid())
        self.assertEqual(metrics["prompts"], 5)
        self.assertEqual(metrics["output_tokens"], 80)
        self.assertEqual(metrics["generation_seconds"], 15.0)
        self.assertAlmostEqual(metrics["output_tokens_per_second"], 80 / 15)
        self.assertAlmostEqual(metrics["questions_per_second"], 5 / 15)
        workers = {item["gpu"]: item for item in metrics["workers"]}
        self.assertEqual(workers[0]["responses"], 24)
        self.assertEqual(workers[1]["responses"], 16)
        self.assertEqual(workers[0]["output_tokens"], 48)
        self.assertEqual(workers[1]["output_tokens"], 32)
        self.assertEqual([workers[gpu]["generation_seconds"] for gpu in (0, 1)], [5.0, 10.0])
        self.assertEqual(sum(len(c.token_ids) for output in outputs for c in output.outputs), 80)
        event = json.loads(printed.getvalue())
        self.assertEqual(event.pop("stage"), "parallel_batch_throughput")
        self.assertEqual(event, metrics)
        self.assertEqual((prompts, params, replies), before)

    def test_incomplete_output_does_not_publish_success_throughput(self):
        backend = self.backend()
        prompts, params, _, replies = fixture(3)
        replies[1]["outputs"][0]["outputs"].pop()
        backend._collect = Mock(return_value={i: dict(reply, batch_id=1) for i, reply in enumerate(replies)})
        with self.assertRaisesRegex(ValueError, "Incomplete worker generation"):
            backend.generate(prompts, params)
        self.assertFalse((backend.output_dir / "throughput.jsonl").exists())

    def test_collect_checks_worker_errors_disconnects_and_identity(self):
        for response in ({"kind": "error", "gpu": 1, "error": "OOM"},
                         {"kind": "result", "gpu": 99},
                         {"kind": "ready", "gpu": 1}, EOFError()):
            with self.subTest(response=response):
                backend = self.backend()
                pipe = backend.workers[1]["pipe"]
                if isinstance(response, Exception):
                    pipe.recv.side_effect = response
                else:
                    pipe.recv.return_value = response
                with patch.object(DRIVER, "wait", return_value=[pipe]):
                    with self.assertRaises(RuntimeError):
                        backend._collect([1], "result", timeout=5)

    def test_collect_detects_dead_worker_without_waiting_for_timeout(self):
        backend = self.backend()
        backend.workers[1]["process"].is_alive.return_value = False
        with patch.object(DRIVER, "wait", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "worker exited 7"):
                backend._collect([1], "result", timeout=5)


class WorkerMetricsTests(unittest.TestCase):
    def exercise_worker(self, fail=False):
        prompts, params, shards, replies = fixture(3, worker_count=1)
        result = DRIVER.assemble_batch(shards, replies, 3)
        for output in result:
            output.finished = True
        # Count actual varying completion lengths, including an empty output.
        result[0].outputs[0].token_ids = []
        result[2].outputs[7].token_ids = [1, 2, 3, 4, 5]
        before = copy.deepcopy((prompts, params, result))
        llm = Mock()
        if fail:
            llm.generate.side_effect = RuntimeError("worker generation failed")
        else:
            llm.generate.return_value = result
        factory = Mock(return_value=llm)
        pipe = Mock()
        pipe.recv.side_effect = [{"kind": "generate", "batch_id": 7, **shards[0]}, {"kind": "close"}]
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {"vllm": SimpleNamespace(LLM=factory)}))
            stack.enter_context(patch.dict(os.environ, {}, clear=False))
            stack.enter_context(patch.object(DRIVER.os, "setsid"))
            stack.enter_context(patch.object(DRIVER, "process_identity", return_value={"pid": 123, "start_ticks": 456}))
            stack.enter_context(patch.object(DRIVER.time, "monotonic", side_effect=[100.0, 104.5]))
            if fail:
                with self.assertRaisesRegex(RuntimeError, "worker generation failed"):
                    DRIVER.worker_main(1, {"tensor_parallel_size": 1}, pipe)
            else:
                DRIVER.worker_main(1, {"tensor_parallel_size": 1}, pipe)
        factory.assert_called_once_with(tensor_parallel_size=1)
        llm.generate.assert_called_once_with(prompts, params, use_tqdm=False)
        pipe.close.assert_called_once()
        self.assertEqual((prompts, params, result), before)
        return [call.args[0] for call in pipe.send.call_args_list]

    def test_worker_measures_actual_generated_counts_and_duration(self):
        messages = self.exercise_worker()
        self.assertEqual([message["kind"] for message in messages], ["ready", "result"])
        reply = messages[1]
        self.assertEqual(reply["gpu"], 1)
        self.assertEqual(reply["batch_id"], 7)
        self.assertEqual(reply["indices"], [0, 1, 2])
        self.assertEqual(reply["metrics"], {"generation_seconds": 4.5, "prompts": 3,
                                            "responses": 24, "output_tokens": 49})
        self.assertEqual(reply["outputs"][0]["outputs"][0]["token_ids"], [])
        self.assertEqual(reply["outputs"][2]["outputs"][7]["token_ids"], [1, 2, 3, 4, 5])

    def test_failed_generation_has_no_success_result_or_throughput_counts(self):
        messages = self.exercise_worker(fail=True)
        self.assertEqual([message["kind"] for message in messages], ["ready", "error"])
        self.assertIn("worker generation failed", messages[1]["error"])
        self.assertNotIn("metrics", messages[1])


class ExecutionReceiptTests(unittest.TestCase):
    def backend(self, path):
        backend = DRIVER.ParallelLLM.__new__(DRIVER.ParallelLLM)
        backend.output_dir = path
        backend.devices = [0, 1]
        backend.receipt_path = None
        backend.receipt = {}
        backend.workers = []
        backend.locks = []
        backend.closed = False
        return backend

    def seed_run(self, path):
        original = b'{"id":"existing-real-group","completions":[1,2,3,4,5,6,7,8]}\n'
        (path / "groups.jsonl").write_bytes(original)
        (path / "run.json").write_text(json.dumps({"run_sha256": "frozen-run",
            "spec": {"driver_source_sha256": DRIVER.core.file_sha(DRIVER.HERE / "filter_perception_v3.py")}}))
        return original

    def test_initial_receipt_preserves_journal_and_does_not_claim_workers_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            original = self.seed_run(path)
            run_before = (path / "run.json").read_bytes()
            backend = self.backend(path)
            backend._start_receipt({"tensor_parallel_size": 1}, 24)
            receipt = json.loads(backend.receipt_path.read_text())
            self.assertEqual(receipt["status"], "initializing")
            self.assertEqual(receipt["workers"], [])
            self.assertEqual(receipt["initial_group_count"], 1)
            self.assertEqual(receipt["initial_groups_bytes"], len(original))
            self.assertEqual(receipt["parallel_driver_sha256"], DRIVER.core.file_sha(DRIVER.__file__))
            archived = path / "executions" / receipt["parallel_driver_source"]
            self.assertEqual(archived.read_bytes(), Path(DRIVER.__file__).read_bytes())
            self.assertEqual(receipt["run_sha256"], "frozen-run")
            self.assertEqual((path / "groups.jsonl").read_bytes(), original)
            self.assertEqual((path / "run.json").read_bytes(), run_before)
            self.assertEqual((path / "executions/legacy-before-parallel.groups.jsonl").read_bytes(), original)
            backend.close(error="worker initialization failed")
            receipt = json.loads(backend.receipt_path.read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["workers"], [])
            self.assertEqual(receipt["final_group_count"], 1)
            self.assertEqual((path / "groups.jsonl").read_bytes(), original)

    def test_changed_core_or_nonprefix_legacy_journal_cannot_publish_execution_receipt(self):
        for changed_core in (False, True):
            with self.subTest(changed_core=changed_core), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)
                self.seed_run(path)
                if changed_core:
                    run = json.loads((path / "run.json").read_text())
                    run["spec"]["driver_source_sha256"] = "changed-core"
                    (path / "run.json").write_text(json.dumps(run))
                else:
                    (path / "executions").mkdir()
                    (path / "executions/legacy-before-parallel.groups.jsonl").write_bytes(b"unrelated journal\n")
                backend = self.backend(path)
                with self.assertRaises(ValueError):
                    backend._start_receipt({"tensor_parallel_size": 1}, 24)
                self.assertIsNone(backend.receipt_path)
                self.assertEqual(list((path / "executions").glob("parallel-*.json")), [])

    def test_startup_failure_closes_owned_workers_and_never_claims_ready(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            path = Path(temporary)
            original = self.seed_run(path)
            context = Mock()
            pipes = [(Mock(), Mock()), (Mock(), Mock())]
            processes = [Mock(pid=900001 + i) for i in range(2)]
            for process in processes:
                process.is_alive.return_value = False
            context.Pipe.side_effect = pipes
            context.Process.side_effect = processes
            lock = Mock()
            get_context = stack.enter_context(patch.object(DRIVER.mp, "get_context", return_value=context))
            stack.enter_context(patch.object(DRIVER, "open", create=True, return_value=lock))
            stack.enter_context(patch.object(DRIVER.fcntl, "flock"))
            stack.enter_context(patch.object(DRIVER.subprocess, "check_output", side_effect=["GPU-FAKE\n", ""]))
            stack.enter_context(patch.object(DRIVER.ParallelLLM, "_collect",
                                            side_effect=RuntimeError("GPU 1 initialization failed")))
            killpg = stack.enter_context(patch.object(DRIVER.os, "killpg"))
            with self.assertRaisesRegex(RuntimeError, "GPU 1 initialization failed"):
                DRIVER.ParallelLLM([0, 1], path, 24, tensor_parallel_size=1)
            get_context.assert_called_once_with("spawn")
            self.assertTrue(all(call.kwargs["daemon"] is False for call in context.Process.call_args_list))
            for process, (parent_pipe, child_pipe) in zip(processes, pipes, strict=True):
                process.start.assert_called_once()
                process.join.assert_called_once()
                parent_pipe.close.assert_called_once()
                child_pipe.close.assert_called_once()
            self.assertEqual({call.args[0] for call in killpg.call_args_list}, {900001, 900002})
            lock.close.assert_called_once()
            receipts = list((path / "executions").glob("parallel-*.json"))
            self.assertEqual(len(receipts), 1)
            receipt = json.loads(receipts[0].read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertEqual(receipt["workers"], [])
            self.assertIn("GPU 1 initialization failed", receipt["error"])
            self.assertEqual((path / "groups.jsonl").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
