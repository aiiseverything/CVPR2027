"""CPU-only tests of capacity-derived admission and boot-scoped OOM evidence."""
from collections import deque
import copy
from pathlib import Path
import tempfile
import unittest

import throughput


def row(prompt=1000, visual=100, identifier=None):
    return {"prompt_tokens": prompt, "visual_tokens": visual, "id": identifier}


class ThroughputTests(unittest.TestCase):
    def test_policy_and_profiles_are_fresh_snapshots(self):
        value = throughput.configuration()
        self.assertEqual(value["version"], "profiled-kv-admission-v1")
        self.assertEqual((value["max_inflight"], value["max_num_seqs"],
                          value["max_num_batched_tokens"], value["gpu_memory_utilization"]),
                         (64, 64, 32768, 0.95))
        self.assertEqual([(p["max_num_seqs"], p["max_num_batched_tokens"], p["gpu_memory_utilization"])
                          for p in value["profiles"]], [(64, 32768, .95), (32, 16384, .95), (32, 16384, .92)])
        value["profiles"][0]["max_num_seqs"] = 0
        self.assertEqual(throughput.configuration()["profiles"][0]["max_num_seqs"], 64)

    def test_capacity_only_uses_this_boot_and_last_ansi_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vllm.log"
            old = b"GPU KV cache size: 9,000,000 tokens\n"
            path.write_bytes(old + b"starting new server\n")
            self.assertIsNone(throughput.read_cache_capacity(path, len(old)))
            with path.open("ab") as stream:
                stream.write(b"GPU KV cache size: 800000 tokens\n")
                stream.write(b"\x1b[1;36m(Engine) GPU KV cache size: \x1b[0m1,802,240 tokens\n")
            self.assertEqual(throughput.read_cache_capacity(path, len(old)), 1802240)
            self.assertIsNone(throughput.read_cache_capacity(path, path.stat().st_size))

    def test_absent_logs_and_invalid_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            self.assertIsNone(throughput.read_cache_capacity(path, 0))
            self.assertFalse(throughput.owned_server_oom(path, 0))
            path.write_text("short")
            for offset in (-1, True, 1.5, 100):
                with self.subTest(offset=offset):
                    with self.assertRaises(ValueError):
                        throughput.read_cache_capacity(path, offset)
                    with self.assertRaises(ValueError):
                        throughput.owned_server_oom(path, offset)

    def test_zero_report_is_not_treated_as_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log"
            path.write_text("GPU KV cache size: 0 tokens\n")
            with self.assertRaises(ValueError):
                throughput.read_cache_capacity(path, 0)

    def test_runtime_capacity_rounding_and_all_fallback_profiles(self):
        for index, expected in enumerate((64, 32, 32)):
            budget = throughput.runtime_budget(1802241, index)
            self.assertEqual(budget["max_inflight"], expected)
            self.assertEqual(budget["reserved_token_budget"], 1622016)
            fallback = throughput.runtime_budget(None, index)
            self.assertEqual(fallback["max_inflight"], 32)
            self.assertIsNone(fallback["reserved_token_budget"])
        for invalid in (0, -1, True, 1.5, "100"):
            with self.assertRaises(ValueError):
                throughput.runtime_budget(invalid)
        for invalid in (-1, 3, True, 0.5):
            with self.assertRaises(ValueError):
                throughput.runtime_budget(100000, invalid)

    def test_token_reservation_exact_boundary(self):
        candidate = row(20000, 18000)
        reservation = 20000 + 8192 + 4096
        budget = {"max_inflight": 64, "reserved_token_budget": reservation * 2}
        self.assertTrue(throughput.can_admit(candidate, [candidate], budget))
        self.assertFalse(throughput.can_admit(row(20001, 18000), [candidate], budget))
        self.assertFalse(throughput.can_admit(candidate, [candidate] * 2, budget))
        with self.assertRaisesRegex(ValueError, "Single row needs"):
            throughput.can_admit(row(40960, 32768), [], {"max_inflight": 64, "reserved_token_budget": 53247})
        self.assertTrue(throughput.can_admit(row(40960, 32768), [], {"max_inflight": 64, "reserved_token_budget": 53248}))

    def test_known_and_unknown_capacity_inflight_limits(self):
        candidate = row()
        known = throughput.runtime_budget(10000000)
        fallback = throughput.runtime_budget(None)
        self.assertTrue(throughput.can_admit(candidate, [candidate] * 63, known))
        self.assertFalse(throughput.can_admit(candidate, [candidate] * 64, known))
        self.assertTrue(throughput.can_admit(candidate, [candidate] * 31, fallback))
        self.assertFalse(throughput.can_admit(candidate, [candidate] * 32, fallback))
        with self.assertRaises(ValueError):
            throughput.can_admit(candidate, [], {"max_inflight": 64, "reserved_token_budget": None})

    def test_invalid_rows_and_budgets_raise(self):
        invalid_rows = [{}, {"prompt_tokens": 100}, {"visual_tokens": 100},
                        row(0, 0), row(-1, 0), row(40961, 0), row(40000, 32769),
                        row(100, -1), row(100, 101), row(True, 0), row(100, True),
                        row(100.0, 0), row(100, "1"), None]
        for candidate in invalid_rows:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    throughput.validate_row(candidate)
                with self.assertRaises(ValueError):
                    throughput.can_admit(row(), [candidate], throughput.runtime_budget(None))
        for budget in ({}, {"max_inflight": 0, "reserved_token_budget": None},
                       {"max_inflight": 32}, {"max_inflight": 32, "reserved_token_budget": -1},
                       {"max_inflight": True, "reserved_token_budget": 100000}, None):
            with self.assertRaises(ValueError):
                throughput.can_admit(row(), [], budget)
        self.assertIsNone(throughput.validate_row(row(1, 0)))

    def test_pure_functions_and_mixed_fifo_complete_once(self):
        source = [row(p, v, i) for i, (p, v) in enumerate(
            [(1000, 100), (40960, 32768), (21000, 20000), (7000, 6000)] * 40)]
        original = copy.deepcopy(source)
        pending, active, submitted, completed = deque(source), [], [], []
        budget = throughput.runtime_budget(200000)
        while pending or active:
            while pending and throughput.can_admit(pending[0], iter(active), budget):
                current = pending.popleft()
                active.append(current)
                submitted.append(current["id"])
            self.assertTrue(active, "An empty set must accept every individually fitting row")
            self.assertLessEqual(sum(r["prompt_tokens"] + 12288 for r in active), 180000)
            completed.append(active.pop(-1 if len(completed) % 2 else 0)["id"])
            self.assertLessEqual(len(completed), len(source))
        self.assertEqual(submitted, list(range(len(source))))
        self.assertEqual(sorted(completed), list(range(len(source))))
        self.assertEqual(source, original)

    def test_oom_is_boot_scoped_and_only_last_mebibyte(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vllm.log"
            old = b"torch.OutOfMemoryError: CUDA out of memory\n"
            path.write_bytes(old + b"Connection refused; HTTP 503; timeout\n")
            self.assertFalse(throughput.owned_server_oom(path, len(old)))
            with path.open("ab") as stream:
                stream.write(b"\x1b[31mtorch.OutOfMemoryError\x1b[0m: failed allocation\n")
            self.assertTrue(throughput.owned_server_oom(path, len(old)))
            path.write_bytes(old + b"x" * (1024 * 1024 + 1))
            self.assertFalse(throughput.owned_server_oom(path, 0))
            for message in (b"CUDA out of memory", b"CUDA error: out of memory", b"torch.cuda.OutOfMemoryError"):
                path.write_bytes(message)
                self.assertTrue(throughput.owned_server_oom(path, 0))


if __name__ == "__main__":
    unittest.main()
