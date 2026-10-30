"""CPU regressions for transport-only retries of immutable source downloads."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "runtime/run_perception_v3_pipeline.py"
SPEC = importlib.util.spec_from_file_location("perception_v3_network_retry_under_test", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class NetworkRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.log = self.root / "download.log"
        self.state_file = self.root / "state.json"
        settings = patch.multiple(pipeline, STATE=self.state_file, STOP=False)
        settings.start()
        self.addCleanup(settings.stop)
        self.state = {"code_sha256": {"source.py": "frozen-source-hash"},
                      "pool_dirs": ["immutable-pool"], "run_sha256": "frozen-run"}
        self.command = ["python", "source.py", "--vqav2-start", "20"]

    def append_log(self, message):
        with self.log.open("a") as stream:
            stream.write(message + "\n")

    def test_transport_and_retryable_http_errors_are_classified(self):
        messages = (
            "requests.exceptions.ProxyError: proxy unavailable",
            "requests.exceptions.ConnectionError: connection closed",
            "requests.exceptions.ConnectTimeout: connect timed out",
            "requests.exceptions.ReadTimeout: read timed out",
            "TimeoutError: timed out", "RemoteDisconnected: remote closed connection",
            "Connection reset by peer", "Connection refused",
            "Temporary failure in name resolution", "NameResolutionError: DNS unavailable",
            "HTTPError: 429 Client Error: Too Many Requests",
            "HTTPError: 502 Server Error: Bad Gateway",
            "HTTPError: 503 Server Error: Service Unavailable",
            "HTTPError: 504 Server Error: Gateway Timeout",
        )
        for message in messages:
            with self.subTest(message=message):
                self.log.write_text(message)
                self.assertTrue(pipeline.transient_network_failure(self.log, 0))

    def test_integrity_auth_and_tls_errors_override_transport_context(self):
        messages = (
            "Upstream LFS verification failed: actual != expected",
            "Metadata identity mismatch: metadata.json",
            "Wrong Content-Range", "Expected HTTP206 range, got 200",
            "Wrong range byte count", "SSLError: certificate invalid",
            "CERTIFICATE_VERIFY_FAILED", "401 Client Error: Unauthorized",
            "403 Client Error: Forbidden", "404 Client Error: Not Found",
            "ValueError: Normalized shard hash mismatch",
        )
        for message in messages:
            with self.subTest(message=message):
                self.log.write_text("ConnectionError: earlier transport context\n" + message)
                self.assertFalse(pipeline.transient_network_failure(self.log, 0))

    def test_missing_empty_or_unknown_error_does_not_retry(self):
        self.assertFalse(pipeline.transient_network_failure(self.log, 0))
        for message in ("", "RuntimeError: insufficient free disk", "normal progress only"):
            self.log.write_text(message)
            self.assertFalse(pipeline.transient_network_failure(self.log, 0))

    def test_old_transport_error_cannot_classify_a_new_unknown_failure(self):
        self.log.write_text("先前尝试\nProxyError: old failure\n")
        offset = self.log.stat().st_size
        self.append_log("RuntimeError: insufficient disk for verified shard assembly")
        self.assertFalse(pipeline.transient_network_failure(self.log, offset))

    def test_old_permanent_error_cannot_block_a_new_transport_retry(self):
        self.log.write_text("ValueError: old invalid input\n")
        offset = self.log.stat().st_size
        self.append_log("ProxyError: current attempt")
        self.assertTrue(pipeline.transient_network_failure(self.log, offset))

    def test_transient_failure_then_success_preserves_pins_and_artifacts(self):
        protected = {"groups.jsonl": b'{"id":"already-committed"}\n',
                     "run.json": b'{"run_sha256":"fixed"}\n',
                     "verified.parquet": b"verified source cache",
                     "source.manifest.json": b'{"sha256":"verified"}\n'}
        for name, content in protected.items():
            (self.root / name).write_bytes(content)
        before = copy.deepcopy(self.state)
        attempts = []

        def run(state, stage, command, log, reason):
            attempts.append((stage, list(command), log, reason))
            pipeline.update(state, stage, child=None)
            if len(attempts) == 1:
                self.append_log("requests.exceptions.ProxyError: proxy temporarily unavailable")
                raise RuntimeError("expanding exited 1")
            self.append_log("All source shards verified")

        with patch.object(pipeline, "run", side_effect=run), \
                patch.object(pipeline, "wait_network_retry") as wait:
            pipeline.run_source_download(self.state, self.command, self.log, "expand original sources")
        self.assertEqual(attempts, [("expanding", self.command, self.log, "expand original sources")] * 2)
        wait.assert_called_once_with(self.state, 30)
        self.assertEqual({key: self.state[key] for key in before}, before)
        self.assertEqual(self.state["network_retry_count"], 0)
        self.assertEqual(self.state["network_retry_in_seconds"], 0)
        self.assertEqual(self.state["stage"], "expanding")
        self.assertEqual(json.loads(self.state_file.read_text()), self.state)
        for name, content in protected.items():
            self.assertEqual((self.root / name).read_bytes(), content)

    def test_permanent_integrity_failure_is_not_retried(self):
        failure = RuntimeError("expanding exited 1")

        def run(*args):
            self.append_log("ConnectionError: recovered earlier in this attempt")
            self.append_log("ValueError: Upstream LFS verification failed: sha256 mismatch")
            raise failure

        with patch.object(pipeline, "run", side_effect=run) as called, \
                patch.object(pipeline, "wait_network_retry") as wait:
            with self.assertRaises(RuntimeError) as caught:
                pipeline.run_source_download(self.state, self.command, self.log, "expand")
        self.assertIs(caught.exception, failure)
        called.assert_called_once()
        wait.assert_not_called()

    def test_each_retry_classifies_only_its_new_log_bytes(self):
        attempts = []

        def run(*args):
            attempts.append(1)
            self.append_log("ProxyError: retryable" if len(attempts) == 1 else
                            "RuntimeError: insufficient free disk")
            raise RuntimeError("expanding exited 1")

        with patch.object(pipeline, "run", side_effect=run), \
                patch.object(pipeline, "wait_network_retry") as wait:
            with self.assertRaises(RuntimeError):
                pipeline.run_source_download(self.state, self.command, self.log, "expand")
        self.assertEqual(len(attempts), 2)
        wait.assert_called_once_with(self.state, 30)

    def test_backoff_increases_and_stays_bounded_at_five_minutes(self):
        attempts = []

        def run(*args):
            attempts.append(1)
            if len(attempts) <= 7:
                self.append_log("ReadTimeout: upstream temporarily unavailable")
                raise RuntimeError("expanding exited 1")

        with patch.object(pipeline, "run", side_effect=run), \
                patch.object(pipeline, "wait_network_retry") as wait:
            pipeline.run_source_download(self.state, self.command, self.log, "expand")
        self.assertEqual([call.args[1] for call in wait.call_args_list], [30, 60, 120, 240, 300, 300, 300])
        self.assertEqual(len(attempts), 8)

    def test_stop_before_attempt_prevents_any_download(self):
        with patch.object(pipeline, "STOP", True), patch.object(pipeline, "run") as run:
            with self.assertRaises(InterruptedError):
                pipeline.run_source_download(self.state, self.command, self.log, "expand")
        run.assert_not_called()

    def test_wait_updates_heartbeat_and_uses_five_second_intervals(self):
        clock = [100.0]
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            clock[0] += seconds

        with patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(pipeline.time, "sleep", side_effect=sleep), \
                patch.object(pipeline, "update", wraps=pipeline.update) as update:
            pipeline.wait_network_retry(self.state, 12)
        self.assertEqual(sleeps, [5, 5, 2])
        self.assertEqual(update.call_count, 3)
        self.assertEqual(self.state["stage"], "waiting_network")
        self.assertEqual(json.loads(self.state_file.read_text()), self.state)

    def test_stop_during_backoff_propagates_without_a_second_attempt(self):
        clock = [0.0]

        def run(*args):
            self.append_log("ProxyError: upstream temporarily unavailable")
            raise RuntimeError("expanding exited 1")

        def stop_during_sleep(seconds):
            clock[0] += seconds
            pipeline.STOP = True

        with patch.object(pipeline, "run", side_effect=run) as called, \
                patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), \
                patch.object(pipeline.time, "sleep", side_effect=stop_during_sleep):
            with self.assertRaises(InterruptedError):
                pipeline.run_source_download(self.state, self.command, self.log, "expand")
        called.assert_called_once()
        self.assertEqual(self.state["network_retry_count"], 1)

    def test_gpu_screening_failure_does_not_use_download_retry(self):
        with patch.object(pipeline, "run", side_effect=RuntimeError("ProxyError during GPU screening")) as run, \
                patch.object(pipeline, "run_source_download") as download:
            with self.assertRaises(RuntimeError):
                pipeline.screening(self.state)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[1], "screening")
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
