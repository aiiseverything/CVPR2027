"""CPU-only queue recovery, ownership, provenance, and capacity checks."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "runtime/run_perception_v3_pipeline.py"
SPEC = importlib.util.spec_from_file_location("perception_v3_pipeline_under_test", SCRIPT)
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.data = self.root / "data_pipeline"
        self.sources = self.data / "perception_v3_sources"
        self.filtered = self.data / "prepared/perception_v3_filter"
        self.sources.mkdir(parents=True)
        paths = patch.multiple(pipeline, DATA=self.data, SOURCES=self.sources,
                               FILTER=self.filtered, STATE=self.root / "state.json")
        paths.start()
        self.addCleanup(paths.stop)
        stat = patch.object(pipeline.os, "statvfs", return_value=SimpleNamespace(f_files=200000, f_favail=100000))
        stat.start()
        self.addCleanup(stat.stop)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def shard(self, index, suffix=".jsonl"):
        folder = self.sources / "normalized_shards/vqav2"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"train-{index:05d}-of-00136{suffix}").write_text("{}\n")

    def test_next_shard_recovers_gap_before_later_completed_files(self):
        for index in (4, 6, 9):
            self.shard(index)
        self.shard(5, ".jsonl.writing")
        self.assertEqual(pipeline.next_shard("vqav2", 4), 5)
        self.shard(5)
        self.assertEqual(pipeline.next_shard("vqav2", 4), 7)

    def test_next_shard_starts_at_original_boundary(self):
        self.assertEqual(pipeline.next_shard("vqav2", 4), 4)
        self.shard(0)
        self.shard(3)
        self.assertEqual(pipeline.next_shard("vqav2", 4), 4)

    def test_identity_recognizes_actual_live_process(self):
        receipt = pipeline.identity(os.getpid())
        self.assertEqual(receipt["pid"], os.getpid())
        self.assertGreater(receipt["start_ticks"], 0)
        self.assertTrue(pipeline.alive(receipt))

    def test_pid_reuse_or_changed_command_is_not_owned(self):
        expected = {"pid": 1234, "start_ticks": 77, "argv": ["python", "screen.py"]}
        variants = (None, {**expected, "start_ticks": 78},
                    {**expected, "argv": ["python", "unrelated.py"]}, {**expected, "pid": 1235})
        for observed in variants:
            with self.subTest(observed=observed), patch.object(pipeline, "identity", return_value=observed):
                self.assertFalse(pipeline.alive(expected))
        with patch.object(pipeline, "identity", return_value=expected):
            self.assertTrue(pipeline.alive(expected))

    def prebuilt(self, predecessor=None):
        base = self.data / "prepared/perception_v3_pool"
        base.mkdir(parents=True)
        out = self.data / "prepared/perception_v3_pool_round2"
        out.mkdir()
        extra = self.sources / "normalized/vqav2.jsonl"
        extra.parent.mkdir()
        extra.write_text('{"id":"extra"}\n')
        (out / "candidates.records.jsonl").write_text('{"id":"candidate"}\n')
        (out / "candidates.parquet").write_bytes(b"owned candidate parquet bytes")
        audit = {"previous_pool": str(predecessor or base), "sources": {"input_files": [
            {"path": str(extra), "sha256": pipeline.sha(extra)}]}}
        self.write_json(out / "audit.json", audit)
        self.write_json(out / "manifest.json", {
            "audit_sha256": pipeline.sha(out / "audit.json"),
            "records_sha256": pipeline.sha(out / "candidates.records.jsonl"),
            "data_sha256": pipeline.sha(out / "candidates.parquet")})
        return {"pool_dirs": [str(base)]}, extra, out

    def test_valid_prebuilt_pool_is_adopted_without_rebuilding(self):
        state, extra, out = self.prebuilt()
        with patch.object(pipeline, "run") as run:
            pipeline.make_pool(state, [extra])
        run.assert_not_called()
        self.assertEqual(state["pool_dirs"][-1], str(out))

    def test_prebuilt_rejects_different_predecessor(self):
        state, extra, out = self.prebuilt(self.root / "unrelated_pool")
        with self.assertRaisesRegex(RuntimeError, "incompatible provenance"):
            pipeline.make_pool(state, [extra])
        self.assertEqual(len(state["pool_dirs"]), 1)

    def test_prebuilt_rejects_changed_source_snapshot(self):
        state, extra, out = self.prebuilt()
        extra.write_text('{"id":"changed"}\n')
        with self.assertRaisesRegex(RuntimeError, "current source snapshot"):
            pipeline.make_pool(state, [extra])

    def test_prebuilt_rejects_tampered_records(self):
        state, extra, out = self.prebuilt()
        (out / "candidates.records.jsonl").write_text('{"id":"tampered"}\n')
        with self.assertRaisesRegex(RuntimeError, "incompatible provenance"):
            pipeline.make_pool(state, [extra])

    def test_presence_shortage_expands_vqa_without_doc(self):
        with patch.object(pipeline.shutil, "disk_usage", return_value=SimpleNamespace(free=100 << 30)), \
                patch.object(pipeline, "run") as run:
            pipeline.expand({}, {"existence_no": 1})
        self.assertEqual(run.call_count, 1)
        command = list(map(str, run.call_args.args[2]))
        self.assertEqual(Path(command[2]).name, "expand_perception_v3_sources.py")
        for flag, value in {"--vqav2-start": "4", "--vqav2-stop": "20",
                            "--docvqa-start": "2", "--docvqa-stop": "2"}.items():
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_ocr_shortage_expands_only_doc(self):
        with patch.object(pipeline.shutil, "disk_usage", return_value=SimpleNamespace(free=100 << 30)), \
                patch.object(pipeline, "run") as run:
            pipeline.expand({}, {"ocr_document": 1})
        command = list(map(str, run.call_args.args[2]))
        for flag, value in {"--vqav2-start": "4", "--vqav2-stop": "4", "--docvqa-stop": "4"}.items():
            self.assertEqual(command[command.index(flag) + 1], value)

    def test_grounding_shortage_uses_refcoco_expander(self):
        with patch.object(pipeline.shutil, "disk_usage", return_value=SimpleNamespace(free=100 << 30)), \
                patch.object(pipeline, "run") as run:
            pipeline.expand({}, {"grounding": 1})
        self.assertEqual(run.call_count, 1)
        self.assertEqual(Path(run.call_args.args[2][2]).name, "expand_perception_v3_refcoco.py")

    def test_projected_disk_floor_blocks_new_expansion(self):
        # Above the old60GiB floor, but below the floor after the next round's reservation.
        with patch.object(pipeline.shutil, "disk_usage", return_value=SimpleNamespace(free=61 << 30)), \
                patch.object(pipeline, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "projected staging disk"):
                pipeline.expand({}, {"object_attribute": 1})
        run.assert_not_called()

    def test_shared_inode_floor_blocks_expansion(self):
        with patch.object(pipeline.os, "statvfs", return_value=SimpleNamespace(f_files=200000, f_favail=100)), \
                patch.object(pipeline, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "filesystem inodes"):
                pipeline.expand({}, {"object_attribute": 1})
        run.assert_not_called()

    def test_image_inode_floor_is_checked_separately(self):
        replies = [SimpleNamespace(f_files=200000, f_favail=5000),
                   SimpleNamespace(f_files=200000, f_favail=1000)]
        with patch.object(pipeline.os, "statvfs", side_effect=replies), patch.object(pipeline, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "image-cache filesystem inodes"):
                pipeline.expand({}, {"grounding": 1})
        run.assert_not_called()

    def test_exhausted_sources_do_not_relax_or_repeat_data(self):
        with patch.object(pipeline, "next_shard", side_effect=[136, 12]), \
                patch.object(pipeline.shutil, "disk_usage", return_value=SimpleNamespace(free=100 << 30)), \
                patch.object(pipeline, "run") as run:
            with self.assertRaisesRegex(RuntimeError, "exhausted"):
                pipeline.expand({}, {"existence_yes": 1, "ocr_document": 1})
        run.assert_not_called()

    def cache_fixture(self, records=3000):
        raw = self.sources / "raw/vqav2/owned.parquet"
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b"reproducible source cache")
        image = self.sources / "images/keep.img"
        image.parent.mkdir()
        image.write_bytes(b"training image")
        self.write_json(self.filtered / "dataset/manifest.json", {"status": "complete", "records": records})
        return raw, image

    def test_retirement_only_deletes_new_transport_cache(self):
        raw, image = self.cache_fixture()
        state = {}
        pipeline.release_owned_raw_cache(state)
        self.assertFalse(raw.exists())
        self.assertTrue(image.exists())
        receipt = json.loads((self.sources / "cache_retirement.json").read_text())
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(len(receipt["files"]), 1)
        self.assertGreater(state["retired_source_cache_bytes"], 0)

    def test_incomplete_dataset_prevents_retirement(self):
        raw, image = self.cache_fixture(records=2999)
        with self.assertRaisesRegex(RuntimeError, "complete filtered dataset"):
            pipeline.release_owned_raw_cache({})
        self.assertTrue(raw.exists())

    def test_symlink_cannot_delete_unowned_file(self):
        raw, image = self.cache_fixture()
        outside = self.root / "unowned.parquet"
        outside.write_bytes(b"do not delete")
        raw.unlink()
        raw.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError, "Unexpected source cache path"):
            pipeline.release_owned_raw_cache({})
        self.assertTrue(outside.exists())


if __name__ == "__main__":
    unittest.main()
