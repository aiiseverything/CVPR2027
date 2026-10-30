import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "runtime/perception_v2_storage.py"
SPEC = importlib.util.spec_from_file_location("perception_v2_storage_test", PATH)
STORAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STORAGE)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / "data/runs/perception_v2"
        self.generalist = self.root / "data/runs/generalist"
        self.tmp = self.root / "tmp/spill"
        self.overlay = self.root / "overlay/spill"
        self.base = self.root / "base"
        self.base.mkdir()
        (self.base / "model.safetensors").write_bytes(b"weights" * 10)
        self.patches = [patch.object(STORAGE, "EXPECTED_RUN", self.run),
                        patch.object(STORAGE, "GENERALIST_RUN", self.generalist),
                        patch.object(STORAGE, "TMP_ROOT", self.tmp),
                        patch.object(STORAGE, "OVERLAY_ROOT", self.overlay),
                        patch.object(STORAGE, "BASE_MODEL", self.base),
                        patch.object(STORAGE, "RESERVE_BYTES", 32),
                        patch.object(STORAGE, "_disk", self.disk)]
        for p in self.patches:
            p.start()
        self.checkpoint(self.generalist, 50)
        self.actor = self.checkpoint(self.run, 25)

    def tearDown(self):
        for future, _ in list(STORAGE._JOBS.values()):
            future.result(timeout=5)
        STORAGE._JOBS.clear()
        for p in reversed(self.patches):
            p.stop()
        self.temp.cleanup()

    def disk(self, path):
        path = str(path)
        device = 2 if "/tmp/spill" in path else 3 if "/overlay/spill" in path else 1
        return {"free_bytes": 1000000, "device": device}

    def checkpoint(self, run, step):
        actor = run / "checkpoints" / f"global_step_{step}" / "actor"
        actor.mkdir(parents=True)
        for i, name in enumerate(STORAGE.CHECKPOINT_FILES):
            (actor / name).write_bytes(bytes([i + 1]) * (64 if name.startswith("model") else 128))
        (actor.parent / "data.pt").write_bytes(b"loader")
        (run / "checkpoints/latest_checkpointed_iteration.txt").write_text(str(step))
        return actor

    def finish(self, final=False):
        function = STORAGE.finalize if final else STORAGE.tick
        for _ in range(1000):
            report = function(self.run)
            if not report["pending"]:
                return report
            time.sleep(.001)
        self.fail("Storage worker did not finish")

    def test_first_spill_hash_receipt_and_repeat(self):
        original = {name: (self.actor / name).read_bytes() for name in STORAGE.SPILL_FILES}
        report = self.finish()
        self.assertEqual(report["spilled_step"], 25)
        state = json.loads((self.run / STORAGE.MANIFEST).read_text())
        for name, data in original.items():
            source = self.actor / name
            self.assertTrue(source.is_symlink())
            self.assertEqual(source.read_bytes(), data)
            entry = state["steps"]["25"]["files"][name]
            self.assertEqual(entry["sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(entry["status"], "linked")
        self.assertFalse((self.actor / STORAGE.OPTIM1).is_symlink())
        self.assertTrue((self.run / "perception_v2_storage_receipts/step_25.json").is_file())
        self.assertEqual(self.finish()["spilled_bytes"], report["spilled_bytes"])

    def test_retired_actor_gc_leaves_global_step_and_other_files(self):
        self.finish()
        state = json.loads((self.run / STORAGE.MANIFEST).read_text())
        owned = [Path(e["destination"]) for e in state["steps"]["25"]["files"].values()]
        stranger = self.tmp / "another_run/keep"
        stranger.parent.mkdir(parents=True)
        stranger.write_bytes(b"keep")
        shutil.rmtree(self.actor)
        self.assertTrue((self.actor.parent / "data.pt").exists())
        report = self.finish()
        self.assertEqual(report["collected_steps"], [25])
        self.assertTrue(all(not p.exists() for p in owned))
        self.assertEqual(stranger.read_bytes(), b"keep")

    def test_partial_actor_rotation_does_not_fail(self):
        self.finish()
        (self.actor / STORAGE.MODEL0).unlink()
        self.assertTrue(self.actor.is_dir())
        self.assertEqual(self.finish()["spilled_step"], 25)

    def test_gc_refuses_same_size_modified_spill(self):
        self.finish()
        destination = (self.actor / STORAGE.MODEL0).resolve()
        data = destination.read_bytes()
        destination.write_bytes(b"x" * len(data))
        shutil.rmtree(self.actor)
        with self.assertRaises(STORAGE.StorageOwnershipError):
            self.finish()
        self.assertTrue(destination.exists())

    def test_unknown_symlink_and_external_run_refused(self):
        external = self.root / "outside"
        external.write_bytes(b"never change")
        (self.actor / STORAGE.MODEL0).unlink()
        (self.actor / STORAGE.MODEL0).symlink_to(external)
        with self.assertRaises(STORAGE.StorageOwnershipError):
            self.finish()
        self.assertEqual(external.read_bytes(), b"never change")
        with self.assertRaises(STORAGE.StorageOwnershipError):
            STORAGE.tick(self.generalist)

    def test_capacity_failure_before_copy(self):
        with patch.object(STORAGE, "_disk", lambda p: {"free_bytes": 1, "device": self.disk(p)["device"]}):
            with self.assertRaises(STORAGE.StorageCapacityError):
                STORAGE.check_capacity("formal")
            with self.assertRaises(STORAGE.StorageCapacityError):
                self.finish()
        for name in STORAGE.CHECKPOINT_FILES:
            self.assertTrue((self.actor / name).is_file())
            self.assertFalse((self.actor / name).is_symlink())
        self.assertFalse(self.tmp.exists())

    def test_capacity_uses_actual_checkpoint_and_base_weights(self):
        formal = STORAGE.check_capacity("formal")
        self.assertEqual(formal["measured"]["checkpoint_bytes"], 390)
        self.assertEqual(formal["measured"]["hf_export_bytes"], 70)
        self.assertEqual(formal["required_free_bytes"], {"data": 556, "tmp": 224, "overlay": 96})
        smoke = STORAGE.check_capacity("smoke")
        self.assertEqual(smoke["required_free_bytes"], {"data": 492})

    def test_final_step_not_spilled_and_finalize_waits_for_worker(self):
        self.assertTrue(STORAGE.tick(self.run)["pending"])
        report = self.finish(final=True)
        self.assertTrue(report["finalized"])
        shutil.rmtree(self.actor)
        final_actor = self.checkpoint(self.run, 100)
        report = self.finish()
        self.assertEqual(report["latest_step"], 100)
        for name in STORAGE.CHECKPOINT_FILES:
            self.assertFalse((final_actor / name).is_symlink())

    def test_published_copy_recovery(self):
        self.finish()
        manifest = self.run / STORAGE.MANIFEST
        state = json.loads(manifest.read_text())
        entry = state["steps"]["25"]["files"][STORAGE.MODEL0]
        source = self.actor / STORAGE.MODEL0
        payload = source.read_bytes()
        source.unlink()
        source.write_bytes(payload)
        entry["source_signature"] = STORAGE._signature(source)
        entry["status"] = "published"
        state["steps"]["25"]["status"] = "pending"
        manifest.write_text(json.dumps(state))
        self.finish()
        self.assertTrue(source.is_symlink())
        self.assertEqual(source.read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
