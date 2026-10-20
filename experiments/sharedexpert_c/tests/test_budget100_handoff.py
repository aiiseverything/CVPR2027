"""CPU-only process fixtures; never signal real training or query GPUs."""
import json
from pathlib import Path
import signal
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import controller_handoff as core
import budget100_handoff as budget


def make_checkpoint(f):
    run = f.here / "runs/perception"
    checkpoint = run / "checkpoints/global_step_100"
    (checkpoint / "actor").mkdir(parents=True, exist_ok=True)
    required = [checkpoint / "data.pt", checkpoint / "actor/fsdp_config.json"]
    required += [checkpoint / "actor" / f"{kind}_world_size_4_rank_{rank}.pt"
                 for kind in ("model", "optim", "extra_state") for rank in range(4)]
    for path in required:
        path.write_bytes(b"CPU fixture only")
    (run / "checkpoints/latest_checkpointed_iteration.txt").write_text("100")
    (run / "rollouts").mkdir(exist_ok=True)
    (run / "rollouts/100.jsonl").write_text("".join(
        json.dumps({"group_uid": str(i // 8)}) + "\n" for i in range(240)))
    return required


class BudgetBoundary(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.object(core, "_FAKE_DRIVER",
            core._FAKE_DRIVER.replace("'reasoning','--gpus'", "'perception','--gpus'"))
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def fixture(self, **kwargs):
        return core.CpuFixture(teacher="perception", **kwargs)

    def execute(self, f, **kwargs):
        f.execute(decision_fn=budget.decision, children_fn=budget.perception_children,
                  progress_fn=budget.unexpected_progress, **kwargs)

    def test_complete_checkpoint_and_rollout_required(self):
        with self.fixture() as f:
            (f.here / "runs/perception/checkpoints/latest_checkpointed_iteration.txt").write_text("99")
            self.assertEqual(budget.decision(f.ctx, f.plan)[0], "wait")
            required = make_checkpoint(f)
            self.assertEqual(budget.decision(f.ctx, f.plan)[0], "ready")
            for path in required:
                contents = path.read_bytes()
                path.unlink()
                self.assertFalse(budget.checkpoint_ready(f.ctx))
                path.write_bytes(contents)
            rollout = f.here / "runs/perception/rollouts/100.jsonl"
            full = rollout.read_text()
            for invalid in ("{}", full[:-20], full.replace('"0"', '"1"')):
                rollout.write_text(invalid)
                self.assertFalse(budget.checkpoint_ready(f.ctx))
            rollout.write_text(full)
            self.assertTrue(budget.checkpoint_ready(f.ctx))
            self.assertTrue(core.alive(f.old))

    def test_overrun_and_other_teacher_progress_refuse(self):
        for role, step in (("perception", 101), ("reasoning", 1), ("generalist", 1)):
            with self.fixture() as f:
                make_checkpoint(f)
                path = f.here / f"runs/{role}/rollouts/{step}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
                self.assertEqual(budget.decision(f.ctx, f.plan)[0], "abort")
                with self.assertRaises(RuntimeError):
                    self.execute(f)
                self.assertTrue(core.alive(f.old))

    def test_actual_fake_process_handoff_preserves_checkpoint(self):
        for missing_child_field in (False, True):
            with self.fixture(empty_child_state=missing_child_field) as f:
                required = make_checkpoint(f)
                hashes = {str(path): core.sha(path) for path in required}
                old_child, descendant = f.child(), f.descendant()
                self.execute(f)
                self.assertEqual(f.state["stage"], "complete")
                self.assertFalse(core.alive(f.old))
                self.assertFalse(core.alive(old_child))
                self.assertFalse(core.alive(descendant))
                self.assertTrue(core.alive(f.state["new_driver"]))
                self.assertEqual(hashes, {str(path): core.sha(path) for path in required})
                with self.assertRaises(RuntimeError):
                    self.execute(f)

    def test_cancel_after_freeze_resumes_only_owned_stop(self):
        with self.fixture() as f:
            make_checkpoint(f)
            def cancelled():
                raise InterruptedError("CPU fixture cancellation")
            with self.assertRaises(InterruptedError):
                self.execute(f, after_freeze=cancelled)
            self.assertTrue(core.alive(f.old))
            self.assertNotIn(core.identity(f.old["pid"])["state"], ("T", "t"))
            self.assertTrue(core.alive(f.child()))

    def test_detached_watch_wait_deduplicate_cancel(self):
        with self.fixture() as f:
            make_checkpoint(f)
            (f.here / "runs/perception/checkpoints/latest_checkpointed_iteration.txt").write_text("99")
            ctx = budget.Context(f.here)
            for relative in ("training/launch_100.py", "training/config/execution_budget.json",
                             "logs/budget100_validation.json"):
                path = f.here / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            core.prepare(ctx, f.old["pid"], f.old["start_ticks"])
            launched = budget.launch(ctx)
            watcher = core.identity(launched["pid"])
            f.extra.append(watcher)
            try:
                core.wait_until(lambda: core.read(ctx.state, {}).get("stage") == "watching")
                self.assertTrue(budget.status(ctx)["watcher_running"])
                self.assertTrue(budget.launch(ctx)["already_running"])
                core.send_exact(watcher, signal.SIGTERM)
                core.wait_until(lambda: core.read(ctx.state, {}).get("stage") == "cancelled")
                self.assertTrue(core.alive(f.old))
                self.assertTrue(core.alive(f.child()))
            finally:
                if core.alive(watcher):
                    core.send_exact(watcher, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
