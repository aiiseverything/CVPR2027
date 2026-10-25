"""CPU-only checks of mixed-image FIFO scheduling and resource boundaries."""
from collections import deque
import copy
import unittest

import scheduling


def row(visual, prompt=None, identifier=None):
    return {"id": identifier, "visual_tokens": visual,
            "prompt_tokens": max(1, visual) if prompt is None else prompt}


class SchedulingTests(unittest.TestCase):
    def test_fixed_policy_snapshot_is_independent(self):
        config = scheduling.configuration()
        self.assertEqual(config["version"], "visual-token-admission-v1")
        self.assertEqual(config["capacity_slots"], 16)
        self.assertEqual(config["max_inflight"], 16)
        self.assertEqual(config["max_visual_tokens_per_row"], 32768)
        self.assertEqual(config["max_inflight_visual_tokens"], 65536)
        self.assertEqual(config["max_inflight_prompt_tokens"], 81920)
        config["visual_token_costs"][0]["slots"] = 999
        self.assertEqual(scheduling.configuration()["visual_token_costs"][0]["slots"], 1)

    def test_cost_thresholds_and_homogeneous_capacity(self):
        for visual, cost in [(0, 1), (4096, 1), (4097, 2), (8192, 2),
                             (8193, 4), (16384, 4), (16385, 8), (32768, 8)]:
            with self.subTest(visual=visual):
                candidate = row(visual)
                self.assertEqual(scheduling.row_cost(candidate), cost)
                count = 16 // cost
                self.assertTrue(scheduling.can_admit(candidate, [candidate] * (count - 1)))
                self.assertFalse(scheduling.can_admit(candidate, [candidate] * count))

    def test_mixed_high_and_low_images_share_capacity(self):
        high, low = row(20000, 20500), row(1500, 1700)
        self.assertTrue(scheduling.can_admit(high, [low] * 8))
        self.assertFalse(scheduling.can_admit(high, [low] * 9))
        self.assertTrue(scheduling.can_admit(low, [high] + [low] * 7))
        self.assertFalse(scheduling.can_admit(low, [high] + [low] * 8))
        self.assertFalse(scheduling.can_admit(high, [high, low]))

    def test_prompt_budget_is_independent_of_visual_slots(self):
        active = [row(100, 40000), row(100, 40000)]
        self.assertTrue(scheduling.can_admit(row(100, 1920), active))
        self.assertFalse(scheduling.can_admit(row(100, 1921), active))
        self.assertTrue(scheduling.can_admit(row(100, 40960), [row(100, 40960)]))
        self.assertFalse(scheduling.can_admit(row(0, 1), [row(100, 40960)] * 2))

    def test_invalid_metadata_fails_before_waiting(self):
        invalid = [{}, {"visual_tokens": 1}, {"prompt_tokens": 100},
                   row(-1, 100), row(1, -1), row(True, 100), row(1, True),
                   row(1.0, 100), row(1, "100"), row(0, 0),
                   row(1, 40961), row(32769, 40000), row(101, 100), None]
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError):
                    scheduling.row_cost(candidate)
                with self.assertRaises(ValueError):
                    scheduling.can_admit(candidate, [])
                with self.assertRaises(ValueError):
                    scheduling.can_admit(row(100), [row(100)] * 16 + [candidate])

    def test_inputs_are_unchanged_and_active_iterators_work(self):
        candidate, active = row(2000, 2100), [row(20000, 20500)]
        original = copy.deepcopy((candidate, active))
        self.assertTrue(scheduling.can_admit(candidate, iter(active)))
        self.assertEqual((candidate, active), original)

    def test_every_valid_row_can_start_alone(self):
        for visual in (0, 4096, 4097, 8192, 8193, 16384, 16385, 32768):
            self.assertTrue(scheduling.can_admit(row(visual, 40960), []))

    def test_fifo_blocked_head_is_not_bypassed_by_small_rows(self):
        pending = deque([row(20000, 21000, "large"), row(100, 200, "small")])
        active = [row(20000, 21000), row(100, 200)]
        self.assertFalse(scheduling.can_admit(pending[0], active))
        self.assertTrue(scheduling.can_admit(pending[1], active))
        # Caller keeps the blocked FIFO head until an active request completes.
        active.pop(0)
        self.assertTrue(scheduling.can_admit(pending[0], active))
        active.append(pending.popleft())
        self.assertEqual(active[-1]["id"], "large")

    def test_mixed_fifo_completes_every_row_once_without_deadlock(self):
        # Include prompt-limited small images and both exact budget boundaries.
        specs = [(4096, 4300)] * 17 + [(32768, 40960), (100, 40000),
                 (16385, 17000), (0, 40960), (8192, 8400), (16384, 16600)] * 12
        source = [row(v, p, i) for i, (v, p) in enumerate(specs)]
        pending, active, submitted, completed = deque(source), [], [], []
        ticks = 0
        while pending or active:
            while pending and scheduling.can_admit(pending[0], active):
                current = pending.popleft()
                active.append(current)
                submitted.append(current["id"])
            self.assertTrue(active, "A valid FIFO head must fit an empty active set")
            self.assertLessEqual(sum(scheduling.row_cost(r) for r in active), 16)
            self.assertLessEqual(sum(r["prompt_tokens"] for r in active), 81920)
            self.assertLessEqual(sum(r["visual_tokens"] for r in active), 65536)
            self.assertLessEqual(len(active), 16)
            # Vary completion order; later requests can finish first.
            done = active.pop(-1 if ticks % 2 else 0)
            completed.append(done["id"])
            ticks += 1
            self.assertLessEqual(ticks, len(source))
        self.assertEqual(submitted, list(range(len(source))))
        self.assertEqual(sorted(completed), list(range(len(source))))


if __name__ == "__main__":
    unittest.main()
