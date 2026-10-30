"""Protect screening decisions and exact mixture quotas before costly inference."""
import copy
import importlib.util
from collections import Counter
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "perception_v3_filtering", ROOT / "data_pipeline/perception_v3_filtering.py")
FILTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILTER)


def record(task="vqa", answer="yes", identity="fixture", category="object_attribute"):
    value = {"id": identity, "source": "fixture", "task": task,
             "answers": [answer] * (10 if task == "vqa" else 1),
             "perception_category": category}
    if task == "bbox":
        value["bbox"] = [0, 0, 100, 100]
    return value


def output(index, answer="yes", finish="stop"):
    return {"index": index, "text": "</think>\\boxed{" + answer + "}", "finish_reason": finish}


def accepted_pool(n=80):
    result = []
    for category, weight in FILTER.WEIGHTS.items():
        for i in range(n * weight // 100):
            answer = "yes" if i % 2 == 0 else "no"
            value = record(answer=answer, identity=f"{category}:{i}", category=category)
            other = "no" if answer == "yes" else "yes"
            value["difficulty_filter"] = FILTER.accept_group(
                value, [output(j, answer if j < 4 else other) for j in range(8)])
            result.append(value)
    return result


class Correctness(unittest.TestCase):
    def test_vqa_partial_consensus_is_not_a_correct_completion(self):
        value = record()
        value["answers"] = ["yes"] * 7 + ["no"] * 3
        scored = FILTER.score_completion(value, "</think>\\boxed{no}", "stop")
        self.assertAlmostEqual(scored["score"], .9)
        self.assertFalse(scored["correct"])
        self.assertTrue(FILTER.score_completion(value, "</think>\\boxed{yes}", "stop")["correct"])

    def test_anls_partial_credit_is_not_exact_correctness(self):
        value = record(task="anls", answer="1234567890")
        partial = FILTER.score_completion(value, "</think>\\boxed{1234567891}", "stop")
        self.assertAlmostEqual(partial["score"], .9)
        self.assertFalse(partial["correct"])
        self.assertTrue(FILTER.score_completion(
            record(task="anls", answer="Example"), "</think>\\boxed{ example }", "stop")["correct"])

    def test_bbox_threshold_is_inclusive(self):
        value = record(task="bbox")
        boundary = FILTER.score_completion(value, "</think>\\boxed{[0,0,50,100]}", "stop")
        self.assertEqual(boundary["score"], .5)
        self.assertTrue(boundary["correct"])
        self.assertFalse(FILTER.score_completion(value, "</think>\\boxed{[0,0,49,100]}", "stop")["correct"])

    def test_truncated_or_unfinished_or_missing_box_is_wrong(self):
        value = record()
        for text, finish in [("</think>\\boxed{yes}", "length"),
                             ("\\boxed{yes}", "stop"), ("</think>yes", "stop")]:
            with self.subTest(text=text, finish=finish):
                scored = FILTER.score_completion(value, text, finish)
                self.assertEqual(scored["score"], 0)
                self.assertFalse(scored["correct"])

    def test_unknown_finish_is_not_a_model_failure(self):
        with self.assertRaisesRegex(ValueError, "finish_reason"):
            FILTER.score_completion(record(), "</think>\\boxed{yes}", None)


class GroupAcceptance(unittest.TestCase):
    def test_accepts_only_two_through_six_correct(self):
        value = record()
        for k in range(9):
            result = FILTER.accept_group(value, [output(i, "yes" if i < k else "no") for i in range(8)])
            self.assertEqual(result["n_correct"], k)
            self.assertEqual(result["accepted"], 2 <= k <= 6)

    def test_fractional_accuracy_cannot_fake_two_successes(self):
        value = record()
        value["answers"] = ["yes"] * 7 + ["no"] * 3
        result = FILTER.accept_group(value, [output(i, "no") for i in range(8)])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["n_correct"], 0)
        self.assertAlmostEqual(result["reward_mean"], .9)
        self.assertEqual(result["reward_range"], 0)

    def test_untrusted_flags_are_recomputed(self):
        completions = [{**output(i, "no"), "correct": True, "score": 1} for i in range(8)]
        self.assertEqual(FILTER.accept_group(record(), completions)["n_correct"], 0)

    def test_incomplete_duplicate_error_and_bad_payload_are_not_wrong_answers(self):
        values = [output(i, "yes" if i < 4 else "no") for i in range(8)]
        cases = [values[:7], [values[0]] + values[:7],
                 [{**values[0], "error": "timeout"}] + values[1:],
                 [{**values[0], "finish_reason": None}] + values[1:]]
        for completions in cases:
            result = FILTER.accept_group(record(), completions)
            self.assertFalse(result["accepted"])
            self.assertIsNone(result["n_correct"])

    def test_completion_order_and_statistics(self):
        result = FILTER.accept_group(record(), list(reversed([
            output(i, "yes" if i < 4 else "no") for i in range(8)])))
        self.assertEqual(result["correct_flags"], [True] * 4 + [False] * 4)
        self.assertEqual(result["reward_mean"], .5)
        self.assertEqual(result["reward_std"], .5)
        self.assertEqual(result["reward_range"], 1)


class MixtureSelection(unittest.TestCase):
    def test_blocks_unique_balanced_deterministic_and_input_unchanged(self):
        pool = accepted_pool()
        before = copy.deepcopy(pool)
        chosen = FILTER.select_records(pool, 40, 19)
        again = FILTER.select_records(list(reversed(pool)), 40, 19)
        self.assertEqual(chosen, again)
        self.assertEqual(pool, before)
        self.assertEqual(len({r["id"] for r in chosen}), 40)
        expected = {key: value // 5 for key, value in FILTER.WEIGHTS.items()}
        for start in (0, 20):
            self.assertEqual(Counter(r["perception_category"] for r in chosen[start:start + 20]), expected)
        self.assertEqual(Counter(r["answers"][0] for r in chosen
                                 if r["perception_category"] == "existence_confusion"), {"yes": 3, "no": 3})

    def test_category_shortage_never_repeats(self):
        pool = [r for r in accepted_pool(40) if r["id"] != "grounding:0"]
        with self.assertRaisesRegex(ValueError, "Quota shortage for grounding"):
            FILTER.select_records(pool, 40, 1)
        self.assertEqual(FILTER.max_feasible_size(pool), 0)

    def test_presence_balance_shortage_and_explicit_opt_out(self):
        pool = accepted_pool(40)
        for row in pool:
            if row["perception_category"] == "existence_confusion":
                row["answers"] = ["yes"] * 10
                row["difficulty_filter"] = FILTER.accept_group(
                    row, [output(i, "yes" if i < 4 else "no") for i in range(8)])
        with self.assertRaisesRegex(ValueError, "existence_confusion:no"):
            FILTER.select_records(pool, 40, 1)
        self.assertEqual(FILTER.max_feasible_size(pool), 0)
        self.assertEqual(FILTER.max_feasible_size(pool, balance_existence=False), 40)
        self.assertEqual(len(FILTER.select_records(pool, 40, 1, balance_existence=False)), 40)

    def test_missing_receipt_duplicate_or_wrong_policy_rejected(self):
        pool = accepted_pool(40)
        with self.assertRaisesRegex(ValueError, "Duplicate accepted record"):
            FILTER.select_records(pool + [pool[0]], 40, 1)
        pool[0]["difficulty_filter"]["policy_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "screening receipt"):
            FILTER.select_records(pool, 40, 1)

    def test_size_constraints_and_maximum(self):
        self.assertEqual(FILTER.max_feasible_size(accepted_pool(80)), 80)
        self.assertEqual(FILTER.max_feasible_size([]), 0)
        with self.assertRaisesRegex(ValueError, "positive multiple of 20"):
            FILTER.select_records(accepted_pool(), 30, 1)
        with self.assertRaisesRegex(ValueError, "divisible by 40"):
            FILTER.select_records(accepted_pool(), 20, 1)
        self.assertEqual(len(FILTER.select_records(accepted_pool(), 20, 1, balance_existence=False)), 20)

    def test_changed_candidate_cannot_reuse_accepted_receipt(self):
        pool = accepted_pool(40)
        pool[0]["answers"] = ["different"] * 10
        with self.assertRaisesRegex(ValueError, "screening receipt"):
            FILTER.select_records(pool, 40, 1)


if __name__ == "__main__":
    unittest.main()
