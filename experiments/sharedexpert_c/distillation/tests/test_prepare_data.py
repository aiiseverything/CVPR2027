"""Mixture, deduplication and leakage tests without GPU or pyarrow."""
from collections import Counter
from copy import deepcopy
import unittest

from experiments.sharedexpert_c.distillation.prepare_data import (
    DOMAIN_ORDER, SOURCE_WEIGHTS, allocate, audit_rows, mix_domains, select_domain,
)


def example(source, number=0):
    return {"extra_info": {"source": source, "record_id": f"{source}:{number}", "split": "train",
                           "question_sha256": f"q-{source}-{number}", "image_sha256": [f"image-{source}-{number}"]},
            "prompt": [{"role": "user", "content": "question"}],
            "reward_model": {"ground_truth": "answer"}, "images": [{"path": "/test/image.jpg"}]}


class DataPreparationTest(unittest.TestCase):
    def inputs(self):
        return {domain: [example(source, number) for source in weights for number in range(12)]
                for domain, weights in SOURCE_WEIGHTS.items()}

    def test_6400_row_quotas_and_exact_96_row_domain_balance(self):
        rows, report = mix_domains(self.inputs())
        self.assertEqual(report["domain_counts"], {"perception": 2134, "reasoning": 2133, "generalist": 2133})
        for start in range(0, len(rows) - 95, 96):
            self.assertEqual(Counter(row["domain"] for row in rows[start:start + 96]), dict.fromkeys(DOMAIN_ORDER, 32))
        self.assertEqual(report["domains"]["perception"]["source_counts"], {"tallyqa": 854, "clevr": 853, "refcoco": 427})
        self.assertEqual(report["domains"]["generalist"]["source_counts"], {"vqav2": 1422, "docvqa": 178, "chartqa": 178, "if_format": 355})

    def test_no_mutation_and_deterministic_sampling(self):
        inputs = self.inputs()
        before = deepcopy(inputs)
        first, _ = mix_domains(inputs, count=90)
        second, _ = mix_domains(inputs, count=90)
        self.assertEqual(first, second)
        self.assertEqual(inputs, before)

    def test_generalist_replay_excluded(self):
        rows = self.inputs()["generalist"] + [example("tallyqa"), example("geometry3k")]
        selected, report = select_domain(rows, "generalist", 12, 17)
        self.assertNotIn("tallyqa", {row["extra_info"]["source"] for row in selected})
        self.assertEqual(report["excluded_teacher_replay_rows"], {"tallyqa": 1, "geometry3k": 1})

    def test_repeated_source_exposures_are_deduplicated_before_sampling(self):
        inputs = self.inputs()["perception"]
        inputs += [deepcopy(inputs[0])] * 50
        selected, report = select_domain(inputs, "perception", 10, 17)
        self.assertEqual(report["sources"]["tallyqa"]["eligible_unique_records"], 12)
        self.assertEqual(len({row["extra_info"]["record_id"] for row in selected}), 10)

    def test_nontrain_and_conflicting_duplicate_fail(self):
        inputs = self.inputs()["perception"]
        inputs[0]["extra_info"]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "non-training"):
            select_domain(inputs, "perception", 10, 17)
        inputs = self.inputs()["perception"]
        bad = deepcopy(inputs[0])
        bad["reward_model"] = {"ground_truth": "different"}
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            select_domain(inputs + [bad], "perception", 10, 17)

    def test_heldout_image_overlap_rejected_independently_of_question(self):
        rows, _ = mix_domains(self.inputs(), count=30)
        overlap = rows[0]["extra_info"]["image_sha256"][0]
        with self.assertRaisesRegex(ValueError, "Evaluation overlap"):
            audit_rows(rows, ({overlap}, set(), set()), check_image_paths=False)
        report = audit_rows(rows, (set(), set(), set()), check_image_paths=False)
        self.assertEqual(report["evaluation_identity_overlap"], 0)
        self.assertEqual(report["repeated_record_exposures"], 0)


if __name__ == "__main__":
    unittest.main()
