"""CPU-only scoring boundary and resume/completeness tests."""
import copy
from collections import Counter
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import protocol
import scoring
from annotation_issues import KNOWN_ANNOTATION_ISSUES


def row(bench="vstar", index=0, answer="A", options=None, **metadata):
    if options is None:
        options = {"A": "first", "B": "second"}
    return {"id": f"{bench}:{index}", "benchmark": bench, "question": "Question?",
            "answer": answer, "options": options, "metadata": metadata,
            "image": {"sha256": "0" * 64, "width": 100, "height": 100},
            "prompt_tokens": 100}


def prediction(r, text="<answer>A</answer>", **kw):
    return {"id": r["id"], "row_sha256": protocol.digest(r), "model": "baseline",
            "raw_response": text, "reasoning": None, "finish_reason": "stop",
            "usage": {"completion_tokens": 20}, "error": None, **kw}


def annotation_row(record_index=0):
    record = KNOWN_ANNOTATION_ISSUES[record_index]
    r = row(record["benchmark"], answer=record["answer"], options=copy.deepcopy(record["options"]))
    r.update({key: record[key] for key in ("id", "question")})
    r["metadata"] = {key: copy.deepcopy(record[key]) for key in ("original_id", "option_lines", "source")}
    r["metadata"]["answer_label_missing_from_options"] = record["missing_gold"]
    if record["benchmark"] == "zoom":
        r["metadata"]["question_type"] = "mcq"
    else:
        r["metadata"].update(category="Perception/Monitoring", **{"l2-category": "vehicle/location"})
    return r


class ExtractionTests(unittest.TestCase):
    def test_thinking_boundary_last_answer_and_input_immutability(self):
        r = row()
        p = prediction(r, '<think><answer>B</answer></think> <answer>B</answer> <answer>A</answer>')
        before = copy.deepcopy((r, p))
        result = scoring.score_prediction(r, p)
        self.assertTrue(result["correct"])
        self.assertEqual(result["prediction"], "A")
        self.assertEqual((r, p), before)

    def test_unfinished_thought_and_final_markers_are_not_answers(self):
        for text in ["<think>Maybe A. <answer>A</answer>",
                     "<answer>A</answer><answer>B", r"\boxed{A}\boxed{B",
                     "<think>Answer A</think>", ""]:
            with self.subTest(text=text):
                result = scoring.score_prediction(row(), prediction(row(), text))
                self.assertFalse(result["correct"])
                self.assertFalse(result["parse_success"])

    def test_never_extract_a_letter_from_long_prose_or_last_line(self):
        for text in ["I first considered A but B is also possible.",
                     "Let me think about all these options.\nA", "A or B", "Apple", "A. first"]:
            with self.subTest(text=text):
                result = scoring.score_prediction(row(), prediction(row(), text))
                self.assertFalse(result["parse_success"])

    def test_short_unambiguous_final_formats(self):
        for text in ["A", "(a)", "[A].", "Answer: A", "最终答案：Ａ", r"\boxed{A}",
                     "<think>Something B.</think>A"]:
            with self.subTest(text=text):
                self.assertTrue(scoring.score_prediction(row(), prediction(row(), text))["correct"])
        self.assertTrue(scoring.score_prediction(row(), prediction(row(), "A", reasoning="Maybe B"))["correct"])

    def test_length_api_and_other_terminal_reasons_zero(self):
        for kwargs, why in [({"finish_reason": "length"}, "length_termination"),
                            ({"error": "timeout"}, "api_error"),
                            ({"finish_reason": "content_filter"}, "non_stop_finish_reason")]:
            result = scoring.score_prediction(row(), prediction(row(), **kwargs))
            self.assertFalse(result["correct"])
            self.assertEqual(result["reason"], why)
        result = scoring.score_prediction(row(), prediction(row(), finish_reason="length"))
        self.assertTrue(result["correct_before_length_penalty"])

    def test_exact_numeric_matches_and_rejects_substrings(self):
        r = row("zoom", answer="2", options={}, question_type="blank")
        for text in ["2", "2.0", "Answer: 2.", "2e0", "4/2", r"\boxed{\frac{4}{2}}", "<answer>２</answer>"]:
            with self.subTest(text=text):
                self.assertTrue(scoring.score_prediction(r, prediction(r, text))["correct"])
        for text in ["12", "2.00001", "There are 2 objects", "12 or 2", "2,0", "nan", "inf", "1e999999999", "1/0"]:
            with self.subTest(text=text):
                self.assertFalse(scoring.score_prediction(r, prediction(r, text))["correct"])

    def test_source_duplicate_option_letters_preserved(self):
        r = row("mme_rw_en", answer="D", options={x: x for x in "ABCDE"}, category="Perception", **{
            "l2-category": "OCR", "option_lines": ["(A) a", "(B) b", "(C) c", "(D) d1", "(D) d2", "(E) e"],
            "option_labels_duplicated": ["D"]})
        self.assertTrue(scoring.score_prediction(r, prediction(r, "D"))["correct"])
        self.assertFalse(scoring.score_prediction(r, prediction(r, "F"))["parse_success"])

    def test_bad_source_schema_fails_closed(self):
        for r in [row("zoom", answer="A", options={}, question_type="mcq"),
                  row("zoom", answer="two", options={}, question_type="blank"),
                  row("hrbench_4k"), row("mme_rw_en")]:
            with self.subTest(row=r):
                with self.assertRaises(ValueError):
                    scoring.score_prediction(r, prediction(r))

    def test_only_exact_known_zoom_annotation_exception_is_allowed(self):
        r = annotation_row()
        self.assertTrue(scoring.score_prediction(r, prediction(r, "D"))["correct"])
        self.assertFalse(scoring.score_prediction(r, prediction(r, "A"))["correct"])
        for changed in [
            {**r, "metadata": {**r["metadata"], "original_id": "another"}},
            {**r, "options": {"A": "a", "C": "c"}},
            {**r, "answer": "E"},
            {**r, "benchmark": "vstar"},
            {**r, "question": "different"},
            {**r, "metadata": {**r["metadata"], "answer_label_missing_from_options": False}},
        ]:
            with self.subTest(row=changed), self.assertRaises(ValueError):
                scoring.score_prediction(changed, prediction(changed, "D"))

    def test_all_source_exceptions_keep_original_gold_and_require_exact_identity(self):
        for index, record in enumerate(KNOWN_ANNOTATION_ISSUES):
            r = annotation_row(index)
            before = copy.deepcopy(r)
            self.assertTrue(scoring.score_prediction(r, prediction(r, record["answer"]))["correct"])
            self.assertEqual(r, before)
            self.assertEqual(scoring.source_annotation_issue(r), record["issue_type"])
            if record["missing_gold"]:
                r["metadata"]["source"]["row_index"] += 1
                with self.assertRaises(ValueError):
                    scoring.score_prediction(r, prediction(r, record["answer"]))
            if record["benchmark"] == "mme_rw_cn":
                r = annotation_row(index)
                self.assertNotIn("A", r["options"])
                self.assertFalse(scoring.score_prediction(r, prediction(r, "一"))["correct"])


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root, self.results = Path(self.tmp.name) / "data", Path(self.tmp.name) / "results"
        self.root.mkdir()
        self.rows = [row(category="direct_attributes"),
            row("zoom", answer="2", options={}, question_type="blank"),
            row("zoom", 1, question_type="mcq"),
            row("hrbench_4k", 0, category="FSP", cycle_category="cycle0"),
            row("hrbench_4k", 1, category="FSP", cycle_category="cycle1"),
            row("hrbench_4k", 2, category="FCP", cycle_category="cycle1"),
            row("hrbench_8k", category="FSP", cycle_category="cycle0"),
            row("mme_rw_en", category="Perception", **{"l2-category": "OCR"}),
            row("mme_rw_cn", category="感知", **{"l2-category": "OCR"})]
        self.models = patch.dict(scoring.MODEL_PATHS, {"baseline": Path("unused"), "reasoning-step100": Path("unused2")}, clear=True)
        self.expected = patch.dict(protocol.EXPECTED, dict(Counter(r["benchmark"] for r in self.rows)), clear=True)
        self.models.start(); self.addCleanup(self.models.stop)
        self.expected.start(); self.addCleanup(self.expected.stop)
        suite = self.root / "suite.jsonl"
        suite.write_text("".join(json.dumps(r) + "\n" for r in self.rows))
        counts = dict(Counter(r['benchmark'] for r in self.rows))
        selection = {'schema_version': 'highres-selection-v1', 'ready': True,
                     'benchmarks': {b: {'expected_count': n, 'filters': []} for b, n in counts.items()}}
        (self.root / 'selection.json').write_text(json.dumps(selection))
        self.meta = {"suite_sha256": protocol.file_sha(suite), "protocol": protocol.PROTOCOL,
                     "protocol_sha256": protocol.digest(protocol.PROTOCOL), "source_manifest_sha256": "test-source",
                     "selection_sha256": protocol.file_sha(self.root / 'selection.json'), 'counts': counts}
        (self.root / "suite.meta.json").write_text(json.dumps(self.meta))
        self.directory = self.results / "full" / "baseline"
        self.directory.mkdir(parents=True)
        self.run = {**self.meta, "name": "baseline", "n_requested": len(self.rows), "smoke": False}
        (self.directory / "run.json").write_text(json.dumps(self.run))
        self.path = self.directory / "predictions.jsonl"
        self.pred = [prediction(r, "2" if r["benchmark"] == "zoom" and not r["options"] else "B" if r["id"] in {"hrbench_4k:1", "hrbench_4k:2"} else "A") for r in self.rows]
        self.write(self.pred)

    def write(self, predictions):
        self.path.write_text("".join(json.dumps(p) + "\n" for p in predictions))

    def report(self):
        return scoring.summarize(self.root, self.results)

    def test_full_summary_hr_cycle_macro_and_native_categories(self):
        report = self.report()
        self.assertEqual(report["models"]["baseline"]["status"], "complete")
        self.assertEqual(report["models"]["reasoning-step100"]["status"], "not_started")
        d = report["models"]["baseline"]["datasets"]
        self.assertEqual(d["hrbench_4k"]["accuracy"], .5)
        self.assertAlmostEqual(d["hrbench_4k"]["micro_accuracy"], 1/3)
        self.assertEqual(d["hrbench_4k"]["category_cycle_macro"]["FSP"]["accuracy"], .5)
        self.assertEqual(d["hrbench_4k"]["category_cycle_macro"]["FCP"]["accuracy"], 0)
        self.assertEqual(d["mme_rw_cn"]["category_l2"]["感知"]["OCR"]["accuracy"], 1)
        self.assertEqual(d["zoom"]["groups"]["question_type"]["blank"]["accuracy"], 1)
        self.assertEqual(protocol.file_sha(self.results / "summary.csv"), report["summary_csv_sha256"])
        with (self.results / "summary.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), 12)

    def test_resume_duplicate_last_wins_even_if_latest_is_error(self):
        bad = {**self.pred[0], "error": "first timeout"}
        self.write([bad] + self.pred)
        report = self.report()["models"]["baseline"]
        self.assertEqual(report["input_snapshot"]["duplicates"], 1)
        self.assertEqual(report["datasets"]["vstar"]["accuracy"], 1)
        self.write(self.pred + [{**self.pred[0], "raw_response": "B"}])
        self.assertEqual(self.report()["models"]["baseline"]["datasets"]["vstar"]["accuracy"], 0)
        self.write(self.pred + [bad])
        result = self.report()["models"]["baseline"]["datasets"]["vstar"]
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["api_errors"], 1)

    def test_known_invalid_zoom_primary_and_excluded_sensitivity_are_separate(self):
        r = annotation_row()
        self.rows[2] = r
        self.pred[2] = prediction(r, "A")
        suite = self.root / "suite.jsonl"
        suite.write_text("".join(json.dumps(x) + "\n" for x in self.rows))
        self.meta["suite_sha256"] = protocol.file_sha(suite)
        (self.root / "suite.meta.json").write_text(json.dumps(self.meta))
        self.run["suite_sha256"] = self.meta["suite_sha256"]
        (self.directory / "run.json").write_text(json.dumps(self.run))
        self.write(self.pred)
        d = self.report()["models"]["baseline"]["datasets"]["zoom"]
        self.assertEqual(d["n_expected"], 2)
        self.assertEqual(d["accuracy"], .5)
        self.assertEqual(len(d["source_anomalies"]), 1)
        sensitivity = d["exclude_known_invalid_annotations"]
        self.assertEqual(sensitivity["n_expected"], 1)
        self.assertEqual(sensitivity["accuracy"], 1)
        self.assertEqual(sensitivity["excluded_ids"], [r["id"]])

    def test_partial_never_reports_full_accuracy(self):
        self.write(self.pred[1:])
        result = self.report()["models"]["baseline"]
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["datasets"]["vstar"]["accuracy"])
        self.assertEqual(result["datasets"]["zoom"]["accuracy"], 1)

    def test_incomplete_tail_is_read_only_and_prevents_complete_claim(self):
        with self.path.open("ab") as f:
            f.write(b'{"id": "vstar:0",')
        before = self.path.read_bytes()
        result = self.report()["models"]["baseline"]
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(result["input_snapshot"]["incomplete_tail"])
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["datasets"]["zoom"]["accuracy"])

    def test_corrupt_middle_or_stale_record_or_run_is_rejected(self):
        original = self.path.read_text()
        self.path.write_text('{broken}\n' + original)
        with self.assertRaises(ValueError): self.report()
        for key, value in [("row_sha256", "stale"), ("model", "wrong"), ("id", "unknown")]:
            self.write([{**self.pred[0], key: value}] + self.pred[1:])
            with self.subTest(key=key), self.assertRaises(ValueError): self.report()
        self.write(self.pred)
        (self.directory / "run.json").write_text(json.dumps({**self.run, "suite_sha256": "stale"}))
        with self.assertRaises(ValueError): self.report()

    def test_missing_run_keeps_observed_score_but_no_full_accuracy(self):
        (self.directory / "run.json").unlink()
        d = self.report()["models"]["baseline"]["datasets"]["zoom"]
        self.assertIsNone(d["accuracy"])
        self.assertEqual(d["observed_accuracy"], 1)


if __name__ == "__main__":
    unittest.main()
