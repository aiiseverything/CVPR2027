"""CPU-only identity boundaries for pinned annotation exceptions."""
import copy
import io
import unittest

from PIL import Image

import annotation_issues as issues
import prepare


def fixture(record):
    row = {key: copy.deepcopy(record[key]) for key in ("id", "benchmark", "question", "answer", "options")}
    row["metadata"] = {key: copy.deepcopy(record[key]) for key in ("original_id", "option_lines", "source")}
    if row["benchmark"] == "zoom":
        row["metadata"]["question_type"] = "mcq"
    return row


class AnnotationIssueTests(unittest.TestCase):
    def test_only_eleven_exact_records_ten_missing_gold(self):
        self.assertEqual(len(issues.KNOWN_ANNOTATION_ISSUES), 11)
        self.assertEqual(sum(issues.known_missing_gold(fixture(r)) for r in issues.KNOWN_ANNOTATION_ISSUES), 10)
        for record in issues.KNOWN_ANNOTATION_ISSUES:
            with self.subTest(id=record["id"]):
                row = fixture(record)
                before = copy.deepcopy(row)
                self.assertEqual(issues.known_annotation_issue(row), record["issue_type"])
                self.assertEqual(row, before)

    def test_every_text_and_source_identity_field_is_required(self):
        for record in issues.KNOWN_ANNOTATION_ISSUES:
            for key in ("id", "benchmark", "question", "answer", "options"):
                row = fixture(record)
                row[key] = {} if key == "options" else str(row[key]) + " altered"
                with self.subTest(id=record["id"], key=key):
                    self.assertIsNone(issues.known_annotation_issue(row))
                    self.assertFalse(issues.known_missing_gold(row))
            for key in ("original_id", "option_lines", "source"):
                row = fixture(record)
                del row["metadata"][key]
                with self.subTest(id=record["id"], missing=key):
                    self.assertIsNone(issues.known_annotation_issue(row))
            for key in record["source"]:
                row = fixture(record)
                row["metadata"]["source"][key] = "different"
                with self.subTest(id=record["id"], source=key):
                    self.assertIsNone(issues.known_annotation_issue(row))

    def test_advisory_flags_cannot_authorize_unknown_rows(self):
        row = fixture(issues.KNOWN_ANNOTATION_ISSUES[0])
        row["question"] += "?"
        row["metadata"].update(answer_label_missing_from_options=True,
                               annotation_issue_type="missing_gold_label")
        self.assertIsNone(issues.known_annotation_issue(row))
        self.assertFalse(issues.known_missing_gold(row))

    def test_non_latin_label_requires_exact_eight_source_identities(self):
        cn = [r for r in issues.KNOWN_ANNOTATION_ISSUES if r["benchmark"] == "mme_rw_cn"]
        self.assertEqual(len(cn), 8)
        for record in cn:
            row = fixture(record)
            parsed, duplicates = prepare.options_from_lines(record["option_lines"], source_row=row)
            self.assertEqual(parsed, record["options"])
            self.assertIn("一", parsed)
            self.assertNotIn("A", parsed)
            self.assertEqual(duplicates, [])
            with self.assertRaises(ValueError):
                prepare.options_from_lines(record["option_lines"])
            altered = record["option_lines"].copy()
            altered[0] += " changed"
            with self.assertRaises(ValueError):
                prepare.options_from_lines(altered, source_row=row)
            row["metadata"]["source"]["row_index"] += 1
            with self.assertRaises(ValueError):
                prepare.options_from_lines(record["option_lines"], source_row=row)

    def test_preparation_keeps_original_text_gold_lines_and_labels(self):
        encoded = io.BytesIO()
        Image.new("RGB", (2, 2)).save(encoded, format="PNG")
        payload = encoded.getvalue()
        for record in issues.KNOWN_ANNOTATION_ISSUES:
            with self.subTest(id=record["id"]):
                entry = {"benchmark": record["benchmark"], **record["source"]}
                if record["benchmark"] == "zoom":
                    source_row = {"id": record["original_id"], "query": record["question"],
                                  "response": record["answer"], "question_type": "mcq", "image": payload}
                else:
                    source_row = {"index": record["original_id"], "question": record["question"],
                                  "answer": record["answer"], "multi-choice options": record["option_lines"],
                                  "category": "Perception/Monitoring", "l2-category": "vehicle/location",
                                  "bytes": payload}
                before = copy.deepcopy(source_row)
                row, image = prepare.normalize_row(source_row, entry, int(record["id"].split(":")[1]),
                                                   record["source"]["row_index"])
                self.assertEqual(source_row, before)
                self.assertEqual(image, payload)
                for key in ("question", "answer", "options"):
                    self.assertEqual(row[key], record[key])
                self.assertEqual(row["metadata"]["option_lines"], record["option_lines"])
                self.assertEqual(row["metadata"]["annotation_issue_type"], record["issue_type"])
                self.assertEqual(row["metadata"].get("answer_label_missing_from_options", False), record["missing_gold"])

    def test_preparation_rejects_neighbor_and_mutated_missing_gold(self):
        for record in issues.KNOWN_ANNOTATION_ISSUES:
            if not record["missing_gold"]:
                continue
            entry = {"benchmark": record["benchmark"], **record["source"]}
            if record["benchmark"] == "zoom":
                source_row = {"id": record["original_id"], "query": record["question"],
                              "response": record["answer"], "question_type": "mcq"}
            else:
                source_row = {"index": record["original_id"], "question": record["question"],
                              "answer": record["answer"], "multi-choice options": record["option_lines"],
                              "category": "Perception/Monitoring", "l2-category": "vehicle/location"}
            with self.subTest(id=record["id"]), self.assertRaises(ValueError):
                prepare.normalize_row(source_row, entry, int(record["id"].split(":")[1]) + 1,
                                      record["source"]["row_index"])


if __name__ == "__main__":
    unittest.main()
