"""CPU-only checks against actual legacy and Transformers-5 exported metadata."""
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from run_checkpoint import ROOT, HERE, validate_hf_export_metadata


class ActualExportMetadata(unittest.TestCase):
    def test_transformers5_smoke_export_loads_on_cpu(self):
        model = HERE.parent / "runs/perception_smoke/hf/global_step_1"
        self.assertFalse((model / "preprocessor_config.json").exists())
        result = validate_hf_export_metadata(model)
        self.assertEqual(result["processor_metadata"], ["processor_config.json"])
        self.assertEqual(result["processor_class"], "Qwen3VLProcessor")
        self.assertEqual(result["model_type"], "qwen3_5")
        self.assertIn("Qwen3_5ForConditionalGeneration", result["architectures"])

    def test_legacy_original_metadata_loads_on_cpu(self):
        model = ROOT / "pre-lab/models/Qwen3.5-4B"
        self.assertFalse((model / "processor_config.json").exists())
        result = validate_hf_export_metadata(model)
        self.assertEqual(result["processor_metadata"], ["preprocessor_config.json"])
        self.assertEqual(result["processor_class"], "Qwen3VLProcessor")
        self.assertEqual(result["model_type"], "qwen3_5")

    def test_missing_processor_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="sharedexpert-export-metadata-") as temp:
            model = Path(temp)
            for name in ("config.json", "tokenizer_config.json"):
                (model / name).write_text(json.dumps({}))
            with self.assertRaisesRegex(ValueError, "missing processor_config.json or preprocessor_config.json"):
                validate_hf_export_metadata(model)


if __name__ == "__main__":
    unittest.main()
