"""CPU checks for full-dataset gating, deterministic exposure order and input schema."""
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile

import prepare_training as prep
from opd_config import HERE, CURATION, SETTINGS, atomic_json, sha


@contextmanager
def fixture(pool_size, correct=True, finish="stop"):
    with tempfile.TemporaryDirectory(prefix="mfr-opd-contract-") as directory:
        root = Path(directory)
        curation = root / "curation"
        export = curation / "export"
        export.mkdir(parents=True)
        with (CURATION / "candidates.jsonl").open() as f:
            sample = json.loads(next(f))
        image = root / "image.img"
        image.write_bytes(Path(sample["images"][0]).read_bytes())
        records = []
        for i in range(pool_size):
            r = deepcopy(sample)
            r["id"] = f"contract:{i:06d}"
            r["images"] = [str(image)]
            r["teachers"] = {t:{"assessment":{"verified_correct":correct},
                "finish_reason":finish,"usage":{"completion_tokens":17}} for t in SETTINGS["teachers"]}
            records.append(r)
        source = export / "dataset.jsonl"
        source.write_text("".join(json.dumps(r)+"\n" for r in records))
        manifest = {"status":"complete","jsonl_sha256":sha(source),"both_verified_correct":pool_size}
        atomic_json(export / "manifest.json",manifest)
        atomic_json(curation / "export_complete.json",{"status":"complete","accepted":pool_size,
                    "manifest_sha256":sha(export / "manifest.json")})
        old = prep.HERE,prep.CURATION,prep.DATASET
        prep.HERE,prep.CURATION,prep.DATASET = root,curation,root / "training.parquet"
        try:
            yield root
        finally:
            prep.HERE,prep.CURATION,prep.DATASET = old


def main():
    import pyarrow.parquet as pq
    from opd_config import EXPERIMENT, compose_config, prepare_model_views
    prepare_model_views()
    spec = importlib.util.spec_from_file_location("dataset_preflight",EXPERIMENT / "training/preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checks = []
    for pool_size in [2,6303]:
        with fixture(pool_size) as folder:
            result = prep.main()
            assert result["exposures"] == 6300 and result["unique_examples"] == min(pool_size,6300)
            exposure = [json.loads(x) for x in (folder / "exposures.jsonl").read_text().splitlines()]
            assert [x["index"] for x in exposure] == list(range(6300))
            assert all(x["step"] == i//63+1 and x["slot"] == i%63 for i,x in enumerate(exposure))
            if pool_size == 2:
                assert all(exposure[i]["id"] != exposure[i+1]["id"] for i in range(0,6300,2))
            else:
                assert all(x["cycle"] == 0 for x in exposure)
            report = module.check_dataset(prep.DATASET,6300)
            assert not report["errors"],report
            rows = pq.read_table(prep.DATASET).to_pylist()
            assert all(r["teacher_images"] == [x["path"] for x in r["images"]] for r in rows)
            assert all(r["extra_info"]["index"] == i for i,r in enumerate(rows))
            if pool_size == 2:
                # Exercise the actual dataset loader consumed by the training dataloader.
                from matched_inputs import MatchedImageDataset
                from verl.utils import hf_processor
                from opd_config import VIEWS
                processor = hf_processor(str(VIEWS["student"]),use_fast=True)
                dataset = MatchedImageDataset([str(prep.DATASET)],processor.tokenizer,
                                             compose_config("perception_v1").data,processor=processor)
                assert len(dataset) == 6300
                item = dataset[0]
                assert "teacher_images" in item and "raw_prompt" in item
                assert item["raw_prompt"][0]["content"][0]["type"] == "image"
            assert prep.main() == result
            (folder / "training_data.json").unlink()
            again = prep.main()
            assert again["exposure_manifest_sha256"] == result["exposure_manifest_sha256"]
            assert again["parquet_sha256"] == result["parquet_sha256"]
            checks.append({"pool":pool_size,"exposures":6300,"deterministic":True,"schema_valid":True})
    for correct,finish in [(False,"stop"),(True,"length")]:
        with fixture(1,correct=correct,finish=finish):
            try:
                prep.main()
            except AssertionError:
                pass
            else:
                raise AssertionError("Unverified or capped teacher answer entered training")
    with fixture(1) as folder:
        (prep.CURATION / "export_complete.json").unlink()
        try:
            prep.main()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("Incomplete curation started training")
    result = {"status":"passed","gpu_smoke_performed":False,"fixtures":checks,
              "incomplete_curation_rejected":True,"wrong_and_length_capped_answers_rejected":True}
    atomic_json(HERE / "data_contract_validation.json",result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
