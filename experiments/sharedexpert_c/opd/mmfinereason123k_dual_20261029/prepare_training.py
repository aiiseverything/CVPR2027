"""Freeze identical 6300 prompt exposures from the completed dual-correct set."""
from collections import Counter
import json
import random

from opd_config import HERE, CURATION, DATASET, SETTINGS, atomic_json, digest, now, sha


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    complete = json.loads((CURATION / "export_complete.json").read_text())
    manifest = json.loads((CURATION / "export/manifest.json").read_text())
    assert complete["status"] == manifest["status"] == "complete"
    assert complete["manifest_sha256"] == sha(CURATION / "export/manifest.json")
    source = CURATION / "export/dataset.jsonl"
    assert sha(source) == manifest["jsonl_sha256"]
    saved = HERE / "training_data.json"
    if saved.exists():
        old = json.loads(saved.read_text())
        assert old["settings_sha256"] == digest(SETTINGS)
        assert old["accepted_source_sha256"] == manifest["jsonl_sha256"]
        assert old["parquet_sha256"] == sha(DATASET)
        return old
    pool = []
    with source.open() as f:
        for line in f:
            row = json.loads(line)
            for teacher in SETTINGS["teachers"]:
                rec = row["teachers"][teacher]
                assert rec["assessment"]["verified_correct"] and rec["finish_reason"] == "stop"
                assert 0 < rec["usage"]["completion_tokens"] <= 8192
            row["images"] = [str((CURATION / "export" / p).resolve()) for p in row["images"]]
            del row["teachers"]
            pool.append(row)
    assert len(pool) == complete["accepted"] == manifest["both_verified_correct"]
    if not pool:
        raise ValueError("No jointly verified examples; cannot start authorized OPD")
    # Sorting before seeded shuffling makes order independent of GPU completion timing.
    pool.sort(key=lambda r:r["id"])
    rng = random.Random(SETTINGS["seed"])
    chosen, cycles = [], 0
    while len(chosen) < 6300:
        order = list(range(len(pool)))
        rng.shuffle(order)
        chosen.extend((pool[i], cycles) for i in order[:6300-len(chosen)])
        cycles += 1
    records, exposure_manifest, checked = [], [], set()
    for index, (row, cycle) in enumerate(chosen):
        for path, expected in zip(row["images"], row["image_file_sha256"]):
            if path not in checked:
                assert sha(path) == expected
                checked.add(path)
        images = [{"path":p,"min_pixels":4096,"max_pixels":786432} for p in row["images"]]
        records.append({
            "data_source": "mmfinereason_dual_correct", "prompt": [{"role":"user","content":"<image>"+row["prompt_text"]}],
            "images": images, "teacher_images": row["images"], "ability":"opd",
            "reward_model":{"style":"rule","ground_truth":json.dumps({"task":"exact","answers":[row["answer"]],"format":"boxed"},ensure_ascii=False)},
            "extra_info":{"record_id":row["id"],"index":index,"source":row["source"],"source_id":row["source_id"],
                          "cycle":cycle,"prompt_tokens":row["prompt_tokens"],"question_sha256":row["question_sha256"],
                          "image_sha256":json.dumps(row["image_sha256"]),"accepted_row_sha256":row["row_sha256"]}})
        exposure_manifest.append({"index":index,"step":index//63+1,"slot":index%63,"id":row["id"],"cycle":cycle,"source":row["source"]})
    pq.write_table(pa.Table.from_pylist(records),DATASET,compression="zstd")
    (HERE / "exposures.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in exposure_manifest))
    result = {"status":"complete","at":now(),"accepted_pool":len(pool),"exposures":6300,"global_batch_size":63,"steps":100,
        "unique_examples":len({r["id"] for r,_ in chosen}),"permutation_cycles":cycles,
        "source_exposures":dict(Counter(r["source"] for r,_ in chosen)),
        "accepted_source_sha256":manifest["jsonl_sha256"],"settings_sha256":digest(SETTINGS),
        "parquet_sha256":sha(DATASET),"exposure_manifest_sha256":sha(HERE / "exposures.jsonl"),
        "same_file_for_both_runs":True,"answer_or_teacher_rollout_in_prompt":False}
    atomic_json(saved,result)
    return result


if __name__ == "__main__":
    print(json.dumps(main(),ensure_ascii=False,indent=2))
