# SharedExpert C: three-teacher RL data pipeline

The current user-approved protocol uses **3,000 prompt exposures per teacher**, 30 prompts per optimizer step, 100 steps, and eight completions per prompt: 24,000 responses per teacher. Each prepared parquet retains its original 12,000 rows and checksum; training consumes the first 3,000 rows without reshuffling. All three teachers share the same exposure budget. The prefix audit in `../logs/budget100_validation.json` finds 3,000 unique record IDs per teacher; image uniqueness is not inferred.

## Retained 12,000-row prepared allocation

| Teacher | Source | Exposures |
|---|---|---:|
| Perception | TallyQA / CLEVR / RefCOCO | 4,800 / 4,800 / 2,400 |
| Reasoning | Geometry3K / MMK12 / ViRL39K | 4,800 / 2,400 / 4,800 |
| Generalist | VQAv2 | 4,800 |
| Generalist | DocVQA / ChartQA | 600 / 600 |
| Generalist | Geometry3K / MMK12 / ViRL39K | 960 / 480 / 960 |
| Generalist | TallyQA / CLEVR / RefCOCO | 960 / 960 / 480 |
| Generalist | Content-correct JSON instruction following | 1,200 |

This table describes the full retained prepared files, not the current 100-step exposure budget. Each 200-prompt block has the exact specified proportions. The trainer must set `shuffle=False`; source order inside each block is deterministically shuffled. At batch 30, each 600-row / 20-step interval covers three complete blocks with exact ratios. The run consumes the first 15 of 60 blocks, so full-run ratios remain exact; individual 30-row batches need not match every ratio. A shuffled deck is exhausted before a source record is reused.

## Current 100-step prefix allocation

| Teacher | Source | Exposures (all distinct record IDs within each teacher) |
|---|---|---:|
| Perception | TallyQA / CLEVR / RefCOCO | 1,200 / 1,200 / 600 |
| Reasoning | Geometry3K / MMK12 / ViRL39K | 1,200 / 600 / 1,200 |
| Generalist | VQAv2 | 1,200 |
| Generalist | DocVQA / ChartQA | 150 / 150 |
| Generalist | Geometry3K / MMK12 / ViRL39K | 240 / 120 / 240 |
| Generalist | TallyQA / CLEVR / RefCOCO | 240 / 240 / 120 |
| Generalist | IF | 300 |

The following 9,000-row audit remains historical evidence for the superseded 300-step budget.

## Historical 300-step batch-30 prefix allocation

| Teacher | Source | Exposures | Unique record IDs |
|---|---|---:|---:|
| Perception | TallyQA / CLEVR / RefCOCO | 3,600 / 3,600 / 1,800 | 3,600 / 3,600 / 1,800 |
| Reasoning | Geometry3K / MMK12 / ViRL39K | 3,600 / 1,800 / 3,600 | 2,100 / 1,800 / 3,600 |
| Generalist | VQAv2 | 3,600 | 3,600 |
| Generalist | DocVQA / ChartQA | 450 / 450 | 450 / 450 |
| Generalist | Geometry3K / MMK12 / ViRL39K | 720 / 360 / 720 | 720 / 360 / 720 |
| Generalist | TallyQA / CLEVR / RefCOCO | 720 / 720 / 360 | 720 / 720 / 360 |
| Generalist | Content-correct JSON instruction following | 900 | 900 |

The actual first-9,000-row audit is in `../logs/visual_batch30_budget_audit.json`,
under `teachers.*.prefix_identity_counts`. Perception/reasoning/generalist contain
9,000/7,500/9,000 distinct record IDs. Reasoning Geometry3K includes 1,500 examples
twice and 600 once, giving 1,500 additional exposures. RefCOCO IDs retain both ref
and sentence IDs; different questions on the same image remain distinct. These are
identifier counts, not semantic deduplication or a new leakage audit. No prefix
unique-image count is inferred from the full-file audit.

The general VQA/caption category is instantiated as VQAv2 only in this pilot, because its ten human annotations support a verifiable reward. No caption proxy reward is substituted. IF data uses a deterministic, disjoint image partition of VQAv2 train; it requires a single JSON `answer` key and correct VQA content.

## Sources and split verification

| Source | Original / distribution | Train-only preparation |
|---|---|---|
| TallyQA | [official annotations](https://github.com/manoja328/TallyQA_dataset), commit `46cdc649ec79c3dcc2720ff227ad07d7ee51da6f`; image distribution [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | All nine original TallyQA train image shards. Each image's question/answer set is matched to a unique image in the official `train.json`, then an untouched original question and answer are used. Ambiguous matches are excluded and counted. |
| CLEVR | [official release](https://cs.stanford.edu/people/jcjohns/clevr/), official v1.0 no-image archive; same Cauldron image distribution | Three of 24 train image shards selected with seed 20261019. All selected question/image identities are verified against official `CLEVR_train_questions.json`. CoGenT, CLEVR-Math, and val/test are not used. |
| RefCOCO | [official REFER definitions](https://github.com/lichengunc/refer), [annotation mirror](https://huggingface.co/datasets/jxu124/refcoco), revision `9fba8200c5326e996f789191f095bd464ef1d09e` | Original UNC `split == train`; 1,500 seeded training images. Original COCO image URLs. Mirror xyxy boxes are checked against original `raw_anns` xywh values before 0–1000 normalization. |
| Geometry3K | [InterGPS](https://github.com/lupantech/InterGPS), [documented conversion](https://huggingface.co/datasets/hiyouga/geometry3k) | Only original 2,101-row train parquet. The mirror documents its official train/val/test conversion. Local original download commit and upstream LFS digest are checked in `raw/reused_source_verification.json`. |
| MMK12 | [official FanqingM/MMK12](https://huggingface.co/datasets/FanqingM/MMK12) | Seeded 8,000-record reservoir from the three official train shards; original test is never used. Local downloaded files are checked against their pinned upstream SHA256. |
| ViRL39K | [official TIGER-Lab/ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K), revision `812ec617dea4bc8a4e751663b88e4ebb7de4d00e` | Seeded 10,000-record reservoir, then exclusion of identified original heldout derivatives. Its image zip must match upstream SHA256 `49e6485a02815430fe4a468a25c03c3514adf8c2b68135068fc36c7656d8ad27`. Image/question/original-UUID overlaps with the actual MMK12 and Geometry3K pools are removed. The release's actual image inputs are preserved. |
| VQAv2 | [lmms-lab/VQAv2](https://huggingface.co/datasets/lmms-lab/VQAv2), revision `32665d35052eb4a6d4414851c3c829a72754915a` | First four pinned original train shards, 13,052 questions; all ten answer annotations, including repeats, retained. This is a fixed source subset, not a claim to use the full VQAv2 train distribution. |
| DocVQA | [lmms-lab/DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA), revision `539088ef8a8ada01ac8e2e6d4e372586748a265e` | Seeded 2,500-record reservoir from the first two `DocVQA/train` shards; assert `data_split == train`. |
| ChartQA | [HuggingFaceM4/ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA), revision `b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5` | Seeded 2,500-record reservoir from all three train shards. |

The Cauldron revision is `847a98a779b1652d65111daf20c972dfcd333605`. Its instruction suffixes are removed only to match against the official annotation; the original question is emitted. No dataset loading script is executed. `hf-mirror.com` is used as a byte transport for the identical pinned Hugging Face revisions; source identity remains the upstream repository. Large downloads use HTTP ranges and validate the complete upstream LFS SHA256 before atomically publishing the local file.

## Leakage checks

- `evaluation/suite.jsonl` freezes 3,423 heldout questions, including 500 RefCOCO UNC validation examples with 439 images. No heldout rows are used to construct RL labels or router training data.
- Image identity is SHA256 of big-endian width/height plus decoded RGB bytes, matching `evaluation/common.py`; all matching images are removed across sources.
- Original RefCOCO validation COCO image IDs are also excluded from VQAv2 and the TallyQA source, protecting against alternative encodings of those COCO images.
- Duplicate image/question pairs are removed inside each source. Repeated question templates with different images are retained and reported. Long exact mathematical question matches to evaluation are excluded separately.
- RefCOCO pairs with identical image/question but conflicting target boxes are excluded entirely, rather than choosing one box arbitrarily. This run found seven such pairs (including both annotation noise and truly different regions). Case-only or equivalent mathematical-answer notation differences are ordinary duplicates; VQA annotation multiplicities remain intact for the selected canonical record.
- ViRL39K overlaps with the other reasoning datasets are removed at image or question level before applying mixture quotas.
- **A release-level training label is insufficient:** original UUID auditing found 1,843 `MMK12` and 209 `MMK12Converted` entries in ViRL39K derived from the original MMK12 **test** split. All 2,052 known test-derived IDs are prohibited in training. See `raw/virl39k_original_split_audit.json`; normalized-pool audit reports how many appeared in this run's seeded reservoir. Remaining MMK12 train derivatives are additionally deduplicated by their original UUID, so image re-encoding cannot hide cross-source repetitions.
- Exact pixel and text matching cannot establish absence of paraphrased, cropped, or visually near-duplicate benchmark images. The audit describes its actual scope; it does not claim a complete semantic decontamination proof.

## Output contract

`prepared/{perception,reasoning,generalist}.parquet` follows the local verl schema:

```text
data_source: sharedexpert_<source>
prompt: [{role: user, content: <image> markers + original question + final-answer instruction}]
images: [{path: absolute path, min_pixels: 4096, max_pixels: min(786432,1572864//n_images)}]
reward_model: {style: rule, ground_truth: JSON string}
extra_info: {source, source_id, split, task, question_sha256, image_sha256, record_id, index}
```

Reward payload tasks are `exact`, `math`, `vqa`, `anls`, `relaxed_numeric`, `bbox`, or `json`. Bboxes are xyxy coordinates in [0,1000]. VQA rewards must preserve annotation multiplicities. JSON rewards include `content_task: vqa`, so format alone is insufficient. Non-JSON tasks request their final answer in `\boxed{...}`.

ViRL39K includes up to eight images in a question. The configured total area cap corresponds to approximately 1,536 visual tokens; one-image records retain the 786,432-pixel cap. The installed HF processor then performs a second resize with a default 65,536-pixel minimum, so the configured 4,096 minimum is not the final effective floor. The header audit applies both installed pure resize functions without decoding the image library: final visual tokens are `sum(H_final*W_final)/1024`, and one full FP32 pixel tensor payload is `sum(H_final*W_final)*24` bytes before multiplying by group size. These payload estimates do not establish runtime copy counts, RSS, or memory feasibility. The training agent checks the final processor-expanded prompt and preserves image spans during text truncation; image/template-only overflow fails explicitly. Data and FP32 transport are unchanged.

`prepared/smoke.parquet` contains four genuine training questions, one each from Geometry3K, MMK12, VQAv2, and DocVQA, for an isolated one-step integration check. It is never used as the formal 300-step dataset.

## Commands

Use `/home/ml-user/workdir/CVPR2027/pre-lab/venv/bin/python` for PyArrow/Pillow, from the workspace root. Download manifests under `raw/` pin files and digests. `download_sources.py` handles the primary releases, and `download_ranges.py` handles large files with upstream SHA256 validation. `prepare_refcoco.py --train-images 1500 --workers 64` downloads official COCO images and prepares original UNC splits, emitting validation first for evaluation setup.

```bash
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/build.py normalize geometry3k
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/build.py normalize mmk12 --limit 8000
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/build.py normalize virl39k --limit 10000
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/build.py normalize vqav2 --limit 13052
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/build.py normalize docvqa --limit 2500
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/build.py normalize chartqa --limit 2500
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/build.py normalize tallyqa --limit 14000
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/build.py normalize clevr --limit 10000
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/audit_sources.py
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/build.py finalize --n 12000 --batch 40 --eval experiments/sharedexpert_c/evaluation/suite.jsonl
pre-lab/venv/bin/python experiments/sharedexpert_c/data_pipeline/audit_prepared.py --eval experiments/sharedexpert_c/evaluation/suite.jsonl --full-image-hash
```

The `finalize --n 12000 --batch 40` command reproduces the original prepared artifact and its historical construction metadata; it is not the current training batch. Do not rebuild or relabel these retained files to implement batch 30. Their existing `prepared/audit.json` reports full-file source counts, unique IDs/images, repeated exposures, filtering counts, fixed evaluation-manifest SHA256, normalized-pool hashes, provenance-artifact hashes, and parquet SHA256. Current consumption is separately documented by the prefix audit above. `prepared/source_validation.json` records an independent SHA256 recheck of all 31 pinned LFS source payloads; the two official non-LFS annotation archives have local reproducibility hashes, with no claim of an unavailable upstream checksum. ViRL normalization verifies selected image CRCs and atomically resumes missing archive extraction. `prepared/validation.json` is written only after strict mixture/schema/heldout-overlap/image-existence checks pass, including IF image disjointness and (in the command above) a recomputed RGB hash of every distinct training image path.
