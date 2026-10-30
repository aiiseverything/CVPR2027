# Perception teacher v2 data

This is a separate 3,000-prompt data version for `perception_v2`. It preserves
the active generalist protocol: 100 steps, batch size 30, group size 8, thinking
enabled, and the existing final-answer rule rewards. It does not modify the v1
perception parquet, any normalized source, or `rewards.py`.

| Category key | Weight | Formal prompts | Annotation source | Unchanged reward |
|---|---:|---:|---|---|
| `object_attribute` | 20% | 600 | VQAv2 train | VQA annotator consensus |
| `grounding` | 20% | 600 | RefCOCO train | Bounding-box IoU |
| `ocr_document` | 20% | 600 | DocVQA train | ANLS |
| `spatial_detail` | 15% | 450 | VQAv2 train | VQA annotator consensus |
| `existence_confusion` | 15% | 450 | VQAv2 train | VQA annotator consensus |
| `general_vqa_replay` | 10% | 300 | VQAv2 train | VQA annotator consensus |

The builder emits shuffled 20-record blocks with counts `4/4/4/3/3/2`. Thus each
60-prompt interval (two 30-prompt steps) and the complete 100-step run have the
requested ratio exactly. A single 30-prompt step cannot have exact 15% quotas.
There are no repeated record IDs or image/question pairs. The same image may
appear with different original questions, including across categories.

## Selection and label fidelity

`prepare_perception_v2.py` reads the existing normalized original **train**
annotations only. It generates no answers, false-image pairs, teacher labels,
reasoning traces, or model-judge scores. The full original selected records are
saved in `prepared/perception_v2.records.jsonl`; the only added fields are
`perception_category` and `selection_rule`. Parquet `extra_info` includes the
original source/record IDs, question/image hashes and the same category/rule.
The original answer lists, their multiplicities, and bounding-box coordinates
are passed unchanged to the existing reward implementation.

The categories use reproducible question rules rather than an LLM classifier:

1. **Object/category/attribute:** explicit color, kind/type, material, shape,
   breed/species and object identification questions. The sample contains 450
   color prompts and 150 other class/attribute prompts to retain some variety.
   This is not a claim of dedicated fine-grained species coverage.
2. **Grounding:** original RefCOCO train expressions and their 0–1000 normalized
   boxes. An identical image/question that has conflicting boxes is excluded.
3. **OCR/document:** DocVQA questions beginning with a direct question word and
   containing a field cue such as date, name, heading, code, address, table, value
   or page. Arithmetic, comparisons, extremes, causal/interpretive wording and
   paragraph-comprehension cues are excluded. ChartQA is not needed for this
   3,000-prompt run. The ANLS reward still gives partial credit for near-matching
   strings, including numeric fields; no stricter reward was silently added.
4. **Spatial/detail:** explicit left/right/behind/above/etc. references or local
   visible-detail questions (for example an item on a desk or in a hand).
   Approximately half of the quota comes from each subtype. Text-reading,
   gaze-direction inference and non-visual semantic uses of spatial words are
   filtered. Object sizes and occlusion are not newly annotated; this is not
   advertised as a dedicated small-object/occlusion benchmark.
5. **Presence/absence:** explicit “is/are there”, “do/can you see” questions with
   at least 8 of 10 original annotators agreeing. Exactly **225 yes and 225 no**
   prompts are selected. Count/comparison, capacity, inferred intention and
   knowledge-dependent presence formulations are excluded. These are genuine
   labeled presence/absence examples, **not measured hard negatives**; no model
   error-rate mining has been performed.
6. **General visual QA replay:** visible activities, held/eaten/worn objects,
   sports and room recognition. Complex counting, arithmetic, “why”, hypothetical
   intentions and broad external-knowledge prompts are excluded.

The non-existence VQAv2 categories require at least **7 of 10** annotators to
agree after the existing reward normalizer. Unknown/none answers are excluded;
non-boolean questions are also excluded when at least two annotations reject
the premise. Boolean spatial questions may still legitimately have answer “no”.
All original answers remain in the reward gold, including minority annotations.
The selected modal reference therefore receives full consensus reward; minority
reference spellings need not receive full reward.

Routing is exclusive: rejected existence questions do not fall through into
another category. Data shortages raise an error; the builder never cycles or
duplicates records to satisfy a quota. Question rules and high agreement improve
precision but do not prove every sample is visually unambiguous. Each category
has 12 deterministic review examples in `perception_v2.samples.json`.

## Evaluation isolation

The builder discovers every current `evaluation/suite*.jsonl`, currently the
core suite, BLINK/MMVP suite and general/OCRBench/IFEval suite. It records their
paths, file hashes, counts and benchmark names in the audit, and excludes:

- Any train image with an exact RGB pixel hash in any suite.
- Every exact normalized evaluation question, even when the training image is
  different. This conservative rule also removes some harmless generic prompts.
- Matching original source record IDs and any exposed COCO image IDs.
- Every original RefCOCO validation COCO image ID, including images outside the
  local 500-record held-out subset.
- Duplicate train record IDs and exact image/question pairs across categories,
  and ambiguous RefCOCO image/question pairs with conflicting target boxes.

Every selected local training image is decoded and its RGB pixel hash recomputed
before the training parquet is written. The comparison uses the pixel hashes
stored in evaluation manifests. Exact-pixel hashes do **not** detect all resized,
cropped, re-encoded or otherwise transformed near-duplicates. Original IDs are
not exposed by every evaluation dataset. The audit does not make claims about
base-model pretraining exposure or unavailable external test sets.

## Rebuild and artifacts

From the repository root, with Python, Pillow and PyArrow available:

```bash
python experiments/sharedexpert_c/data_pipeline/prepare_perception_v2.py
python -m unittest discover -s experiments/sharedexpert_c/tests -p test_perception_v2_data.py
```

The builder compares `training/config/common_local_100.json` with the protocol
recorded in `runs/generalist/protocol.json`; it fails if they differ. Inspection
with `--skip-image-verification` writes pool counts and review examples only and
does not create train parquets.

Generated artifacts are separate from v1:

- `prepared/perception_v2.parquet`: the 3,000 formal prompts.
- `prepared/perception_v2_smoke.parquet`: six real labeled prompts, one per
  category, for the separate smoke run. Their rollout difficulty and reward
  variance are not assumed; the training runner checks the actual smoke result.
- `prepared/perception_v2.records.jsonl`: full original selected records with
  selection metadata, ordered exactly like the parquet.
- `prepared/perception_v2.samples.json`: 12 review samples per category.
- `prepared/perception_v2.audit.json`: source hashes, all exclusion counts,
  eligible/selected counts, unique image/record counts, yes/no balance,
  recomputed image validation, unchanged reward hash and reference-score checks.
- `prepared/perception_v2.manifest.json`: compact hashes and paths used to pin
  the queued run to the prepared artifacts.

The reference check sends `</think>\boxed{original_reference}` through the
actual reward function for all 3,000 examples, requiring valid parsing and
positive reward, and reports any score below 1. Tests additionally verify all
20/60/3,000-row ratios, no repeated records/pairs, unchanged original labels,
the 225/225 balance, current evaluation exclusion, and six-category smoke input.
