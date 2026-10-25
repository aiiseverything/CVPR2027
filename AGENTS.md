# Project instructions

## Training workflow — user decision, 2026-10-24 (Asia/Shanghai)

- For future training jobs, run CPU configuration, dataset, dependency, and
  resource checks, then start the authorized formal training directly.
- Do not schedule a separate GPU smoke run, preliminary one-step training run,
  or smoke checkpoint/HF export by default. The user explicitly requested that
  these additional runs and their save/export/cleanup delays be removed.
- Do not request confirmation to skip smoke. Add a separate GPU smoke only if
  the user explicitly requests it later. Keep normal formal-training monitoring,
  checkpointing, and final model export.
- Some existing launchers and queues still require a smoke receipt. Before
  reusing them for a NEW job, remove that prerequisite from the new execution
  path before freezing its inputs. Never fabricate a passing smoke receipt or
  claim a GPU check was performed when it was not.
- The already-running perception_v2 queue has frozen source/config/data hashes.
  Do not edit its pinned inputs to apply this future-job preference mid-run;
  allow its existing execution to finish. Preserve the user's selected data
  mixture and formal training settings.
