# SharedExpert 训练监控状态

更新时间（UTC）：2026-10-22T14:11:12.218204+00:00

状态：**STOPPED**；阶段：`failed`；教师：`generalist`。

原因：MemoryError: Cgroup working set exceeded 94% of its actual memory limit; sent SIGTERM only to the current experiment child group

| 教师 | 最新 TensorBoard step | 已保存 checkpoint step |
|---|---:|---:|
| perception | 100 | 100 |
| reasoning | 100 | 100 |
| generalist | 14 | 0 |

共享容器 OOM 增量（相对本实验首次监控基线；不直接归因教师）：{"oom": 0, "oom_kill": 0}

首次基线：{"low": 0, "high": 0, "max": 28579648, "oom": 19, "oom_kill": 12}

最近告警：
```json
{
  "id": "bf2f835f87db42898233d1975e5626ab4f6e7c93228d2682e5c92ea8c359fc61",
  "observed_at": "2026-10-22T11:25:11.630837+00:00",
  "kind": "training_stopped",
  "test_mode": false,
  "source": "/home/ml-user/workdir/CVPR2027/experiments/sharedexpert_c/runtime/experiment_state.json",
  "experiment_stage": "failed",
  "teacher": "generalist",
  "driver_pid": 2671251,
  "driver_start_ticks": 362238149,
  "state_updated_at": "2026-10-22T19:25:09.800228+08:00",
  "reason": "MemoryError: Cgroup working set exceeded 94% of its actual memory limit; sent SIGTERM only to the current experiment child group",
  "attribution": "driver state/process identity evidence; does not imply OOM"
}
```

最近监控错误：
```json
null
```

曲线：[training_metrics.png](training_metrics.png)。历史采样（gzip）：[samples.jsonl.gz](samples.jsonl.gz)。
监控存活不代表训练正在运行；checkpoint 与 scalar step 分开记录。
