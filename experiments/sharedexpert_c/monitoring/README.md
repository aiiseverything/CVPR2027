# 独立训练监控

解释器固定使用已有系统 **`/opt/venv/main/bin/python3`**：matplotlib 3.7.5、TensorBoard 2.14.0、NumPy 1.26.4。训练 `.venv-train` 没有 matplotlib；本监控不安装包、不修改训练环境或 freeze。进程设置 `CUDA_VISIBLE_DEVICES=''`、CPU 线程数 1、matplotlib Agg，只使用 `nvidia-smi` 读取 GPU 状态。

从工作区根目录运行：

```bash
/opt/venv/main/bin/python3 experiments/sharedexpert_c/monitor_training.py launch
/opt/venv/main/bin/python3 experiments/sharedexpert_c/monitor_training.py status
```

`launch` 使用独立 session 后台运行，关闭终端后继续；同一输出目录由本机 `/tmp/sharedexpert-monitor-*.lock` 排他锁保护。默认只监控 `runs/perception`、`runs/reasoning`、`runs/generalist`，不自动搜索 smoke 或历史归档。监控仅检测、记录和绘图，不向训练或其他服务发信号；root driver 已有的 94% working-set 保护保持独立。

每 5 秒采样 cgroup 的真实 limit/current/inactive_file/working_set、memory.events、各 GPU 的显存和利用率、实验阶段、正式 trainer 及子进程。`cgroup.memory_stat` 保存完整原始计数，并提供 `anon_bytes`、`file_bytes`、`active_file_bytes`、`inactive_file_bytes`、`shmem_bytes`、`slab_unreclaimable_bytes`；`memory_pressure` 保存 PSI 原始文本。旧字段保持兼容，新增字段只适用于升级后的采样，不能反推之前的匿名内存或缓存构成。

默认每 60 秒更新实际 TensorBoard scalar CSV 和 PNG；阶段、教师或失败状态变化时立即更新，不等待下一个绘图周期。训练最终 checkpoint 齐备时立即额外 render，随后评测期间继续监控；当 root driver 的 `experiment_stage=complete` 时最后 render 并退出，状态记为 `complete`。收到监控自身停止信号时也最后 render。后台监控自身异常记录到独立 errors 文件，不修改训练结果。

| 文件 | 内容 |
|---|---|
| `state.json` | 原子保存的 PID、heartbeat、首次/本实例 OOM 基线、最新计数、日志 byte offsets、训练进度和绘图状态 |
| `status.md` | 每次采样原子更新的可读摘要：阶段、教师、失败原因、scalar/checkpoint step、最近告警/监控错误、OOM 基线与增量 |
| `samples.jsonl` | cgroup、`nvidia-smi`、阶段及进程原始采样 |
| `alerts.jsonl` | 去重的异常证据，含观测时间、来源、byte offset 和相邻日志上下文 |
| `errors.jsonl`、`monitor.log` | 监控自身错误及后台 stdout/stderr |
| `metrics.csv` | 实际 scalar 的长表：run、step、tag、value、wall_time、来源；同 run/tag/step 取最新事件 |
| `scalar_inventory.json` | 实际存在的 tags 和每教师最新 step |
| `training_metrics.png` | reward/score、accuracy、response length、pg loss、grad norm、KL 诊断、IS 诊断、step time 八面板 |
| `render_status.json` | 等待/成功/错误状态；不存在的 tag 不生成伪曲线 |

图表标题直接显示真实 `experiment_stage`、教师和失败原因摘要；失败/异常退出显示红色 **STOPPED**，即使一个 scalar 都没有也不会继续显示 WAITING。有历史 scalar 时保留真实曲线并标注停止。只有活动阶段没有匹配 scalar 时显示 WAITING；其他没有数据的状态显示 NO DATA。某些训练版本没有 accuracy tag，不把 reward 假装成 accuracy。CSV 没有数据时仅有表头。NaN/Inf 原值留在 CSV，同时记录去重的 `nonfinite_scalar` 告警，不把它们伪装成有限曲线点。TensorBoard 的 `perf/*memory*` 可能采用不同统计范围，**不作为实际物理资源峰值**；实际资源证据只采用 samples 中的 `nvidia-smi` 和 cgroup。

异常类别分开记录：`cgroup_oom_counter_increment` 是共享容器内核计数增加，不能据此归因某教师；`cuda_oom_log` 是日志明确 CUDA OOM；`host_or_ray_oom_log` / `host_memory_error_log` 是 host/Ray 内存错误；`preventive_memory_guard` 是 root driver 的预防性停止，不声称已经发生 OOM。准备阶段历史计数为 oom=19、oom_kill=12，作为历史参考保存，启动时已有计数不会报警。首次监控的实际 baseline 和 last_events 在重启时保留，因此能记录监控离线期间的增量；每个实例另记自己的 baseline。

`training_stopped` 单独记录 root 状态为 `failed` 或 `interrupted` 的事件；活动阶段 root PID 不存在、已为 zombie，或命令/已记录的启动时钟不匹配时，记录 `driver_unexpected_exit`。首次观察到有效 driver 时保存其 `start_ticks`，后续检查可识别 PID 复用。进程扫描失败单独记监控错误，不能据此断言 driver 死亡。事件按 driver 身份、状态时间与失败原因去重，跨重启仍只记一次。它们表示流程停止，不表示 OOM；监控继续运行以便观察恢复，且不自动重启训练或发送外部消息。

日志游标跨重启保存；告警 ID 包含文件身份和字节位置，重启不会重复输出已记录事件。首次接管 `runtime/experiment.log` 从当时文件末尾开始，避免把此前 smoke/probe 失败当作正式事故；三个正式 training.log 从各自起点读。归档目录从不被正式模式扫描。

driver 集成接口：`launch` 返回 JSON，包含 `pid`、`state`、`python_executable`（重复启动返回 `already_running`）。`status` 返回 `running`、`pid`、`start_ticks`（`/proc/PID/stat` 的启动时钟，防 PID 复用）、`heartbeat_epoch`、`heartbeat_age_seconds`、`status`、`render_status`、`latest_metric_steps`、`last_error`；新增 `display_status`、`experiment_stage`、`experiment_teacher`、`experiment_error`、`driver_health`、`plot_banner`。`running=true` 只表示 monitor 自身存活，不能解释为训练仍在运行。直接读取 `monitoring/state.json` 或 `status.md` 也可；建议 heartbeat 超过 **120 秒**才视为失联，容许首次 matplotlib 初始化/文件系统延迟。可在监控进程死亡后重复 `launch`，不会重置首次基线或日志 offset。监控不提供杀训练接口。

隔离自测只读指定的真实 smoke，输出到与正式 monitoring 不同的目录；`--run-dir` 仅允许搭配 `--once`，不会被持久模式误用于正式曲线：

```bash
/opt/venv/main/bin/python3 experiments/sharedexpert_c/monitor_training.py --once \
  --run-dir experiments/sharedexpert_c/runs/perception_smoke_reward_superseded_20261020_0000 \
  --output experiments/sharedexpert_c/monitoring_test
```

该输出带 `test_mode=true` 和 SELF-TEST 图题，仅验证读取、采样与绘图链路；旧 smoke 的任何分数都不是正式训练结果。
