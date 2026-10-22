# 2026 年 10 月 22 日训练监控快照

本快照保存北京时间 2026-10-22 22:11 左右的已有监控结果。generalist 停在 step 14，触发原 driver 的 94% 容器 working-set 保护；没有执行训练重启。

| 文件 | 内容 |
| --- | --- |
| [status.md](status.md) | 停止原因、scalar/checkpoint 进度、最近告警 |
| [training_metrics.png](training_metrics.png) | 三教师训练指标曲线 |
| [metrics.csv](metrics.csv) | 22,478 条原始 TensorBoard scalar 指标 |
| [samples.jsonl.gz](samples.jsonl.gz) | 50,073 条完整资源采样，gzip 无损压缩 |
| [alerts.jsonl](alerts.jsonl) | 去重告警及上下文 |
| [state.json](state.json) | 监控状态和进程证据 |
| [render_status.json](render_status.json) | 绘图状态与各教师最新 step |
| [scalar_inventory.json](scalar_inventory.json) | 实际存在的指标 tags |
| [driver/](driver/) | driver 状态和停止日志 |
| [runs/](runs/) | 三教师训练日志、配置、TensorBoard 事件及 checkpoint/导出凭据 |
| [budget100/](budget100/) | 调整到 100 steps 的执行预算与历史验证记录 |
| [capture.json](capture.json) | 快照时间、原始采样范围及解压后的 SHA256 |
| [manifest.json](manifest.json) | 快照文件大小和 SHA256 |

原始采样从 2026-10-19T16:38:11 UTC 到 2026-10-22T14:11:12 UTC，解压后为 923,983,419 字节。恢复采样文件：

```bash
gzip -dc samples.jsonl.gz > samples.jsonl
```

monitor 仍在独立采样，快照文件不会随其变化。采集过程中只读取源文件，按采样源文件读取前的大小截取完整 JSONL 记录。各文件的更新时间保留在原有内容中；快照不是跨文件事务。status.md 中的采样链接改为本快照的 gzip 文件。

perception 的原始 protocol.json 保留启动时的 300-step 配置；后续在 100 steps 结束的记录见 budget100。旧监控的 exported 字段检查 global_step_300，可能显示 false；实际 global_step_100 导出完成凭据已保存在 runs/perception/exports 和 runs/reasoning/exports。

resource_recovery.png 为 10 月 20 日已有的资源恢复图，保留作历史记录；当前曲线请看 training_metrics.png。共享容器的内存及 OOM 计数不能直接归因某个教师。

[返回训练与评测总览](../../../RESULTS_20261022.md)
