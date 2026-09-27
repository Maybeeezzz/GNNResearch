# 官方代码复现结果

此目录存作者官方代码在本机 MPS 上的运行结果。`official/` 是逐折 JSON；`manifest_*.json` 记录配置、环境与划分哈希；`logs/` 是训练日志；`memory/` 是显存采样和正常结束时的摘要；`summary/` 是准确率、显存汇总及图表。

- [`summary/core_nodeclass_mps_memory/`](summary/core_nodeclass_mps_memory/accuracy_summary.json)：CiteSeer/CoraML 核心五折准确率和显存结果，已完成。
- [`summary/github_nodeclass_mps_memory/`](summary/github_nodeclass_mps_memory/report.md)：GitHub MPS 复现。用户要求停止；进度记录 3/80 个独立拟运行项已保存，完整五折配置为 0/16。部分记录不代表最终结果。
- `github_mps_smoke/`、`mps_smoke/` 和 `mps_memory_smoke/`：短训练流程/设备检查，不并入正式准确率结论。

显存文件是进程内采样值；中断 worker 的 JSONL 没有完整结束摘要，应标为不完整。
