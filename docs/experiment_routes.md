# 保留的实验与官方结果

| 路线 | 设备/来源 | 内容 | 入口 |
|---|---|---|---|
| Appendix E | 原论文 H100 | Table 3–16；SF/BP accuracy、ROC-AUC、GPU memory、训练时间 | [论文图表](../results/original_paper/appendix_e/report.md) |
| 核心官方复现 | MPS | CiteSeer / CoraML，GCN，五折，1～4 层 | [协议](forwardgnn_reproduction_protocol.md) |
| GitHub 官方复现 | MPS | GCN / GAT，五折，1～4 层，含 SF 独立深度显存 | [动态报告](../results/reproduction/summary/github_nodeclass_mps_memory/report.md) |
| 虚拟节点消融 | MPS | SF-VN / SF-Proto，Cora / CiteSeer，固定公开划分 | [报告](../results/own_experiments/sf_vn_ablation_mps/report.md) |

论文发布结果、本机官方代码复现与独立消融分别保存。不同划分、任务、硬件及显存测量口径不得混作同一实验。图中的均值差不表示统计显著性。

旧 CPU 与探索性实验已按用户要求清理，见[清单](cleanup_20260927.json)。GitHub 实验仍按原预算运行，完整性以动态报告为准。
