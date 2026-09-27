# ForwardGNN：官方结果与 MPS 复现

本仓库保留原论文结果、官方源码及划分、Apple MPS 实验与显存监控。旧 CPU/pilot/staged 和自定义图级、链接预测实验已清理；[清理清单](docs/cleanup_20260927.json)记录删除范围和仓库外恢复包位置。

## 项目已有结果

| 结果 | 当前结果 | 状态与口径 |
|---|---|---|
| 原论文附录 E，Table 3–16 | 节点分类 accuracy、链接预测 ROC-AUC、GPU 显存、训练时间及数据集统计；覆盖 5 个图数据集和 GCN/SAGE/GAT | 已提取论文报告值，非本机运行；SF/BP 配置级数据和图见[附录报告](results/original_paper/appendix_e/report.md) |
| 论文四层节点分类 SF/BP | 15 组数据集/骨干比较中，SF 的均值 accuracy 较高 6 组，GPU 显存较低 7 组 | 论文五次运行的均值；没有做显著性检验。SF 优势取决于图和骨干 |
| 核心官方代码 MPS 复现 | CiteSeer/CoraML × GCN × BP/SF × 1～4 层，16 个配置均完成五折；本机均值与论文均值最大差 0.302 个百分点 | 正式 MPS 结果；禁用 CPU 算子回退。SF 主准确率来自官方四层训练保存的前缀 |
| 核心 MPS 显存 | 两数据集 × BP/SF × 1～4 层，共 16 个已完成进程配置；另行记录张量与 Metal 驱动分配峰值 | 0.5 秒进程内采样。SF 绝对显存并非总低于 BP；此数据不能与论文 H100 的 MB 直接比较 |
| GitHub 官方代码 MPS 复现 | GCN/GAT，BP/SF，1～4 层，官方五折；已保存 3 次独立训练结果（其中完整五折配置为 0/16）；用户要求停止，未完成 | 已停止，尚无最终复现结论；保留的进度与部分结果见[动态报告](results/reproduction/summary/github_nodeclass_mps_memory/report.md) |
| SF 虚拟节点消融 | Cora/CiteSeer，2/4 层，VN 与训练节点原型读出，每条件 5 个配对初始化；8 个条件已完成 | 独立 MPS 消融，不是原论文完整复现。固定公开划分，结果见[消融报告](results/own_experiments/sf_vn_ablation_mps/report.md) |

核心准确率逐配置对照见[复现结果报告](docs/forwardgnn_reproduction_results_core_nodeclass_mps_memory.md)，核心显存明细和口径见[显存报告](results/reproduction/summary/core_nodeclass_mps_memory/memory_comparison/report.md)。MPS 虚拟节点消融比较 SF-VN 与无虚拟节点的原型读出，不能解释为完全等价架构只删除虚拟节点。

项目级结论限于上述实验：论文结果中，SF 不会在所有数据集和骨干上同时提高准确率并降低显存；本机 MPS 核心结果接近论文报告的准确率，但苹果统一内存采样与 H100 显存口径不同；GitHub 全矩阵已按用户要求停止。

## 结果目录

- `results/original_paper/`：原论文发布的表格与图表；入口为 [Appendix E 报告](results/original_paper/appendix_e/report.md)。
- `results/reproduction/`：作者官方代码在本机 MPS 上的复现结果，包括逐折输出、配置 manifest、日志、显存采样和汇总。核心结果报告见上表；GitHub 队列已停止，部分折结果不是最终复现结论。
- `results/own_experiments/`：本项目自行设计的实验，目前是 [SF 虚拟节点消融](results/own_experiments/sf_vn_ablation_mps/report.md)，不属于原论文完整复现。

MPS 短训练仅作流程检查，不纳入正式精度统计。

## 运行

```bash
conda activate gnn-research
python -m paper_reproduction.experiments.plot_appendix_results
python -m paper_reproduction.experiments.watch_mps_memory --setting github_nodeclass_mps_memory --watch
python -m paper_reproduction.experiments.summarize_github --watch
```

[核心复现协议](docs/forwardgnn_reproduction_protocol.md) · [GitHub 数据来源与运行说明](docs/forwardgnn_github_reproduction.md) · [论文方法总结](docs/forwardgnn_paper_summary.md) · [目录说明](RESEARCH_LAYOUT.md)

正式训练入口只接受 MPS，禁用 CPU 算子 fallback；数据准备和指标汇总在 CPU。显存每 0.5 秒采样，区分张量分配与驱动分配；本机采样峰值与论文 H100 的 MB 不能直接当作同口径结果。
