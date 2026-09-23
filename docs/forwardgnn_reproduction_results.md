# ForwardGNN：本项目复现结果报告

本文记录本机实际执行的复现配置、测试结果及局限。原论文的独立总结见 [原论文结果总结](forwardgnn_paper_summary.md)；下表论文数值仅作为复现对照，不是本机测量值。

配置：`core_nodeclass`。状态：当前指定矩阵已完成。

## 范围与可追溯性

本次结果与此前使用 Planetoid 公开划分、两层 64 维模型、100 步等设置的项目实验分开保存；旧实验与 integration_smoke 流程检查均未纳入汇总。

- 数据：CitationFull-CiteSeer, CitationFull-Cora_ML；骨干：GCN；深度 1～4。
- 每个配置 5 个作者原始划分；隐藏维度 128，Adam lr=0.001、weight_decay=0.0005。
- 最多 1000 epoch，SF 为每层预算；每 2 epoch 验证，100 次验证未提升则早停，恢复最佳验证准确率权重。
- 使用官方训练代码；只增加未使用的 legacy cached 算子的可选导入适配，标准 GCN/SAGE/GAT 保持官方调用。
- 代码提交：`73fe25720afd0f854e49c099f9c2517c0549617d`；划分提交：`48997da6fc96fd11ee3dfed47e720d942853e729`。
- 环境：macOS-27.0-arm64-arm-64bit，PyTorch 2.10.0，PyG 2.8.0.post1，每个训练进程 CPU 单线程；部分独立配置并行执行。
- 原始 JSON 与日志位于 `results/paper_reproduction/official/` 和 `logs/`；manifest 包含划分文件 SHA256。
- 每个 SF 四层训练过程保存 1～4 层前缀的结果。这些前缀不是独立初始化的四次训练，遵循官方脚本。

## 准确率与论文锚点

所有准确率均为百分数。标准差使用总体标准差 ddof=0，与官方汇总代码一致；不是置信区间。未完成五折的行不能当作最终结果。

| 数据 | 骨干 | 方法 | 层数 | 完成折数 | 本机均值 ± 标准差 | 论文均值 ± 标准差 | 差值（百分点） |
|---|---|---|---:|---:|---:|---:|---:|
| CitationFull-CiteSeer | GCN | bp | 1 | 5/5 | 84.28 ± 2.99 | 84.28 ± 3.0 | -0.00 |
| CitationFull-CiteSeer | GCN | sf | 1 | 5/5 | 92.62 ± 0.73 | 92.60 ± 0.7 | +0.02 |
| CitationFull-CiteSeer | GCN | bp | 2 | 5/5 | 94.28 ± 0.71 | 94.28 ± 0.7 | -0.00 |
| CitationFull-CiteSeer | GCN | sf | 2 | 5/5 | 94.18 ± 0.68 | 94.18 ± 0.7 | +0.00 |
| CitationFull-CiteSeer | GCN | bp | 3 | 5/5 | 94.89 ± 0.78 | 94.89 ± 0.8 | +0.00 |
| CitationFull-CiteSeer | GCN | sf | 3 | 5/5 | 94.35 ± 0.71 | 94.33 ± 0.7 | +0.02 |
| CitationFull-CiteSeer | GCN | bp | 4 | 5/5 | 94.87 ± 0.52 | 94.87 ± 0.5 | +0.00 |
| CitationFull-CiteSeer | GCN | sf | 4 | 5/5 | 94.63 ± 0.61 | 94.66 ± 0.6 | -0.03 |
| CitationFull-Cora_ML | GCN | bp | 1 | 5/5 | 31.72 ± 4.80 | 31.72 ± 4.8 | -0.00 |
| CitationFull-Cora_ML | GCN | sf | 1 | 5/5 | 87.68 ± 1.45 | 87.75 ± 1.4 | -0.07 |
| CitationFull-Cora_ML | GCN | bp | 2 | 5/5 | 86.84 ± 0.98 | 86.84 ± 1.0 | +0.00 |
| CitationFull-Cora_ML | GCN | sf | 2 | 5/5 | 88.01 ± 1.32 | 87.95 ± 1.4 | +0.06 |
| CitationFull-Cora_ML | GCN | bp | 3 | 5/5 | 88.65 ± 1.22 | 88.65 ± 1.4 | -0.00 |
| CitationFull-Cora_ML | GCN | sf | 3 | 5/5 | 88.41 ± 1.14 | 88.15 ± 1.5 | +0.26 |
| CitationFull-Cora_ML | GCN | bp | 4 | 5/5 | 86.81 ± 1.88 | 86.61 ± 2.3 | +0.20 |
| CitationFull-Cora_ML | GCN | sf | 4 | 5/5 | 88.28 ± 1.28 | 88.48 ± 1.2 | -0.20 |

论文锚点来源：[附录 E 的 Tables 4、5](https://arxiv.org/html/2403.11004v1#A5.SS1)。差值仅描述数值接近程度，不是显著性检验。

## 结论与未覆盖项

本报告仅验证上述核心节点分类矩阵，不宣称复现完整论文。尚未覆盖其他数据/骨干组合、FF 对照、top-down 扩展、链接预测及 H100 显存结果。
CPU 没有提供论文 GPU 显存指标。日志时间保留供审计，但 SF 含逐层验证/测试等额外开销，BP 的计时范围不同，不以两者原始时间比宣称优化器加速。软件版本也与论文不同。

论文使用 PyTorch 1.13.1 / PyG 2.2.0 / H100，本机环境不同；本次使用官方准确率脚本的标准算子，未复现 Table 15 的缓存邻域聚合加速实验。

本次与论文对应的 16 项配置中，平均测试准确率的最大绝对差为 0.264 个百分点。


![深度与准确率](/Users/maybe/GNN/results/paper_reproduction/summary/core_nodeclass/accuracy_vs_depth.png)

在本次完成的相同深度对照中：

- CitationFull-CiteSeer / GCN / 1 层：SF − BP = +8.35 个百分点。
- CitationFull-CiteSeer / GCN / 2 层：SF − BP = -0.09 个百分点。
- CitationFull-CiteSeer / GCN / 3 层：SF − BP = -0.54 个百分点。
- CitationFull-CiteSeer / GCN / 4 层：SF − BP = -0.24 个百分点。
- CitationFull-Cora_ML / GCN / 1 层：SF − BP = +55.96 个百分点。
- CitationFull-Cora_ML / GCN / 2 层：SF − BP = +1.17 个百分点。
- CitationFull-Cora_ML / GCN / 3 层：SF − BP = -0.23 个百分点。
- CitationFull-Cora_ML / GCN / 4 层：SF − BP = +1.47 个百分点。

结论仅限这些配置；应结合上表逐项判断与论文是否一致。
