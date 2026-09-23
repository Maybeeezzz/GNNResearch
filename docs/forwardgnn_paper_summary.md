# Forward Learning of Graph Neural Networks：原论文结果总结

本文仅总结原论文的方法、实验设置、报告结果及结论，不包含本机复现结果。实际复现另见 [复现结果报告](forwardgnn_reproduction_results.md)。

论文：Park 等，ICLR 2024，[arXiv:2403.11004](https://arxiv.org/abs/2403.11004)。
数值核对版本：arXiv v1 的正文、附录 B、附录 E；不将本项目此前实验当作论文结果。
官方代码：[facebookresearch/forwardgnn](https://github.com/facebookresearch/forwardgnn)，提交 `73fe25720afd0f854e49c099f9c2517c0549617d`。

## 1. 论文主要回答的问题

通过逐层局部学习代替端到端误差反传，能否训练具有实用预测能力的 GNN，并改善深度增加时的训练内存需求？

- FF-LA / FF-VN：分别通过标签特征拼接、类别虚拟节点构造正负输入。
- SF：单个增强图前向产生真实节点和类别代表，以局部对比交叉熵训练当前层，冻结后再训练下一层。
- SF-Top-To-Input / SF-Top-To-Loss：通过缓存的高层表示引入反馈，仍切断跨层梯度。
- 链接预测：一次节点表示计算后对正负边评分，以每层局部目标训练。

“Forward-only”不等于没有梯度；官方实现对当前层损失调用 backward()。Single-Forward 不表示整次训练只有一次前向。

## 2. 实验设置

| 项目 | 论文设置 |
| --- | --- |
| 任务与指标 | 节点分类 Accuracy；链接预测 ROC-AUC |
| 数据 | CitationFull-CiteSeer、CitationFull-Cora_ML、CitationFull-PubMed、Amazon-Photo、GitHub |
| 划分 | 训练/验证/测试 = 64%/16%/20%；5 次划分 |
| 骨干、深度 | GCN、GraphSAGE、GAT；1～4 层 |
| 隐藏维度 | 128；GAT 4 个头 |
| 优化 | Adam，lr=0.001，weight_decay=0.0005 |
| 预算 | 最多 1000 epoch；验证早停 patience=100 |
| 官方执行细节 | 每 2 epoch 验证；patience 按验证次数计；基础 SF 对每层分别早停 |
| SF | temperature=1；官方脚本使用双向虚拟边，不拼接标签特征 |
| 硬件 | NVIDIA H100，AMD EPYC 9654 |

官方脚本、数据代码还给出以下可复现细节：输入 NormalizeFeatures；节点划分采用 KFold(5, shuffle=True, random_state=101)，每折的其余 80% 再按 train_test_split(test_size=0.2, random_state=split_i*127) 划分验证；不是五次固定公开划分上的换种子。训练种子为 `100*101 + split_i*3`。

来源：[数据加载](https://github.com/facebookresearch/forwardgnn/blob/main/src/datasets/dataloader.py)、[划分代码](https://github.com/facebookresearch/forwardgnn/blob/main/src/datasets/datasplit.py)、[BP 脚本](https://github.com/facebookresearch/forwardgnn/blob/main/exp/nodeclass/nodeclass-bp.sh)、[SF 脚本](https://github.com/facebookresearch/forwardgnn/blob/main/exp/nodeclass/nodeclass-sf.sh)。

## 3. 主要结果

论文的总体证据支持：SF 在许多节点分类配置中接近或超过 BP；链接预测中多数比较有利于 SF，特别是 BP 随深度增加退化时。SF 的训练内存随层数增加更稳定，但不代表所有配置下的绝对内存都低于 BP，也不代表必然更快。

下面是附录节点分类结果中的 GCN 对照锚点。数值为百分数，均值 ± 标准差，尚不是本机复现值。

| 数据 | 层数 | BP-GCN | SF-GCN |
| --- | ---: | ---: | ---: |
| CiteSeer | 2 | 94.28 ± 0.7 | 94.18 ± 0.7 |
| CiteSeer | 4 | 94.87 ± 0.5 | 94.66 ± 0.6 |
| CoraML | 2 | 86.84 ± 1.0 | 87.95 ± 1.4 |
| CoraML | 4 | 86.61 ± 2.3 | 88.48 ± 1.2 |
| PubMed | 2 | 86.74 ± 0.3 | 87.61 ± 0.3 |
| PubMed | 4 | 86.01 ± 0.5 | 87.61 ± 0.4 |
| Amazon-Photo | 2 | 64.88 ± 16.1 | 93.73 ± 0.4 |
| Amazon-Photo | 4 | 90.21 ± 2.2 | 93.48 ± 0.3 |
| GitHub | 2 | 74.17 ± 0.5 | 85.90 ± 0.2 |
| GitHub | 4 | 86.34 ± 0.2 | 85.87 ± 0.3 |

来源：[论文附录 E，Tables 3–5](https://arxiv.org/html/2403.11004v1#A5.SS1)。可见 SF 并非对每个数据集、深度都占优。

内存示例：CiteSeer 的 GCN，BP 从 1 层的 0.52 MB 增至 4 层的 9.21 MB；SF 在 1～4 层均为 15.51 MB（Table 6）。这说明“随深度增长平稳”和“绝对内存更小”必须分开判断。

Top-down 的总体分类结果有改善，但具体骨干和深度仍有例外，尤其不能把 Top-To-Input 与 Top-To-Loss 当作效果相同的算法。FF 系列对配置较敏感；构造多个负输入增加计算和内存成本。

Table 1 的汇总准确率（不是单个 GCN 配置；其中 SF-TopDown 指 Top-To-Input）：

| 方法 | CiteSeer | CoraML | PubMed | Amazon | GitHub |
| --- | ---: | ---: | ---: | ---: | ---: |
| SF | 91.14 | 84.95 | 81.90 | 89.69 | 83.16 |
| SF-TopDown | 93.58 | 86.01 | 82.21 | 92.09 | 83.68 |

链接预测锚点（Table 9，CiteSeer，ROC-AUC %）：2 层 BP-GCN 为 84.31 ± 0.6，ForwardGNN-CE-GCN 为 93.61 ± 1.0；4 层分别为 80.05 ± 0.9 和 93.61 ± 1.0。这里的损失是边预测交叉熵，不能与节点分类 Accuracy 混为同一指标。

时间实验（附录 E.3 / Table 15）测量 100 epoch。GitHub 四层 GCN：BP 13.04 秒、SF 2.52 秒；四层 SAGE：BP 14.91 秒、SF 1.61 秒。其加速解释依赖固定输入时缓存邻域聚合，而不是单靠删除跨层反传。GAT 的可学习注意力和动态 top-down 输入不能直接复用这种缓存。

## 4. 结论应如何使用

1. 论文支持局部学习是 GNN 训练的可行选择，不支持其对所有任务、数据和骨干统一优于 BP。
2. 性能、内存、耗时是不同维度；达到相同准确率的耗时需要单独实验。
3. 节点分类与链接预测使用不同目标，不能用一个任务的结果代替另一个任务。
4. 论文 GPU 内存指标不能用 CPU 进程 RSS 替代；不同软件与硬件上的时间只作独立观察。
