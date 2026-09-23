# GNN 训练方法性能实验报告

## 实验范围

- 运行数：23
- 数据集：CiteSeer, Cora，Planetoid 公开划分
- 方法：bp
- 种子：41, 42, 43, 44, 45
- 各配置的实际深度、宽度、预算和学习率见下表及 runs.csv；标准差为样本标准差，单次运行的标准差为空。

## 实测结果

| dataset | model | method | layers | hidden | steps | step_unit | learning_rate | initial_test_accuracy_mean | accuracy_delta_mean | test_accuracy_mean | test_accuracy_std | test_macro_f1_mean | wall_seconds_mean | peak_rss_mb_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CiteSeer | gcn | bp | 2 | 64 | 500 | full_network_update | 0.01 | 0.1778 | 0.5346 | 0.7124 | 0.0017 | 0.6859 | 18.1245 | 478.3469 |
| CiteSeer | gcn | bp | 2 | 64 | 500 | full_network_update | 0.01 | 0.6724 | -0.0010 | 0.6714 | 0.0073 | 0.6372 | 18.0822 | 479.5063 |
| Cora | gcn | bp | 2 | 64 | 500 | full_network_update | 0.01 | 0.1580 | 0.6474 | 0.8054 | 0.0023 | 0.7941 | 6.6475 | 412.4156 |
| Cora | gcn | bp | 2 | 64 | 500 | full_network_update | 0.01 | 0.7748 | -0.0094 | 0.7654 | 0.0030 | 0.7587 | 6.6533 | 412.4781 |
| Cora | residual-gcn | bp | 8 | 64 | 500 | full_network_update | 0.01 | 0.6093 | 0.0877 | 0.6970 | 0.0291 | 0.6967 | 16.8129 | 552.2344 |

## 比较口径

- forwardgnn 实现基础 SF-GCN（逐层局部学习）。其 steps 为每层预算，step_unit 为 local_layer_update；总更新次数为 steps × layers。
- BP 的 step 为完整网络更新；不要跨这两种步数单位比较 milliseconds_per_step。
- 同样的每层更新预算下，可以比较完整墙钟时间，但这是固定预算比较，不是达到相同准确率所需的时间比较。
- forward-gcn + bp 为相同虚拟节点架构的端到端对照；gcn + bp 是常规两层 GCN，两者参数量不同。
- 墙钟包含训练过程中的定期评估开销；ForwardGNN 还包括层间缓存和训练函数末尾评估。初始化模型、初始指标和训练后的独立汇总评估不计入。
- 峰值 RSS 为 CPU 进程内存，不能视为 CUDA 峰值显存；分阶段实验的墙钟不包含前置预训练。

## 局限

- 这些记录只覆盖所列配置的全批量节点分类，不构成跨任务或硬件的普遍性能结论。
- ForwardGNN 的本地实现及 Planetoid 数据划分与官方完整实验配置不同，不能直接比较论文数值。
