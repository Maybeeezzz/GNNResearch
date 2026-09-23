# GNN 节点分类基线

论文结果总结见 [docs/forwardgnn_paper_summary.md](docs/forwardgnn_paper_summary.md)，官方代码复现操作见 [docs/forwardgnn_reproduction_protocol.md](docs/forwardgnn_reproduction_protocol.md)，本机复现状态与结果见 [docs/forwardgnn_reproduction_results.md](docs/forwardgnn_reproduction_results.md)。这些实验与下方原有 Planetoid 基线分开存储。

各项研究按 GNN 骨干、优化方法（BP/FF/SF）和任务类型整理的实验路线见 [docs/experiment_routes.md](docs/experiment_routes.md)。
脚本命名及 Python 格式约定见 [docs/code_style.md](docs/code_style.md)；Ruff 配置位于 `pyproject.toml`。
研究代码已统一按任务功能归档，目录索引见 [RESEARCH_LAYOUT.md](RESEARCH_LAYOUT.md)。

项目包含两层 GCN、深层 ResidualGCN 和使用逐层局部学习的 ForwardGNN。模型和合成数据训练只依赖 PyTorch；加载真实 Planetoid 数据集需要 PyTorch Geometric。支持稀疏消息传递、分层数据切分、早停、评估、模型保存和测试。

## 快速运行

```bash
conda env create -f environment.yml
conda activate gnn-research
python train.py
```

`environment.yml` 还包含 JupyterLab、scikit-learn、pandas、NetworkX、绘图库和 TensorBoard，适合后续实验研究。如果只需要最小运行环境，也可以执行 `python -m pip install -r requirements.txt`。

默认生成一个随机块模型（SBM）图，使用标准 BP 并保存到 `checkpoints/gcn_standard.pt`。常用参数：

```bash
python train.py --epochs 500 --hidden 128 --dropout 0.4 --device auto
pytest -q
```

## ForwardGNN Single-Forward 局部学习（当前方法）

实现论文 [Forward Learning of Graph Neural Networks](https://arxiv.org/abs/2403.11004) 第 3.2 节 / Algorithm 2 的 SF-GCN。核心实现位于 `node_classification/models/forward_gnn.py`，入口为 `train_forwardgnn()`。

```bash
python train.py --method forwardgnn --epochs 100 --hidden 64 --device cpu
python train.py --dataset Cora --method forwardgnn --layers 2 --epochs 300 --lr 0.001 --device cpu
python train.py --dataset Cora --method forwardgnn --layers 8 --hidden 128 --device cpu
```

`--method forwardgnn` 自动选择独立模型 `forward-gcn`。其各层均输出隐藏表示，不使用普通 GCN 的分类输出层或 ResidualGCN 的残差块。实现使用每类一个固定随机特征的虚拟节点，只连接训练节点与其对应类别；默认双向边，可以通过 `--edge-direction unidirection` 改为单向边。每层输入先作行 L2 归一化，再执行 GCN + ReLU；用真实节点和虚拟节点表示的点积除以 `--temperature` 计算交叉熵。逐层训练、冻结已训练层并缓存其输出；推理平均各层的类别概率。

这是**局部一阶梯度学习**：每层仍调用局部 `backward()`，层间使用 `detach()`，不执行端到端反传。它与下文复用验证输出的 single-forward BP 不同。这里实现的是基础 SF-GCN，不含第 3.3 节的 top-down 扩展、标签特征拼接或 GAT/GraphSAGE。

`--epochs` 是**每层**更新次数上限；两层各 100 次，共 200 次局部更新。验证、缓存构建和最终推理额外执行前向计算，不能将“single-forward”理解为整次实验仅做一次前向。早停按每层验证准确率选择权重，`--patience -1` 关闭早停、使用最后权重。检查点包含训练标签上下文与虚拟节点特征；恢复时按检查点中的模型配置构造 `ForwardGNN` 并加载 `model_state_dict`，然后在同一节点顺序的原图上推理。

算法设计核对了[官方实现](https://github.com/facebookresearch/forwardgnn/blob/73fe25720afd0f854e49c099f9c2517c0549617d/src/forward_learning/nodeclass/gnn_sf.py)。本项目是独立 PyTorch 实现，未使用官方 CachedGCN 的传播缓存优化，未复现其完整超参数及数据划分；下方实验使用 Planetoid 公开划分，不能直接与论文数值比较。

## 复用验证前向的 BP 效率实验（旧方法）

标准训练每个 epoch 在参数更新后额外执行一次验证前向传播；single-forward 会复用训练时的输出计算验证损失，因此每个 epoch 只进行一次完整图前向传播：

```bash
python train.py --method single-forward
python benchmark.py --epochs 100 --repeats 5 --nodes 1200
```

基准脚本会预热运行时、交替方法的测试顺序，并报告平均耗时、标准差、每 epoch 延迟、吞吐率、加速比和测试准确率。默认关闭 dropout，以减少两种验证方式之间的随机差异；可以用 `--dropout 0.5` 测试实际训练配置。

复杂模型可以使用带 LayerNorm 和残差连接的深层 GCN：

```bash
python train.py --model residual-gcn --layers 8 --hidden 128
python benchmark.py --model residual-gcn --layers 8 --hidden 128 \
  --nodes 2000 --features 128 --epochs 80 --repeats 3
```

## 真实数据集实验

正式实验使用 Cora、CiteSeer 的 Planetoid 公开划分，记录 Accuracy、Macro-F1、墙钟时间、峰值 RSS 和每 50 步的收敛轨迹。每个配置在独立进程中运行：

```bash
# 标准 GCN/BP、相同虚拟节点架构的 BP、ForwardGNN 局部训练
python -m node_classification.experiments.baselines.run_study \
  --datasets Cora CiteSeer --models gcn forward-gcn \
  --methods bp forwardgnn --scopes all \
  --seeds 41 42 43 --steps 300 --results results/forwardgnn

# 深层 SF-GCN 与相同架构 BP
python -m node_classification.experiments.baselines.run_study \
  --datasets Cora --models forward-gcn --layers 8 \
  --methods bp forwardgnn --steps 300 --results results/forwardgnn_deep

# 汇总 CSV、收敛图和 Markdown 报告
python -m node_classification.experiments.baselines.summarize \
  --inputs results/forwardgnn results/forwardgnn_deep \
  --output results/forwardgnn_summary
```

实验脚本使用固定预算的最终权重，不作验证早停。ForwardGNN 的 `--steps` 为每层更新次数，曲线 `step` 为累计局部更新次数；BP 为全网络更新次数。JSON 分别记录 `total_updates`、`step_unit`、`local_backward_passes`、`optimization_layer_forwards`，墙钟包含评估开销。比较完整训练耗时和最终 Accuracy/Macro-F1，不能直接用两种方法的 ms/step 宣称加速。相同隐藏宽度下，标准 GCN 和 SF-GCN 的参数量不同；`forward-gcn + bp` 提供相同虚拟节点架构、相同初始化方式的端到端对照，其目标是各层平均概率的负对数似然。当前批量脚本不支持 ForwardGNN 的输出层微调或 BP 预训练迁移。

已有 ForwardGNN 与 BP 的验证结果位于 `results/forwardgnn_validation_summary/report.md`。

## 使用自己的数据

传入一个 `.npz` 文件：

```bash
python train.py --data path/to/graph.npz
```

文件必须包含：

- `x`：`float32 [节点数, 特征数]`
- `edge_index`：`int64 [2, 边数]`，每列是 `[源节点, 目标节点]`
- `y`：`int64 [节点数]`，类别编号应从 0 开始

可选包含布尔数组 `train_mask`、`val_mask`、`test_mask`；三者必须同时提供。若省略，程序会按类别分层生成 20%/20%/60% 切分。对于无向图，请在 `edge_index` 中同时加入两个方向；自环会由模型自动添加。

示例：

```python
import numpy as np

np.savez(
    "graph.npz",
    x=node_features,
    edge_index=edges,
    y=node_labels,
)
```

核心节点分类模型位于 `node_classification/models/`，训练算法位于 `node_classification/training/`，图级任务实现位于 `graph_level/model/graph_level.py`。若任务是异构图，需要相应地替换读出层、损失函数或消息传递结构。

## 无虚拟节点的局部训练扩展

运行 `python -m link_prediction.experiments.local_link_prediction`，比较逐层局部边预测损失与端到端 BP。使用 CitationFull-CiteSeer / Cora_ML，2、4 层 GCN，3 个随机划分；现有相同配置结果自动续用。

协议及实际结果见 [无虚拟节点链接预测实验](docs/sf_without_virtual_nodes_link_prediction.md)。该实验采用独立配置，不等同于原论文链接预测复现。

### 图分类与图回归

运行 `python -m graph_level.experiments.local_graph_tasks`：BACE 分子图二分类、ESOL 分子溶解度回归，按分子骨架分组划分，比较 BP 与不使用虚拟节点的逐层局部训练。默认 2/4 层、隐藏维度 32、三个随机种子，每层局部训练 60 步，BP 匹配总优化器调用数。

依赖 RDKit；初次运行由 PyG 下载 MoleculeNet 数据。每个分子独立消息传递，每层通过 mean/sum 汇聚和辅助预测头获得图级损失。推理使用最终层的预测头。

- 模型及训练：`graph_level/model/graph_level.py`
- 数据划分及实验入口：`graph_level/experiments/local_graph_tasks.py`
- [图分类与回归实验报告](docs/sf_graph_tasks_results.md)
- [三类任务扩展总结与失败案例](docs/sf_task_extensions.md)
- 原始划分、预测、权重和逐轮日志：`results/graph_local_v1/`

完整配置和源文件哈希相同才允许续跑。修改模型或配置时使用新的 `--setting`。只刷新现有报告可运行 `python -m graph_level.experiments.local_graph_tasks --report-only`。

### 等质量时间与内存

`python -m graph_level.experiments.graph_time_to_target` 以已保存 BP 验证性能作为每组共同目标，在独立进程中重新测量 BP/SF 首次达标时间及 RSS 峰值。SF 的低层预训练计入累计时间；未达标会保留为未达标，不参与成功配对的速度比。

详见 [同等验证性能时间与内存报告](docs/sf_time_to_target_memory.md)。这项测试沿用 `results/graph_local_v1` 的划分及超参数，不修改原实验结果。
