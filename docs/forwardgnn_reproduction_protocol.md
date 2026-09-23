# 官方代码复现操作说明

## 文件

- `docs/forwardgnn_paper_summary.md`：论文结果及解释。
- `docs/forwardgnn_reproduction_results.md`：实际执行结果，与论文锚点比较。
- `paper_reproduction/experiments/reproduce_paper.py`：串行启动官方实验，支持官方已有结果续跑。
- `paper_reproduction/experiments/summarize_paper.py`：统一准确率单位，检查五折完整性，计算总体标准差。
- `third_party/forwardgnn`：官方代码，保留原许可证。
- `third_party/forwardgnn-datasplits`：作者发布的原始划分，保留原来源。
- `results/paper_reproduction`：配置、文件哈希、逐折原始 JSON 和完整日志。

## 执行

```bash
conda activate gnn-research
python -m paper_reproduction.experiments.reproduce_paper
python -m paper_reproduction.experiments.summarize_paper
```

默认矩阵是 CitationFull-CiteSeer / CitationFull-Cora_ML，GCN，BP 与 SF，1～4 层，每项 5 折。BP 各深度独立训练；SF 按官方脚本训练 4 层，逐层保存前缀结果。共 40 个 BP 训练和 10 个四层 SF 训练，形成 80 条深度结果，不能称为 80 次独立训练。

先前 Planetoid 实验、2 epoch 的 `integration_smoke` 流程检查均不纳入正式汇总。正式预算不因本机 CPU 较慢而减少：最多 1000 epoch，patience=100 个验证检查，每 2 epoch 验证。

更换数据、骨干或预算时使用独立的 `--setting`，防止与已有结果混用。非默认数据需要先用官方数据加载器下载。现有运行只覆盖核心节点分类，不包含全论文矩阵。

## 最小兼容修改

官方标准 GCN/SAGE/GAT 的导入被未使用的 legacy CachedGCN/CachedSAGE 的扩展依赖阻塞。新增 `src/gnn/optional_cached.py`，并调整 `src/gnn/__init__.py` 和 `src/gnn/gnn_conv.py` 的导入。

如果旧缓存算子无法导入，用显式报错的占位类保留类型检查接口；若真正请求 Cached 模型则立即报错，不自动替换其算法。正式脚本选择的 GCN 使用当前 PyG 的 GCNConv，符合官方脚本的算子选择。未修改官方损失、初始化流程、早停或训练循环。

软件版本与论文不同，因此这是作者原始代码在新环境的复现，不是冻结原始环境的位级重放。CPU 无法验证 H100 显存。官方 SF 训练中记录测试指标，但早停由验证准确率决定；本次不根据测试结果选超参数。

## 指标口径

官方 BP JSON 的 `perf` 是比例，SF 是百分数；汇总统一为百分数。标准差 ddof=0，与官方 NumPy 汇总代码一致。每条原始记录保留 run_i、run_seed、层数、训练预算和训练时间。

SF 的 `train_time` 是训练到该前缀层的累计墙钟，包含其评估开销；不能将其与 BP 不同计时范围简单相除宣称速度优势。比较论文准确率时只使用恢复最佳验证权重后的测试结果。

本次尾段将 Cora_ML 的 3、4 层 BP 与其他独立配置并行执行，每个进程限制为 1 个 CPU 线程。worker 使用配置级文件锁防止同一实验重复写入；默认驱动仍按顺序启动。资源竞争也使本次墙钟时间不适合作为严格速度基准。
