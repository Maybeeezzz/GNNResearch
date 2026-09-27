# 官方代码复现操作说明

## 文件

- `docs/forwardgnn_paper_summary.md`：论文结果及解释。
- `docs/forwardgnn_reproduction_results_core_nodeclass_mps_memory.md`：实际执行结果，与论文锚点比较。
- `paper_reproduction/experiments/reproduce_paper.py`：串行启动官方实验，支持官方已有结果续跑。
- `paper_reproduction/experiments/summarize_paper.py`：统一准确率单位，检查五折完整性，计算总体标准差。
- `third_party/forwardgnn`：官方代码，保留原许可证。
- `third_party/forwardgnn-datasplits`：作者发布的原始划分，保留原来源。
- `results/paper_reproduction`：配置、文件哈希、逐折原始 JSON 和完整日志。

## 执行

```bash
conda activate gnn-research
python -m paper_reproduction.experiments.reproduce_paper --device mps --setting core_nodeclass_mps_memory
python -m paper_reproduction.experiments.summarize_paper --setting core_nodeclass_mps_memory
```

默认重跑使用 MPS；MPS 不可用时会直接报错，不会回退到 CPU。旧 CPU 结果已清理，现行完整精度及显存结果位于 `core_nodeclass_mps_memory`。若在受限 shell 中运行 MPS，进程必须获得 Metal/MPS 设备访问权限。

默认矩阵是 CitationFull-CiteSeer / CitationFull-Cora_ML，GCN，BP 与 SF，1～4 层，每项 5 折。BP 各深度独立训练；SF 按官方脚本训练 4 层，逐层保存前缀结果。共 40 个 BP 训练和 10 个四层 SF 训练，形成 80 条深度结果，不能称为 80 次独立训练。

MPS 短训练检查不纳入正式汇总。正式预算为：最多 1000 epoch，patience=100 个验证检查，每 2 epoch 验证。

更换数据、骨干或预算时使用独立的 `--setting`，防止与已有结果混用。非默认数据需要先用官方数据加载器下载。现有运行只覆盖核心节点分类，不包含全论文矩阵。

## 最小兼容修改

官方标准 GCN/SAGE/GAT 的导入被未使用的 legacy CachedGCN/CachedSAGE 的扩展依赖阻塞。新增 `src/gnn/optional_cached.py`，并调整 `src/gnn/__init__.py` 和 `src/gnn/gnn_conv.py` 的导入。

如果旧缓存算子无法导入，用显式报错的占位类保留类型检查接口；若真正请求 Cached 模型则立即报错，不自动替换其算法。正式脚本选择的 GCN 使用当前 PyG 的 GCNConv，符合官方脚本的算子选择。未修改官方损失、初始化流程、早停或训练循环。

软件版本与论文不同，因此这是作者原始代码在新环境的复现，不是冻结原始环境的位级重放。MPS 采样口径不能等同于 H100 显存。官方 SF 训练中记录测试指标，但早停由验证准确率决定；本次不根据测试结果选超参数。

## 指标口径

官方 BP JSON 的 `perf` 是比例，SF 是百分数；汇总统一为百分数。标准差 ddof=0，与官方 NumPy 汇总代码一致。每条原始记录保留 run_i、run_seed、层数、训练预算和训练时间。

SF 的 `train_time` 是训练到该前缀层的累计墙钟，包含其评估开销；不能将其与 BP 不同计时范围简单相除宣称速度优势。比较论文准确率时只使用恢复最佳验证权重后的测试结果。

本次尾段将 Cora_ML 的 3、4 层 BP 与其他独立配置并行执行，每个进程限制为 1 个 CPU 线程。worker 使用配置级文件锁防止同一实验重复写入；默认驱动仍按顺序启动。资源竞争也使本次墙钟时间不适合作为严格速度基准。

## 仅 MPS 与显存监控

复现入口现在只接受 `--device mps`，MPS 不可用就失败。每个新 worker 在导入
PyTorch 前强制 `PYTORCH_ENABLE_MPS_FALLBACK=0`，不支持的 MPS 算子会报错。
数据加载、划分读取及指标汇总仍在 CPU；模型训练与推理使用 MPS。

带监控的正式重跑使用独立 setting，避免直接跳过原来已完成但没有显存记录的训练：

```bash
conda run --no-capture-output -n gnn-research python -m paper_reproduction.experiments.reproduce_paper --device mps --setting core_nodeclass_mps_memory
conda run --no-capture-output -n gnn-research python -m paper_reproduction.experiments.watch_mps_memory --setting core_nodeclass_mps_memory --watch
conda run -n gnn-research python -m paper_reproduction.experiments.summarize_paper --setting core_nodeclass_mps_memory
```

第一条命令默认使用上述两个数据集、GCN、1～4 层、五折及正式训练预算。
`--memory-interval 0.5` 控制采样间隔，单位秒。第二条命令每两秒显示已记录进程的
当前用量和采样峰值，Ctrl-C 只退出查看，不会停止训练。

数据写入 `results/paper_reproduction/memory/<setting>/`。每次 worker 执行创建独立
时间戳 JSONL，逐条刷新，可在训练过程中读取；结束或 Python 异常时另存
`.summary.json`，记录 completed/failed。强制终止可能没有汇总，此时查看器只显示
最后采样的时间，不把陈旧记录当作仍在运行。续跑不覆盖既有采样；跳过的训练
不会重新产生训练显存，不能将该次低用量当作完整训练测量。

- `tensor_bytes`：当前 MPS 张量分配，不含分配器缓存。
- `driver_bytes`：本进程 Metal 驱动分配，包括缓存和 MPS/MPSGraph 分配。
- `recommended_max_bytes`：Metal 建议的最大工作集，不是剩余显存。
- `sampled_peak_*`：上述计数器的采样最大值，只是实际瞬时峰值的下界。

口径依据 [PyTorch MPS API](https://docs.pytorch.org/docs/stable/mps.html)。
采样来自训练进程内部，不是另一个空进程；后台线程不主动同步 GPU、不清理缓存。
范围为整个 worker，包含五折、数据加载和评估；SF 包含全部前缀层，不能将其
解释为某一折或单独一层的峰值。Apple 统一内存与论文 H100 显存口径不同。
这次重跑仍是核心节点分类矩阵，不等于完整论文全部实验。

绘图比较时，为测量 SF 的完整深度曲线，另以 `--methods sf --max-layers N`
运行 N=1、2、3；setting 分别为 `core_nodeclass_mps_memory_sf_depthN`。
这些实验保持五折、1000 epoch 上限及原早停规则，独立初始化、独立进程；其显存
用于深度比较，准确率主汇总仍使用四层 SF 的官方前缀结果。完成后重绘：

```bash
conda run -n gnn-research python -m paper_reproduction.experiments.plot_memory_comparison --extra-settings core_nodeclass_mps_memory_sf_depth1 core_nodeclass_mps_memory_sf_depth2 core_nodeclass_mps_memory_sf_depth3
```

生成的 PNG、SVG、CSV 与中文比较报告位于
`results/paper_reproduction/summary/core_nodeclass_mps_memory/memory_comparison/`。
