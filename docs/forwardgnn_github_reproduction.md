# GitHub 数据集复现

论文数据已经整理；正式 MPS 训练于 2026-09-27 启动。完成状态以[自动更新报告](../results/reproduction/summary/github_nodeclass_mps_memory/report.md)和其中的进度计数为准，不能将启动或短训练通过等同于五折复现完成。

## 实验配置

本轮运行 GCN、GAT 的 BP/SF，深度 1～4、作者五折，隐藏维度 128。采用官方训练代码和完整训练预算：最多 1000 epoch（SF 每层），每 2 epoch 验证、100 次验证无提升早停。训练与推理仅使用 MPS，`PYTORCH_ENABLE_MPS_FALLBACK=0`；数据准备及指标处理在 CPU 上。

主队列共 10 个工作进程：每种骨干 SF 四层一次逐层训练及 BP 四种深度。SF 前缀准确率按官方脚本保存。第二条队列独立运行 SF 深度 1～3，以比较不同深度的进程显存。合计 80 次独立训练、140 条结果记录、16 个显存配置。两条队列并行，运行时间不用于性能比较。

论文整理覆盖 GCN、GraphSAGE、GAT，来源为 [Table 3 和 Table 8](https://arxiv.org/html/2403.11004v1)。GraphSAGE 未纳入本轮实测；FF、top-down 和链接预测也不属于本轮范围。

## 数据恢复

PyG 原始 GitHub 下载地址无法解析，因此从[作者发布的完整边划分文件](https://github.com/NamyongPark/forwardgnn-datasplits)恢复数据。对每折训练、验证、测试正边取并集，并验证五折恢复出的特征、标签和完整正边集合逐项一致。保留作者已经归一化的 128 维特征，没有再次归一化。

恢复后为 37,700 节点、578,006 条有向边、2 类，所有官方节点划分恰好覆盖全部节点。原始边顺序无法恢复，按源/目标编号排序可能造成浮点求和差异。

数据恢复实现见 `paper_reproduction/experiments/prepare_github_data.py`。15 个边划分源文件及缓存的 SHA256 见[数据来源记录](../results/reproduction/github_data_provenance.json)；节点划分哈希和环境见各运行 manifest。每次启动工作进程校验恢复缓存哈希。

## 运行与监控

以下训练命令用于恢复中断的队列；已有同配置任务运行时不应重复启动。训练器会拒绝同配置并发，并跳过已经保存的折。

```bash
conda run --no-capture-output -n gnn-research python -m paper_reproduction.experiments.reproduce_paper --datasets GitHub --backbones GCN GAT --setting github_nodeclass_mps_memory

for depth in 1 2 3; do
  conda run --no-capture-output -n gnn-research python -m paper_reproduction.experiments.reproduce_paper --datasets GitHub --backbones GCN GAT --methods sf --max-layers "$depth" --setting "github_nodeclass_mps_memory_sf_depth$depth" || break
done
```

汇总、绘图并每分钟刷新（全部完成或检测到失败后退出）：

```bash
conda run --no-capture-output -n gnn-research python -m paper_reproduction.experiments.summarize_github --watch
```

主队列显存实时监控；独立深度可替换 `--setting`：

```bash
conda run --no-capture-output -n gnn-research python -m paper_reproduction.experiments.watch_mps_memory --setting github_nodeclass_mps_memory --watch
```

采样间隔 0.5 秒，分别记录张量分配和 Metal 驱动分配，单位 MiB。采样峰值覆盖整个进程的加载、训练、验证和五折过程，可能漏掉瞬时峰值。驱动分配包含缓存和框架分配；论文 H100 的 MB 与此测量口径不同，绝对值之差不作为复现误差。图中的星号表示尚未完成，缺失点不进行插值。

## 产物

- [论文准确率与显存图](../results/reproduction/summary/github_nodeclass_mps_memory/github_paper_results.png)
- [当前 MPS 显存图](../results/reproduction/summary/github_nodeclass_mps_memory/github_mps_memory.png)
- [完整对比报告](../results/reproduction/summary/github_nodeclass_mps_memory/report.md)
- [机器可读进度](../results/reproduction/summary/github_nodeclass_mps_memory/progress.json)

同目录提供 SVG 和显存 JSON。逐折准确率、训练日志、原始显存采样分别位于 `results/reproduction/official/`、`logs/`、`memory/` 下对应 setting 目录。
