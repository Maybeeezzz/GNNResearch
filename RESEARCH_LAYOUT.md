# 当前目录

- `paper_reproduction/experiments/`：MPS 官方训练入口、监控、数据适配、论文表格提取和绘图。
- `paper_reproduction/reference/`：论文结构化数据及固定版本官方 HTML；数值带表号、来源锚点。
- `paper_reproduction/tests/`：显存记录、准确率单位、恢复数据与表格提取校验。
- `results/paper_reproduction/published/`：论文发布的数值和图表。
- `results/paper_reproduction/official/`：本机 MPS 执行官方代码的原始结果。
- `results/paper_reproduction/{memory,logs,summary}/`：MPS 采样、日志与汇总。
- `results/sf_vn_ablation_mps/`：独立 MPS 消融结果。
- `node_classification/`：MPS 消融入口及其共享数据、模型、训练和单元测试依赖。
- `third_party/`：作者官方代码、数据划分与运行所需数据缓存。
- `mechanism_studies/`：与保留的 MPS 消融相关的机制研究说明。
- `docs/`：现行协议、报告索引、清理清单和代码规范。

早期 CPU、pilot/staged、自定义图级和链接预测入口及产物已移除。清理文件和 SHA256 见 [cleanup_20260927.json](docs/cleanup_20260927.json)。通用单元测试使用小型 CPU 张量校验算法，不属于论文 CPU 训练实验。
