# 项目脚本命名与 Python 代码规范

本约定适用于项目自有代码：根目录脚本，以及按任务分类的 `node_classification/`、`graph_level/`、`link_prediction/` 和 `paper_reproduction/`。旧 `gnn/`、`experiments/` 入口已移除。`third_party/` 是上游代码或数据仓库，保留其原始命名和风格，不做批量重命名/格式化，以维持来源可追溯性。

## 文件与标识符命名

- Python 文件统一使用小写 `snake_case.py`。所有当前项目自有 Python 文件都已满足这一形式，因此不为形式统一而改动已被 README、命令行或导入路径引用的文件名。
- 新脚本用动词说明职责：`train_<scope>.py`、`benchmark_<metric>.py`、`reproduce_<paper_or_result>.py`、`summarize_<result>.py`；实验模块以任务/比较对象命名，如 `local_link_prediction.py`、`compare_sf_ff_graph_tasks.py`。
- 测试文件使用 `test_<module_or_behavior>.py`，并与实现模块保持对应。
- 函数、变量使用 `snake_case`；类使用 `PascalCase`；模块常量使用 `UPPER_SNAKE_CASE`。缩写在文件名中小写（`gnn`、`sf`、`ff`），在正文/报告中可按惯例大写。
- 命令行公开入口保留 `main()` 和 `if __name__ == "__main__":`，不要让导入模块触发训练或数据下载。

## 格式与静态检查

- Python 3.11；UTF-8；LF；4 空格缩进；Ruff formatter 行宽 88、双引号。
- 导入按标准库、第三方依赖、项目内部模块分组；用绝对项目导入或包内相对导入保持同一模块内一致。
- 公共函数和复杂实验入口标注参数/返回类型；路径用 `pathlib.Path`；配置和结果记录优先使用明确的数据结构/JSON 键，不依赖位置隐含约定。
- `pyproject.toml` 是 Ruff 格式和 lint 规则的唯一配置源；第三方上游目录不参加 Ruff 检查。

在环境中安装开发工具后，可以对项目自有 Python 文件统一格式化和检查：

```bash
python -m pip install -r requirements-dev.txt
ruff format .
ruff check --select I --fix .
ruff check .
```

`ruff format .` 遵循配置并排除 `third_party/`。本次没有运行格式化器、静态检查或测试；当前执行环境没有安装 Ruff，因此先提交了统一配置和约定，而没有声称已有全部源码经过自动格式化。

## 训练方法术语

- `bp`：端到端反向传播。
- `ff`：Forward-Forward；如果是图级实验里的 FF-style 适配，应在脚本/报告中注明样本构造与 goodness 定义。
- `sf` / `forwardgnn`：逐层局部 Single-Forward 学习。
- `single-forward`（验证前向复用）：项目旧 BP 实现中复用训练输出计算验证损失，不与 SF/ForwardGNN 混称。

结果目录和 CLI 中保留已有公开配置名，方法值统一使用小写 `bp`、`ff`、`sf`；需要区分 SF 目标时采用限定名（如 `sf_ce`、`sf_ff_objective`），不把目标函数变体误标成另一种完整训练算法。
