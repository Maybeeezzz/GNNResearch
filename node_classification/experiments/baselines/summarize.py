"""Aggregate JSON runs into CSV tables, plots, and a Markdown report."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/gnn-xdg-cache")
import matplotlib.pyplot as plt
import pandas as pd


GROUP = [
    "dataset", "model", "method", "scope", "pretrain_steps",
    "pretrain_fraction", "learning_rate", "steps", "layers", "hidden",
    "step_unit", "temperature", "edge_direction", "dropout", "eval_every",
]
METRICS = [
    "initial_test_accuracy", "accuracy_delta", "test_accuracy", "test_macro_f1", "val_accuracy", "wall_seconds",
    "milliseconds_per_step", "peak_rss_mb", "incremental_peak_rss_mb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/summary"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    curves = []
    for directory in args.inputs:
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text())
            record.setdefault("step_unit", "full_network_update")
            record.setdefault("temperature", None)
            record.setdefault("edge_direction", None)
            record["initial_test_accuracy"] = record["history"][0]["test_accuracy"]
            record["accuracy_delta"] = (
                record["test_accuracy"] - record["initial_test_accuracy"]
            )
            record["source_file"] = str(path)
            records.append({key: value for key, value in record.items() if key != "history"})
            for point in record["history"]:
                curves.append({
                    **{key: record[key] for key in GROUP + ["seed"]},
                    **point,
                })
    if not records:
        raise ValueError("no JSON result files found")

    args.output.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(records)
    curve_frame = pd.DataFrame(curves)
    aggregations = {metric: ["mean", "std"] for metric in METRICS}
    summary = runs.groupby(GROUP, dropna=False).agg(aggregations).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple) else column
        for column in summary.columns
    ]
    runs.to_csv(args.output / "runs.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)
    curve_frame.to_csv(args.output / "curves.csv", index=False)

    # Historical plots assume fixed depth/budgets and full-network step units.
    # Do not mix local layer updates with those axes.
    if "forwardgnn" not in set(runs["method"]):
        plot_gcn_curves(curve_frame, args.output / "gcn_convergence.png")
        plot_deep_sensitivity(summary, args.output / "deep_gcn_sensitivity.png")
    write_report(runs, summary, args.output / "report.md")
    print(f"runs={len(runs)} groups={len(summary)} output={args.output}")


def plot_gcn_curves(curves: pd.DataFrame, output: Path) -> None:
    selected = curves[(curves["model"] == "gcn") & (curves["scope"] == "all")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey="row")
    for row, pretrain_steps in enumerate((0, 200)):
        for column, dataset in enumerate(("Cora", "CiteSeer")):
            axis = axes[row, column]
            panel = selected[
                (selected["dataset"] == dataset)
                & (selected["pretrain_steps"] == pretrain_steps)
            ]
            for method, method_frame in panel.groupby("method"):
                stats = method_frame.groupby("step")["test_accuracy"].agg(["mean", "std"])
                axis.plot(stats.index, stats["mean"], label=method)
                axis.fill_between(
                    stats.index,
                    stats["mean"] - stats["std"].fillna(0),
                    stats["mean"] + stats["std"].fillna(0),
                    alpha=0.16,
                )
            regime = "scratch" if pretrain_steps == 0 else "staged fine-tuning"
            axis.set_title(f"{dataset}: {regime}")
            axis.set_ylabel("test accuracy")
            axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("optimization step")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_deep_sensitivity(summary: pd.DataFrame, output: Path) -> None:
    panel = summary[
        (summary["model"] == "residual-gcn")
        & (summary["pretrain_steps"] == 200)
    ].copy()
    if panel.empty:
        return
    panel["label"] = panel.apply(
        lambda row: f"{row['method']}\nlr={row['learning_rate']:g}", axis=1
    )
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(
        panel["label"], panel["test_accuracy_mean"],
        yerr=panel["test_accuracy_std"].fillna(0), capsize=5,
    )
    axis.set_ylabel("test accuracy (mean ± std)")
    axis.set_title("Residual GCN: learning-rate sensitivity")
    axis.set_ylim(0, 0.9)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(runs: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    def table(frame: pd.DataFrame) -> str:
        columns = [
            "dataset", "model", "method", "layers", "hidden", "steps", "step_unit", "learning_rate",
            "initial_test_accuracy_mean", "accuracy_delta_mean",
            "test_accuracy_mean", "test_accuracy_std", "test_macro_f1_mean",
            "wall_seconds_mean", "peak_rss_mb_mean",
        ]
        shown = frame[columns].copy()
        for column in columns[8:]:
            shown[column] = shown[column].map(lambda value: f"{value:.4f}")
        headers = list(shown.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in shown.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return "\n".join(lines)

    text = f"""# GNN 训练方法性能实验报告

## 实验范围

- 运行数：{len(runs)}
- 数据集：{', '.join(sorted(set(runs['dataset'])))}，Planetoid 公开划分
- 方法：{', '.join(sorted(set(runs['method'])))}
- 种子：{', '.join(str(seed) for seed in sorted(set(runs['seed'])))}
- 各配置的实际深度、宽度、预算和学习率见下表及 runs.csv；标准差为样本标准差，单次运行的标准差为空。

## 实测结果

{table(summary)}

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
"""
    output.write_text(text)


if __name__ == "__main__":
    main()
