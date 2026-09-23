"""Summarize only official-code replication records, with explicit units/coverage."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/paper_reproduction"
# Published GCN node-classification anchors (percent), Tables 4 and 5.
PAPER = {
    ("CitationFull-CiteSeer", "bp", 1): (84.28, 3.0),
    ("CitationFull-CiteSeer", "bp", 2): (94.28, 0.7),
    ("CitationFull-CiteSeer", "bp", 4): (94.87, 0.5),
    ("CitationFull-CiteSeer", "bp", 3): (94.89, 0.8),
    ("CitationFull-CiteSeer", "sf", 1): (92.60, 0.7),
    ("CitationFull-CiteSeer", "sf", 2): (94.18, 0.7),
    ("CitationFull-CiteSeer", "sf", 4): (94.66, 0.6),
    ("CitationFull-CiteSeer", "sf", 3): (94.33, 0.7),
    ("CitationFull-Cora_ML", "bp", 1): (31.72, 4.8),
    ("CitationFull-Cora_ML", "bp", 2): (86.84, 1.0),
    ("CitationFull-Cora_ML", "bp", 4): (86.61, 2.3),
    ("CitationFull-Cora_ML", "bp", 3): (88.65, 1.4),
    ("CitationFull-Cora_ML", "sf", 1): (87.75, 1.4),
    ("CitationFull-Cora_ML", "sf", 2): (87.95, 1.4),
    ("CitationFull-Cora_ML", "sf", 4): (88.48, 1.2),
    ("CitationFull-Cora_ML", "sf", 3): (88.15, 1.5),
}


def collect(setting):
    groups = defaultdict(list)
    for path in sorted((RESULTS / "official" / setting).glob("*/node-class/*.json")):
        row = json.loads(path.read_text())
        method = "sf" if "SingleForward" in row["model"] else "bp"
        backbone = row["model"].split("-")[-1]
        # Upstream BP writes fractions while SF writes percentages.
        row["accuracy_percent"] = row["perf"] * (100 if method == "bp" else 1)
        if not 0 <= row["accuracy_percent"] <= 100:
            raise ValueError(f"Invalid accuracy units in {path}")
        if row["run_seed"] != 10100 + 3 * row["run_i"]:
            raise ValueError(f"Unexpected seed in {path}")
        key = (row["dataset"], backbone, method, row["num_layers"])
        groups[key].append(row)
    return groups


def plot_results(summaries, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/gnn-paper-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = sorted({(r['dataset'], r['backbone']) for r in summaries})
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.5), squeeze=False)
    for axis, (dataset, backbone) in zip(axes[0], panels):
        for method, color in (("bp", "#3454a5"), ("sf", "#c05c19")):
            points = [r for r in summaries if r['dataset'] == dataset and r['backbone'] == backbone and r['method'] == method]
            axis.errorbar([r['depth'] for r in points], [r['mean_percent'] for r in points],
                          yerr=[r['std_percent_ddof0'] for r in points], fmt='o-', capsize=4,
                          label=f"{method.upper()} local (mean +/- std)", color=color)
            anchors = [r for r in points if r['paper_mean_percent'] is not None]
            axis.scatter([r['depth'] for r in anchors], [r['paper_mean_percent'] for r in anchors],
                         marker='D', facecolors='none', edgecolors=color, s=70,
                         label=f"{method.upper()} paper anchors")
        axis.set_title(f"{dataset.replace('CitationFull-', '')} / {backbone}")
        axis.set_xlabel("Number of GNN layers")
        axis.set_ylabel("Test accuracy (%)")
        axis.set_xticks(sorted({r['depth'] for r in summaries}))
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Official-code reproduction: five author splits, CPU")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", default="core_nodeclass")
    args = parser.parse_args()
    manifest = json.loads((RESULTS / f"manifest_{args.setting}.json").read_text())
    config = manifest["configuration"]
    groups = collect(args.setting)
    summaries = []
    for dataset in config["datasets"]:
        for backbone in config["backbones"]:
            for depth in range(1, config["max_layers"] + 1):
                for method in ("bp", "sf"):
                    rows = groups.get((dataset, backbone, method, depth), [])
                    ids = [r["run_i"] for r in rows]
                    if len(ids) != len(set(ids)):
                        raise ValueError("Duplicate split records")
                    complete = set(ids) == set(range(config["runs"]))
                    values = [r["accuracy_percent"] for r in rows]
                    anchor = PAPER.get((dataset, method, depth)) if backbone == "GCN" else None
                    mean = statistics.mean(values) if values else None
                    summaries.append({
                        "dataset": dataset, "backbone": backbone, "method": method,
                        "depth": depth, "n": len(rows), "complete": complete,
                        "mean_percent": mean,
                        "std_percent_ddof0": statistics.pstdev(values) if values else None,
                        "paper_mean_percent": anchor[0] if anchor else None,
                        "paper_std_percent": anchor[1] if anchor else None,
                        "delta_percentage_points": mean - anchor[0] if values and anchor else None,
                        "mean_logged_seconds": statistics.mean(r["train_time"] for r in rows) if rows else None,
                        "splits": sorted(ids),
                    })
    output = RESULTS / "summary" / args.setting
    output.mkdir(parents=True, exist_ok=True)
    (output / "accuracy_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    lines = ["# ForwardGNN：本项目复现结果报告", "",
             "本文记录本机实际执行的复现配置、测试结果及局限。原论文的独立总结见 [原论文结果总结](forwardgnn_paper_summary.md)；下表论文数值仅作为复现对照，不是本机测量值。", "",
             f"配置：`{args.setting}`。状态：" + ("当前指定矩阵已完成。" if all(r['complete'] for r in summaries) else "正在运行，以下仅为已完成记录。"), "",
             "## 范围与可追溯性", "",
             "本次结果与此前使用 Planetoid 公开划分、两层 64 维模型、100 步等设置的项目实验分开保存；旧实验与 integration_smoke 流程检查均未纳入汇总。", "",
             f"- 数据：{', '.join(config['datasets'])}；骨干：{', '.join(config['backbones'])}；深度 1～{config['max_layers']}。",
             f"- 每个配置 {config['runs']} 个作者原始划分；隐藏维度 128，Adam lr=0.001、weight_decay=0.0005。",
             f"- 最多 {config['epochs']} epoch，SF 为每层预算；每 2 epoch 验证，100 次验证未提升则早停，恢复最佳验证准确率权重。",
             "- 使用官方训练代码；只增加未使用的 legacy cached 算子的可选导入适配，标准 GCN/SAGE/GAT 保持官方调用。",
             f"- 代码提交：`{manifest['upstream_commit']}`；划分提交：`{manifest['split_commit']}`。",
             f"- 环境：{manifest['platform']}，PyTorch {manifest['torch']}，PyG {manifest['pyg']}，每个训练进程 CPU 单线程；部分独立配置并行执行。",
             "- 原始 JSON 与日志位于 `results/paper_reproduction/official/` 和 `logs/`；manifest 包含划分文件 SHA256。",
             "- 每个 SF 四层训练过程保存 1～4 层前缀的结果。这些前缀不是独立初始化的四次训练，遵循官方脚本。",
             "", "## 准确率与论文锚点", "",
             "所有准确率均为百分数。标准差使用总体标准差 ddof=0，与官方汇总代码一致；不是置信区间。未完成五折的行不能当作最终结果。", "",
             "| 数据 | 骨干 | 方法 | 层数 | 完成折数 | 本机均值 ± 标准差 | 论文均值 ± 标准差 | 差值（百分点） |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        local = f"{row['mean_percent']:.2f} ± {row['std_percent_ddof0']:.2f}" if row['n'] else "待完成"
        paper = f"{row['paper_mean_percent']:.2f} ± {row['paper_std_percent']:.1f}" if row['paper_mean_percent'] is not None else "未摘录"
        delta = f"{row['delta_percentage_points']:+.2f}" if row['delta_percentage_points'] is not None else "—"
        lines.append(f"| {row['dataset']} | {row['backbone']} | {row['method']} | {row['depth']} | {row['n']}/{config['runs']} | {local} | {paper} | {delta} |")
    lines += ["", "论文锚点来源：[附录 E 的 Tables 4、5](https://arxiv.org/html/2403.11004v1#A5.SS1)。差值仅描述数值接近程度，不是显著性检验。", "",
              "## 结论与未覆盖项", "",
              "本报告仅验证上述核心节点分类矩阵，不宣称复现完整论文。尚未覆盖其他数据/骨干组合、FF 对照、top-down 扩展、链接预测及 H100 显存结果。",
              "CPU 没有提供论文 GPU 显存指标。日志时间保留供审计，但 SF 含逐层验证/测试等额外开销，BP 的计时范围不同，不以两者原始时间比宣称优化器加速。软件版本也与论文不同。", ""]
    lines += ["论文使用 PyTorch 1.13.1 / PyG 2.2.0 / H100，本机环境不同；本次使用官方准确率脚本的标准算子，未复现 Table 15 的缓存邻域聚合加速实验。", ""]
    if all(r['complete'] for r in summaries):
        gaps = [abs(r['delta_percentage_points']) for r in summaries if r['delta_percentage_points'] is not None]
        if gaps:
            lines += [f"本次与论文对应的 {len(gaps)} 项配置中，平均测试准确率的最大绝对差为 {max(gaps):.3f} 个百分点。", ""]
        if config['runs'] == 5:
            plot = output / "accuracy_vs_depth.png"
            plot_results(summaries, plot)
            lines += ["", f"![深度与准确率]({plot})", ""]
        lines += ["在本次完成的相同深度对照中：", ""]
        for row in summaries:
            if row['method'] != 'sf':
                continue
            baseline = next(r for r in summaries if r['dataset'] == row['dataset'] and r['backbone'] == row['backbone'] and r['depth'] == row['depth'] and r['method'] == 'bp')
            difference = row['mean_percent'] - baseline['mean_percent']
            lines.append(f"- {row['dataset']} / {row['backbone']} / {row['depth']} 层：SF − BP = {difference:+.2f} 个百分点。")
        lines += ["", "结论仅限这些配置；应结合上表逐项判断与论文是否一致。"]
    report = ROOT / "docs/forwardgnn_reproduction_results.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"records={sum(r['n'] for r in summaries)} complete_groups={sum(r['complete'] for r in summaries)}/{len(summaries)} report={report}")


if __name__ == "__main__":
    main()
