"""Plot paper-reported memory and measured MPS memory with distinct scopes."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_reproduction.experiments.watch_mps_memory import latest_sample

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/reproduction"
COLORS = {"bp": "#2463a6", "sf": "#df7635"}
LABELS = {"CitationFull-CiteSeer": "CiteSeer", "CitationFull-Cora_ML": "CoraML"}


def collect_memory(setting):
    rows = []
    for path in sorted((RESULTS / "memory" / setting).glob("*.jsonl")):
        sample = latest_sample(path)
        if sample is None:
            continue
        dataset, suffix = path.stem.split("_GNN", 1)
        model, depth, stamp, pid = ("GNN" + suffix).rsplit("_", 3)
        if model not in tuple(f"{prefix}-{backbone}" for prefix in ("GNN", "GNN_SingleForward") for backbone in ("GCN", "GAT", "SAGE")):
            continue
        summary_path = path.with_suffix(".summary.json")
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        rows.append(dict(dataset=dataset, backbone=model.rsplit("-", 1)[-1], method="sf" if "SingleForward" in model else "bp",
                         layers=int(depth), status=summary.get("status", "incomplete"),
                         tensor_mib=summary.get("sampled_peak_tensor_bytes", sample["sampled_peak_tensor_bytes"]) / 2**20,
                         driver_mib=summary.get("sampled_peak_driver_bytes", sample["sampled_peak_driver_bytes"]) / 2**20,
                         timestamp=stamp, pid=pid, source=str(path)))
    # Do not let a resumed worker that skips training replace an earlier peak.
    groups = {}
    for row in rows:
        key = (row['dataset'], row['backbone'], row['method'], row['layers'])
        groups.setdefault(key, []).append(row)
    combined = []
    for group in groups.values():
        completed = [row for row in group if row['status'] == 'completed']
        candidates = completed or group
        row = dict(candidates[-1])
        for metric in ('tensor_mib', 'driver_mib'):
            row[metric] = max(r[metric] for r in candidates)
        row['source'] = ';'.join(r['source'] for r in candidates)
        combined.append(row)
    return sorted(combined, key=lambda row: (row['dataset'], row['method'], row['layers']))


def style(ax, title, ylabel):
    ax.set(title=title, xlabel="Number of layers", ylabel=ylabel, xticks=[1, 2, 3, 4])
    ax.set_xlim(0.7, 4.5)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"{name}.svg", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', default='core_nodeclass_mps_memory')
    parser.add_argument('--extra-settings', nargs='*', default=[])
    args = parser.parse_args()
    out = RESULTS / 'summary' / args.setting / 'memory_comparison'
    out.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'paper_reproduction/reference/paper_gcn_memory.csv').open() as handle:
        paper = list(csv.DictReader(handle))
    base_manifest = json.loads((RESULTS / f'manifest_{args.setting}.json').read_text())
    for extra in args.extra_settings:
        manifest_path = RESULTS / f'manifest_{extra}.json'
        if not manifest_path.exists():
            continue  # A queued depth may not have started yet.
        other = json.loads(manifest_path.read_text())
        for key in ('datasets', 'backbones', 'runs', 'epochs', 'device', 'memory_interval'):
            if other['configuration'][key] != base_manifest['configuration'][key]:
                raise ValueError(f'Incompatible setting {extra}: {key}')
    measured = collect_memory(args.setting)
    for extra in args.extra_settings:
        measured.extend(collect_memory(extra))
    keys = [(r['dataset'], r['method'], r['layers']) for r in measured]
    if len(set(keys)) != len(keys):
        raise ValueError('Overlapping configurations across settings')
    measured.sort(key=lambda r: (r['dataset'], r['method'], r['layers']))
    sf_complete = all(any(r['dataset'] == d and r['method'] == 'sf' and r['layers'] == depth and r['status'] == 'completed' for r in measured) for d in LABELS for depth in range(1, 5))
    manifest = json.loads((RESULTS / f'manifest_{args.setting}.json').read_text())
    interval = manifest['configuration']['memory_interval']
    snapshot = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    completed_count = sum(r['status'] == 'completed' for r in measured)
    plt.rcParams.update({'font.size': 11, 'axes.titleweight': 'bold', 'figure.titlesize': 17})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (dataset, label) in zip(axes, LABELS.items()):
        for method in COLORS:
            selected = [r for r in paper if r['dataset'] == dataset and r['method'] == method]
            x = [int(r['layers']) for r in selected]
            y = [float(r['memory_mb_as_reported']) for r in selected]
            ax.plot(x, y, 'o-', color=COLORS[method], label=method.upper(), lw=2)
            for depth, value in zip(x, y):
                ax.annotate(f'{value:.2f}', (depth, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
        style(ax, label, 'GPU memory (MB, paper units)')
        ax.set_ylim(0, max(float(r['memory_mb_as_reported']) for r in paper if r['dataset'] == dataset) * 1.24)
        ax.legend(loc='upper center', ncol=2, frameon=False)
    fig.suptitle('Original paper | GCN node classification', y=0.99)
    fig.text(0.5, 0.015, 'Source: arXiv:2403.11004v1, Tables 6 and 8 | H100 | Values as published', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save(fig, out, 'paper_memory')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, (dataset, label) in enumerate(LABELS.items()):
        for i, (metric, title) in enumerate([('tensor_mib', 'Tensor allocation'), ('driver_mib', 'Metal driver allocation')]):
            ax = axes[i, col]
            for method in COLORS:
                selected = [r for r in measured if r['dataset'] == dataset and r['method'] == method]
                completed = [r for r in selected if r['status'] == 'completed']
                ax.plot([r['layers'] for r in completed], [r[metric] for r in completed], 'o-',
                        color=COLORS[method], label=method.upper(), lw=2)
                for row in selected:
                    if row['status'] != 'completed':
                        ax.scatter(row['layers'], row[metric], marker='x', color=COLORS[method], s=65)
                    suffix = '*' if row['status'] != 'completed' else ''
                    ax.annotate(f"{row[metric]:.1f}{suffix}", (row['layers'], row[metric]),
                                xytext=(0, -16 if method == 'bp' else 9), textcoords='offset points', ha='center', fontsize=9)
            style(ax, f'{label} | {title}', 'Sampled process peak (MiB)')
            values = [r[metric] for r in measured if r['dataset'] == dataset]
            ax.set_ylim(0, max(values, default=1) * 1.3)
            ax.legend(loc='upper center', ncol=2, frameon=False)
            if not values:
                ax.text(0.5, 0.5, 'Not measured yet', transform=ax.transAxes, ha='center', color='#666666')
    fig.suptitle('Current reproduction | MPS only, GCN', y=0.99)
    fig.text(0.5, 0.94, f'{snapshot} | Completed depth configurations: {completed_count}/16', ha='center', fontsize=10)
    fig.text(0.5, 0.045, f'{interval:g} s sampling; whole worker, all folds, loading + training + evaluation. Sampled peaks only.', ha='center', fontsize=10)
    fig.text(0.5, 0.018, ('Independent depth measurements; MiB = bytes / 2^20.' if completed_count == 16 else 'Missing depths are not inferred; * incomplete. MiB = bytes / 2^20.'), ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.075, 1, 0.92))
    save(fig, out, 'mps_memory')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, (dataset, label) in enumerate(LABELS.items()):
        for i, method in enumerate(COLORS):
            ax = axes[i, col]
            reference = [r for r in paper if r['dataset'] == dataset and r['method'] == method]
            values = [float(r['memory_mb_as_reported']) for r in reference]
            ax.plot([1, 2, 3, 4], [v / values[0] for v in values], 'o--', color='#555555', label='Paper')
            local = [r for r in measured if r['dataset'] == dataset and r['method'] == method and r['status'] == 'completed']
            first = next((r for r in local if r['layers'] == 1), None)
            if first is not None:
                for metric, color, metric_label in [('tensor_mib', '#2463a6', 'MPS tensor'), ('driver_mib', '#df7635', 'MPS driver')]:
                    ax.plot([r['layers'] for r in local], [r[metric] / first[metric] for r in local], 'o-', color=color, label=metric_label)
            style(ax, f'{label} | {method.upper()}', 'Memory / own depth-1 value')
            maximum = max(float(value) for line in ax.lines for value in line.get_ydata())
            ax.set_ylim(0, max(1.2, maximum * 1.18))
            ax.legend(loc='best', fontsize=9)
    fig.suptitle('Depth scaling | Each series normalized to its own L1', y=0.99)
    fig.text(0.5, 0.02, 'Normalization compares shapes only; it does not align hardware, allocators or measurement scopes.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, 0.055, 1, 0.95))
    save(fig, out, 'memory_depth_scaling')

    with (out / 'mps_memory.csv').open('w') as handle:
        if measured:
            writer = csv.DictWriter(handle, fieldnames=list(measured[0]))
            writer.writeheader()
            writer.writerows(measured)
    (out / 'paper_memory.csv').write_text((ROOT / 'paper_reproduction/reference/paper_gcn_memory.csv').read_text())
    lines = ['# 原论文与 MPS 复现显存比较', '', f'生成时间：{snapshot}；完整显存配置 {completed_count}/16。', '',
             '范围：CiteSeer / CoraML，GCN 节点分类，BP / SF。不是全论文全部骨干和任务。', '',
             f"本机环境：PyTorch {manifest['torch']} / PyG {manifest['pyg']}，{manifest['platform']}，MPS。每配置 {manifest['configuration']['runs']} 折，隐藏维度 128，最多 {manifest['configuration']['epochs']} epoch，每两轮验证、patience=100，采样间隔 {interval:g} 秒。", '',
             '![论文结果](paper_memory.png)', '', '![MPS 实测](mps_memory.png)', '', '![按各自一层归一化](memory_depth_scaling.png)', '',
             '## 数据与口径', '',
             '- 论文数据：[arXiv v1，Tables 6、8](https://arxiv.org/html/2403.11004v1#A5.T6)，保留原文 MB 单位。原文表格及已检查的官方训练代码未明确给出对应采样 API、基线扣除方法和缓存范围，不能假定与当前指标一致。',
             '- [PyTorch MPS API](https://docs.pytorch.org/docs/stable/mps.html)：张量内存不含分配器缓存；驱动分配包含缓存及 MPS/MPSGraph 分配。单位 MiB，按字节 / 2^20 换算。',
             '- 当前值为每个配置进程的采样最大值，包含五折、加载、训练、验证和测试，不是五折均值或精确瞬时峰值。短暂张量尖峰可能漏采，采样曲线不保证单调；驱动曲线还受缓存影响。',
             '- 主复现的 SF 进程运行四层配置；SF L1～L3 使用额外独立 setting、独立进程，仍为五折和完整预算。未测点不插值，未结束 worker 标为 incomplete，数值只是目前下界。',
             '- 训练使用 MPS，关闭算子 CPU fallback。数据加载和汇总在 CPU。主矩阵和 SF 补测各自串行，两个驱动并行运行；记录的是各进程自身的内存，时间不用于性能基准，资源竞争可能影响采样时刻。', '',
             '## 实测数据', '', '| 数据 | 方法 | 层数 | 张量峰值 MiB | 驱动峰值 MiB | 状态 |', '|---|---|---:|---:|---:|---|']
    for r in measured:
        lines.append(f"| {LABELS.get(r['dataset'], r['dataset'])} | {r['method'].upper()} | {r['layers']} | {r['tensor_mib']:.2f} | {r['driver_mib']:.2f} | {r['status']} |")
    lines += ['', '## 比较', '',
              '论文中 BP 的 L4/L1 倍数为 CiteSeer 17.71、CoraML 15.16；SF 在两数据集均为 1.00。这里 SF 的绝对值仍高于 BP，稳定的深度趋势不等于绝对显存更低。', '']
    for dataset, label in LABELS.items():
        complete = {r['layers']: r for r in measured if r['dataset'] == dataset and r['method'] == 'bp' and r['status'] == 'completed'}
        if 1 in complete and 4 in complete:
            a, b = complete[1], complete[4]
            lines.append(f"{label} 的 MPS BP 从 L1 到 L4：张量采样峰值 {a['tensor_mib']:.2f} → {b['tensor_mib']:.2f} MiB（{b['tensor_mib']/a['tensor_mib']:.2f} 倍）；驱动峰值 {a['driver_mib']:.2f} → {b['driver_mib']:.2f} MiB（{b['driver_mib']/a['driver_mib']:.2f} 倍）。")
        else:
            lines.append(f'{label} 尚缺完整 BP L1/L4 显存，暂不比较增长倍数。')
    if sf_complete:
        for dataset, label in LABELS.items():
            rows = {r['layers']: r for r in measured if r['dataset'] == dataset and r['method'] == 'sf'}
            a, b = rows[1], rows[4]
            lines.append(f"{label} 的 MPS SF L4/L1：张量峰值 {b['tensor_mib']/a['tensor_mib']:.2f} 倍，驱动峰值 {b['driver_mib']/a['driver_mib']:.2f} 倍。")
    else:
        lines.append('本机 SF 的独立深度测量尚未全部完成，暂不判断其深度趋势。')
    lines += ['', '硬件、软件、单位和测量范围不同，不计算“对论文显存误差百分比”，也不据此宣称复现或推翻 H100 显存结论。', '',
              '重绘命令：`conda run -n gnn-research python -m paper_reproduction.experiments.plot_memory_comparison --setting ' + args.setting + (' --extra-settings ' + ' '.join(args.extra_settings) if args.extra_settings else '') + '`', '',
              '原始测量来源见 [MPS CSV](mps_memory.csv) 的 source 列；原论文数值见 [论文 CSV](paper_memory.csv)。', '', '矢量图：[论文](paper_memory.svg)、[MPS](mps_memory.svg)、[深度归一化比较](memory_depth_scaling.svg)。']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print(out)


if __name__ == '__main__':
    main()
