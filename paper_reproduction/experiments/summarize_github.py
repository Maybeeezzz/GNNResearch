"""Generate GitHub paper/local figures and a live, explicitly partial report."""
import argparse
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/gnn-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paper_reproduction.experiments.plot_memory_comparison import collect_memory
from paper_reproduction.experiments.summarize_paper import collect

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results/paper_reproduction'
SETTING = 'github_nodeclass_mps_memory'
SETTINGS = [SETTING] + [f'{SETTING}_sf_depth{i}' for i in (1, 2, 3)]
BACKBONES = ('GCN', 'GAT')
COLORS = {'bp': '#2463a6', 'sf': '#df7635'}


def save(fig, output, name):
    for extension in ('png', 'svg'):
        fig.savefig(output / f'{name}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def format_axis(ax, title, ylabel):
    ax.set(title=title, xlabel='Number of layers', ylabel=ylabel, xticks=[1, 2, 3, 4])
    ax.set_xlim(.7, 4.4)
    ax.grid(axis='y', alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)


def reference_figures(paper, output):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, backbone in enumerate(('GCN', 'SAGE', 'GAT')):
        for method in COLORS:
            rows = [r for r in paper if r['backbone'] == backbone and r['method'] == method]
            x = [r['layers'] for r in rows]
            axes[0, col].errorbar(x, [r['accuracy_percent'] for r in rows],
                                 yerr=[r['std_percent'] for r in rows], fmt='o-',
                                 capsize=4, color=COLORS[method], label=method.upper())
            axes[1, col].plot(x, [r['memory_mb'] for r in rows], 'o-',
                              color=COLORS[method], label=method.upper())
        format_axis(axes[0, col], f'GitHub / {backbone}', 'Accuracy (%)')
        format_axis(axes[1, col], f'GitHub / {backbone}', 'GPU memory (MB, paper units)')
        axes[0, col].set_ylim(65, 91)
        axes[1, col].set_ylim(bottom=0)
        axes[0, col].legend()
        axes[1, col].legend()
    fig.suptitle('Original paper | GitHub node classification', fontsize=18)
    fig.text(.5, .02, 'arXiv:2403.11004v1 | Table 3: five-run accuracy mean +/- std | Table 8: H100 memory', ha='center')
    fig.tight_layout(rect=(0, .05, 1, .94))
    save(fig, output, 'github_paper_results')


def snapshot():
    output = RESULTS / 'summary' / SETTING
    output.mkdir(parents=True, exist_ok=True)
    paper = json.loads((ROOT / 'paper_reproduction/reference/github_paper_results.json').read_text())['rows']
    groups = collect(SETTING)
    memory = [row for setting in SETTINGS for row in collect_memory(setting) if row['dataset'] == 'GitHub']
    memory.sort(key=lambda r: (r['backbone'], r['method'], r['layers']))
    complete_groups = sum(len(groups.get(('GitHub', b, m, d), [])) == 5
                          for b in BACKBONES for m in COLORS for d in range(1, 5))
    completed_memory = sum(r['status'] == 'completed' for r in memory)
    failures = [r for r in memory if r['status'] == 'failed']
    fits = 0
    for setting in SETTINGS:
        manifest_path = RESULTS / f'manifest_{setting}.json'
        if not manifest_path.exists():
            continue
        config = json.loads(manifest_path.read_text())['configuration']
        for (dataset, backbone, method, depth), rows in collect(setting).items():
            if method == 'bp' or depth == config['max_layers']:
                fits += len(rows)
    completed = complete_groups == 16 and completed_memory == 16
    state = 'complete' if completed else ('failed' if failures else 'running')
    status = dict(updated_utc=datetime.now(timezone.utc).isoformat(), state=state,
                  saved_independent_fits=fits, expected_independent_fits=80,
                  complete_accuracy_groups=complete_groups, expected_accuracy_groups=16,
                  completed_memory_configurations=completed_memory, expected_memory_configurations=16,
                  failures=failures)
    temp = output / 'progress.json.tmp'
    temp.write_text(json.dumps(status, indent=2) + '\n')
    temp.replace(output / 'progress.json')
    (output / 'memory_summary.json').write_text(json.dumps(memory, indent=2) + '\n')
    reference_figures(paper, output)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, backbone in enumerate(BACKBONES):
        for i, (metric, label) in enumerate((('tensor_mib', 'Tensor allocations'), ('driver_mib', 'Metal driver allocations'))):
            ax = axes[i, col]
            for method in COLORS:
                rows = [r for r in memory if r['backbone'] == backbone and r['method'] == method]
                done = [r for r in rows if r['status'] == 'completed']
                ax.plot([r['layers'] for r in done], [r[metric] for r in done], 'o-', color=COLORS[method], label=method.upper())
                for row in rows:
                    partial = row['status'] != 'completed'
                    if partial:
                        ax.scatter(row['layers'], row[metric], marker='x', color=COLORS[method], s=65)
                    ax.annotate(f"{row[metric]:.1f}" + ('*' if partial else ''), (row['layers'], row[metric]),
                                xytext=(0, -16 if method == 'bp' else 9), textcoords='offset points', ha='center', fontsize=9)
            format_axis(ax, f'{backbone} | {label}', 'Sampled worker peak (MiB)')
            values = [r[metric] for r in memory if r['backbone'] == backbone]
            ax.set_ylim(0, max(values, default=1) * 1.35)
            if not values:
                ax.text(.5, .5, 'Not measured yet', transform=ax.transAxes, ha='center', color='#666666')
            ax.legend(loc='upper center', ncol=2, frameon=False)
    fig.suptitle(f'GitHub | MPS only | {completed_memory}/16 memory configurations complete', fontsize=16)
    fig.text(.5, .025, '0.5 s sampling; all folds + loading/training/evaluation. * Incomplete; missing depths are not inferred.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .94))
    save(fig, output, 'github_mps_memory')

    lines = ['# GitHub 数据集：论文结果与 MPS 复现', '',
             f"更新时间：{status['updated_utc']}。状态：**{state}**。", '',
             f'已保存 {fits}/80 次独立训练结果；准确率完整配置 {complete_groups}/16；显存完整配置 {completed_memory}/16。', '',
             '## 范围与数据来源', '',
             '- 论文整理：GCN / GraphSAGE / GAT，BP 与 SF，节点分类 1～4 层。来源：[Table 3 准确率、Table 8 显存](https://arxiv.org/html/2403.11004v1)。',
             '- 本轮复现：GCN / GAT，BP / SF，1～4 层、官方五折、128 维、Adam lr=0.001、weight_decay=0.0005；最多 1000 epoch（SF 为每层），每 2 epoch 验证，patience=100。没有降低正式训练预算。',
             '- 训练及推理仅 MPS，禁用 CPU 算子 fallback。数据准备与指标汇总使用 CPU。两条训练队列并行、每条内部串行；时间不作为性能基准。',
             '- PyG 的 graphmining.ai 下载地址无法解析。改从[作者发布的数据划分](https://github.com/NamyongPark/forwardgnn-datasplits)恢复：五组完整正边集合、节点特征和标签逐项完全一致。37,700 节点、578,006 条有向边、128 维特征、2 类。',
             '- 保留作者已经归一化的特征，避免二次归一化。边按源/目标编号排序，原始顺序不可恢复，可能影响浮点求和。数据恢复代码、15 个源文件哈希和缓存哈希可审计，见 results/paper_reproduction/github_data_provenance.json。',
             '- 不覆盖 FF、top-down、链接预测、GraphSAGE 实测或 H100 运行。', '',
             f'![论文结果]({output / "github_paper_results.png"})', '',
             '## 论文数值', '',
             '| 骨干 | 方法 | 层数 | 准确率 %（均值 ± 标准差） | 显存 MB |', '|---|---|---:|---:|---:|']
    for r in paper:
        lines.append(f"| {r['backbone']} | {r['method'].upper()} | {r['layers']} | {r['accuracy_percent']:.2f} ± {r['std_percent']:.1f} | {r['memory_mb']:.2f} |")
    lines += ['', '论文四层 SF 相对 BP 的显存下降：GCN 约 39.55%，GraphSAGE 约 48.20%，GAT 约 63.80%。这些是论文数值，不是本机测量。', '',
              '## 本机准确率', '', '| 骨干 | 方法 | 层数 | 折数 | 本机 % | 论文 % | 差值（百分点） |', '|---|---|---:|---:|---:|---:|---:|']
    for backbone in BACKBONES:
        for method in COLORS:
            for depth in range(1, 5):
                rows = groups.get(('GitHub', backbone, method, depth), [])
                ref = next(r for r in paper if r['backbone'] == backbone and r['method'] == method and r['layers'] == depth)
                local = delta = '待完成'
                if rows:
                    values = [r['accuracy_percent'] for r in rows]
                    mean = statistics.mean(values)
                    local = f'{mean:.2f} ± {statistics.pstdev(values):.2f}'
                    delta = f"{mean-ref['accuracy_percent']:+.2f}"
                lines.append(f"| {backbone} | {method.upper()} | {depth} | {len(rows)}/5 | {local} | {ref['accuracy_percent']:.2f} | {delta} |")
    lines += ['', '未完成五折的均值只是中间结果，不能作最终复现结论。SF 主准确率来自四层逐层训练保存的前缀；独立深度补测只用于显存比较。', '',
              f'![MPS 显存]({output / "github_mps_memory.png"})', '',
              '## 显存比较', '',
              '当前计数为整个训练进程的采样峰值，包含五折及评估，单位 MiB=字节/2^20。张量分配不含分配器缓存，驱动分配包含缓存和 MPS/MPSGraph 分配；短暂峰值可能漏采。与论文 H100 的 MB 数值不具有相同测量口径，不将二者绝对差当作复现误差。', '',
              '| 骨干 | 方法 | 层数 | 张量峰值 MiB | 驱动峰值 MiB | 状态 |', '|---|---|---:|---:|---:|---|']
    for r in memory:
        lines.append(f"| {r['backbone']} | {r['method'].upper()} | {r['layers']} | {r['tensor_mib']:.2f} | {r['driver_mib']:.2f} | {r['status']} |")
    for backbone in BACKBONES:
        four = {r['method']: r for r in memory if r['backbone'] == backbone and r['layers'] == 4 and r['status'] == 'completed'}
        if set(four) == {'bp', 'sf'}:
            lines += ['', f"本机四层 {backbone}：SF/BP 张量峰值比为 {four['sf']['tensor_mib']/four['bp']['tensor_mib']:.3f}，驱动峰值比为 {four['sf']['driver_mib']/four['bp']['driver_mib']:.3f}（小于 1 表示 SF 更低）。"]
        else:
            lines += ['', f'{backbone} 四层显存对照尚未完成，暂不判断 SF 是否更省。']
    lines += ['', '原始数据：[显存 JSON](memory_summary.json)、[进度 JSON](progress.json)。图表同时提供同名 SVG。', '',
              '重绘：`conda run -n gnn-research python -m paper_reproduction.experiments.summarize_github`。',
              '持续更新：在上述命令后添加 `--watch`；每分钟更新，完成或检测到失败时退出。']
    temp = output / 'report.md.tmp'
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(output / 'report.md')
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        status = snapshot()
        print(json.dumps(status), flush=True)
        if not args.watch or status['state'] in ('complete', 'failed'):
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
