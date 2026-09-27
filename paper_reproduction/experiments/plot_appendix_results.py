"""Extract Appendix E tables from the pinned arXiv HTML and plot published data."""
import hashlib
import json
import os
import re
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/gnn-paper-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'paper_reproduction/reference'
OUTPUT = ROOT / 'results/original_paper/appendix_e'
SOURCE = 'https://arxiv.org/html/2403.11004v1'
DATASETS = ('Amazon', 'GitHub', 'CiteSeer', 'PubMed', 'CoraML')
BACKBONES = ('GCN', 'SAGE', 'GAT')
COLORS = {'bp': '#2463a6', 'sf': '#df7635', 'sf-ce': '#df7635', 'sf-ff': '#29937c', 'sf-symba': '#935cb0'}
LABELS = {'bp': 'BP', 'sf': 'SF', 'sf-ce': 'SF-CE', 'sf-ff': 'SF-FF objective', 'sf-symba': 'SF-SymBa objective'}


def clean(element):
    return ' '.join(element.get_text(' ', strip=True).split())


def extract(source_path):
    soup = BeautifulSoup(source_path.read_text(), 'html.parser')
    # MathML annotations duplicate the visible numeric text in get_text().
    for annotation in soup.select('annotation'):
        annotation.decompose()
    tables, records = [], []
    for number in range(3, 17):
        figure = soup.find(id=f'A5.T{number}')
        assert figure is not None
        panels = figure.select('figure.ltx_table') or [figure]
        for panel in panels:
            grid = panel.select_one('.ltx_tabular')
            cells = [[clean(c) for c in tr.select(':scope > .ltx_td')]
                     for tr in grid.select('.ltx_tr')]
            caption = clean(panel.find('figcaption'))
            dataset = re.sub(r'^\([a-z]\)\s*', '', caption) if number < 16 else None
            tables.append(dict(table=number, panel_id=panel['id'], dataset=dataset,
                               caption=clean(figure.find('figcaption')), cells=cells,
                               source=f'{SOURCE}#{panel["id"]}'))
            if number == 16:
                continue
            task = 'link_prediction' if 9 <= number <= 14 else 'node_classification'
            metric = ('accuracy_percent' if number <= 5 else 'gpu_memory_mb' if number <= 8
                      else 'roc_auc_percent' if number <= 11 else 'gpu_memory_mb' if number <= 14
                      else 'training_seconds_100_epochs')
            for row in cells[2:]:
                assert len(row) == 5, (number, row)
                method_text = re.sub(r'\s*\(Sec.*', '', row[0])
                match = re.fullmatch(r'(Backpropagation|SF|ForwardGNN-CE|ForwardGNN-FF|ForwardGNN-SymBa)-(GCN|SAGE|GAT)', method_text)
                if not match:
                    continue
                method = {'Backpropagation': 'bp', 'SF': 'sf', 'ForwardGNN-CE': 'sf-ce',
                          'ForwardGNN-FF': 'sf-ff', 'ForwardGNN-SymBa': 'sf-symba'}[match[1]]
                for depth, cell in enumerate(row[1:], 1):
                    numeric = re.fullmatch(r'(\d+(?:\.\d+)?)(?:\s*±\s*(\d+(?:\.\d+)?))?', cell)
                    assert numeric, (number, row[0], cell)
                    records.append(dict(task=task, dataset=dataset, backbone=match[2], method=method,
                                        layers=depth, metric=metric, value=float(numeric[1]),
                                        std=float(numeric[2]) if numeric[2] else None,
                                        original_method=method_text, original_cell=cell, table=number,
                                        source=f'{SOURCE}#{panel["id"]}'))
    assert len(tables) == 23
    keys = [(r['task'], r['dataset'], r['backbone'], r['method'], r['layers'], r['metric']) for r in records]
    assert len(keys) == len(set(keys)), 'Duplicate measurements'
    return tables, records


def validate(records):
    # Independently transcribed earlier references must agree with extraction.
    github = json.loads((REFERENCE / 'github_paper_results.json').read_text())['rows']
    for row in github:
        for metric, field in [('accuracy_percent', 'accuracy_percent'), ('gpu_memory_mb', 'memory_mb')]:
            actual = [r for r in records if r['task'] == 'node_classification' and r['dataset'] == 'GitHub'
                      and all(r[k] == row[k] for k in ('backbone', 'method', 'layers')) and r['metric'] == metric]
            assert len(actual) == 1 and actual[0]['value'] == row[field]
            if metric == 'accuracy_percent':
                assert actual[0]['std'] == row['std_percent']
    for task, metric in [('node_classification', 'accuracy_percent'), ('node_classification', 'gpu_memory_mb'),
                         ('link_prediction', 'roc_auc_percent'), ('link_prediction', 'gpu_memory_mb')]:
        for method in ('bp', 'sf' if task == 'node_classification' else 'sf-ce'):
            rows = [r for r in records if r['task'] == task and r['metric'] == metric and r['method'] == method]
            assert len(rows) == 60, (task, metric, method, len(rows))
            assert {r['dataset'] for r in rows} == set(DATASETS)


def save(fig, name):
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(OUTPUT / f'{name}.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


def grid_plot(records, task, metric, methods, name, title, datasets=DATASETS):
    fig, axes = plt.subplots(len(datasets), 3, figsize=(14, 2.75 * len(datasets)), squeeze=False)
    for i, dataset in enumerate(datasets):
        for j, backbone in enumerate(BACKBONES):
            ax = axes[i, j]
            for method in methods:
                rows = sorted([r for r in records if r['task'] == task and r['metric'] == metric
                               and r['dataset'] == dataset and r['backbone'] == backbone and r['method'] == method], key=lambda r: r['layers'])
                assert len(rows) == 4
                ax.errorbar([r['layers'] for r in rows], [r['value'] for r in rows],
                            yerr=[r['std'] or 0 for r in rows], fmt='o-', capsize=3,
                            color=COLORS[method], label=LABELS[method], linewidth=1.8, markersize=4)
            ax.set(title=f'{dataset} / {backbone}', xticks=[1, 2, 3, 4], xlim=(.8, 4.2))
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            if metric == 'gpu_memory_mb' or metric.startswith('training'):
                ax.set_ylim(bottom=0)
            if i == len(datasets)-1:
                ax.set_xlabel('Layers')
            if j == 0:
                ax.set_ylabel({'accuracy_percent': 'Accuracy (%)', 'roc_auc_percent': 'ROC-AUC (%)',
                               'gpu_memory_mb': 'GPU memory (MB)', 'training_seconds_100_epochs': 'Seconds / 100 epochs'}[metric])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .958), ncol=len(methods), frameon=False)
    fig.suptitle(title, fontsize=17, y=.987)
    uncertainty = 'No standard deviations reported' if metric == 'gpu_memory_mb' else 'Error bars: reported standard deviation'
    fig.text(.5, .01, f'Published values only | arXiv:2403.11004v1, Appendix E | {uncertainty}; panel y-scales differ', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .025, 1, .93))
    save(fig, name)


def comparisons(records):
    lookup = {(r['task'], r['dataset'], r['backbone'], r['method'], r['layers'], r['metric']): r for r in records}
    result = []
    for task, metric, method in [('node_classification', 'accuracy_percent', 'sf'), ('link_prediction', 'roc_auc_percent', 'sf-ce')]:
        for dataset in DATASETS:
            for backbone in BACKBONES:
                for depth in range(1, 5):
                    def value(m, k):
                        return lookup[task, dataset, backbone, m, depth, k]['value']
                    result.append(dict(task=task, dataset=dataset, backbone=backbone, layers=depth, sf_variant=method,
                                       score_metric=metric, bp_score=value('bp', metric), sf_score=value(method, metric),
                                       score_delta_pp=value(method, metric)-value('bp', metric),
                                       bp_memory_mb=value('bp', 'gpu_memory_mb'), sf_memory_mb=value(method, 'gpu_memory_mb'),
                                       memory_saving_percent=100*(1-value(method, 'gpu_memory_mb')/value('bp', 'gpu_memory_mb'))))
    return result


def overview(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    node = [r for r in rows if r['task'] == 'node_classification' and r['layers'] == 4]
    for ax, metric, title, unit in zip(axes, ('score_delta_pp', 'memory_saving_percent'),
                                      ('Accuracy: SF minus BP', 'GPU memory saved by SF'), ('pp', '%')):
        values = np.array([[next(r[metric] for r in node if r['dataset'] == d and r['backbone'] == b) for b in BACKBONES] for d in DATASETS])
        limit = max(abs(values.min()), abs(values.max()))
        heat = ax.imshow(values, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
        ax.set(xticks=range(3), xticklabels=BACKBONES, yticks=range(5), yticklabels=DATASETS, title=title)
        for i in range(5):
            for j in range(3):
                ax.text(j, i, f'{values[i,j]:+.2f} {unit}', ha='center', va='center',
                        color='white' if abs(values[i,j]) > limit*.6 else '#111111', fontsize=12)
        fig.colorbar(heat, ax=ax, shrink=.7)
    fig.suptitle('Original paper | Four-layer node classification | SF vs BP', fontsize=17)
    fig.text(.5, .025, 'Positive values favor SF. Memory saving = 100 × (1 − SF/BP). Mean differences are not significance tests.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .94))
    save(fig, 'node_four_layer_comparison')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_path = REFERENCE / 'source/forwardgnn_2403.11004v1.html'
    tables, records = extract(source_path)
    validate(records)
    metadata = dict(source=SOURCE, appendix='E', source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    extraction='HTML visible cells; exact method names; no digitization or inferred values')
    (REFERENCE / 'appendix_e_tables.json').write_text(json.dumps(dict(**metadata, tables=tables), ensure_ascii=False, indent=2)+'\n')
    (REFERENCE / 'appendix_e_sf_bp.json').write_text(json.dumps(dict(**metadata, records=records), ensure_ascii=False, indent=2)+'\n')
    table_md = ['# Appendix E：原表数据', '', f'来源：[{SOURCE}]({SOURCE})。保留所有方法及原始数值；`±` 为原表标准差。', '']
    for table in tables:
        table_md += [f"## Table {table['table']} — {table['dataset'] or 'Dataset statistics'}", '', f"[原表]({table['source']})", '']
        cells = table['cells']
        headers = cells[0] if table['table'] == 16 else ['Method', 'Layers=1', 'Layers=2', 'Layers=3', 'Layers=4']
        data = cells[1:] if table['table'] == 16 else cells[2:]
        table_md += ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
        table_md += ['| '+' | '.join(row)+' |' for row in data]
        table_md += ['']
    (OUTPUT / 'tables.md').write_text('\n'.join(table_md))
    grid_plot(records, 'node_classification', 'accuracy_percent', ('bp', 'sf'), 'node_accuracy', 'Original paper | Node classification accuracy | Tables 3–5')
    grid_plot(records, 'node_classification', 'gpu_memory_mb', ('bp', 'sf'), 'node_gpu_memory', 'Original paper | Node classification GPU memory | Tables 6–8')
    grid_plot(records, 'link_prediction', 'roc_auc_percent', ('bp', 'sf-ce', 'sf-ff', 'sf-symba'), 'link_roc_auc', 'Original paper | Link prediction ROC-AUC | Tables 9–11')
    grid_plot(records, 'link_prediction', 'gpu_memory_mb', ('bp', 'sf-ce', 'sf-ff', 'sf-symba'), 'link_gpu_memory', 'Original paper | Link prediction GPU memory | Tables 12–14')
    grid_plot(records, 'node_classification', 'training_seconds_100_epochs', ('bp', 'sf'), 'node_training_time', 'Original paper | Training time | Table 15', datasets=('GitHub', 'CiteSeer'))
    rows = comparisons(records)
    overview(rows)
    (OUTPUT / 'paired_comparisons.json').write_text(json.dumps(rows, indent=2)+'\n')
    node4 = [r for r in rows if r['task'] == 'node_classification' and r['layers'] == 4]
    lines = ['# 原论文附录 E：SF 与 BP 的准确率和 GPU 显存', '',
             f'来源：[Forward Learning of Graph Neural Networks，arXiv v1，Appendix E]({SOURCE}#A5)。全部为原论文发布值，不是本机 MPS 实测。', '',
             '## 覆盖范围与口径', '',
             f'- Table 3–16 全部 14 张编号表、{len(tables)} 个表格面板已提取，完整保留所有方法的原始单元格，见 [原表数据](tables.md)。',
             '- 节点分类：5 个数据集 × 3 个骨干 × 4 个深度，比较基础 SF（Sec. 3.2）与 BP；top-down 变体未混入基础 SF。',
             '- 链接预测指标是 ROC-AUC，不是 Accuracy。原表以 ForwardGNN-CE、ForwardGNN-FF、ForwardGNN-SymBa 命名局部目标变体，图中分别标为 SF-CE、SF-FF objective、SF-SymBa objective，均与 BP 单独比较；不把 FF objective 当作节点分类的 FF-LA/FF-VN。',
             '- 准确率与 AUC 使用百分数，误差条为五次运行的原表标准差，均值差不代表显著性。GPU 显存保留论文 MB 单位，不附加不存在的误差条。',
             '- E.4 的 Table 16 仅包含数据集统计；该节精度/显存以及 E.5 方向性实验由图片给出，本次不从图中估读数值。',
             '- Table 15 的训练时间另行绘图。论文 H100 显存与本机 MPS 的采样张量/驱动峰值分开保存，不直接混比绝对数值。', '',
             '## 四层节点分类', '',
             '![四层对比](node_four_layer_comparison.png)', '',
             '| 数据集 | 骨干 | BP accuracy % | SF accuracy % | SF−BP（百分点） | BP MB | SF MB | SF 显存下降 % |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in node4:
        lines.append(f"| {r['dataset']} | {r['backbone']} | {r['bp_score']:.2f} | {r['sf_score']:.2f} | {r['score_delta_pp']:+.2f} | {r['bp_memory_mb']:.2f} | {r['sf_memory_mb']:.2f} | {r['memory_saving_percent']:+.2f} |")
    wins = sum(r['memory_saving_percent'] > 0 for r in node4)
    accuracy_wins = sum(r['score_delta_pp'] > 0 for r in node4)
    lines += ['', f'四层的 15 组数据集/骨干组合中，SF 显存更低 {wins} 组，准确率均值更高 {accuracy_wins} 组。SF 的显存随层数增长较平稳，但绝对显存和准确率优势都依赖具体配置。不能将均值胜出计数视为显著性检验。', '']
    for name, title in [('node_accuracy', '节点分类 Accuracy'), ('node_gpu_memory', '节点分类 GPU memory'),
                        ('link_roc_auc', '链接预测 ROC-AUC'), ('link_gpu_memory', '链接预测 GPU memory'), ('node_training_time', '训练时间')]:
        lines += [f'## {title}', '', f'![{title}]({name}.png)', '', f'[SVG]({name}.svg) · [PDF]({name}.pdf)', '']
    lines += ['## 可追溯数据与重绘', '',
              '[全部表格 JSON](../../../../paper_reproduction/reference/appendix_e_tables.json) · [SF/BP 数值 JSON](../../../../paper_reproduction/reference/appendix_e_sf_bp.json) · [逐配置差值](paired_comparisons.json)', '',
              '源码 HTML 固定在 `paper_reproduction/reference/source/forwardgnn_2403.11004v1.html`；JSON 含来源锚点及 SHA256。已与先前独立整理的 GitHub 全部准确率/标准差/显存值交叉核验。', '',
              '重绘：`conda run -n gnn-research python -m paper_reproduction.experiments.plot_appendix_results`。']
    (OUTPUT / 'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(panels=len(tables), selected_measurements=len(records), paired_comparisons=len(rows), four_layer_memory_wins=wins, four_layer_accuracy_wins=accuracy_wins)))


if __name__ == '__main__':
    main()
