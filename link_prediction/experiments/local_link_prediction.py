"""Virtual-node-free local GNN training versus end-to-end BP.

Independent extension experiment, not the paper's exact reproduction protocol.
"""
import argparse
import copy
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.datasets import CitationFull
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures

ROOT = Path(__file__).resolve().parents[2]


def edge_split(edge_index, n, seed):
    """Canonical undirected pairs; disjoint negatives exclude ALL observed edges."""
    rng = np.random.default_rng(seed)
    pairs = sorted({tuple(sorted(e)) for e in edge_index.t().tolist() if e[0] != e[1]})
    positive = set(pairs)
    if n * (n - 1) // 2 - len(pairs) < len(pairs):
        raise ValueError('Not enough distinct non-edges for the 1:1 protocol')
    rng.shuffle(pairs)
    negatives = set()
    while len(negatives) < len(pairs):
        a, b = sorted(rng.integers(n, size=2).tolist())
        if a != b and (a, b) not in positive:
            negatives.add((a, b))
    negatives = sorted(negatives)
    rng.shuffle(negatives)
    boundaries = [0, int(.64 * len(pairs)), int(.80 * len(pairs)), len(pairs)]
    result = {}
    for name, lo, hi in zip(('train', 'val', 'test'), boundaries, boundaries[1:]):
        result[name] = (
            torch.tensor(pairs[lo:hi] + negatives[lo:hi], dtype=torch.long).t(),
            torch.cat([torch.ones(hi-lo), torch.zeros(hi-lo)]),
        )
    train_pos = torch.tensor(pairs[:boundaries[1]], dtype=torch.long).t()
    result['message_edges'] = torch.cat([train_pos, train_pos.flip(0)], dim=1)
    return result


class Encoder(nn.Module):
    def __init__(self, features, hidden, depth):
        super().__init__()
        self.layers = nn.ModuleList([
            GCNConv(features if i == 0 else hidden, hidden, cached=True)
            for i in range(depth)
        ])

    @staticmethod
    def layer_forward(layer, x, edges):
        return torch.tanh(layer(x, edges))

    def forward(self, x, edges):
        for layer in self.layers:
            x = self.layer_forward(layer, x, edges)
        return x


def logits(z, pairs):
    return (z[pairs[0]] * z[pairs[1]]).sum(-1)


@torch.no_grad()
def evaluate(z, split):
    pairs, labels = split
    score = logits(z, pairs).numpy()
    return dict(auc=float(roc_auc_score(labels.numpy(), score)),
                ap=float(average_precision_score(labels.numpy(), score)))


def fit(initial, x, split, method, epochs, lr):
    model = copy.deepcopy(initial)
    depth = len(model.layers)
    trace, stage_counts = [], []
    start = time.perf_counter()
    edges, (pairs, labels) = split['message_edges'], split['train']
    # SF has epochs updates per layer; BP has epochs*depth total updates.
    stages = list(model.layers) if method == 'sf' else [model]
    fixed = x
    for stage_i, module in enumerate(stages):
        opt = torch.optim.Adam(module.parameters(), lr=lr, weight_decay=5e-4)
        best_auc, best_state, best_epoch = -1., None, None
        budget = epochs if method == 'sf' else epochs * depth
        for epoch in range(budget):
            module.train()
            opt.zero_grad(set_to_none=True)
            z = model.layer_forward(module, fixed.detach(), edges) if method == 'sf' else model(x, edges)
            loss = F.binary_cross_entropy_with_logits(logits(z, pairs), labels)
            loss.backward()
            opt.step()
            if (epoch + 1) % 5 == 0 or epoch + 1 == budget:
                module.eval()
                with torch.no_grad():
                    z = model.layer_forward(module, fixed, edges) if method == 'sf' else model(x, edges)
                    metric = evaluate(z, split['val'])
                trace.append(dict(stage=stage_i+1, epoch=epoch+1,
                                  elapsed=time.perf_counter()-start, loss=float(loss.detach()), **metric))
                if metric['auc'] > best_auc:
                    best_auc = metric['auc']
                    best_state = copy.deepcopy(module.state_dict())
                    best_epoch = epoch+1
        module.load_state_dict(best_state)
        stage_counts.append(dict(updates=budget, selected_epoch=best_epoch, val_auc=best_auc))
        if method == 'sf':
            # Clear gradients and freeze each completed layer explicitly.
            for p in module.parameters():
                p.grad = None
                p.requires_grad_(False)
            with torch.no_grad():
                fixed = model.layer_forward(module, fixed, edges).detach()
    train_seconds = time.perf_counter()-start
    model.eval()
    with torch.no_grad():
        z = model(x, edges)
        validation = evaluate(z, split['val'])
        test = evaluate(z, split['test'])
    return dict(method=method, train_seconds=train_seconds, validation=validation,
                test=test, stages=stage_counts, curve=trace)


def report(rows, config, target):
    lines = ['# 不使用虚拟节点的 Single-Forward：链接预测实验', '',
             '状态：全部配置完成。' if len(rows) == 2*len(config.datasets)*len(config.depths)*len(config.seeds) else '状态：运行中，结果未齐。', '',
             '这是独立扩展实验，不是原论文链接预测数值的复现。SF 保留逐层局部优化思想，不添加虚拟节点，也不使用节点类别标签。', '',
             '每层输出节点表示，正负边都由同一次图前向的节点表示打分：', '',
             r'$$s_{ij}^{(\ell)}=(h_i^{(\ell)})^T h_j^{(\ell)},\qquad \mathcal L_\ell=\operatorname{BCEWithLogits}(s_{ij}^{(\ell)},y_{ij}).$$', '',
             'SF 当前层输入 detach，完成后冻结并缓存其输出；BP 对最终层损失进行端到端反传。两者均用最后一层预测，不采用多层集成。负边仍然存在，但不需要为负边构造另一张输入图。', '',
             '## 实验协议', '',
             f'- 数据：{config.datasets}；深度：{config.depths}；隐藏维度：{config.hidden}；种子：{config.seeds}。',
             f'- 每层 tanh、GCNConv；Adam lr={config.lr}，weight_decay=0.0005；CPU 单线程，串行运行。',
             f'- SF 每层 {config.epochs} 次更新；BP 共 深度×{config.epochs} 次更新。匹配优化器调用总次数，不代表 FLOPs、参数更新次数或时间相同；两种方法未分别搜索超参数。',
             '- 无向边先去重，再按 64%/16%/20% 划分；负边与所有正边不重叠，三个集合的负边互不重叠，正负比 1:1。消息传递仅使用训练正边，验证/测试均不加入保留正边。',
             '- 每 5 步验证，按验证 AUC 选权重。SF 每层独立选择，低层固定后训练高层；BP 按最终层选权重。没有按测试集调参。',
             '- 每对 BP/SF 使用完全相同的初始化及划分，执行顺序随种子交替。计时从模型复制之后开始，含优化器创建、训练、验证、最佳权重复制和恢复；不含数据加载与最终测试。首轮实验期间曾执行约两秒的单元测试，因此时间属于探索性测量，严格基准应在独占资源下重测。',
             '- 两者都缓存固定图的 GCN 邻接归一化。SF 只缓存冻结层输出，本实验没有实现预计算当前层邻域聚合的额外加速。',
             '- AUC/AP 在固定的 1:1 负采样下计算，AP 不等于全图真实稀疏率下的精度。各随机种子是重复随机边划分，不是作者原始五折。',
             '', '## 测试集结果', '',
             '| 数据 | 层数 | 方法 | 次数 | ROC-AUC % | AP % | 训练秒数 |',
             '|---|---:|---|---:|---:|---:|---:|']
    for dataset in config.datasets:
        for depth in config.depths:
            for method in ('bp','sf'):
                subset = [r for r in rows if (r['dataset'],r['depth'],r['method']) == (dataset,depth,method)]
                if not subset:
                    continue
                def fmt(values):
                    return f'{statistics.mean(values):.2f} ± {statistics.pstdev(values):.2f}'
                lines.append(f'| {dataset} | {depth} | {method.upper()} | {len(subset)} | '+
                             ' | '.join([fmt([100*r['test'][k] for r in subset]) for k in ('auc','ap')]+[fmt([r['train_seconds'] for r in subset])])+' |')
    lines += ['', '## 同配置对照', '']
    for dataset in config.datasets:
        for depth in config.depths:
            bp = [r for r in rows if (r['dataset'],r['depth'],r['method']) == (dataset,depth,'bp')]
            sf = [r for r in rows if (r['dataset'],r['depth'],r['method']) == (dataset,depth,'sf')]
            if len(bp) == len(sf) == len(config.seeds):
                delta = 100*(statistics.mean(r['test']['auc'] for r in sf)-statistics.mean(r['test']['auc'] for r in bp))
                speed = statistics.mean(r['train_seconds'] for r in bp)/statistics.mean(r['train_seconds'] for r in sf)
                lines.append(f'- {dataset}，{depth} 层：SF − BP 的 AUC 为 {delta:+.2f} 个百分点；BP/SF 平均耗时比为 {speed:.2f}。')
    lines += ['', '均值 ± 总体标准差；少量种子结果不构成统计显著性证明。CPU 墙钟只适用于本机本配置，不代表 GPU 加速。SF 是否更好需要分别比较 AUC、AP 和时间，不能由局部损失下降推出泛化优势。', '',
              '本轮只检验链接预测，尚不能外推到图回归、图分类或动态任务。', '',
              '复现命令：`conda run -n gnn-research python -m link_prediction.experiments.local_link_prediction`。',
              f'原始记录与学习曲线：`{target}`。']
    (ROOT/'docs/sf_without_virtual_nodes_link_prediction.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['CiteSeer','Cora_ML'])
    parser.add_argument('--depths', nargs='+', type=int, default=[2,4])
    parser.add_argument('--seeds', nargs='+', type=int, default=[11,22,33])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--lr', type=float, default=.005)
    parser.add_argument('--setting', default='link_no_virtual_v1')
    args = parser.parse_args()
    torch.set_num_threads(1)
    target = ROOT/'results'/args.setting
    target.mkdir(parents=True, exist_ok=True)
    manifest = target/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())['config'] != vars(args):
        raise ValueError('Different configuration; use a different --setting')
    manifest.write_text(json.dumps(dict(config=vars(args), platform=platform.platform(), torch=torch.__version__), indent=2))
    rows=[]
    for dataset in args.datasets:
        data = CitationFull(str(ROOT/'third_party/forwardgnn/data/CitationFull'), dataset,
                            transform=NormalizeFeatures())[0]
        for depth in args.depths:
            for seed in args.seeds:
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)
                split=edge_split(data.edge_index, data.num_nodes, seed)
                initial=Encoder(data.num_features,args.hidden,depth)
                for method in (('bp','sf') if seed%2 else ('sf','bp')):
                    path=target/f'{dataset}_{depth}_{seed}_{method}.json'
                    if path.exists():
                        row=json.loads(path.read_text())
                    else:
                        row=dict(dataset=dataset,depth=depth,seed=seed,
                                 **fit(initial,data.x,split,method,args.epochs,args.lr))
                        path.write_text(json.dumps(row,indent=2))
                    rows.append(row)
                    print(f"{dataset} L{depth} seed{seed} {method}: AUC={row['test']['auc']:.4f} AP={row['test']['ap']:.4f} seconds={row['train_seconds']:.1f}",flush=True)
                    report(rows,args,target)


if __name__ == '__main__':
    main()
