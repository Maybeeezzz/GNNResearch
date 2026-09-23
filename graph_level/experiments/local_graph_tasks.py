"""Real molecular graph classification/regression with BP and local training."""
import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils.smiles import x_map

from graph_level.model.graph_level import GraphBatch, GraphNetwork, train_graph

ROOT = Path(__file__).resolve().parents[2]


def load_graphs(name):
    dataset = MoleculeNet(str(ROOT/'data/MoleculeNet'), name)
    graphs, groups = [], []
    for graph in dataset:
        graph = graph.clone()
        if graph.num_nodes == 0 or not torch.isfinite(graph.y).all():
            raise ValueError('Empty molecule or missing target requires an explicit policy')
        graph.x = torch.cat([F.one_hot(graph.x[:,i],len(values)).float()
                             for i,values in enumerate(x_map.values())],dim=1)
        # No target-derived features, no virtual node, no cross-molecule edges.
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(graph.smiles),includeChirality=False)
        groups.append(scaffold)  # all acyclic molecules remain in one group
        graphs.append(graph)
    return graphs, groups


def split_graphs(graphs, groups, seed):
    ids = np.arange(len(graphs))
    train, held = next(GroupShuffleSplit(n_splits=1,test_size=.30,random_state=seed).split(ids,groups=groups))
    a,b = next(GroupShuffleSplit(n_splits=1,test_size=.50,random_state=seed+1000).split(held,groups=np.asarray(groups)[held]))
    indices = dict(train=train.tolist(),val=held[a].tolist(),test=held[b].tolist())
    group_sets = [set(groups[i] for i in indices[k]) for k in indices]
    assert all(not group_sets[i]&group_sets[j] for i in range(3) for j in range(i))
    return indices


def write_report(target, config):
    rows = [json.loads(p.read_text()) for p in sorted(target.glob('run_*.json'))]
    expected = len(config['datasets'])*len(config['depths'])*len(config['seeds'])*2
    lines = ['# 无虚拟节点的局部学习：图分类与图回归','',
             f'完成状态：{len(rows)}/{expected} 次训练。','',
             '本实验扩展 Single-Forward 的一次前向、逐层局部监督思想，使用辅助预测头代替类别虚拟节点。它不是原论文节点分类算法的原样移植，也不声称提出新的算法。','',
             '## 方法','',
             r'$$H^{(\ell)}=\tanh(\hat A H^{(\ell-1)}W_\ell+b_\ell),\quad g^{(\ell)}=[\operatorname{mean}(H^{(\ell)})\|\operatorname{sum}(H^{(\ell)})],\quad \hat y^{(\ell)}=q_\ell(g^{(\ell)}).$$','',
             '分类局部目标为交叉熵；回归局部目标为标准化目标的 MSE。训练当前层及其预测头，输入 detach，低层完成后冻结；推理使用最后一层及最后一个头。BP 使用相同初始化、骨干和最终头，对最终任务损失端到端优化。','',
             '## 数据与公平性','',
             '- BACE：整张分子图的二分类，主指标 ROC-AUC；另报 AP、Accuracy。ESOL：整张分子图的实测溶解度回归，主指标 RMSE，另报 MAE、R²，单位为原始 log-solubility。',
             '- 数据来自 PyG MoleculeNet，原子类别特征做固定 one-hot。该 GCN 基线只使用键的连接，不使用键类别或三维坐标。没有节点类别监督，也没有虚拟节点。',
             '- 按无手性 Murcko 骨架分组，种子分别随机将约 70%/15%/15% 的骨架组分配到训练/验证/测试。分子比例不一定相等；所有无环分子留在同一个空骨架组。划分记录保存精确样本索引和各集合数量。不是官方 MoleculeNet 固定基准划分。',
             '- 每个集合用独立的块对角图批次，分子之间无边；测试图在训练期间不参与消息传递。回归标签的均值与标准差只由训练集计算。',
             f"- 隐藏维度 {config['hidden']}，深度 {config['depths']}，种子 {config['seeds']}，Adam lr={config['lr']}，weight_decay=0.0005；CPU 单线程，方法串行运行。",
             f"- SF 每层 {config['epochs']} 次全批次更新；BP 更新次数为 深度×{config['epochs']}。因此总优化器调用数及样本呈现次数相同，但参数更新次数、FLOPs 和训练秒数不同。",
             '- 每 5 步验证；分类选验证 AUC 最大、回归选验证 RMSE 最小权重。SF 每层选择，BP 最终输出选择，不早停、不基于测试选择超参。',
             '- 两种方法都预计算固定第一层输入的邻域聚合。SF 还会缓存后续冻结层输入的邻域聚合；这些预计算计入训练耗时。SF 辅助头的训练和前缀输出缓存亦计入。',
             '- 时间从模型复制后开始，包含缓存构造、优化器创建、训练、验证、最佳权重保存/恢复；不含数据下载、数据预处理、最终测试与磁盘保存。CPU 墙钟为本次机器实测，非 GPU 基准。',
             '- 相同划分和初始化配对比较，执行顺序随种子交替；三个种子未做显著性检验，未分别搜索两种方法超参数。','',
             '## 图分类结果','',
             '| 数据 | 层数 | 方法 | 次数 | AUC % | AP % | Accuracy % | 秒 |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    def fmt(vals):
        return f'{statistics.mean(vals):.3f} ± {statistics.pstdev(vals):.3f}'
    def table(task, keys):
        out=[]
        for dataset in config['datasets']:
            for depth in config['depths']:
                for method in ('bp','sf'):
                    subset=[r for r in rows if (r['task'],r['dataset'],r['depth'],r['method'])==(task,dataset,depth,method)]
                    if subset:
                        vals=[fmt([r['test'][k]*(100 if task=='classification' else 1) for r in subset]) for k in keys]
                        out.append(f'| {dataset} | {depth} | {method.upper()} | {len(subset)} | '+' | '.join(vals+[fmt([r['train_seconds'] for r in subset])])+' |')
        return out
    lines += table('classification',('auc','ap','accuracy'))
    lines += ['', '## 图回归结果','',
              '| 数据 | 层数 | 方法 | 次数 | RMSE ↓ | MAE ↓ | R² ↑ | 秒 |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    lines += table('regression',('rmse','mae','r2'))
    lines += ['', '## 简单基线与实际划分','']
    for p in sorted(target.glob('split_*.json')):
        s=json.loads(p.read_text())
        lines.append(f"- {p.stem}：分子数 {s['counts']}；测试基线 {s['baseline']}。")
    lines += ['', '分类基线使用训练多数类/训练正类比例；回归基线始终预测训练目标均值。标准差为总体标准差 ddof=0。', '',
              '本报告只支持上述任务、模型和预算下的比较；最终层局部目标可能压制后续层需要的信息。性能差异不能直接证明局部学习普遍优于端到端 BP。','',
              '运行：`conda run -n gnn-research python -m graph_level.experiments.local_graph_tasks`。','',
              '数据来源：[PyG MoleculeNet](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MoleculeNet.html)。',
              f'原始结果、逐轮曲线、预测、权重及配置：`{target}`。']
    (ROOT/'docs/sf_graph_tasks_results.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=['BACE','ESOL'],default=['BACE','ESOL'])
    p.add_argument('--depths',nargs='+',type=int,default=[2,4])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--setting',default='graph_local_v1')
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    config=vars(args).copy(); config.pop('report_only')
    target=ROOT/'results'/args.setting
    if args.report_only:
        write_report(target,json.loads((target/'manifest.json').read_text())['config'])
        return
    if args.epochs<1 or args.hidden<1 or min(args.depths)<1:
        p.error('budgets and dimensions must be positive')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    target.mkdir(parents=True,exist_ok=True)
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__),ROOT/'graph_level/model/graph_level.py')}
    manifest_path=target/'manifest.json'
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['config']!=config or old['source_sha256']!=hashes:
            raise ValueError('Configuration/code changed; choose a fresh setting')
    manifest=dict(config=config,source_sha256=hashes,torch=torch.__version__,pyg=torch_geometric.__version__,
                  rdkit=rdBase.rdkitVersion,platform=platform.platform(),device='cpu',threads=1)
    manifest['raw_data_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted((ROOT/'data/MoleculeNet').glob('*/raw/*.csv'))}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    for name in args.datasets:
        task='classification' if name=='BACE' else 'regression'
        graphs,groups=load_graphs(name)
        for seed in args.seeds:
            indices=split_graphs(graphs,groups,seed)
            batches={k:GraphBatch.from_graphs([graphs[i] for i in ids]) for k,ids in indices.items()}
            if task=='classification' and any(b.y.unique().numel()!=2 for b in batches.values()):
                raise ValueError('Both classes required for AUC; revise split protocol explicitly')
            train_y,test_y=batches['train'].y,batches['test'].y
            if task=='classification':
                baseline=dict(auc=.5,ap=float(test_y.mean()),accuracy=float((test_y==int(train_y.mean()>=.5)).float().mean()))
            else:
                baseline=dict(rmse=float(((test_y-train_y.mean())**2).mean().sqrt()),
                              mae=float((test_y-train_y.mean()).abs().mean()))
            split_record=dict(indices=indices,counts={k:len(v) for k,v in indices.items()},baseline=baseline,
                              scaffold_groups={k:sorted(set(groups[i] for i in ids)) for k,ids in indices.items()})
            (target/f'split_{name}_{seed}.json').write_text(json.dumps(split_record,indent=2))
            for depth in args.depths:
                torch.manual_seed(seed)
                model=GraphNetwork(graphs[0].x.size(1),args.hidden,depth,2 if task=='classification' else 1)
                for method in (('bp','sf') if seed%2 else ('sf','bp')):
                    path=target/f'run_{name}_{depth}_{seed}_{method}.json'
                    if path.exists():
                        row=json.loads(path.read_text())
                    else:
                        row,fitted=train_graph(model,batches,task,method,args.epochs,args.lr)
                        row.update(dataset=name,task=task,depth=depth,seed=seed)
                        torch.save(fitted.state_dict(),path.with_suffix('.pt'))
                        path.write_text(json.dumps(row,indent=2,allow_nan=False))
                    print(f"{name} depth={depth} seed={seed} {method}: {row['test']} seconds={row['train_seconds']:.2f}",flush=True)
                    write_report(target,config)
    print(f'Completed: {target}',flush=True)


if __name__=='__main__':
    main()
