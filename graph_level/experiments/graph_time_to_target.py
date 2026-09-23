"""Fresh-process, paired graph-task time-to-validation-target benchmark."""
import argparse
import copy
import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import torch

from graph_level.experiments.local_graph_tasks import load_graphs
from graph_level.model.graph_level import GraphBatch, GraphNetwork, metrics, objective

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/graph_time_to_target_v1'
REFERENCE = ROOT/'results/graph_local_v1'


class Memory:
    """10 ms RSS sampling; process lifetime high-water also reported separately."""
    def __init__(self):
        self.proc=psutil.Process()
        self.baseline=self.proc.memory_info().rss
        self.peak=self.baseline
        self.stop=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True)

    def sample(self):
        self.peak=max(self.peak,self.proc.memory_info().rss)

    def run(self):
        while not self.stop.wait(.01):
            self.sample()

    def finish(self):
        self.sample()
        self.stop.set()
        self.thread.join()
        high=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform!='darwin':
            high*=1024
        return dict(baseline_rss_mib=self.baseline/2**20,
                    sampled_training_peak_rss_mib=self.peak/2**20,
                    sampled_training_increment_mib=(self.peak-self.baseline)/2**20,
                    process_lifetime_peak_until_training_end_mib=high/2**20)


def attained(task,value,target):
    return value>=target-1e-10 if task=='classification' else value<=target+1e-10


def worker(args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    name,depth,seed,method=args.dataset,args.depth,args.seed,args.method
    task='classification' if name=='BACE' else 'regression'
    key='auc' if task=='classification' else 'rmse'
    ref=json.loads((REFERENCE/f'run_{name}_{depth}_{seed}_bp.json').read_text())
    target=ref['validation'][key]
    cfg=json.loads((REFERENCE/'manifest.json').read_text())['config']
    split=json.loads((REFERENCE/f'split_{name}_{seed}.json').read_text())['indices']
    graphs,_=load_graphs(name)
    batches={k:GraphBatch.from_graphs([graphs[i] for i in ids]) for k,ids in split.items()}
    del graphs
    torch.manual_seed(seed)
    model=GraphNetwork(batches['train'].x.size(1),cfg['hidden'],depth,2 if task=='classification' else 1)
    mean=float(batches['train'].y.mean()) if task=='regression' else 0.
    scale=max(float(batches['train'].y.std(unbiased=False)),1e-8) if task=='regression' else 1.
    monitor=Memory(); monitor.thread.start()
    start=time.perf_counter()
    trace=[]; hit=None; updates=0; final_best=None; final_best_value=None
    with torch.no_grad():
        aggregates={k:torch.sparse.mm(batches[k].adj,batches[k].x) for k in ('train','val')}
    stages=range(depth) if method=='sf' else range(1)
    timed_out=False
    for stage in stages:
        final_stage=method=='bp' or stage==depth-1
        if method=='sf':
            params=list(model.layers[stage].parameters())+list(model.heads[stage].parameters())
        else:
            params=list(model.layers.parameters())+list(model.heads[-1].parameters())
        opt=torch.optim.Adam(params,lr=cfg['lr'],weight_decay=5e-4)
        budget=(args.final_steps if final_stage else cfg['epochs']) if method=='sf' else args.final_steps+(depth-1)*cfg['epochs']
        best=None; best_value=None
        def forward(k):
            if method=='bp':
                return model(batches[k],aggregates[k])
            h=torch.tanh(model.layers[stage](aggregates[k].detach()))
            return model.heads[stage](batches[k].readout(h))
        for step in range(1,budget+1):
            model.train(); opt.zero_grad(set_to_none=True)
            loss=objective(forward('train'),batches['train'].y,task,mean,scale)
            loss.backward(); opt.step(); updates+=1
            if step%5==0 or step==budget:
                model.eval()
                with torch.no_grad():
                    value=metrics(forward('val'),batches['val'].y,task,mean,scale)[key]
                elapsed=time.perf_counter()-start
                trace.append(dict(stage=stage+1,step=step,total_updates=updates,seconds=elapsed,value=value,eligible=final_stage))
                if best_value is None or (value>best_value if task=='classification' else value<best_value):
                    best_value=value
                    best=copy.deepcopy(model.state_dict())
                monitor.sample()
                if final_stage and elapsed<=args.seconds and attained(task,value,target):
                    hit=dict(seconds=elapsed,value=value,step=step,total_updates=updates)
                    break
            if time.perf_counter()-start>=args.seconds:
                timed_out=True
                break
        if final_stage:
            final_best,final_best_value=best,best_value
        if hit is not None or timed_out:
            break
        model.load_state_dict(best)
        if method=='sf' and not final_stage:
            for p in params:
                p.requires_grad_(False); p.grad=None
            with torch.no_grad():
                aggregates={k:torch.sparse.mm(batches[k].adj,torch.tanh(model.layers[stage](aggregates[k]))) for k in aggregates}
    train_end=time.perf_counter()-start
    memory=monitor.finish()
    # All test operations occur after the timer and memory measurement stop.
    test=None
    if hit is not None or final_best is not None:
        if hit is None:
            model.load_state_dict(final_best)
        with torch.no_grad():
            test=metrics(model(batches['test']),batches['test'].y,task,mean,scale)
        torch.save(model.state_dict(),OUT/f'{name}_{depth}_{seed}_{method}.pt')
    row=dict(dataset=name,depth=depth,seed=seed,method=method,task=task,metric=key,target=target,
             reached=hit is not None,hit=hit,observed_seconds=train_end,updates=updates,
             stop_reason='target' if hit else ('time_limit' if timed_out else 'update_limit'),
             best_final_validation=final_best_value,test=test,memory=memory,curve=trace,
             final_steps=args.final_steps,time_limit_seconds=args.seconds)
    (OUT/f'{name}_{depth}_{seed}_{method}.json').write_text(json.dumps(row,indent=2,allow_nan=False))
    print(f'{name} L{depth} seed{seed} {method} hit={hit} peakRSS={memory["sampled_training_peak_rss_mib"]:.1f}MiB',flush=True)


def report():
    rows=[json.loads(p.read_text()) for p in OUT.glob('*.json') if p.name!='manifest.json']
    lines=['# 达到同等验证性能的时间与内存峰值','',f'完成运行：{len(rows)}/24。','',
           '## 指标口径','',
           '- BACE 以验证 ROC-AUC 越高越好，ESOL 以验证 RMSE 越低越好。每个数据/深度/种子的共同目标固定为前一轮 BP 最佳验证指标；目标不读取测试集。这个目标是 BP 参考线，不是对称或外部业务目标。',
           '- 两种方法从相同初始化开始，沿用前轮精确骨架划分和超参数。SF 低层仍各训练 60 步并按验证选择，最后一层最多 600 步；BP 最多 600+60×(深度−1) 步。每次另有 30 秒训练上限。',
           '- 每 5 步验证一次。只用指定深度最终层的指标判定达标，SF 低层的好成绩不提前算作成功；SF 低层训练、缓存和选权重时间全部累计。记录首次检测到达标的时间，不要求连续多次达标。',
           '- 独立新进程、CPU 单线程、串行运行；计时含首次优化器初始化、训练、验证、缓存和权重选择，排除加载数据、构造批次/初始模型、最终测试、磁盘写入。不是预热后的纯 kernel 耗时。',
           '- 每 10 ms 采样进程 RSS，并在验证时追加采样。报告训练窗口绝对 RSS 峰值和相对训练开始 RSS 的增量；采样可能漏掉短暂尖峰。RSS 包含 Python/库、数据、缓存和模型，不能当作 GPU 显存或纯张量内存。',
           '- 原始 JSON 同时保存操作系统 ru_maxrss（进程启动到训练结束的高水位），其范围含导入和数据预处理，不能视为纯训练峰值。已达标行的内存截止达标停止，未达标行覆盖完整尝试预算。',
           '- 未达标保留预算/停止原因，不能删除后只比较成功样本。只有同一配置与种子双方都达标时才计算配对时间比。小样本未做显著性检验。','',
           '## 逐次结果','',
           '| 数据 | 层 | 种子 | 方法 | 共同验证目标 | 达标秒数 | 尝试秒数 | 停止原因 | RSS峰值 MiB | RSS增量 MiB |',
           '|---|---:|---:|---|---:|---:|---:|---|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['dataset'],r['depth'],r['seed'],r['method'])):
        t=f"{r['hit']['seconds']:.3f}" if r['reached'] else '未达标'
        lines.append(f"| {r['dataset']} | {r['depth']} | {r['seed']} | {r['method'].upper()} | {r['target']:.6f} | {t} | {r['observed_seconds']:.3f} | {r['stop_reason']} | {r['memory']['sampled_training_peak_rss_mib']:.1f} | {r['memory']['sampled_training_increment_mib']:.1f} |")
    lines+=['','## 配对汇总','','| 数据 | 层 | BP 达标 | SF 达标 | 双方达标配对数 | BP/SF 时间比几何均值 |','|---|---:|---:|---:|---:|---:|']
    for name in ('BACE','ESOL'):
        for depth in (2,4):
            group=[r for r in rows if (r['dataset'],r['depth'])==(name,depth)]
            ratios=[]
            for seed in (11,22,33):
                pair={r['method']:r for r in group if r['seed']==seed}
                if len(pair)==2 and all(r['reached'] for r in pair.values()):
                    ratios.append(pair['bp']['hit']['seconds']/pair['sf']['hit']['seconds'])
            ratio=f'{statistics.geometric_mean(ratios):.3f}' if ratios else '—'
            b=sum(r['reached'] for r in group if r['method']=='bp');s=sum(r['reached'] for r in group if r['method']=='sf')
            lines.append(f'| {name} | {depth} | {b}/3 | {s}/3 | {len(ratios)} | {ratio} |')
    lines+=['','时间比 >1 表示在双方达标的配对中 SF 更快；不能外推到未达标运行。验证达标不保证测试表现相同，原始记录保存了达标权重的测试指标。','',
            f'原始记录和权重：`{OUT}`。','',
           '运行：`python -m graph_level.experiments.graph_time_to_target`；仅重建报告：`python -m graph_level.experiments.graph_time_to_target --report-only`。']
    (ROOT/'docs/sf_time_to_target_memory.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',action='store_true')
    p.add_argument('--dataset');p.add_argument('--depth',type=int);p.add_argument('--seed',type=int)
    p.add_argument('--method',choices=['bp','sf'])
    p.add_argument('--final-steps',type=int,default=600)
    p.add_argument('--seconds',type=float,default=30.)
    p.add_argument('--report-only',action='store_true')
    args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.worker:
        worker(args);return
    if args.report_only:
        report();return
    if args.final_steps!=600 or args.seconds!=30:
        p.error('This version fixes budgets to avoid mixing configurations')
    paths=[Path(__file__),ROOT/'graph_level/model/graph_level.py',ROOT/'graph_level/experiments/local_graph_tasks.py']
    paths+=sorted(REFERENCE.glob('run_*_bp.json'))+sorted(REFERENCE.glob('split_*.json'))+[REFERENCE/'manifest.json']
    manifest=dict(platform=platform.platform(),torch=torch.__version__,threads=1,final_steps=600,seconds=30,
                  sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    mp=OUT/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:
        raise ValueError('Code/reference changed; preserve previous results and use another version')
    mp.write_text(json.dumps(manifest,indent=2))
    for name in ('BACE','ESOL'):
        for depth in (2,4):
            for seed in (11,22,33):
                for method in (('bp','sf') if seed%2 else ('sf','bp')):
                    path=OUT/f'{name}_{depth}_{seed}_{method}.json'
                    if not path.exists():
                        subprocess.run([sys.executable,'-u','-m','graph_level.experiments.graph_time_to_target','--worker',
                                        '--dataset',name,'--depth',str(depth),'--seed',str(seed),'--method',method],check=True,cwd=ROOT)
                    report()


if __name__=='__main__':
    main()
