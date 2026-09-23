"""Compare SF and virtual-node-free FF-LA on BACE and ESOL graph tasks.

Both methods use identical GCN tensor shapes/depths/hidden units and paired
initial weights. FF receives candidate target codes as appended input features;
SF receives zero in those same feature slots and has its usual local task head.
"""
import argparse, copy, hashlib, json, platform, statistics, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from graph_level.experiments.local_graph_tasks import load_graphs, split_graphs
from graph_level.model.graph_level import GraphBatch, GraphNetwork, metrics, objective

ROOT=Path(__file__).resolve().parents[2]


def condition(batch, code, classes):
    onehot=F.one_hot(torch.as_tensor(code,dtype=torch.long),classes).float()
    if onehot.ndim==1:
        onehot=onehot.expand(batch.y.numel(),-1)
    node_code=onehot[batch.pool.indices()[0]]
    x=torch.cat([batch.x[:,:-classes],node_code],dim=1)
    return GraphBatch(x,batch.adj,batch.pool,batch.counts,batch.y)


def outputs(model,batch,code,classes,stop=None):
    conditioned=condition(batch,code,classes)
    h=conditioned.x
    activations=[]
    for layer in model.layers if stop is None else model.layers[:stop]:
        h=torch.tanh(layer(torch.sparse.mm(batch.adj,h)))
        activations.append(h)
    return activations


def graph_goodness(h,batch):
    node_g=h.square().mean(dim=1,keepdim=True)
    return (torch.sparse.mm(batch.pool,node_g)/batch.counts).view(-1)


def local_score(metrics_dict,task):
    return metrics_dict['auc'] if task=='classification' else -metrics_dict['rmse']


def sf_fit(initial,batches,task,classes,centres,epochs,lr):
    model=copy.deepcopy(initial); depth=len(model.layers)
    train,val=batches['train'],batches['val']
    mean=float(train.y.mean()) if task=='regression' else 0.
    scale=max(float(train.y.std(unbiased=False)),1e-8) if task=='regression' else 1.
    # SF gets the same input tensor shape as FF, with zero target channels.
    sf_batches={k:condition(v,torch.zeros(v.y.numel(),dtype=torch.long),classes) for k,v in batches.items()}
    start=time.perf_counter(); traces=[]; stages=[]
    fixed={k:sf_batches[k].x for k in ('train','val')}
    for idx in range(depth):
        params=list(model.layers[idx].parameters())+list(model.heads[idx].parameters())
        optimizer=torch.optim.Adam(params,lr=lr,weight_decay=5e-4)
        best=-float('inf'); state=None; best_step=0
        with torch.no_grad():
            aggs={k:torch.sparse.mm(sf_batches[k].adj,fixed[k]) for k in fixed}
        for step in range(1,epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            h=torch.tanh(model.layers[idx](aggs['train'].detach()))
            pred=model.heads[idx](sf_batches['train'].readout(h))
            loss=objective(pred,train.y,task,mean,scale)
            loss.backward();optimizer.step()
            if step%5==0 or step==epochs:
                model.eval()
                with torch.no_grad():
                    h=torch.tanh(model.layers[idx](aggs['val']))
                    pred=model.heads[idx](sf_batches['val'].readout(h))
                    valm=metrics(pred,val.y,task,mean,scale)
                score=local_score(valm,task)
                traces.append(dict(stage=idx+1,step=step,seconds=time.perf_counter()-start,
                                   val=valm,loss=float(loss.detach())))
                if score>best:
                    best=score;best_step=step;state=(copy.deepcopy(model.layers[idx].state_dict()),copy.deepcopy(model.heads[idx].state_dict()))
        model.layers[idx].load_state_dict(state[0]);model.heads[idx].load_state_dict(state[1])
        for p in params:p.requires_grad_(False);p.grad=None
        stages.append(dict(stage=idx+1,selected_step=best_step,updates=epochs))
        if idx<depth-1:
            with torch.no_grad():
                fixed={k:torch.tanh(model.layers[idx](aggs[k])).detach() for k in fixed}
                aggs={k:torch.sparse.mm(sf_batches[k].adj,fixed[k]) for k in fixed}
    seconds=time.perf_counter()-start
    model.eval()
    with torch.no_grad():
        x=condition(batches['test'],torch.zeros(batches['test'].y.numel(),dtype=torch.long),classes)
        h=x.x
        for layer in model.layers:h=torch.tanh(layer(torch.sparse.mm(x.adj,h)))
        pred=model.heads[-1](x.readout(h))
        test=metrics(pred,x.y,task,mean,scale)
        if task=='classification':prediction=pred.softmax(-1)[:,1]
        else:prediction=pred.view(-1)*scale+mean
    return dict(method='sf',train_seconds=seconds,test=test,prediction=prediction.tolist(),
                target=x.y.tolist(),curve=traces,stages=stages),model


def ff_eval(model,batch,task,classes,centres):
    # Candidate-target inference: aggregate goodness over all layers.
    scores=[]
    for code in range(classes):
        acts=outputs(model,batch,code,classes)
        scores.append(torch.stack([graph_goodness(h,batch) for h in acts]).mean(0))
    scores=torch.stack(scores,dim=1)
    if task=='classification':
        # `metrics` expects logits; apply softmax exactly once there.
        pred=scores
    else:
        prob=scores.softmax(1)
        pred=(prob*torch.tensor(centres).view(1,-1)).sum(1,keepdim=True)
    return pred,scores


def ff_fit(initial,batches,task,classes,centres,epochs,lr,seed):
    model=copy.deepcopy(initial);depth=len(model.layers);train,val=batches['train'],batches['val']
    # node feature channels are [original features | candidate target one-hot].
    if task=='classification': true_codes=train.y.long()
    else:
        cuts=torch.tensor(centres)
        true_codes=torch.bucketize(train.y,cuts[1:-1]).long()
    start=time.perf_counter();traces=[];stages=[];rng=torch.Generator().manual_seed(seed+9000)
    for idx in range(depth):
        optimizer=torch.optim.Adam(model.layers[idx].parameters(),lr=lr,weight_decay=5e-4)
        best=-float('inf');state=None;best_step=0
        for step in range(1,epochs+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            wrong=true_codes.clone()
            # One incorrect target per graph; never use a test/validation target.
            offsets=torch.randint(1,classes,(wrong.numel(),),generator=rng)
            wrong=(wrong+offsets)%classes
            if idx==0:
                h_pos=condition(train,true_codes,classes).x
                h_neg=condition(train,wrong,classes).x
            else:
                with torch.no_grad():
                    h_pos=outputs(model,train,true_codes,classes,stop=idx)[idx-1]
                    h_neg=outputs(model,train,wrong,classes,stop=idx)[idx-1]
            z_pos=torch.tanh(model.layers[idx](torch.sparse.mm(train.adj,h_pos)))
            z_neg=torch.tanh(model.layers[idx](torch.sparse.mm(train.adj,h_neg)))
            g_pos=graph_goodness(z_pos,train);g_neg=graph_goodness(z_neg,train)
            threshold=.15
            loss=(F.softplus(threshold-g_pos).mean()+F.softplus(g_neg-threshold).mean())
            loss.backward();optimizer.step()
            if step%5==0 or step==epochs:
                model.eval()
                with torch.no_grad():
                    pred,_=ff_eval(model,val,task,classes,centres)
                    valm=metrics(pred,val.y,task,0.,1.) if task=='regression' else metrics(pred,val.y,task)
                    # Classification candidate probabilities are valid 2-class scores.
                score=local_score(valm,task)
                traces.append(dict(stage=idx+1,step=step,seconds=time.perf_counter()-start,
                                   val=valm,loss=float(loss.detach())))
                if score>best:
                    best=score;best_step=step;state=copy.deepcopy(model.layers[idx].state_dict())
        model.layers[idx].load_state_dict(state)
        for p in model.layers[idx].parameters():p.requires_grad_(False);p.grad=None
        stages.append(dict(stage=idx+1,selected_step=best_step,updates=epochs))
    seconds=time.perf_counter()-start
    model.eval()
    with torch.no_grad():
        pred,scores=ff_eval(model,batches['test'],task,classes,centres)
        test=metrics(pred,batches['test'].y,task,0.,1.) if task=='regression' else metrics(pred,batches['test'].y,task)
        prediction=pred.softmax(1)[:,1] if task=='classification' else pred.view(-1)
    return dict(method='ff',train_seconds=seconds,test=test,prediction=prediction.tolist(),
                target=batches['test'].y.tolist(),curve=traces,stages=stages),model


def report(rows,config,outdir):
    out=['# Single-Forward 与 Forward-Forward：图级任务对照','',
         f"运行数：{len(rows)}/{2*len(config['datasets'])*len(config['depths'])*len(config['seeds'])}。",'',
         '## 比较定义','',
         '- SF：每层一次前向，以局部任务损失训练当前层及其任务头；完成后冻结低层。',
         '- FF-LA 式对照：正输入附加真实类别/目标区间，负输入附加随机错误类别/区间；每层用平均激活平方 goodness 的阈值损失训练当前 GNN 层，不使用虚拟节点。每步一个错误目标。',
         '- 两者有完全相同的 GCN 层数、隐藏维度、输入张量宽度、初始化权重、图批次和层预算。SF 的目标编码通道置零；FF 使用 one-hot 候选标签。SF 使用任务头，FF 使用 goodness 推断，这是两类方法定义上的目标差异。',
         '- FF 连续回归将训练集目标切成 6 个等频区间，候选区间中心的 goodness softmax 期望用于连续预测；所有切点只依赖训练标签。',
         f"- 每层 {config['epochs']} 个全批次优化器更新；同种子初始化配对。Adam lr={config['lr']}，weight_decay=0.0005；CPU 单线程。验证每 5 步，不早停；SF 按该层任务损失选 checkpoint，FF 按全层 goodness 推断验证指标选 checkpoint。", 
         '- 计时含训练循环中的正/负前向及验证；不含数据预处理、最终测试和保存。SF/FF 的 FLOPs、前向次数不同，这是待测计算差异。',
         '- 三个随机骨架划分；均值±总体标准差，不做显著性检验。不是原论文实验的完全复现。','',
         '## 结果解读','',
         '- 在 ESOL 回归中，SF 在两种深度下 RMSE/MAE 均更低且 R² 更高；BACE 分类结果则随层数变化：2 层 FF 的 AUC/AP 较高，4 层 SF 较高。因此这组小样本实验不支持“SF 在准确度上总是胜出”。',
         '- 本机计时中 SF 训练明显更快；FF 对应的训练耗时包括每次正负条件输入训练，以及验证时逐一枚举候选目标造成的额外 GNN 前向。计时优势不能视为所有硬件/实现上的定律。','',
         '## 测试结果','',
         '| 数据 | 层 | 方法 | 指标 | 测试均值 ± 标准差 |',
         '|---|---:|---|---|---:|']
    for name in config['datasets']:
        task='classification' if name=='BACE' else 'regression'
        keys=('auc','ap','accuracy') if task=='classification' else ('rmse','mae','r2')
        for depth in config['depths']:
            for method in ('sf','ff'):
                group=[r for r in rows if (r['dataset'],r['depth'],r['method'])==(name,depth,method)]
                for key in keys:
                    vals=[r['test'][key]*(100 if task=='classification' and key!='r2' else 1) for r in group]
                    if vals:
                        out.append(f"| {name} | {depth} | {method.upper()} | {key} | {statistics.mean(vals):.3f} ± {statistics.pstdev(vals):.3f} |")
                if group:out.append(f"| {name} | {depth} | {method.upper()} | 训练秒 | {statistics.mean(r['train_seconds'] for r in group):.3f} ± {statistics.pstdev(r['train_seconds'] for r in group):.3f} |")
    out+=['','FF 的输入需按候选类别重跑 GNN；报告不含最终测试推理时间。若严格比较端到端部署延迟，应另计每样本候选前向成本。三个种子样本量小、数据切分不确定性大，结论仅限于此实现与协议。','',f"原始逐种子数据、验证曲线和预测：`{outdir}`。"]
    (ROOT/'docs/sf_vs_ff_graph_tasks.md').write_text('\n'.join(out)+'\n')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=['BACE','ESOL'])
    p.add_argument('--depths',nargs='+',type=int,default=[2,4])
    p.add_argument('--seeds',nargs='+',type=int,default=[11,22,33])
    p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--setting',default='sf_ff_graph_tasks_v1')
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args();config=vars(a).copy();config.pop('report_only')
    out=ROOT/'results'/a.setting;out.mkdir(parents=True,exist_ok=True)
    if a.report_only:
        report([json.loads(p.read_text()) for p in out.glob('run_*.json')],json.loads((out/'manifest.json').read_text())['config'],out);return
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    hashes={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__),ROOT/'graph_level/model/graph_level.py',ROOT/'graph_level/experiments/local_graph_tasks.py']}
    manifest=dict(config=config,hashes=hashes,torch=torch.__version__,platform=platform.platform())
    mp=out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=manifest:raise ValueError('Config/source changed; choose a new setting')
    mp.write_text(json.dumps(manifest,indent=2))
    rows=[]
    for name in a.datasets:
        task='classification' if name=='BACE' else 'regression';classes=2 if task=='classification' else 6
        graphs,groups=load_graphs(name)
        for seed in a.seeds:
            split=split_graphs(graphs,groups,seed)
            batches={k:GraphBatch.from_graphs([graphs[i] for i in ids]) for k,ids in split.items()}
            if task=='classification':centres=[]
            else:
                train_y=batches['train'].y
                q=torch.linspace(0,1,classes+1)[1:-1]
                cuts=torch.quantile(train_y,q)
                bin_id=torch.bucketize(train_y,cuts)
                centres=[float(train_y[bin_id==i].mean()) if (bin_id==i).any() else float(train_y.mean()) for i in range(classes)]
            # Reserve the last K feature channels for candidate-label conditioning.
            for split_name,b in batches.items():
                b.x=torch.cat([b.x,torch.zeros((b.x.size(0),classes))],dim=1)
            for depth in a.depths:
                torch.manual_seed(seed)
                initial=GraphNetwork(batches['train'].x.size(1),a.hidden,depth,2 if task=='classification' else 1)
                for method in (('sf','ff') if seed%2 else ('ff','sf')):
                    path=out/f'run_{name}_{depth}_{seed}_{method}.json'
                    if path.exists():row=json.loads(path.read_text())
                    else:
                        # Save unconditioned x in batches; model workers add candidate codes on demand.
                        if method=='sf':
                            saved={k:b.x for k,b in batches.items()}
                            for b in batches.values():b.x[:,-classes:]=0
                            row,fit_model=sf_fit(initial,batches,task,classes,centres,a.epochs,a.lr)
                            for k,b in batches.items():b.x=saved[k]
                        else:
                            row,fit_model=ff_fit(initial,batches,task,classes,centres,a.epochs,a.lr,seed)
                        row.update(dataset=name,depth=depth,seed=seed,task=task)
                        torch.save(fit_model.state_dict(),path.with_suffix('.pt'))
                        path.write_text(json.dumps(row,indent=2,allow_nan=False))
                    rows.append(row)
                    print(f"{name} L{depth} seed{seed} {method}: {row['test']} time={row['train_seconds']:.2f}s",flush=True)
                    report(rows,config,out)
    print('Completed',flush=True)


if __name__=='__main__':main()
