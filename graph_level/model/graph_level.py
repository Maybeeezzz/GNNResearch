"""Graph-level supervised local learning without class/virtual nodes."""
import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, r2_score
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn.conv.gcn_conv import gcn_norm


@dataclass
class GraphBatch:
    x: torch.Tensor
    adj: torch.Tensor
    pool: torch.Tensor
    counts: torch.Tensor
    y: torch.Tensor

    @classmethod
    def from_graphs(cls, graphs):
        b = Batch.from_data_list(graphs)
        edges, values = gcn_norm(b.edge_index, num_nodes=b.num_nodes,
                                 add_self_loops=True, dtype=torch.float32)
        adj = torch.sparse_coo_tensor(edges.flip(0), values,
                                     (b.num_nodes, b.num_nodes)).coalesce()
        indices = torch.stack([b.batch, torch.arange(b.num_nodes)])
        pool = torch.sparse_coo_tensor(indices, torch.ones(b.num_nodes),
                                      (len(graphs), b.num_nodes)).coalesce()
        counts = torch.bincount(b.batch).float().view(-1, 1)
        return cls(b.x.float(), adj, pool, counts, b.y.view(-1).float())

    def readout(self, h):
        total = torch.sparse.mm(self.pool, h)
        # Both local and BP heads see identical, size-sensitive readout.
        return torch.cat([total / self.counts, total], dim=1)


class GraphNetwork(nn.Module):
    def __init__(self, input_dim, hidden, depth, outputs):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden, hidden) for i in range(depth)
        ])
        self.heads = nn.ModuleList([nn.Linear(2 * hidden, outputs) for _ in range(depth)])

    def forward(self, batch, first_aggregate=None):
        h = batch.x
        for i, layer in enumerate(self.layers):
            a = first_aggregate if i == 0 and first_aggregate is not None else torch.sparse.mm(batch.adj, h)
            h = torch.tanh(layer(a))
        return self.heads[-1](batch.readout(h))


def objective(pred, y, task, mean, scale):
    if task == 'classification':
        return F.cross_entropy(pred, y.long())
    return F.mse_loss(pred.view(-1), (y-mean)/scale)


def metrics(pred, y, task, mean=0., scale=1.):
    labels = y.detach().cpu().numpy()
    if task == 'classification':
        prob = pred.softmax(-1)[:, 1].detach().cpu().numpy()
        return dict(auc=float(roc_auc_score(labels, prob)),
                    ap=float(average_precision_score(labels, prob)),
                    accuracy=float(accuracy_score(labels, prob >= .5)))
    prediction = pred.detach().view(-1).cpu().numpy()*scale+mean
    error = prediction-labels
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))), r2=float(r2_score(labels, prediction)))


def train_graph(initial, batches, task, method, epochs, lr):
    """Match optimizer-call count; validate per stage and test only final model."""
    model = copy.deepcopy(initial)
    train, val, test = (batches[k] for k in ('train','val','test'))
    mean = float(train.y.mean()) if task == 'regression' else 0.
    scale = max(float(train.y.std(unbiased=False)), 1e-8) if task == 'regression' else 1.
    start = time.perf_counter()
    depth = len(model.layers)
    fixed = {'train': train.x, 'val': val.x}
    stages, curve = [], []
    # Fixed first-layer aggregation is available to BOTH methods.
    with torch.no_grad():
        aggregates = {k: torch.sparse.mm(batches[k].adj, fixed[k]) for k in fixed}
    for stage in range(depth if method == 'sf' else 1):
        if method == 'sf':
            params = list(model.layers[stage].parameters())+list(model.heads[stage].parameters())
            if stage:
                with torch.no_grad():
                    aggregates = {k: torch.sparse.mm(batches[k].adj, fixed[k]) for k in fixed}
        else:
            params = list(model.layers.parameters())+list(model.heads[-1].parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=5e-4)
        budget = epochs if method == 'sf' else epochs*depth
        best_score, best_state, best_epoch = -float('inf'), None, None

        def forward(name):
            if method == 'bp':
                return model(batches[name], aggregates[name])
            h = torch.tanh(model.layers[stage](aggregates[name].detach()))
            return model.heads[stage](batches[name].readout(h))

        for epoch in range(1, budget+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = objective(forward('train'),train.y,task,mean,scale)
            loss.backward()
            optimizer.step()
            if epoch % 5 == 0 or epoch == budget:
                model.eval()
                with torch.no_grad():
                    v = metrics(forward('val'),val.y,task,mean,scale)
                score = v['auc'] if task == 'classification' else -v['rmse']
                curve.append(dict(stage=stage+1, epoch=epoch, elapsed=time.perf_counter()-start,
                                  train_loss=float(loss.detach()), validation=v))
                if score > best_score:
                    best_score, best_epoch = score, epoch
                    if method == 'sf':
                        best_state = (copy.deepcopy(model.layers[stage].state_dict()),
                                      copy.deepcopy(model.heads[stage].state_dict()))
                    else:
                        best_state = copy.deepcopy(model.state_dict())
        if method == 'sf':
            model.layers[stage].load_state_dict(best_state[0])
            model.heads[stage].load_state_dict(best_state[1])
            for p in params:
                p.requires_grad_(False)
                p.grad = None
            with torch.no_grad():
                fixed = {k: torch.tanh(model.layers[stage](aggregates[k])).detach() for k in fixed}
        else:
            model.load_state_dict(best_state)
        stages.append(dict(stage=stage+1, updates=budget, selected_epoch=best_epoch, selection_score=best_score))
    train_seconds = time.perf_counter()-start
    model.eval()
    with torch.no_grad():
        validation = metrics(model(val),val.y,task,mean,scale)
        prediction = model(test)
        testing = metrics(prediction,test.y,task,mean,scale)
        saved_pred = prediction.softmax(-1)[:,1] if task == 'classification' else prediction.view(-1)*scale+mean
    return dict(method=method, train_seconds=train_seconds, validation=validation,test=testing,
                stages=stages, curve=curve, target_mean=mean, target_scale=scale,
                test_prediction=saved_pred.tolist(), test_target=test.y.tolist()), model
