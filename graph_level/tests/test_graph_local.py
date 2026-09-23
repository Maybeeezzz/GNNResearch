import copy

import pytest
import torch
from torch_geometric.data import Data

from graph_level.experiments.local_graph_tasks import split_graphs
from graph_level.model.graph_level import GraphBatch, GraphNetwork, train_graph


def tiny_batches():
    torch.manual_seed(7)
    graphs = [Data(x=torch.randn(3,4),edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]]),
                   y=torch.tensor([i%2],dtype=torch.float)) for i in range(12)]
    return {k:GraphBatch.from_graphs(graphs[j:j+4]) for k,j in [('train',0),('val',4),('test',8)]}


def test_batch_isolation():
    b=tiny_batches()['train']
    torch.manual_seed(2)
    model=GraphNetwork(4,8,2,2)
    original=model(b).detach()
    changed=copy.deepcopy(b)
    changed.x[:3]+=100
    modified=model(changed).detach()
    torch.testing.assert_close(original[1:],modified[1:])


@pytest.mark.parametrize('task,outputs',[('classification',2),('regression',1)])
def test_one_layer_bp_sf_equivalence(task,outputs):
    torch.set_num_threads(1)
    batches=tiny_batches()
    model=GraphNetwork(4,8,1,outputs)
    bp,_=train_graph(model,batches,task,'bp',5,.001)
    sf,frozen=train_graph(model,batches,task,'sf',5,.001)
    assert bp['test']==sf['test']
    torch.testing.assert_close(torch.tensor(bp['test_prediction']),torch.tensor(sf['test_prediction']))
    assert all(not p.requires_grad and p.grad is None for p in frozen.parameters())


def test_scaffolds_are_disjoint_and_indices_exhaustive():
    groups=[str(i//2) for i in range(40)]
    split=split_graphs([None]*40,groups,11)
    assert sorted(sum(split.values(),[]))==list(range(40))
    sets=[{groups[i] for i in indices} for indices in split.values()]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))


def test_regression_scaler_uses_training_targets_only():
    batches=tiny_batches()
    batches['test'].y+=1000
    model=GraphNetwork(4,8,2,1)
    result,_=train_graph(model,batches,'regression','sf',5,.001)
    assert result['target_mean']==.5
    assert result['target_scale']==.5
