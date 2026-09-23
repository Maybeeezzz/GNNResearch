import torch

from link_prediction.experiments.local_link_prediction import Encoder, edge_split, fit


def test_split_excludes_held_out_edges_and_negative_leakage():
    edges = torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8],
                          [1,0,2,1,3,2,4,3,5,4,6,5,7,6,8,7]])
    split = edge_split(edges, 12, 7)
    positives, negatives = [], []
    for name in ('train','val','test'):
        pairs, y = split[name]
        positives.append(set(map(tuple,pairs[:,y.bool()].t().tolist())))
        negatives.append(set(map(tuple,pairs[:,~y.bool()].t().tolist())))
    assert not set.union(*positives) & set.union(*negatives)
    assert sum(map(len, negatives)) == len(set.union(*negatives))
    messages = {tuple(sorted(e)) for e in split['message_edges'].t().tolist()}
    assert messages == positives[0]
    assert not messages & (positives[1] | positives[2])


def test_one_layer_local_training_matches_bp():
    torch.set_num_threads(1)
    torch.manual_seed(9)
    x = torch.randn(12, 4)
    edges = torch.stack([torch.arange(11),torch.arange(1,12)])
    split = edge_split(edges,12,7)
    model = Encoder(4,8,1)
    bp = fit(model,x,split,'bp',5,.001)
    sf = fit(model,x,split,'sf',5,.001)
    assert bp['test'] == sf['test']
    assert bp['validation'] == sf['validation']
    assert bp['curve'][0]['loss'] == sf['curve'][0]['loss']
