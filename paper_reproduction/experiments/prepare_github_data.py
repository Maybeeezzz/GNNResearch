"""Recover the normalized GitHub graph from the authors' released edge splits.

This is a data-source adapter, not a replacement dataset. All five reconstructions
must have identical features, node labels and complete positive edge sets.
"""
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.utils import is_undirected

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'third_party/forwardgnn/data/GitHub/author_recovered.pt'
PROVENANCE = ROOT / 'results/reproduction/github_data_provenance.json'


def recover():
    source = ROOT / 'third_party/forwardgnn-datasplits/datasplits/GitHub/edge-5splits'
    hashes = {}
    reference = None
    with torch.serialization.safe_globals([Data, DataTensorAttr, DataEdgeAttr, GlobalStorage]):
        for fold in range(5):
            positives = []
            features = labels = None
            for mode in ('train', 'val', 'test'):
                path = source / f'{mode}_data-split{fold}.pt'
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                data = torch.load(path, weights_only=True, map_location='cpu')
                if features is None:
                    features, labels = data.x, data.y
                assert torch.equal(data.x, features) and torch.equal(data.y, labels)
                assert data.x.shape == (37700, 128) and data.y.shape == (37700,)
                assert torch.isfinite(data.x).all() and data.x.min() >= 0
                assert torch.allclose(data.x.sum(1), torch.ones(37700), atol=1e-6)
                edges = data.edge_label_index[:, data.edge_label == 1]
                assert (edges[0] < edges[1]).all(), 'Expected upper-triangular positive edges'
                positives.append(edges)
            edges = torch.cat(positives, dim=1)
            codes = torch.unique(edges[0] * 37700 + edges[1], sorted=True)
            assert len(codes) == edges.shape[1] == 289003
            if reference is None:
                reference = (features, labels, codes)
            else:
                assert all(torch.equal(a, b) for a, b in zip(reference, (features, labels, codes)))
    features, labels, codes = reference
    edges = torch.stack((codes // 37700, codes % 37700))
    edges = torch.cat((edges, edges.flip(0)), dim=1)
    order = torch.argsort(edges[0] * 37700 + edges[1])
    edges = edges[:, order].contiguous()
    assert edges.shape == (2, 578006) and is_undirected(edges)
    assert torch.equal(torch.bincount(labels), torch.tensor([27961, 9739]))
    # Confirm all published node partitions refer to precisely this node universe.
    node_source = source.parent / 'node-5splits'
    for fold in range(5):
        parts = [torch.load(node_source / f'{mode}-node-index-split{fold}.pt', weights_only=True)
                 for mode in ('train', 'val', 'test')]
        assert torch.equal(torch.sort(torch.cat(parts).long()).values, torch.arange(37700))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(x=features, y=labels, edge_index=edges), CACHE)
    metadata = dict(source='author-published ForwardGNN edge splits',
                    source_repository='https://github.com/NamyongPark/forwardgnn-datasplits',
                    source_commit='48997da6fc96fd11ee3dfed47e720d942853e729',
                    node_count=37700, directed_edge_count=578006, feature_count=128,
                    normalized_features=True, normalization='Preserve author tensors; do not normalize again',
                    graph_recovery='Union of positive train/validation/test edges, made symmetric; all five folds agree exactly',
                    limitation='Original edge ordering unavailable; edges sorted by source/destination',
                    source_sha256=hashes, cache_sha256=hashlib.sha256(CACHE.read_bytes()).hexdigest())
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k:v for k,v in metadata.items() if k != 'source_sha256'}, indent=2))


class RecoveredGitHub:
    num_classes = 2
    num_features = 128

    def __init__(self):
        metadata = json.loads(PROVENANCE.read_text())
        if hashlib.sha256(CACHE.read_bytes()).hexdigest() != metadata['cache_sha256']:
            raise ValueError('Recovered GitHub data hash mismatch')
        self.data = Data(**torch.load(CACHE, weights_only=True, map_location='cpu'))

    def __len__(self):
        return 1

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self.data.clone()


def install_adapter():
    import datasets.dataloader as loader
    original = loader.load_dataset

    def load_dataset(name):
        return RecoveredGitHub() if name == 'GitHub' else original(name)

    loader.load_dataset = load_dataset


if __name__ == '__main__':
    recover()
