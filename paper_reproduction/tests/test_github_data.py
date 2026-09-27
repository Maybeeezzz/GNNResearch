import hashlib
import json

import pytest
import torch

from paper_reproduction.experiments import prepare_github_data as github


def test_adapter_preserves_author_features_and_rejects_corruption(tmp_path, monkeypatch):
    cache = tmp_path / 'graph.pt'
    provenance = tmp_path / 'provenance.json'
    features = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    torch.save(dict(x=features, y=torch.tensor([0, 1]),
                    edge_index=torch.tensor([[0, 1], [1, 0]])), cache)
    provenance.write_text(json.dumps({'cache_sha256': hashlib.sha256(cache.read_bytes()).hexdigest()}))
    monkeypatch.setattr(github, 'CACHE', cache)
    monkeypatch.setattr(github, 'PROVENANCE', provenance)
    dataset = github.RecoveredGitHub()
    assert torch.equal(dataset[0].x, features)
    item = dataset[0]
    item.x.zero_()
    assert torch.equal(dataset[0].x, features)
    cache.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        github.RecoveredGitHub()
