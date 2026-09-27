import json

from paper_reproduction.experiments import plot_memory_comparison as plots


def test_memory_collection_preserves_depth_and_peak_on_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, 'RESULTS', tmp_path)
    folder = tmp_path / 'memory' / 'test'
    folder.mkdir(parents=True)
    for stamp, tensor in [('20260101', 20), ('20260102', 1)]:
        path = folder / f'CitationFull-Cora_ML_GNN_SingleForward-GCN_4_{stamp}_100.jsonl'
        path.write_text(json.dumps({'sampled_peak_tensor_bytes': tensor * 2**20,
                                    'sampled_peak_driver_bytes': 40 * 2**20}) + '\n')
        path.with_suffix('.summary.json').write_text(json.dumps({'status': 'completed'}))
    rows = plots.collect_memory('test')
    assert len(rows) == 1
    assert rows[0]['dataset'] == 'CitationFull-Cora_ML'
    assert rows[0]['layers'] == 4
    assert rows[0]['tensor_mib'] == 20
    assert rows[0]['method'] == 'sf'
    assert rows[0]['status'] == 'completed'


def test_unfinished_sampling_not_marked_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, 'RESULTS', tmp_path)
    folder = tmp_path / 'memory' / 'test'
    folder.mkdir(parents=True)
    path = folder / 'CitationFull-CiteSeer_GNN-GCN_2_20260101_100.jsonl'
    path.write_text(json.dumps({'sampled_peak_tensor_bytes': 2**20,
                                'sampled_peak_driver_bytes': 2**21}) + '\n{"partial":')
    row, = plots.collect_memory('test')
    assert row['status'] == 'incomplete'
    assert row['tensor_mib'] == 1
    assert row['driver_mib'] == 2


def test_github_backbones_remain_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, 'RESULTS', tmp_path)
    folder = tmp_path / 'memory' / 'test'
    folder.mkdir(parents=True)
    for backbone, peak in [('GCN', 20), ('GAT', 400)]:
        path = folder / f'GitHub_GNN_SingleForward-{backbone}_4_20260101_100.jsonl'
        path.write_text(json.dumps({'sampled_peak_tensor_bytes': peak * 2**20,
                                    'sampled_peak_driver_bytes': peak * 2**21}) + '\n')
    rows = plots.collect_memory('test')
    assert {r['backbone']: r['tensor_mib'] for r in rows} == {'GCN': 20, 'GAT': 400}
