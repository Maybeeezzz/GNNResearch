import json

import pytest

from paper_reproduction.experiments import summarize_paper


def test_official_accuracy_units(tmp_path, monkeypatch):
    monkeypatch.setattr(summarize_paper, "RESULTS", tmp_path)
    folder = tmp_path / "official" / "test" / "dataset" / "node-class"
    folder.mkdir(parents=True)
    for model, perf in (("GNN-GCN", 0.9), ("GNN_SingleForward-GCN", 90.0)):
        row = dict(model=model, perf=perf, run_seed=10100, run_i=0,
                   dataset="dataset", num_layers=2)
        (folder / f"{model}.json").write_text(json.dumps(row))
    groups = summarize_paper.collect("test")
    assert groups[("dataset", "GCN", "bp", 2)][0]["accuracy_percent"] == 90.0
    assert groups[("dataset", "GCN", "sf", 2)][0]["accuracy_percent"] == 90.0


@pytest.mark.parametrize("perf,seed,message", [(90, 10100, "units"), (0.9, 0, "seed")])
def test_invalid_official_record(tmp_path, monkeypatch, perf, seed, message):
    monkeypatch.setattr(summarize_paper, "RESULTS", tmp_path)
    folder = tmp_path / "official" / "test" / "dataset" / "node-class"
    folder.mkdir(parents=True)
    row = dict(model="GNN-GCN", perf=perf, run_seed=seed, run_i=0,
               dataset="dataset", num_layers=2)
    (folder / "record.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match=message):
        summarize_paper.collect("test")
