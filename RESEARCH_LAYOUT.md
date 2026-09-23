# Research code layout

Model code, training routines, experiment runners and tests from the former
`gnn/`, `experiments/` and `tests/` trees are grouped by task domain:

```text
node_classification/
  data.py
  models/                    # GCN and ForwardGNN
  training/                  # BP and local forward-learning training
  experiments/baselines/     # study runner, worker, result summary
  tests/
graph_level/
  model/                     # graph batching, readout, model, objective
  experiments/               # graph learning and SF/FF benchmarks
  tests/
link_prediction/
  experiments/
  tests/
paper_reproduction/
  experiments/               # original-paper reproduction and summaries
  tests/
mechanism_studies/           # plans for explaining why methods work
```

The old `gnn.*` and `experiments.*` compatibility modules have been removed.
Use task-domain packages for imports and `python -m` entry points.
