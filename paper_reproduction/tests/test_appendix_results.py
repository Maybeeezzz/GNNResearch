import pytest

from paper_reproduction.experiments.plot_appendix_results import REFERENCE, comparisons, extract, validate


def test_source_tables_and_independent_anchors():
    tables, records = extract(REFERENCE / 'source/forwardgnn_2403.11004v1.html')
    validate(records)
    assert {t['table'] for t in tables} == set(range(3, 17))
    table16 = next(t for t in tables if t['table'] == 16)
    assert table16['cells'][1] == ['Planetoid-Cora', '2,708', '10,556', '140', '500', '1,000', '1,433', '7']
    pairs = comparisons(records)
    assert len(pairs) == 120
    github = next(r for r in pairs if r['task'] == 'node_classification' and r['dataset'] == 'GitHub'
                  and r['backbone'] == 'GAT' and r['layers'] == 4)
    assert github['score_delta_pp'] == pytest.approx(-1.53)
    assert github['memory_saving_percent'] == pytest.approx(100 * (1 - 444.83 / 1228.98))
    citeseer = next(r for r in pairs if r['task'] == 'node_classification' and r['dataset'] == 'CiteSeer'
                    and r['backbone'] == 'GCN' and r['layers'] == 4)
    assert citeseer['memory_saving_percent'] < 0
    link = next(r for r in pairs if r['task'] == 'link_prediction' and r['dataset'] == 'GitHub'
                and r['backbone'] == 'GCN' and r['layers'] == 4)
    assert link['score_metric'] == 'roc_auc_percent'
    assert link['sf_memory_mb'] == 569.88
