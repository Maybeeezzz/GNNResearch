from graph_level.experiments.graph_time_to_target import attained


def test_target_metric_direction_and_boundary():
    assert attained('classification', .8, .8)
    assert attained('classification', .81, .8)
    assert not attained('classification', .79, .8)
    assert attained('regression', 1.2, 1.2)
    assert attained('regression', 1.1, 1.2)
    assert not attained('regression', 1.3, 1.2)
