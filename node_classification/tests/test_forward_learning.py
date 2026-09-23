from copy import deepcopy
from io import BytesIO

import pytest
import torch
import torch.nn.functional as F

from node_classification.data import generate_sbm_graph
from node_classification.models.forward_gnn import (
    ForwardGNN,
    SingleForwardLayer,
    train_forwardgnn,
)


def graph_and_model():
    torch.manual_seed(17)
    data = generate_sbm_graph(num_nodes=90, num_features=8, p_in=0.2, p_out=0.005, seed=17)
    model = ForwardGNN(8, 16, data.num_classes, num_layers=2)
    model.bind_graph(data)
    return data, model


def test_virtual_edges_use_only_training_labels():
    data, model = graph_and_model()
    x, edges = model.augment(data.x, data.edge_index)
    n = data.x.size(0)
    assert x.size(0) == n + data.num_classes
    extra = edges[:, data.edge_index.size(1):]
    count = int(data.train_mask.sum())
    assert torch.equal(extra[:, count:], extra[:, :count].flip(0))
    assert torch.equal(extra[0, :count], torch.where(data.train_mask)[0])
    assert torch.equal(extra[1, :count] - n, data.y[data.train_mask])
    changed = deepcopy(data)
    changed.y[~data.train_mask] = (changed.y[~data.train_mask] + 1) % data.num_classes
    before = model(data.x, data.edge_index).detach()
    model.bind_graph(changed)
    assert torch.equal(edges, model.augment(changed.x, changed.edge_index)[1])
    assert torch.equal(before, model(changed.x, changed.edge_index).detach())


def test_local_loss_does_not_backpropagate_to_previous_layer():
    data, model = graph_and_model()
    x, edges = model.augment(data.x, data.edge_index)
    first_output = model.layers[0](x, edges)
    first_output.retain_grad()
    model.local_loss(1, first_output, edges).backward()
    assert first_output.grad is None
    assert all(p.grad is None for p in model.layers[0].parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.layers[1].parameters())


def test_gcn_operator_matches_dense_normalized_adjacency():
    torch.manual_seed(3)
    layer = SingleForwardLayer(3, 4).double()
    x = torch.randn(4, 3, dtype=torch.float64)
    edges = torch.tensor([[0, 1, 2, 3, 0], [1, 0, 1, 2, 0]])
    adjacency = torch.eye(4, dtype=torch.float64)
    for source, target in edges.T:
        if source != target:
            adjacency[target, source] += 1
    inverse = adjacency.sum(1).rsqrt()
    normalized = inverse[:, None] * adjacency * inverse[None, :]
    expected = F.relu(normalized @ (x / (x.norm(dim=1, keepdim=True) + 1e-8)) @ layer.linear.weight.T + layer.bias)
    torch.testing.assert_close(layer(x, edges), expected)


def test_prediction_averages_probabilities_not_logits():
    data, model = graph_and_model()
    x, edges = model.augment(data.x, data.edge_index)
    probabilities = []
    for layer in model.layers:
        x = layer(x, edges)
        probabilities.append(model.class_logits(x).softmax(1))
    torch.testing.assert_close(model(data.x, data.edge_index).exp(), torch.stack(probabilities).mean(0))


def test_training_learns_freezes_layers_and_checkpoint_roundtrip():
    data, model = graph_and_model()
    states = []

    def record(step, loss, current):
        if step in (60, 120):
            states.append(deepcopy(current.layers[0].state_dict()))

    result = train_forwardgnn(model, data, epochs=60, learning_rate=0.01,
                              patience=-1, verbose=False, callback=record)
    assert result.test_accuracy > 0.65
    assert result.local_backward_passes == result.optimization_forwards == 120
    assert result.cache_layer_forwards == 1
    assert result.validation_layer_forwards == 0
    assert result.layer_epochs == [60, 60]
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    saved = BytesIO()
    torch.save(model.state_dict(), saved)
    saved.seek(0)
    restored = ForwardGNN(8, 16, data.num_classes, num_layers=2)
    restored.load_state_dict(torch.load(saved, weights_only=True))
    torch.testing.assert_close(restored(data.x, data.edge_index), model(data.x, data.edge_index))


def test_validation_checkpoint_selection_and_retraining():
    data, model = graph_and_model()
    result = train_forwardgnn(model, data, epochs=8, patience=2, eval_every=2, verbose=False)
    assert all(1 <= best <= epochs <= 8 for best, epochs in zip(result.layer_best_epochs, result.layer_epochs))
    assert result.validation_layer_forwards > 0
    # A second fit must unfreeze the next active layer itself.
    train_forwardgnn(model, data, epochs=1, patience=-1, verbose=False)
    assert int(model.trained_layers) == 2


def test_missing_training_class_is_rejected():
    data, model = graph_and_model()
    data.train_mask[data.y == 0] = False
    with pytest.raises(ValueError, match="every class"):
        model.bind_graph(data)
