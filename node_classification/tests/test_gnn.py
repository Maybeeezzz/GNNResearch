import torch

from node_classification.data import generate_sbm_graph
from node_classification.models.gcn import GCN, GraphConvolution, ResidualGCN
from node_classification.training import classification_metrics, evaluate, macro_f1, train_model
from node_classification.experiments.baselines.worker import staged_masks


def test_graph_convolution_shape_and_gradients():
    layer = GraphConvolution(3, 5)
    x = torch.randn(4, 3, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
    output = layer(x, edge_index)
    assert output.shape == (4, 5)
    output.sum().backward()
    assert x.grad is not None


def test_deep_residual_gcn_shape_and_gradients():
    model = ResidualGCN(6, 16, 4, num_layers=6, dropout=0.0)
    x = torch.randn(10, 6, requires_grad=True)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]]
    )
    output = model(x, edge_index)
    assert output.shape == (10, 4)
    output.sum().backward()
    assert x.grad is not None


def test_generated_data_is_valid_and_reproducible():
    first = generate_sbm_graph(num_nodes=60, seed=7)
    second = generate_sbm_graph(num_nodes=60, seed=7)
    first.validate()
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.edge_index, second.edge_index)
    assert (first.train_mask | first.val_mask | first.test_mask).all()


def test_training_improves_over_random_baseline():
    torch.manual_seed(3)
    data = generate_sbm_graph(
        num_nodes=120, num_features=12, p_in=0.18, p_out=0.005, seed=3
    )
    model = GCN(12, 24, data.num_classes, dropout=0.0)
    before = evaluate(model, data)["train"]
    result = train_model(model, data, epochs=80, patience=25, verbose=False)
    assert result.train_accuracy > before
    assert result.test_accuracy > 0.65


def test_single_forward_training_improves_over_random_baseline():
    torch.manual_seed(4)
    data = generate_sbm_graph(
        num_nodes=90, num_features=8, p_in=0.2, p_out=0.005, seed=4
    )
    model = GCN(8, 16, data.num_classes, dropout=0.0)
    before = evaluate(model, data)["train"]
    result = train_model(
        model,
        data,
        epochs=60,
        patience=20,
        verbose=False,
        method="single-forward",
    )
    assert result.train_accuracy > before
    assert result.test_accuracy > 0.65


def test_macro_f1_and_classification_metrics():
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0], [2.0, 1.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 0, 1])
    assert macro_f1(logits, labels) == 1.0
    data = generate_sbm_graph(num_nodes=30, num_features=6, seed=8)
    model = GCN(6, 8, data.num_classes, dropout=0.0)
    metrics = classification_metrics(model, data)
    assert set(metrics) == {
        "train_accuracy", "train_macro_f1", "val_accuracy", "val_macro_f1",
        "test_accuracy", "test_macro_f1",
    }


def test_staged_masks_are_disjoint_and_cover_training_nodes():
    data = generate_sbm_graph(num_nodes=60, num_features=6, seed=9)
    original = data.train_mask.clone()
    pretrain, finetune = staged_masks(data, 0.5, seed=9)
    assert not (pretrain & finetune).any()
    assert torch.equal(pretrain | finetune, original)
    for class_id in torch.unique(data.y):
        assert (pretrain & (data.y == class_id)).any()
        assert (finetune & (data.y == class_id)).any()
