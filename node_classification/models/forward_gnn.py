"""ForwardGNN Single-Forward node classification (Park et al., ICLR 2024).

Independent implementation of Section 3.2 / Algorithm 2: class virtual nodes,
layer-local contrastive learning, greedy layer training and probability averaging.
Reference: https://arxiv.org/abs/2403.11004
This uses layer-local first-order gradients.
"""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..data import GraphData
from ..training.supervised import TrainResult, evaluate


class SingleForwardLayer(nn.Module):
    """Row L2 normalization followed by GCN and ReLU (official SF-GCN)."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
        # Exactly one self-loop per node. GCN applies its bias after aggregation.
        edge_index = edge_index[:, edge_index[0] != edge_index[1]]
        loops = torch.arange(x.size(0), device=x.device)
        source = torch.cat((edge_index[0], loops))
        target = torch.cat((edge_index[1], loops))
        degree = x.new_zeros(x.size(0))
        degree.index_add_(0, target, x.new_ones(target.numel()))
        inverse = degree.clamp_min(1).rsqrt()
        transformed = self.linear(x)
        output = torch.zeros_like(transformed)
        output.index_add_(0, target, transformed[source] * (inverse[source] * inverse[target])[:, None])
        return F.relu(output + self.bias)


class ForwardGNN(nn.Module):
    """SF-GCN with fixed random virtual features and training-label-only edges.

    All layers emit hidden embeddings; there is no conventional classifier head.
    ``forward`` returns log(mean(layer probabilities)), usable as class logits.
    Call ``bind_graph`` before prediction. The training label context is included
    in state_dict, so loading a checkpoint needs no validation/test labels.
    """

    def __init__(self, in_features: int, hidden_features: int, num_classes: int,
                 num_layers: int = 2, temperature: float = 1.0,
                 edge_direction: str = "bidirection"):
        super().__init__()
        if min(in_features, hidden_features, num_classes, num_layers) < 1:
            raise ValueError("feature sizes, classes and num_layers must be positive")
        if not 0 < temperature < float("inf"):
            raise ValueError("temperature must be finite and positive")
        if edge_direction not in ("bidirection", "unidirection"):
            raise ValueError("invalid virtual edge direction")
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.temperature = temperature
        self.edge_direction = edge_direction
        self.layers = nn.ModuleList(
            SingleForwardLayer(in_features if i == 0 else hidden_features, hidden_features)
            for i in range(num_layers)
        )
        self.register_buffer("virtual_features", torch.randn(num_classes, in_features))
        self.register_buffer("train_indices", torch.empty(0, dtype=torch.long))
        self.register_buffer("train_labels", torch.empty(0, dtype=torch.long))
        self.register_buffer("graph_nodes", torch.tensor(0, dtype=torch.long))
        self.register_buffer("trained_layers", torch.tensor(0, dtype=torch.long))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Training split size varies between graphs, unlike parameter shapes.
        for name in ("train_indices", "train_labels"):
            if prefix + name in state_dict:
                setattr(self, name, torch.empty_like(state_dict[prefix + name], device=self.virtual_features.device))
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def bind_graph(self, data: GraphData) -> None:
        data.validate()
        indices = data.train_mask.nonzero(as_tuple=True)[0]
        labels = data.y[indices]
        if not indices.numel() or labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError("valid nonempty training labels are required")
        if torch.unique(labels).numel() != self.num_classes:
            raise ValueError("ForwardGNN needs a training node for every class")
        self.train_indices = indices.detach().clone()
        self.train_labels = labels.detach().clone()
        self.graph_nodes.fill_(data.x.size(0))

    def augment(self, x: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        if x.size(0) != int(self.graph_nodes) or not self.train_indices.numel():
            raise ValueError("call bind_graph with this transductive graph first")
        virtual_ids = self.train_labels + x.size(0)
        extra = torch.stack((self.train_indices, virtual_ids))
        if self.edge_direction == "bidirection":
            extra = torch.cat((extra, extra.flip(0)), dim=1)
        return torch.cat((x, self.virtual_features)), torch.cat((edge_index, extra), dim=1)

    def class_logits(self, embeddings: Tensor) -> Tensor:
        real = embeddings[:-self.num_classes]
        representatives = embeddings[-self.num_classes:]
        return real @ representatives.T / self.temperature

    def local_loss(self, layer_index: int, inputs: Tensor, edges: Tensor) -> Tensor:
        embeddings = self.layers[layer_index](inputs.detach(), edges)
        return F.cross_entropy(self.class_logits(embeddings)[self.train_indices], self.train_labels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x, edges = self.augment(x, edge_index)
        count = int(self.trained_layers) or self.num_layers
        probabilities = None
        for layer in self.layers[:count]:
            x = layer(x, edges)
            current = self.class_logits(x).softmax(dim=1)
            probabilities = current if probabilities is None else probabilities + current
        return (probabilities / count).clamp_min(torch.finfo(x.dtype).tiny).log()


@dataclass
class ForwardTrainResult(TrainResult):
    layer_epochs: list[int]
    layer_best_epochs: list[int]
    local_backward_passes: int
    optimization_forwards: int
    cache_layer_forwards: int
    validation_layer_forwards: int


def train_forwardgnn(
    model: ForwardGNN, data: GraphData, epochs: int = 300,
    learning_rate: float = 1e-3, weight_decay: float = 5e-4,
    patience: int = 50, eval_every: int = 1, verbose: bool = True,
    callback: Optional[Callable[[int, float, nn.Module], None]] = None,
) -> ForwardTrainResult:
    """Greedily train each layer for at most ``epochs`` local updates.

    Earlier layers are frozen; their detached output is cached. Each local update
    uses one layer forward and a local backward. Validation, cache construction,
    callbacks and final evaluation are additional computation. patience=-1 skips
    validation/checkpoint selection (fixed-budget experiments).
    """
    if not isinstance(model, ForwardGNN):
        raise TypeError("train_forwardgnn requires a ForwardGNN model")
    if epochs < 1 or eval_every < 1 or patience == 0 or patience < -1:
        raise ValueError("epochs/eval_every must be positive; patience positive or -1")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    if patience > 0 and not data.val_mask.any():
        raise ValueError("early stopping requires validation nodes")
    model.bind_graph(data)
    model.trained_layers.zero_()
    inputs, edges = model.augment(data.x, data.edge_index)
    inputs = inputs.detach()
    layer_epochs, best_epochs = [], []
    updates = validation_forwards = cache_forwards = 0
    for i, layer in enumerate(model.layers):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
        optimizer = torch.optim.Adam(layer.parameters(), lr=learning_rate, weight_decay=weight_decay)
        best_state, best_score, best_epoch, stale = None, -float("inf"), 0, 0
        model.trained_layers.fill_(i + 1)
        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = model.local_loss(i, inputs, edges)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite local loss at layer {i + 1}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            updates += 1
            if callback is not None:
                callback(updates, float(loss.detach()), model)
            if patience > 0 and (epoch % eval_every == 0 or epoch == epochs):
                model.eval()
                with torch.no_grad():
                    prediction = model(data.x, data.edge_index).argmax(dim=1)
                    score = float((prediction[data.val_mask] == data.y[data.val_mask]).float().mean())
                validation_forwards += i + 1
                if score > best_score:
                    best_score, best_epoch, stale = score, epoch, 0
                    best_state = deepcopy(layer.state_dict())
                else:
                    stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            layer.load_state_dict(best_state)
        layer_epochs.append(epoch)
        best_epochs.append(best_epoch if best_state is not None else epoch)
        model.eval()
        if i + 1 < model.num_layers:
            with torch.no_grad():
                inputs = layer(inputs, edges).detach()
            cache_forwards += 1
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if verbose:
            print(f"method=forwardgnn layer={i + 1}/{model.num_layers} epochs={epoch} best_epoch={best_epochs[-1]}")
    scores = evaluate(model, data)
    return ForwardTrainResult(
        best_epoch=best_epochs[-1], train_accuracy=scores["train"],
        val_accuracy=scores["val"], test_accuracy=scores["test"],
        layer_epochs=layer_epochs, layer_best_epochs=best_epochs,
        local_backward_passes=updates, optimization_forwards=updates,
        cache_layer_forwards=cache_forwards, validation_layer_forwards=validation_forwards,
    )
