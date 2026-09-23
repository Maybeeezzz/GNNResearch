from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor


@dataclass
class GraphData:
    """All tensors needed for transductive node classification."""

    x: Tensor
    edge_index: Tensor
    y: Tensor
    train_mask: Tensor
    val_mask: Tensor
    test_mask: Tensor

    def validate(self) -> None:
        num_nodes = self.x.size(0)
        if self.x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, num_features]")
        if self.edge_index.ndim != 2 or self.edge_index.size(0) != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.edge_index.dtype != torch.long:
            raise ValueError("edge_index must use torch.long")
        if self.y.shape != (num_nodes,):
            raise ValueError("y must have shape [num_nodes]")
        if self.edge_index.numel() and (
            self.edge_index.min() < 0 or self.edge_index.max() >= num_nodes
        ):
            raise ValueError("edge_index contains an invalid node id")
        for name in ("train_mask", "val_mask", "test_mask"):
            mask = getattr(self, name)
            if mask.dtype != torch.bool or mask.shape != (num_nodes,):
                raise ValueError(f"{name} must be a boolean [num_nodes] tensor")
        if (self.train_mask & self.val_mask).any() or (
            self.train_mask & self.test_mask
        ).any() or (self.val_mask & self.test_mask).any():
            raise ValueError("train/validation/test masks must not overlap")

    def to(self, device: Union[str, torch.device]) -> "GraphData":
        return GraphData(**{
            name: value.to(device) for name, value in vars(self).items()
        })

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1


def _stratified_masks(
    y: Tensor, train_ratio: float, val_ratio: float, generator: torch.Generator
) -> tuple[Tensor, Tensor, Tensor]:
    n = y.numel()
    masks = [torch.zeros(n, dtype=torch.bool) for _ in range(3)]
    for class_id in torch.unique(y):
        indices = torch.where(y == class_id)[0]
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        n_train = max(1, int(indices.numel() * train_ratio))
        n_val = max(1, int(indices.numel() * val_ratio))
        n_train = min(n_train, indices.numel() - 2)
        n_val = min(n_val, indices.numel() - n_train - 1)
        masks[0][indices[:n_train]] = True
        masks[1][indices[n_train : n_train + n_val]] = True
        masks[2][indices[n_train + n_val :]] = True
    return masks[0], masks[1], masks[2]


def generate_sbm_graph(
    num_nodes: int = 600,
    num_features: int = 32,
    num_classes: int = 3,
    p_in: float = 0.08,
    p_out: float = 0.008,
    train_ratio: float = 0.2,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> GraphData:
    """Generate a stochastic-block graph with learnable node features."""
    if num_nodes < num_classes * 3:
        raise ValueError("num_nodes must leave at least 3 nodes per class")
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1):
        raise ValueError("train_ratio and val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1")

    generator = torch.Generator().manual_seed(seed)
    y = torch.arange(num_nodes) % num_classes
    y = y[torch.randperm(num_nodes, generator=generator)]

    centers = torch.randn(num_classes, num_features, generator=generator)
    x = centers[y] + 0.8 * torch.randn(
        num_nodes, num_features, generator=generator
    )
    x = (x - x.mean(dim=0)) / x.std(dim=0).clamp_min(1e-6)

    row, col = torch.triu_indices(num_nodes, num_nodes, offset=1)
    probabilities = torch.where(y[row] == y[col], p_in, p_out)
    selected = torch.rand(row.numel(), generator=generator) < probabilities
    row, col = row[selected], col[selected]
    edge_index = torch.stack(
        [torch.cat([row, col]), torch.cat([col, row])], dim=0
    ).long()

    train_mask, val_mask, test_mask = _stratified_masks(
        y, train_ratio, val_ratio, generator
    )
    data = GraphData(x, edge_index, y.long(), train_mask, val_mask, test_mask)
    data.validate()
    return data


def load_npz_graph(path: Union[str, Path], seed: int = 42) -> GraphData:
    """Load x, edge_index and y (plus optional masks) from an ``.npz`` file."""
    arrays = np.load(Path(path), allow_pickle=False)
    required = {"x", "edge_index", "y"}
    missing = required.difference(arrays.files)
    if missing:
        raise ValueError(f"missing arrays in npz file: {sorted(missing)}")

    x = torch.as_tensor(arrays["x"], dtype=torch.float32)
    edge_index = torch.as_tensor(arrays["edge_index"], dtype=torch.long)
    y = torch.as_tensor(arrays["y"], dtype=torch.long)
    mask_names = ("train_mask", "val_mask", "test_mask")
    if all(name in arrays.files for name in mask_names):
        masks = tuple(torch.as_tensor(arrays[name], dtype=torch.bool) for name in mask_names)
    elif any(name in arrays.files for name in mask_names):
        raise ValueError("provide either all three masks or none of them")
    else:
        masks = _stratified_masks(y, 0.2, 0.2, torch.Generator().manual_seed(seed))

    data = GraphData(x, edge_index, y, *masks)
    data.validate()
    return data


def load_planetoid_graph(
    name: str, root: Union[str, Path] = "data/Planetoid"
) -> GraphData:
    """Load a standard Planetoid citation network with its public split."""
    canonical = {value.lower(): value for value in ("Cora", "CiteSeer", "PubMed")}
    if name.lower() not in canonical:
        raise ValueError("name must be Cora, CiteSeer, or PubMed")
    try:
        from torch_geometric.datasets import Planetoid
        from torch_geometric.transforms import NormalizeFeatures
    except ImportError as error:
        raise ImportError(
            "load_planetoid_graph requires torch-geometric; install environment.yml"
        ) from error

    dataset = Planetoid(
        root=str(Path(root)),
        name=canonical[name.lower()],
        split="public",
        transform=NormalizeFeatures(),
    )
    graph = dataset[0]
    data = GraphData(
        x=graph.x.float(),
        edge_index=graph.edge_index.long(),
        y=graph.y.long(),
        train_mask=graph.train_mask.bool(),
        val_mask=graph.val_mask.bool(),
        test_mask=graph.test_mask.bool(),
    )
    data.validate()
    return data
