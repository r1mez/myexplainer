"""Base interface for counterfactual graph explainers."""
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


@dataclass
class CFResult:
    """Result of a counterfactual explanation."""
    cf_edge_index: Tensor       # [2, E_cf]
    cf_edge_weight: Tensor      # [E_cf]
    oracle_calls: int = 0
    runtime: float = 0.0


class BaseExplainer(nn.Module):
    """Abstract base for counterfactual explainers.

    Subclasses must implement:
        - explain_graph(data, device) -> CFResult
    Subclasses may optionally implement:
        - fit(train_dataset, gnn, device) for trainable explainers
    """

    def explain_graph(self, data: Data, device: str = "cpu") -> CFResult:
        raise NotImplementedError

    def fit(self, train_dataset, gnn, device: str = "cpu") -> None:
        pass
