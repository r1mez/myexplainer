"""Base class for all GNN classifiers in the explainer pipeline."""
from abc import abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


class BaseGNNClassifier(torch.nn.Module):
    """Abstract base for GNN graph classifiers.

    Subclasses implement:
        - _forward_convs(x, edge_index, edge_weight): node embeddings [N, hidden_dim]
        - _pool(node_emb, batch): graph embeddings [B, hidden_dim]
        - _classify(graph_emb): logits [B, num_classes]
    """

    def __init__(self, in_channels: int, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    @abstractmethod
    def _forward_convs(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        ...

    @abstractmethod
    def _pool(self, node_emb: Tensor, batch: Tensor) -> Tensor:
        ...

    @abstractmethod
    def _classify(self, graph_emb: Tensor) -> Tensor:
        ...

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        node_emb = self._forward_convs(x, edge_index)
        graph_emb = self._pool(node_emb, batch)
        return self._classify(graph_emb)

    def get_node_reps(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor = None) -> Tensor:
        return self._forward_convs(x, edge_index, edge_weight)

    def get_graph_rep(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_weight: Tensor = None) -> Tensor:
        node_emb = self._forward_convs(x, edge_index, edge_weight)
        return self._pool(node_emb, batch)

    def get_pred(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        logits = self.forward(x, edge_index, batch)
        probs = F.softmax(logits, dim=-1)
        self.readout = probs
        return probs, logits

    def get_pred_explain(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor]:
        node_emb = self._forward_convs(x, edge_index, edge_weight)
        graph_emb = self._pool(node_emb, batch)
        logits = self._classify(graph_emb)
        probs = F.softmax(logits, dim=-1)
        self.readout = probs
        return probs, logits
