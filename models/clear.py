# Active implementation starts here.
import copy
import os
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGCNConv
from torch_geometric.utils import dense_to_sparse, sort_edge_index, to_dense_adj, to_undirected
from tqdm import tqdm

from utils import get_datasets
from utils.baseline_eval_metrics import (
    OracleWrappedModel,
    compute_fidelity_prob_from_probs,
    compute_proximity_from_edge_index,
    compute_sparsity_from_edge_index,
)

from gnns import *

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def _get_unique_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    edge_index = sort_edge_index(edge_index)
    row, col = edge_index
    unique_mask = row < col
    return edge_index[:, unique_mask]


def _select_desired_labels(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    probs = F.softmax(logits, dim=-1)
    ori_pred = probs.argmax(dim=-1)

    desired_scores = probs.clone()
    desired_scores[torch.arange(probs.size(0), device=logits.device), ori_pred] = -1.0
    desired_label = desired_scores.argmax(dim=-1)
    return ori_pred, desired_label


def _build_dense_inputs_from_batch(
    x_sparse: torch.Tensor,
    edge_index: torch.Tensor,
    batch_vec: torch.Tensor,
    max_num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from torch_geometric.utils import to_dense_batch

    x_dense, node_mask = to_dense_batch(
        x_sparse,
        batch_vec,
        max_num_nodes=max_num_nodes,
    )
    adj = to_dense_adj(
        edge_index=edge_index,
        batch=batch_vec,
        max_num_nodes=max_num_nodes,
    )
    return x_dense, adj, node_mask


def _prepare_single_graph_inputs(
    data,
    max_num_nodes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    num_nodes = x.size(0)

    x_dense = torch.zeros(
        (1, max_num_nodes, x.size(1)),
        device=device,
        dtype=x.dtype,
    )
    adj = torch.zeros(
        (1, max_num_nodes, max_num_nodes),
        device=device,
        dtype=x.dtype,
    )
    node_mask = torch.zeros((1, max_num_nodes), device=device, dtype=torch.bool)

    x_dense[0, :num_nodes] = x
    adj[0, :num_nodes, :num_nodes] = to_dense_adj(
        edge_index,
        max_num_nodes=num_nodes,
    ).squeeze(0)
    node_mask[0, :num_nodes] = True
    return x_dense, adj, node_mask, num_nodes


def _batched_dense_to_sparse(
    feat: torch.Tensor,
    adj: torch.Tensor,
    node_mask: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_chunks: List[torch.Tensor] = []
    edge_index_chunks: List[torch.Tensor] = []
    edge_weight_chunks: List[torch.Tensor] = []
    batch_chunks: List[torch.Tensor] = []
    node_counts: List[int] = []
    offset = 0

    batch_size = feat.size(0)
    full_num_nodes = feat.size(1)

    for graph_idx in range(batch_size):
        if node_mask is None:
            num_nodes = full_num_nodes
        else:
            num_nodes = int(node_mask[graph_idx].sum().item())

        node_counts.append(num_nodes)
        if num_nodes == 0:
            continue

        feat_b = feat[graph_idx, :num_nodes]
        adj_b = adj[graph_idx, :num_nodes, :num_nodes]
        edge_index_b, edge_weight_b = dense_to_sparse(adj_b)

        x_chunks.append(feat_b)
        batch_chunks.append(
            torch.full((num_nodes,), graph_idx, dtype=torch.long, device=device)
        )

        if edge_index_b.numel() > 0:
            edge_index_chunks.append(edge_index_b + offset)
            edge_weight_chunks.append(edge_weight_b)

        offset += num_nodes

    if x_chunks:
        x_sparse = torch.cat(x_chunks, dim=0)
        batch_vec = torch.cat(batch_chunks, dim=0)
    else:
        x_sparse = torch.empty((0, feat.size(-1)), device=device, dtype=feat.dtype)
        batch_vec = torch.empty((0,), device=device, dtype=torch.long)

    if edge_index_chunks:
        edge_index = torch.cat(edge_index_chunks, dim=1)
        edge_weight = torch.cat(edge_weight_chunks, dim=0)
    else:
        edge_index = torch.empty((2, 0), device=device, dtype=torch.long)
        edge_weight = torch.empty((0,), device=device, dtype=feat.dtype)

    node_count_tensor = torch.tensor(node_counts, device=device, dtype=torch.long)
    return x_sparse, edge_index, edge_weight, batch_vec, node_count_tensor


def _predict_from_dense_format(
    pred_model: nn.Module,
    feat: torch.Tensor,
    adj: torch.Tensor,
    node_mask: Optional[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_sparse, edge_index, edge_weight, batch_vec, _ = _batched_dense_to_sparse(
        feat=feat,
        adj=adj,
        node_mask=node_mask,
        device=device,
    )

    probs, logits = pred_model.get_pred_explain(
        x_sparse,
        edge_index,
        edge_weight,
        batch_vec,
    )
    return probs, logits


def predict_with_dense_format(
    pred_model: nn.Module,
    feat: torch.Tensor,
    adj: torch.Tensor,
    device: str,
    node_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device_obj = torch.device(device)
    return _predict_from_dense_format(
        pred_model=pred_model,
        feat=feat.to(device_obj),
        adj=adj.to(device_obj),
        node_mask=node_mask.to(device_obj) if node_mask is not None else None,
        device=device_obj,
    )


@torch.no_grad()
def _predict_single_graph(
    pred_model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=device)
    edge_mask = torch.ones(edge_index.size(1), device=device, dtype=x.dtype)

    probs, logits = pred_model.get_pred_explain(
        x,
        edge_index,
        edge_mask,
        batch_vec,
    )

    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    pred = int(logits.argmax(dim=-1).item())
    return probs[0], logits, pred


def _choose_search_budget(num_edges: int) -> int:
    if num_edges <= 12:
        return num_edges
    return min(num_edges, max(12, int(round(num_edges * 0.25))))


@torch.no_grad()
def _search_minimal_cf_edge_index(
    pred_model: nn.Module,
    x: torch.Tensor,
    ori_edge_index: torch.Tensor,
    edge_keep_scores: torch.Tensor,
    ori_pred: int,
    desired_label: int,
    device: torch.device,
    max_removals: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    unique_edges = _get_unique_undirected_edges(ori_edge_index)
    num_unique_edges = unique_edges.size(1)

    if num_unique_edges == 0:
        probs, _, pred = _predict_single_graph(pred_model, x, ori_edge_index, device)
        return ori_edge_index.clone(), probs, pred

    keep_scores = edge_keep_scores[unique_edges[0], unique_edges[1]]
    max_removals = _choose_search_budget(num_unique_edges) if max_removals is None else min(max_removals, num_unique_edges)

    all_keep = torch.ones(num_unique_edges, dtype=torch.bool, device=device)
    ranked_edges = torch.argsort(keep_scores)
    threshold_keep = keep_scores > 0.5

    candidate_masks: List[torch.Tensor] = [all_keep]
    for removal_count in range(1, max_removals + 1):
        keep_mask = torch.ones(num_unique_edges, dtype=torch.bool, device=device)
        keep_mask[ranked_edges[:removal_count]] = False
        candidate_masks.append(keep_mask)
    candidate_masks.append(threshold_keep)

    seen_masks = set()
    best_fallback_edge_index = ori_edge_index.clone()
    best_fallback_probs, _, best_fallback_pred = _predict_single_graph(
        pred_model, x, best_fallback_edge_index, device
    )
    best_fallback_prob = float(best_fallback_probs[ori_pred].item())

    best_success = None

    for keep_mask in candidate_masks:
        mask_key = tuple(bool(v) for v in keep_mask.detach().cpu().tolist())
        if mask_key in seen_masks:
            continue
        seen_masks.add(mask_key)

        cf_unique_edges = unique_edges[:, keep_mask]
        cf_edge_index = to_undirected(cf_unique_edges, num_nodes=x.size(0))
        cf_probs, _, cf_pred = _predict_single_graph(pred_model, x, cf_edge_index, device)

        removed_edges = int((~keep_mask).sum().item())
        ori_class_prob = float(cf_probs[ori_pred].item())

        if cf_pred == desired_label:
            if best_success is None or removed_edges < best_success["removed"] or (
                removed_edges == best_success["removed"] and ori_class_prob < best_success["ori_prob"]
            ):
                best_success = {
                    "edge_index": cf_edge_index.clone(),
                    "probs": cf_probs.clone(),
                    "pred": cf_pred,
                    "removed": removed_edges,
                    "ori_prob": ori_class_prob,
                }
            continue

        if ori_class_prob < best_fallback_prob:
            best_fallback_edge_index = cf_edge_index.clone()
            best_fallback_probs = cf_probs.clone()
            best_fallback_pred = cf_pred
            best_fallback_prob = ori_class_prob

    if best_success is not None:
        return best_success["edge_index"], best_success["probs"], best_success["pred"]

    return best_fallback_edge_index, best_fallback_probs, best_fallback_pred


class GraphCFE(nn.Module):
    def __init__(
        self,
        pred_model: nn.Module,
        x_dim: int,
        edge_attr_dim: int,
        h_dim: int,
        z_dim: int,
        max_num_nodes: int,
        dropout: float,
        device: str,
        lambda_kl: float = 0.05,
        lambda_sim: float = 2.0,
        lambda_cfe: float = 1.0,
    ):
        super().__init__()
        self.pred_model = pred_model
        self.x_dim = x_dim
        self.edge_attr_dim = edge_attr_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.max_num_nodes = max_num_nodes
        self.dropout = dropout
        self.device = torch.device(device)
        self.lambda_kl = lambda_kl
        self.lambda_sim = lambda_sim
        self.lambda_cfe = lambda_cfe

        self.graph_model = DenseGCNConv(x_dim, h_dim)
        self.graph_norm = nn.LayerNorm(h_dim)
        self.encoder_mean = nn.Linear(h_dim, z_dim)
        self.encoder_logvar = nn.Linear(h_dim, z_dim)

        self.decoder_a = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, int(max_num_nodes * (max_num_nodes - 1) / 2)),
        )

        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def encoder(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        graph_rep = self.graph_model(features, adj)

        if node_mask is not None:
            graph_rep = graph_rep * node_mask.unsqueeze(-1)
            denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(graph_rep.dtype)
            graph_rep = graph_rep.sum(dim=1) / denom
        else:
            graph_rep = graph_rep.mean(dim=1)

        graph_rep = self.graph_norm(graph_rep)
        z_mu = self.encoder_mean(graph_rep)
        z_logvar = self.encoder_logvar(graph_rep).clamp(-8.0, 8.0)
        return z_mu, z_logvar

    def convert_to_symmetric_tensor(self, num_nodes: int, adj_vec: torch.Tensor) -> torch.Tensor:
        upper_triangular = torch.zeros(
            (adj_vec.shape[0], num_nodes, num_nodes),
            device=adj_vec.device,
            dtype=adj_vec.dtype,
        )
        mask = torch.triu_indices(num_nodes, num_nodes, offset=1, device=adj_vec.device)
        upper_triangular[:, mask[0], mask[1]] = adj_vec
        return upper_triangular + upper_triangular.transpose(1, 2)

    def decoder(
        self,
        z: torch.Tensor,
        adj: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_logits_half = self.decoder_a(z)
        edge_logits = self.convert_to_symmetric_tensor(self.max_num_nodes, edge_logits_half)
        edge_keep_mask = torch.sigmoid(edge_logits)

        if node_mask is not None:
            valid_pair_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
            edge_keep_mask = edge_keep_mask * valid_pair_mask.to(edge_keep_mask.dtype)

        eye = torch.eye(self.max_num_nodes, device=edge_keep_mask.device, dtype=edge_keep_mask.dtype)
        edge_keep_mask = edge_keep_mask * (1.0 - eye.unsqueeze(0))
        cf_adj = adj * edge_keep_mask
        return edge_keep_mask, cf_adj

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        features: torch.Tensor,
        adj: torch.Tensor,
        node_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        z_mu, z_logvar = self.encoder(features, adj, node_mask=node_mask)
        z_sample = self.reparameterize(z_mu, z_logvar)
        edge_keep_mask, cf_adj = self.decoder(z_sample, adj, node_mask=node_mask)

        return {
            "z_mu": z_mu,
            "z_logvar": z_logvar,
            "edge_keep_mask": edge_keep_mask,
            "adj_reconst": cf_adj,
            "feat_reconst": features,
        }

    def loss(
        self,
        feat: torch.Tensor,
        adj: torch.Tensor,
        node_mask: torch.Tensor,
        explainer_output: Dict[str, torch.Tensor],
        cf_label: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        z_mu = explainer_output["z_mu"]
        z_logvar = explainer_output["z_logvar"]
        raw_kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar), dim=1).mean()

        existing_edges = torch.triu(adj, diagonal=1) > 0
        if existing_edges.any():
            keep_mask = explainer_output["edge_keep_mask"][existing_edges]
            raw_sim = (1.0 - keep_mask).mean()
        else:
            raw_sim = torch.zeros((), device=feat.device)

        _, cf_logits = _predict_from_dense_format(
            pred_model=self.pred_model,
            feat=explainer_output["feat_reconst"],
            adj=explainer_output["adj_reconst"],
            node_mask=node_mask,
            device=self.device,
        )
        raw_cfe = F.cross_entropy(cf_logits, cf_label.view(-1).long())

        loss_kl = self.lambda_kl * raw_kl
        loss_sim = self.lambda_sim * raw_sim
        loss_cfe = self.lambda_cfe * raw_cfe
        loss = loss_kl + loss_sim + loss_cfe

        return {
            "loss": loss,
            "loss_kl": loss_kl,
            "loss_sim": loss_sim,
            "loss_cfe": loss_cfe,
        }

    def run_one_batch(self, batch) -> Dict[str, torch.Tensor]:
        x_sparse = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device)
        batch_vec = batch.batch.to(self.device)

        x_dense, adj, node_mask = _build_dense_inputs_from_batch(
            x_sparse=x_sparse,
            edge_index=edge_index,
            batch_vec=batch_vec,
            max_num_nodes=self.max_num_nodes,
        )

        with torch.no_grad():
            ori_logits = self.pred_model(x_sparse, edge_index, batch_vec)
            _, y_desired = _select_desired_labels(ori_logits)

        explainer_output = self(x_dense, adj, node_mask=node_mask)
        return self.loss(
            feat=x_dense,
            adj=adj,
            node_mask=node_mask,
            explainer_output=explainer_output,
            cf_label=y_desired,
        )


def train_explainer_inner(
    epochs: int,
    model: GraphCFE,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    model_path: str,
) -> GraphCFE:
    print("Start training GraphCFE explainer...")
    time_begin = time.time()
    best_loss = float("inf")
    best_model_state = None

    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_sim = 0.0
        epoch_cfe = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = model.run_one_batch(batch)
            batch_loss["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            epoch_loss += batch_loss["loss"].item() / len(train_loader)
            epoch_kl += batch_loss["loss_kl"].item() / len(train_loader)
            epoch_sim += batch_loss["loss_sim"].item() / len(train_loader)
            epoch_cfe += batch_loss["loss_cfe"].item() / len(train_loader)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:03d} | "
            f"loss: {epoch_loss:.4f} | "
            f"KL: {epoch_kl:.4f} | "
            f"sim: {epoch_sim:.4f} | "
            f"cfe: {epoch_cfe:.4f}"
        )

    if best_model_state is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(best_model_state, model_path)
        model.load_state_dict(best_model_state)

    print(f"Training done in {(time.time() - time_begin):.1f}s, best loss={best_loss:.4f}")
    return model


def train_graphcfe(
    pred_model: nn.Module,
    train_dataset: InMemoryDataset,
    epochs: int,
    device: str,
    lr: float,
    model_path: str,
    max_num_nodes: int,
) -> GraphCFE:
    pred_model.eval()
    for param in pred_model.parameters():
        param.requires_grad = False

    explainer = GraphCFE(
        pred_model=pred_model,
        x_dim=train_dataset[0].x.size(1),
        edge_attr_dim=(
            train_dataset[0].edge_attr.size(1)
            if hasattr(train_dataset[0], "edge_attr") and train_dataset[0].edge_attr is not None
            else 0
        ),
        h_dim=32,
        z_dim=32,
        max_num_nodes=max_num_nodes,
        dropout=0.1,
        device=device,
        lambda_kl=0.05,
        lambda_sim=2.0,
        lambda_cfe=1.0,
    ).to(device)

    optimizer = torch.optim.Adam(
        explainer.parameters(),
        lr=lr,
        weight_decay=1e-5,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(32, len(train_dataset)),
        shuffle=True,
    )

    return train_explainer_inner(
        epochs=epochs,
        model=explainer,
        optimizer=optimizer,
        train_loader=train_loader,
        model_path=model_path,
    )


@torch.no_grad()
def _generate_single_cf(
    pred_model: nn.Module,
    explainer: GraphCFE,
    data,
    device: str,
    oracle_model: Optional[nn.Module] = None,
    ori_pred: Optional[int] = None,
    desired_label: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    device_obj = torch.device(device)
    x = data.x.to(device_obj)
    edge_index = data.edge_index.to(device_obj)

    if ori_pred is None or desired_label is None:
        _, ori_logits, ori_pred = _predict_single_graph(pred_model, x, edge_index, device_obj)
        _, desired = _select_desired_labels(ori_logits)
        desired_label = int(desired.item())

    x_dense, adj, node_mask, num_nodes = _prepare_single_graph_inputs(
        data=data,
        max_num_nodes=explainer.max_num_nodes,
        device=device_obj,
    )

    output = explainer(x_dense, adj, node_mask=node_mask)
    edge_keep_scores = output["edge_keep_mask"][0, :num_nodes, :num_nodes]

    search_model = oracle_model if oracle_model is not None else pred_model
    return _search_minimal_cf_edge_index(
        pred_model=search_model,
        x=x,
        ori_edge_index=edge_index,
        edge_keep_scores=edge_keep_scores,
        ori_pred=int(ori_pred),
        desired_label=int(desired_label),
        device=device_obj,
    )


@torch.no_grad()
def generate_cfs_with_graphcfe(
    pred_model: nn.Module,
    explainer: GraphCFE,
    eval_dataset: InMemoryDataset,
    device: str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    explainer.eval()
    pred_model.eval()

    cf_feat_list: List[torch.Tensor] = []
    cf_adj_list: List[torch.Tensor] = []
    graph_idx_list: List[int] = []

    for idx in tqdm(range(len(eval_dataset)), desc="Generating CFs"):
        data = eval_dataset[idx]
        cf_edge_index, _, _ = _generate_single_cf(
            pred_model=pred_model,
            explainer=explainer,
            data=data,
            device=device,
        )

        num_nodes = data.x.size(0)
        cf_adj = to_dense_adj(
            cf_edge_index.cpu(),
            max_num_nodes=num_nodes,
        ).squeeze(0)

        cf_feat_list.append(data.x.detach().cpu())
        cf_adj_list.append(cf_adj)
        graph_idx_list.append(idx)

    return cf_feat_list, cf_adj_list, graph_idx_list


@torch.no_grad()
def evaluate_graphcfe(
    pred_model: nn.Module,
    explainer: GraphCFE,
    eval_dataset: InMemoryDataset,
    device: str,
) -> Dict[str, float]:
    print("\n" + "=" * 60)
    print("Evaluating GraphCFE (Edge-Mask Structural CF)")
    print("=" * 60)

    wrapped_model = OracleWrappedModel(pred_model)
    wrapped_model.eval()
    explainer.eval()
    device_obj = torch.device(device)

    valid_cf = 0
    proximity_sum = 0.0
    fidelity_prob_sum = 0.0
    sparsity_sum = 0.0
    total_graphs = len(eval_dataset)

    total_cf_time = 0.0
    total_cf_oracle_calls = 0

    for idx in tqdm(range(total_graphs), desc="Evaluating"):
        data = eval_dataset[idx].to(device_obj)

        ori_prob, ori_logits, ori_pred = _predict_single_graph(
            pred_model=pred_model,
            x=data.x,
            edge_index=data.edge_index,
            device=device_obj,
        )
        _, desired = _select_desired_labels(ori_logits)
        desired_label = int(desired.item())

        calls_before = wrapped_model.oracle_calls
        t0 = time.time()
        cf_edge_index, cf_prob, cf_pred = _generate_single_cf(
            pred_model=pred_model,
            explainer=explainer,
            data=data,
            device=device,
            oracle_model=wrapped_model,
            ori_pred=ori_pred,
            desired_label=desired_label,
        )
        total_cf_time += time.time() - t0
        total_cf_oracle_calls += wrapped_model.oracle_calls - calls_before

        if cf_pred == desired_label:
            valid_cf += 1

        proximity_sum += compute_proximity_from_edge_index(
            ori_edge_index=data.edge_index,
            cf_edge_index=cf_edge_index,
            num_nodes=data.x.size(0),
            device=device_obj,
        )

        fidelity_prob_sum += compute_fidelity_prob_from_probs(
            ori_probs=ori_prob,
            cf_probs=cf_prob,
        )

        sparsity_sum += compute_sparsity_from_edge_index(
            ori_edge_index=data.edge_index,
            cf_edge_index=cf_edge_index,
        )

    avg_runtime_per_graph = total_cf_time / total_graphs if total_graphs > 0 else 0.0
    avg_oracle_calls_per_graph = total_cf_oracle_calls / total_graphs if total_graphs > 0 else 0.0

    results = {
        "validity": valid_cf / total_graphs if total_graphs > 0 else 0.0,
        "proximity": proximity_sum / total_graphs if total_graphs > 0 else 0.0,
        "fidelity_prob": fidelity_prob_sum / total_graphs if total_graphs > 0 else 0.0,
        "sparsity": sparsity_sum / total_graphs if total_graphs > 0 else 0.0,
        "runtime": avg_runtime_per_graph,
        "oracle_calls": avg_oracle_calls_per_graph,
    }

    print(f"Results: {results}")
    return results


def to_dense_adj_sparse_format_helper(adj_binary: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return dense_to_sparse(adj_binary)


def calculate_sparsity(ori_edge_index: torch.Tensor, cf_edge_index: torch.Tensor) -> float:
    return compute_sparsity_from_edge_index(ori_edge_index, cf_edge_index)


if __name__ == "__main__":
    dataset_name = os.environ.get("MYEXPLAINER_DATASET", "fluoride_carbonyl")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset, val_dataset, test_dataset = get_datasets(
        name=dataset_name,
        root=_resolve_project_path("data"),
    )

    train_dataset = [data for data in train_dataset if data.num_nodes <= 300]
    val_dataset = [data for data in val_dataset if data.num_nodes <= 300]
    test_dataset = [data for data in test_dataset if data.num_nodes <= 300]

    gnn_path = _resolve_project_path("param", "gnns", f"{dataset_name}_gcn.pt")
    pred_model = torch.load(gnn_path, map_location=device)
    pred_model = pred_model.to(device)
    pred_model.eval()
    print("GNN classifier loaded.")

    explainer_model_path = _resolve_project_path(
        "param",
        "explainers",
        f"{dataset_name}_graphcfe.pt",
    )

    max_num_nodes = max(
        sample.num_nodes
        for dataset in (train_dataset, val_dataset, test_dataset)
        for sample in dataset
    )

    explainer = train_graphcfe(
        pred_model=pred_model,
        train_dataset=train_dataset,
        epochs=100,
        device=device,
        lr=1e-3,
        model_path=explainer_model_path,
        max_num_nodes=max_num_nodes,
    )

    metrics = evaluate_graphcfe(
        pred_model=pred_model,
        explainer=explainer,
        eval_dataset=val_dataset,
        device=device,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results (Calculated on ALL processed graphs):")
    print("=" * 60)
    print(f"  Validity ↑: {metrics['validity']:.4f}")
    print(f"  Proximity (Adj Diff) ↓: {metrics['proximity']:.4f}")
    print(f"  Fidelity (Prob Drop) ↑: {metrics['fidelity_prob']:.4f}")
    print(f"  Sparsity (Structure) ↑: {metrics['sparsity']:.4f}")
    print(f"  Runtime per graph (s) ↓: {metrics['runtime']:.6f}")
    print(f"  Oracle calls per graph ↓: {metrics['oracle_calls']:.4f}")
    print("=" * 60 + "\n")
