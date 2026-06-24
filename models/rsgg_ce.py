import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_undirected
from tqdm import tqdm

from gnns import *
from utils import get_datasets
from models.base import BaseExplainer, CFResult


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _resolve_project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def _as_device(device: Union[torch.device, str]) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def _ensure_batch_vector(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.zeros(x.size(0), dtype=torch.long, device=device)


def _call_logits(
    pred_model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    batch: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if batch is None:
        batch = _ensure_batch_vector(x, x.device)

    output = pred_model(x, edge_index, batch)
    if isinstance(output, tuple):
        logits = output[-1]
    else:
        logits = output
    return logits


@torch.no_grad()
def _predict_single_graph(
    pred_model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    batch = _ensure_batch_vector(x, device)
    logits = _call_logits(pred_model, x, edge_index, batch=batch)
    if logits.dim() > 1:
        logits_single = logits[0]
    else:
        logits_single = logits
    probs = F.softmax(logits_single, dim=-1)
    pred = int(probs.argmax().item())
    return probs, logits_single, pred


def _select_desired_label(logits: torch.Tensor, ori_pred: int) -> int:
    if logits.dim() > 1:
        logits = logits[0]
    probs = F.softmax(logits, dim=-1).detach().clone()
    probs[ori_pred] = -1.0
    return int(probs.argmax().item())


def _num_classes_from_model_or_data(pred_model: nn.Module, sample: Data, device: torch.device) -> int:
    with torch.no_grad():
        logits = _call_logits(
            pred_model,
            sample.x.to(device),
            sample.edge_index.to(device),
            batch=_ensure_batch_vector(sample.x.to(device), device),
        )
    if logits.dim() == 1:
        return int(logits.numel())
    return int(logits.size(-1))


def _unique_undirected_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    edge_index = edge_index.to(device)
    row, col = edge_index
    mask = row != col
    row = row[mask]
    col = col[mask]

    if row.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    u = torch.minimum(row, col)
    v = torch.maximum(row, col)
    linear = u * num_nodes + v
    unique_linear = torch.unique(linear)
    unique_u = torch.div(unique_linear, num_nodes, rounding_mode="floor")
    unique_v = unique_linear % num_nodes
    keep = unique_u < unique_v
    return torch.stack([unique_u[keep], unique_v[keep]], dim=0).long()


def _complete_non_edges(
    existing_unique_edges: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    row, col = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
    if existing_unique_edges.numel() == 0:
        return torch.stack([row, col], dim=0)

    existing_mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    existing_mask[existing_unique_edges[0], existing_unique_edges[1]] = True
    mask = ~existing_mask[row, col]
    return torch.stack([row[mask], col[mask]], dim=0).long()


def _to_full_undirected_edge_index(unique_edges: torch.Tensor) -> torch.Tensor:
    if unique_edges.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=unique_edges.device)
    return to_undirected(unique_edges, num_nodes=None)


def _scores_for_edges(edge_scores: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    if edges.numel() == 0:
        return torch.empty((0,), dtype=edge_scores.dtype, device=edge_scores.device)
    forward = edge_scores[edges[0], edges[1]]
    backward = edge_scores[edges[1], edges[0]]
    return 0.5 * (forward + backward)


def _edge_probabilities_to_valid_matrix(edge_probs: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_probs = edge_probs[:num_nodes, :num_nodes]
    edge_probs = torch.nan_to_num(edge_probs, nan=0.0, posinf=1.0, neginf=0.0)
    edge_probs = edge_probs.clamp(0.0, 1.0)
    edge_probs = 0.5 * (edge_probs + edge_probs.t())
    edge_probs.fill_diagonal_(0.0)
    return edge_probs


class RSGGCEGenerator(nn.Module):
    """RSGG-CE graph generator adapted from GRETEL's ResGenerator.

    The official implementation uses a residual GCN encoder plus an inner-product
    graph auto-encoder decoder. This PyG-native version keeps the same idea while
    avoiding GRETEL's project-specific Dataset/GraphInstance abstractions.
    """

    def __init__(
        self,
        node_features: int,
        hidden_dim: Optional[int] = None,
        num_conv_layers: int = 2,
        residuals: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or max(node_features, 8))
        self.node_features = int(node_features)
        self.hidden_dim = hidden_dim
        self.residuals = bool(residuals)

        layers: List[GCNConv] = []
        in_dim = self.node_features
        for _ in range(max(1, int(num_conv_layers))):
            layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.encoder_layers = nn.ModuleList(layers)
        self.feature_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.node_features),
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x.float()
        for conv in self.encoder_layers:
            h = conv(h, edge_index.long(), edge_weight=edge_weight)
            h = F.relu(h)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = x.size(0)
        if edge_weight is None and edge_index.numel() > 0:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)

        z = self.encode(x, edge_index, edge_weight=edge_weight)
        edge_logits = torch.matmul(z, z.t()) / max(float(self.hidden_dim) ** 0.5, 1.0)

        if self.residuals:
            residual_adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(edge_logits)
            edge_logits = edge_logits + residual_adj

        edge_probs = torch.sigmoid(edge_logits)
        edge_probs = _edge_probabilities_to_valid_matrix(edge_probs, num_nodes)

        decoded_x = self.feature_decoder(z)
        if self.residuals:
            decoded_x = decoded_x + x.float()

        return decoded_x, edge_probs


class RSGGCEDiscriminator(nn.Module):
    """Simple graph discriminator following GRETEL's SimpleDiscriminator shape."""

    def __init__(
        self,
        node_features: int,
        max_num_nodes: int,
        hidden_dim: int = 16,
        noise_std: float = 0.2,
    ) -> None:
        super().__init__()
        self.max_num_nodes = int(max_num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.noise_std = float(noise_std)
        self.conv = GCNConv(int(node_features), self.hidden_dim)
        self.fc = nn.Linear(self.max_num_nodes * self.hidden_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.conv(x.float(), edge_index.long(), edge_weight=edge_weight)
        if self.training and self.noise_std > 0:
            h = h + torch.randn_like(h) * self.noise_std
        h = F.relu(h)
        dense_h, _ = to_dense_batch(h, batch, max_num_nodes=self.max_num_nodes)
        dense_h = dense_h.reshape(dense_h.size(0), -1)
        return self.fc(dense_h).view(-1)


class RSGGCEClassGAN(nn.Module):
    """One per-class RSGG-CE GAN.

    GRETEL trains one generator/discriminator pair for each class. A pair for
    class k learns to make source graphs from other classes look like class k.
    """

    def __init__(
        self,
        model_label: int,
        node_features: int,
        max_num_nodes: int,
        generator_hidden_dim: Optional[int] = None,
        discriminator_hidden_dim: int = 16,
        num_conv_layers: int = 2,
        residuals: bool = True,
    ) -> None:
        super().__init__()
        self.model_label = int(model_label)
        self.generator = RSGGCEGenerator(
            node_features=node_features,
            hidden_dim=generator_hidden_dim,
            num_conv_layers=num_conv_layers,
            residuals=residuals,
        )
        self.discriminator = RSGGCEDiscriminator(
            node_features=node_features,
            max_num_nodes=max_num_nodes,
            hidden_dim=discriminator_hidden_dim,
        )

    def generate(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.generator(data.x, data.edge_index)


class RSGGCEExplainer(BaseExplainer):
    """RSGG-CE baseline with the same external style as existing baselines."""

    def __init__(
        self,
        node_features: int,
        num_classes: int,
        max_num_nodes: int,
        generator_hidden_dim: Optional[int] = None,
        discriminator_hidden_dim: int = 16,
        num_conv_layers: int = 2,
        residuals: bool = True,
        sampling_iterations: int = 500,
        use_generated_features: bool = True,
    ) -> None:
        super().__init__()
        self.node_features = int(node_features)
        self.num_classes = int(num_classes)
        self.max_num_nodes = int(max_num_nodes)
        self.sampling_iterations = int(sampling_iterations)
        self.use_generated_features = bool(use_generated_features)

        self.class_gans = nn.ModuleDict(
            {
                str(label): RSGGCEClassGAN(
                    model_label=label,
                    node_features=self.node_features,
                    max_num_nodes=self.max_num_nodes,
                    generator_hidden_dim=generator_hidden_dim,
                    discriminator_hidden_dim=discriminator_hidden_dim,
                    num_conv_layers=num_conv_layers,
                    residuals=residuals,
                )
                for label in range(self.num_classes)
            }
        )
        self._oracle_model: Optional[nn.Module] = None

    def _gan_for_label(self, label: int) -> RSGGCEClassGAN:
        return self.class_gans[str(int(label))]

    @torch.no_grad()
    def _explain_graph_core(
        self,
        data: Data,
        oracle_model: nn.Module,
        device: Union[torch.device, str],
        ori_pred: Optional[int] = None,
        desired_label: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device_obj = _as_device(device)
        graph = data.to(device_obj)

        if ori_pred is None or desired_label is None:
            _, logits, pred = _predict_single_graph(
                oracle_model,
                graph.x,
                graph.edge_index,
                device_obj,
            )
            ori_pred = pred
            desired_label = _select_desired_label(logits, ori_pred)

        gan = self._gan_for_label(ori_pred)
        gan.eval()
        generated_x, edge_probs = gan.generate(graph)
        cf_x = generated_x if self.use_generated_features else graph.x

        cf_edge_index = self._sample_counterfactual_edges(
            x=cf_x,
            ori_edge_index=graph.edge_index,
            edge_probs=edge_probs,
            oracle_model=oracle_model,
            ori_pred=int(ori_pred),
            desired_label=int(desired_label),
            device=device_obj,
        )
        return cf_edge_index, cf_x

    @torch.no_grad()
    def _sample_counterfactual_edges(
        self,
        x: torch.Tensor,
        ori_edge_index: torch.Tensor,
        edge_probs: torch.Tensor,
        oracle_model: nn.Module,
        ori_pred: int,
        desired_label: int,
        device: torch.device,
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        edge_probs = _edge_probabilities_to_valid_matrix(edge_probs, num_nodes)

        existing = _unique_undirected_edges(ori_edge_index, num_nodes=num_nodes, device=device)
        missing = _complete_non_edges(existing, num_nodes=num_nodes, device=device)

        keep_scores = _scores_for_edges(edge_probs, existing)
        add_scores = _scores_for_edges(edge_probs, missing)

        original_unique = existing.clone()
        original_full = _to_full_undirected_edge_index(original_unique).to(device)
        best_edge_index = original_full
        best_score = float("-inf")

        def evaluate_candidate(unique_edges: torch.Tensor) -> Tuple[bool, float]:
            nonlocal best_edge_index, best_score
            cf_edge_index = _to_full_undirected_edge_index(unique_edges).to(device)
            probs, _, cf_pred = _predict_single_graph(
                oracle_model,
                x,
                cf_edge_index,
                device,
            )
            target_score = float(probs[desired_label].item() - probs[ori_pred].item())
            if target_score > best_score:
                best_score = target_score
                best_edge_index = cf_edge_index
            return cf_pred == desired_label, target_score

        if existing.numel() == 0 and missing.numel() == 0:
            return best_edge_index

        # Stochastic phase: mirrors the official positive/negative edge sampler.
        current_unique = original_unique.clone()
        if existing.numel() > 0:
            keep_probs = keep_scores.clamp(1e-6, 1.0)
        else:
            keep_probs = keep_scores
        if missing.numel() > 0:
            add_probs = add_scores.clamp(1e-6, 1.0)
        else:
            add_probs = add_scores

        for iteration in range(max(0, self.sampling_iterations)):
            if existing.size(1) > 0:
                bernoulli = torch.rand(existing.size(1), device=device)
                keep_mask = bernoulli <= keep_probs
                if not keep_mask.any():
                    best_idx = int(torch.argmax(keep_probs).item())
                    keep_mask[best_idx] = True
                current_unique = existing[:, keep_mask]

            if iteration > 0 and missing.size(1) > 0:
                add_count = min(iteration, missing.size(1))
                sampled = torch.multinomial(add_probs, num_samples=add_count, replacement=False)
                current_unique = torch.cat([current_unique, missing[:, sampled]], dim=1)

            found, _ = evaluate_candidate(current_unique)
            if found:
                return best_edge_index

        # Deterministic fallback: remove low-score original edges, then add high-score missing edges.
        current_unique = original_unique.clone()
        if existing.size(1) > 0:
            remove_order = torch.argsort(keep_scores, descending=False)
            keep_mask = torch.ones(existing.size(1), dtype=torch.bool, device=device)
            for edge_idx in remove_order:
                keep_mask[edge_idx] = False
                current_unique = existing[:, keep_mask]
                found, _ = evaluate_candidate(current_unique)
                if found:
                    return best_edge_index

        if missing.size(1) > 0:
            current_unique = original_unique.clone()
            add_order = torch.argsort(add_scores, descending=True)
            add_budget = min(missing.size(1), max(1, existing.size(1)))
            for edge_idx in add_order[:add_budget]:
                current_unique = torch.cat([current_unique, missing[:, edge_idx.view(1)]], dim=1)
                found, _ = evaluate_candidate(current_unique)
                if found:
                    return best_edge_index

        return best_edge_index

    def explain_graph(self, data, device="cpu"):
        """Generate a counterfactual explanation for a single graph.

        Args:
            data: PyG Data object with x, edge_index.
            device: Device string for computation.

        Returns:
            CFResult with cf_edge_index and cf_edge_weight.
        """
        cf_edge_index, cf_x = self._explain_graph_core(
            data=data,
            oracle_model=self._oracle_model,
            device=device,
        )

        return CFResult(
            cf_edge_index=cf_edge_index,
            cf_edge_weight=torch.ones(cf_edge_index.size(1), device=_as_device(device)),
        )

    def fit(self, train_dataset, gnn, device="cpu", epochs=100, lr=1e-3, model_path=None, **kwargs):
        """Train the RSGG-CE explainer on a dataset.

        Args:
            train_dataset: Training dataset.
            gnn: Pre-trained GNN classifier.
            device: Device string.
            epochs: Number of training epochs.
            lr: Learning rate.
            model_path: Path to save the best model checkpoint.
        """
        trained = train_rsgg_ce(
            pred_model=gnn,
            train_dataset=train_dataset,
            epochs=epochs,
            device=device,
            lr=lr,
            model_path=model_path,
            **kwargs,
        )
        self.load_state_dict(trained.state_dict())
        self._oracle_model = gnn


def _partition_dataset_by_prediction(
    pred_model: nn.Module,
    dataset: Sequence[Data],
    num_classes: int,
    device: torch.device,
) -> Tuple[Dict[int, List[Data]], List[int]]:
    by_label: Dict[int, List[Data]] = {label: [] for label in range(num_classes)}
    pred_labels: List[int] = []

    pred_model.eval()
    for idx in tqdm(range(len(dataset)), desc="Pre-computing train predictions"):
        data = dataset[idx].to(device)
        _, _, pred = _predict_single_graph(pred_model, data.x, data.edge_index, device)
        pred_labels.append(pred)
        by_label.setdefault(pred, []).append(dataset[idx].cpu())

    return by_label, pred_labels


def _infinite_loader(data_list: Sequence[Data], device: torch.device) -> Iterable[Data]:
    loader = DataLoader(data_list, batch_size=1, shuffle=True)
    while True:
        for batch in loader:
            yield batch.to(device)


def _edge_weights_from_prob_matrix(edge_probs: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((0,), dtype=edge_probs.dtype, device=edge_probs.device)
    return edge_probs[edge_index[0], edge_index[1]].clamp(0.0, 1.0)


def _train_single_class_gan(
    class_gan: RSGGCEClassGAN,
    real_data: Sequence[Data],
    source_data: Sequence[Data],
    epochs: int,
    lr: float,
    device: torch.device,
    feature_recon_weight: float,
    edge_recon_weight: float,
) -> None:
    if not real_data or not source_data:
        return

    class_gan.train()
    class_gan.to(device)

    real_stream = _infinite_loader(real_data, device)
    source_stream = _infinite_loader(source_data, device)
    gen_opt = torch.optim.SGD(class_gan.generator.parameters(), lr=lr)
    disc_opt = torch.optim.SGD(class_gan.discriminator.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    progress = tqdm(range(epochs), desc=f"Training RSGG-CE class {class_gan.model_label}")
    for _ in progress:
        real_batch = next(real_stream)
        source_batch = next(source_stream)

        disc_opt.zero_grad()
        real_weight = torch.ones(real_batch.edge_index.size(1), device=device)
        real_logit = class_gan.discriminator(
            real_batch.x,
            real_batch.edge_index,
            real_batch.batch,
            edge_weight=real_weight,
        )

        with torch.no_grad():
            fake_x_detached, fake_edge_probs_detached = class_gan.generator(
                source_batch.x,
                source_batch.edge_index,
            )
            fake_edge_weight_detached = _edge_weights_from_prob_matrix(
                fake_edge_probs_detached,
                source_batch.edge_index,
            )
        fake_logit = class_gan.discriminator(
            fake_x_detached.detach(),
            source_batch.edge_index,
            source_batch.batch,
            edge_weight=fake_edge_weight_detached.detach(),
        )

        real_target = torch.ones_like(real_logit)
        fake_target = torch.zeros_like(fake_logit)
        disc_loss = 0.5 * (bce(real_logit, real_target) + bce(fake_logit, fake_target))
        disc_loss.backward()
        disc_opt.step()

        gen_opt.zero_grad()
        fake_x, fake_edge_probs = class_gan.generator(source_batch.x, source_batch.edge_index)
        fake_edge_weight = _edge_weights_from_prob_matrix(fake_edge_probs, source_batch.edge_index)
        gen_logit = class_gan.discriminator(
            fake_x,
            source_batch.edge_index,
            source_batch.batch,
            edge_weight=fake_edge_weight,
        )
        adv_loss = bce(gen_logit, torch.ones_like(gen_logit))
        feature_loss = F.mse_loss(fake_x, source_batch.x.float())

        edge_loss = torch.zeros((), device=device)
        if edge_recon_weight > 0.0:
            adj = to_dense_adj(source_batch.edge_index, max_num_nodes=source_batch.num_nodes).squeeze(0)
            edge_loss = F.binary_cross_entropy(fake_edge_probs, adj.to(fake_edge_probs))

        gen_loss = adv_loss + feature_recon_weight * feature_loss + edge_recon_weight * edge_loss
        gen_loss.backward()
        gen_opt.step()

        progress.set_postfix(
            {
                "D": f"{disc_loss.item():.4f}",
                "G": f"{gen_loss.item():.4f}",
            }
        )


def train_rsgg_ce(
    pred_model: nn.Module,
    train_dataset: Union[InMemoryDataset, Sequence[Data]],
    epochs: int,
    device: Union[torch.device, str],
    lr: float = 1e-3,
    model_path: Optional[str] = None,
    num_classes: Optional[int] = None,
    max_num_nodes: Optional[int] = None,
    generator_hidden_dim: Optional[int] = None,
    discriminator_hidden_dim: int = 16,
    num_conv_layers: int = 2,
    residuals: bool = True,
    sampling_iterations: int = 500,
    use_generated_features: bool = True,
    feature_recon_weight: float = 1.0,
    edge_recon_weight: float = 0.0,
) -> RSGGCEExplainer:
    """Train RSGG-CE on a project dataset.

    This mirrors the baseline style used in models/clear.py: train on the train
    split once, save an explainer checkpoint if requested, and reuse it on any
    eval split via BaseExplainer.explain_graph().
    """

    device_obj = _as_device(device)
    pred_model = pred_model.to(device_obj)
    pred_model.eval()
    for param in pred_model.parameters():
        param.requires_grad = False

    if len(train_dataset) == 0:
        raise ValueError("train_dataset is empty")

    sample = train_dataset[0]
    node_features = int(sample.x.size(1))
    if max_num_nodes is None:
        max_num_nodes = max(int(train_dataset[idx].num_nodes) for idx in range(len(train_dataset)))
    if num_classes is None:
        num_classes = _num_classes_from_model_or_data(pred_model, sample.to(device_obj), device_obj)

    explainer = RSGGCEExplainer(
        node_features=node_features,
        num_classes=num_classes,
        max_num_nodes=max_num_nodes,
        generator_hidden_dim=generator_hidden_dim,
        discriminator_hidden_dim=discriminator_hidden_dim,
        num_conv_layers=num_conv_layers,
        residuals=residuals,
        sampling_iterations=sampling_iterations,
        use_generated_features=use_generated_features,
    ).to(device_obj)

    by_label, _ = _partition_dataset_by_prediction(
        pred_model=pred_model,
        dataset=train_dataset,
        num_classes=num_classes,
        device=device_obj,
    )

    for label in range(num_classes):
        real_data = by_label.get(label, [])
        source_data: List[Data] = []
        for other_label, data_list in by_label.items():
            if other_label != label:
                source_data.extend(data_list)

        print(
            f"RSGG-CE class {label}: real={len(real_data)}, "
            f"source={len(source_data)}"
        )
        if not real_data or not source_data:
            print(f"Skipping class {label} because one side is empty.")
            continue

        _train_single_class_gan(
            class_gan=explainer._gan_for_label(label),
            real_data=real_data,
            source_data=source_data,
            epochs=epochs,
            lr=lr,
            device=device_obj,
            feature_recon_weight=feature_recon_weight,
            edge_recon_weight=edge_recon_weight,
        )

    if model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
            {
                "state_dict": explainer.state_dict(),
                "config": {
                    "node_features": node_features,
                    "num_classes": num_classes,
                    "max_num_nodes": max_num_nodes,
                    "generator_hidden_dim": generator_hidden_dim,
                    "discriminator_hidden_dim": discriminator_hidden_dim,
                    "num_conv_layers": num_conv_layers,
                    "residuals": residuals,
                    "sampling_iterations": sampling_iterations,
                    "use_generated_features": use_generated_features,
                },
            },
            model_path,
        )

    explainer.eval()
    return explainer


def load_rsgg_ce(
    model_path: str,
    map_location: Union[torch.device, str] = "cpu",
) -> RSGGCEExplainer:
    checkpoint = torch.load(model_path, map_location=map_location)
    config = checkpoint["config"]
    explainer = RSGGCEExplainer(**config)
    explainer.load_state_dict(checkpoint["state_dict"])
    explainer.to(_as_device(map_location))
    explainer.eval()
    return explainer


if __name__ == "__main__":
    dataset_name = os.environ.get("MYEXPLAINER_DATASET", "alkane_carbonyl")
    epochs = int(os.environ.get("RSGG_CE_EPOCHS", "1000"))
    lr = float(os.environ.get("RSGG_CE_LR", "0.001"))
    sampling_iterations = int(os.environ.get("RSGG_CE_SAMPLING_ITERATIONS", "500"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Using dataset: {dataset_name}")

    train_dataset, val_dataset, test_dataset = get_datasets(
        name=dataset_name,
        root=_resolve_project_path("data"),
    )

    gnn_path = _resolve_project_path("param", "gnns", f"{dataset_name}_gcn.pt")
    pred_model = torch.load(gnn_path, map_location=device)
    pred_model = pred_model.to(device)
    pred_model.eval()
    print("GNN classifier loaded.")

    print("Training RSGG-CE explainer from scratch.")
    explainer = train_rsgg_ce(
        pred_model=pred_model,
        train_dataset=train_dataset,
        epochs=epochs,
        device=device,
        lr=lr,
        model_path=None,
        sampling_iterations=sampling_iterations,
    )
    explainer._oracle_model = pred_model

    for idx in range(min(10, len(val_dataset))):
        data = val_dataset[idx]
        result = explainer.explain_graph(data, device=device)
        print(f"Graph {idx}: CF edges={result.cf_edge_index.size(1)}")
