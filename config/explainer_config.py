"""Immutable configuration for MyExplainer pipeline.

Loads CLI args and YAML configs into a single validated object.
Replaces the mutable ``args`` Namespace that was threaded through every module.
"""
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch

from utils.simple_yaml import load_yaml_file

# Paths to per-dataset YAML config files (resolved relative to repo root).
_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
_LOSS_HPARAMS_PATH = _CONFIGS_DIR / "loss_hparams.yaml"
_EXPLAINER_HPARAMS_PATH = _CONFIGS_DIR / "explainer_hparams.yaml"

# Integer-typed explainer hyperparameter keys (all others are cast to float).
_EXPLAINER_INT_KEYS = frozenset({
    "oracle_del_topk",
    "oracle_del_random_negatives",
    "oracle_del_probe_graphs_per_batch",
})


def _load_dataset_yaml(path: Path, dataset_key: str) -> dict:
    """Load the sub-dict for *dataset_key* from a ``datasets:`` YAML file."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = load_yaml_file(path)
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        raise ValueError(f"Config must contain a top-level 'datasets' mapping: {path}")

    if dataset_key not in datasets:
        available = ", ".join(sorted(datasets))
        raise KeyError(
            f"No configuration for dataset '{dataset_key}' in {path.name}. "
            f"Available: {available}"
        )

    params = datasets[dataset_key]
    if not isinstance(params, dict):
        raise ValueError(
            f"Configuration for dataset '{dataset_key}' in {path.name} must be a mapping."
        )
    return params


@dataclass(frozen=True)
class ExplainerConfig:
    # --- Core settings ---
    dataset: str
    device: torch.device
    cuda: int = 0
    train_mode: bool = True
    task: str = "graph"

    # --- Data parameters ---
    top_k: int = 1
    threshold: float = 0.0
    batch_size: int = 256

    # --- Model parameters ---
    h_dim: int = 256
    z_dim: int = 32
    max_num_nodes: int = 25
    dropout: float = 0.1

    # --- Training parameters ---
    epochs: int = 1
    lr: float = 0.01
    weight_decay: float = 1e-5
    grad_clip_max_norm: float = 1.0
    scheduler_factor: float = 0.8
    scheduler_patience: int = 15
    scheduler_min_lr: float = 1e-6

    # --- Subgraph parameters ---
    subgraph_method: str = "genGraphEx"

    # --- GNN path ---
    gnn_path: str = "param/"

    # --- Loss hyperparameters (loaded from YAML) ---
    w_cf: float = 5.0
    w_l1_add: float = 0.1
    w_l1_del: float = 1.0
    w_oracle_del_rank: float = 1.0
    w_vgae_recon: float = 5.0
    w_vgae_kl: float = 1.0
    enable_fs_feature_recon: bool = False
    w_vgae_feat_recon: float = 1.0
    w_proto: float = 0.0
    cf_margin: float = 0.5
    lambda_cf_margin: float = 1.0

    # --- Explainer hyperparameters (loaded from YAML) ---
    oracle_del_topk: int = 6
    oracle_del_random_negatives: int = 2
    oracle_del_probe_graphs_per_batch: int = 4
    oracle_del_reward_tie_eps: float = 1e-6

    # --- Dynamic fields (set after dataset load) ---
    x_dim: int = 0
    edge_attr_dim: int = 0

    @classmethod
    def from_args(cls, args) -> "ExplainerConfig":
        """Build config from argparse Namespace, loading YAML hparams."""
        device = torch.device(
            f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
        )

        dataset_key = str(args.dataset).lower()

        # Load per-dataset loss hyperparameters from YAML
        loss_raw = _load_dataset_yaml(_LOSS_HPARAMS_PATH, dataset_key)

        # Load per-dataset explainer hyperparameters from YAML
        try:
            explainer_raw = _load_dataset_yaml(_EXPLAINER_HPARAMS_PATH, dataset_key)
        except (FileNotFoundError, KeyError):
            explainer_raw = {}

        # Build keyword dict from CLI args
        kw = dict(
            dataset=args.dataset,
            device=device,
            cuda=args.cuda,
            train_mode=getattr(args, "train_mode", True),
            task=getattr(args, "task", "graph"),
            top_k=args.top_k,
            threshold=args.threshold,
            batch_size=args.batch_size,
            h_dim=args.h_dim,
            z_dim=args.z_dim,
            max_num_nodes=args.max_num_nodes,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            subgraph_method=args.subgraph_method,
            gnn_path=args.gnn_path,
        )

        # Apply loss YAML values with proper typing
        for key, value in loss_raw.items():
            if key == "enable_fs_feature_recon":
                kw[key] = bool(value)
            else:
                kw[key] = float(value)

        # Apply explainer YAML values with proper typing
        for key, value in explainer_raw.items():
            if key in _EXPLAINER_INT_KEYS:
                kw[key] = int(value)
            else:
                kw[key] = float(value)

        return cls(**kw)

    def with_dataset_dims(self, x_dim: int, edge_attr_dim: int) -> "ExplainerConfig":
        """Return a new config with dataset-derived dimensions set."""
        return dataclasses.replace(self, x_dim=x_dim, edge_attr_dim=edge_attr_dim)
