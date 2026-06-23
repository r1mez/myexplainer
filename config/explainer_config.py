"""Immutable configuration for MyExplainer pipeline.

Loads CLI args and YAML configs into a single validated object.
Replaces the mutable ``args`` Namespace that was threaded through every module.
"""
import dataclasses
from dataclasses import dataclass

import torch

from utils.loss_hparams import load_loss_hparams
from utils.explainer_hparams import load_explainer_hparams


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

        # Load per-dataset loss hyperparameters from YAML
        loss_hparams = load_loss_hparams(args.dataset)

        # Load per-dataset explainer hyperparameters from YAML
        try:
            explainer_hparams = load_explainer_hparams(args.dataset)
        except (FileNotFoundError, KeyError):
            explainer_hparams = {}

        # Merge: CLI args override YAML defaults
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
        # Apply YAML loss hparams
        kw.update(loss_hparams)
        # Apply YAML explainer hparams
        kw.update(explainer_hparams)

        return cls(**kw)

    def with_dataset_dims(self, x_dim: int, edge_attr_dim: int) -> "ExplainerConfig":
        """Return a new config with dataset-derived dimensions set."""
        return dataclasses.replace(self, x_dim=x_dim, edge_attr_dim=edge_attr_dim)
