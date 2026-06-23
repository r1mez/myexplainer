"""
Case Study: t-SNE In-Distribution Check for Generated CF Graphs

Plots original graphs and MyExplainerV2-generated CF graphs on the same
t-SNE plot to visually assess whether the generated CF graphs fall within
the data distribution.

4 groups, 4 colors:
  - Original Predicted Class 0  (solid red)
  - Original Predicted Class 1  (solid blue)
  - Generated CF Target Class 0 (hollow orange)
  - Generated CF Target Class 1 (hollow green)

Only 10% of the dataset is randomly sampled for CF generation.

Usage:
    python case_study/tsne_indistribution_vis.py --dataset ba2motif
    python case_study/tsne_indistribution_vis.py --dataset mutag --sample_ratio 0.1
"""

import argparse
import sys
import os
import warnings
import types
import random

import torch
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnns import *
from models.myexplainerV2 import MyExplainerV2
from utils.dataset import get_datasets
from utils.pair_data import MappedDataset, train_collate_fn
from utils.batch_utils import output_to_batch
from utils.subgraph_method import (
    GraphRepModel, graphsampler,
    GraphRepModelDiscrete, graphsamplerDiscrete,
)


# ---------------------------------------------------------------------------
# Custom subgraph mining (same as tsne_subgraph_vis.py, no filtering)
# ---------------------------------------------------------------------------

def mine_subgraphs(args, datasets, num_samples=100):
    dataset_name = args.dataset.lower()
    patterns_0 = []
    patterns_1 = []

    if dataset_name == 'ba2motif':
        print(f"  Sampling {num_samples} patterns per class (continuous)")
        Bdist, mean_estimate, result, Adj = GraphRepModel(datasets[0], 25)
        for _ in range(num_samples):
            patterns_0.append(graphsampler(25, Bdist, mean_estimate, result, Adj))
        Bdist, mean_estimate, result, Adj = GraphRepModel(datasets[1], 25)
        for _ in range(num_samples):
            patterns_1.append(graphsampler(25, Bdist, mean_estimate, result, Adj))
    else:
        feature_dim = max(
            max((int(d.x.shape[1]) for d in datasets[0]), default=1),
            max((int(d.x.shape[1]) for d in datasets[1]), default=1),
        )
        max_nodes_0 = max((int(d.num_nodes) for d in datasets[0]), default=1)
        max_nodes_1 = max((int(d.num_nodes) for d in datasets[1]), default=1)
        print(f"  Sampling {num_samples} patterns per class (discrete, feat_dim={feature_dim})")
        X, Adj = GraphRepModelDiscrete(datasets[0], max_nodes_0)
        for _ in range(num_samples):
            patterns_0.append(graphsamplerDiscrete(max_nodes_0, X, Adj, num_node_features=feature_dim))
        X, Adj = GraphRepModelDiscrete(datasets[1], max_nodes_1)
        for _ in range(num_samples):
            patterns_1.append(graphsamplerDiscrete(max_nodes_1, X, Adj, num_node_features=feature_dim))

    # Sort by size (largest first)
    sort_key = lambda G: (G.number_of_nodes(), G.number_of_edges())
    patterns_0.sort(key=sort_key, reverse=True)
    patterns_1.sort(key=sort_key, reverse=True)

    print(f"  Class 0: {len(patterns_0)} patterns, Class 1: {len(patterns_1)} patterns")
    return {0: patterns_0, 1: patterns_1}


def build_mining_args(dataset_name, x_dim, device):
    args = types.SimpleNamespace()
    args.dataset = dataset_name
    args.device = device
    args.subgraph_method = "genGraphEx"
    args.x_dim = x_dim
    args.proto_topk = 1000
    return args


# ---------------------------------------------------------------------------
# t-SNE helpers
# ---------------------------------------------------------------------------

def run_tsne(features, perplexity=30, random_state=42, use_pca_pre=False):
    if use_pca_pre and features.shape[1] > 50:
        n_comp = min(50, features.shape[0] - 1, features.shape[1])
        pca = PCA(n_components=n_comp, random_state=random_state)
        features = pca.fit_transform(features)

    n_samples = features.shape[0]
    adjusted_perplexity = min(perplexity, n_samples - 1)
    tsne = TSNE(
        n_components=2,
        perplexity=adjusted_perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(features)


def plot_tsne_4groups(
    embeddings_2d,
    group_labels,
    title="t-SNE: Original vs Generated CF Graphs",
    save_path=None,
    flip_rate=None,
    gnn_accuracy=None,
):
    """Plot 4 groups with distinct colors and marker styles."""
    config = {
        0: {"label": "Original Class 0",     "color": "#E74C3C", "marker": "o", "alpha": 0.5,  "s": 30, "edgecolors": "none"},
        1: {"label": "Original Class 1",     "color": "#3498DB", "marker": "o", "alpha": 0.5,  "s": 30, "edgecolors": "none"},
        2: {"label": "Generated CF → Class 0", "color": "#F39C12", "marker": "^", "alpha": 1.0,  "s": 60, "edgecolors": "#E67E22", "linewidths": 1.2, "face_alpha": 0.3},
        3: {"label": "Generated CF → Class 1", "color": "#2ECC71", "marker": "^", "alpha": 1.0,  "s": 60, "edgecolors": "#27AE60", "linewidths": 1.2, "face_alpha": 0.3},
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for gid, cfg in config.items():
        mask = group_labels == gid
        if mask.sum() == 0:
            continue
        face_alpha = cfg.get("face_alpha")
        scatter_kwargs = dict(
            marker=cfg["marker"],
            label=f'{cfg["label"]} ({mask.sum()})',
            alpha=cfg["alpha"],
            s=cfg["s"],
            edgecolors=cfg["edgecolors"],
            linewidths=cfg.get("linewidths", 0),
        )
        if face_alpha is not None:
            scatter_kwargs["facecolors"] = (*matplotlib.colors.to_rgb(cfg["color"]), face_alpha)
        else:
            scatter_kwargs["c"] = cfg["color"]
        ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], **scatter_kwargs)

    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.tick_params(labelsize=10)

    # Annotate metrics
    info_lines = []
    if gnn_accuracy is not None:
        info_lines.append(f"GNN Accuracy: {gnn_accuracy:.2f}%")
    if flip_rate is not None:
        info_lines.append(f"CF Flip Rate: {flip_rate:.2f}%")
    if info_lines:
        info_text = "\n".join(info_lines)
        ax.text(
            0.98, 0.02, info_text,
            transform=ax.transAxes, fontsize=11,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="gray", alpha=0.9),
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="t-SNE in-distribution check for generated CF graphs"
    )
    parser.add_argument(
        "--dataset", type=str, default="mutag",
        choices=[
            "ba2motif", "mutag", "mutag188", "nci1",
            "bbbp", "alkane_carbonyl", "fluoride_carbonyl", "proteins",
        ],
    )
    parser.add_argument("--perplexity", type=float, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--sample_ratio", type=float, default=0.05,
        help="Fraction of dataset to generate CF graphs for",
    )
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument(
        "--proto_topk", type=int, default=100,
        help="Top-K pattern families per class for MappedDataset (keeps VF2 fast)",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load dataset & GNN
    # ------------------------------------------------------------------
    print(f"\n[1] Loading dataset: {args.dataset}")
    train_dataset, val_dataset, test_dataset = get_datasets(args.dataset)
    dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])
    x_dim = train_dataset[0].x.shape[1]
    print(f"  x_dim={x_dim}, total={len(dataset)} "
          f"(train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)})")

    gnn_path = f"param/gnns/{args.dataset.lower()}_gcn.pt"
    print(f"  Loading GNN from: {gnn_path}")
    gnn = torch.load(gnn_path, map_location=device)
    for p in gnn.parameters():
        p.requires_grad_(False)
    gnn.to(device)
    gnn.eval()

    # ------------------------------------------------------------------
    # 2. Get GNN predictions for all graphs
    # ------------------------------------------------------------------
    print("\n[2] Getting GNN predictions...")
    all_loader = PyGDataLoader(dataset, batch_size=256, shuffle=False)
    all_pred_labels = []
    all_pred_probs = []
    with torch.no_grad():
        for batch in all_loader:
            batch = batch.to(device)
            probs, _ = gnn.get_pred(batch.x, batch.edge_index, batch.batch)
            all_pred_labels.append(probs.argmax(dim=1).cpu())
            all_pred_probs.append(probs.cpu())
    all_pred_labels = torch.cat(all_pred_labels).numpy()
    all_pred_probs = torch.cat(all_pred_probs)

    # Split by predicted class
    train_dataset_0 = [dataset[i] for i in range(len(dataset)) if all_pred_labels[i] == 0]
    train_dataset_1 = [dataset[i] for i in range(len(dataset)) if all_pred_labels[i] == 1]
    splited = {0: train_dataset_0, 1: train_dataset_1}
    print(f"  Predicted class 0: {len(train_dataset_0)}, class 1: {len(train_dataset_1)}")

    # ------------------------------------------------------------------
    # 3. Mine subgraphs + build MappedDataset
    # ------------------------------------------------------------------
    print("\n[3] Mining frequent subgraphs...")
    mining_args = build_mining_args(args.dataset, x_dim, device)
    patterns = mine_subgraphs(mining_args, splited, num_samples=args.num_samples)

    print(f"\n[4] Building MappedDataset (VF2 subgraph matching)...")

    md_args = types.SimpleNamespace()
    md_args.dataset = args.dataset
    md_args.device = device
    md_args.threshold = 0.5
    md_args.x_dim = x_dim
    md_args.h_dim = 256
    md_args.z_dim = 31
    md_args.num_classes = 2

    mapped_dataset = MappedDataset(
        md_args, dataset, patterns,
        pred_labels=all_pred_labels,
        pred_probs=all_pred_probs,
    )

    # ------------------------------------------------------------------
    # 5. Load MyExplainerV2 checkpoint
    # ------------------------------------------------------------------
    print("\n[5] Loading MyExplainerV2 checkpoint...")
    ckpt_path = f"param/myexplainer_{args.dataset.lower()}_best.pt"
    state_dict = torch.load(ckpt_path, map_location=device)

    # Infer correct hyperparameters from checkpoint weight shapes
    ckpt_h_dim = state_dict["conv1.lin.weight"].shape[0]
    ckpt_z_dim = state_dict["add_net.encoder_mu.weight"].shape[0]
    # DeleteNet input dim tells us the edge feature concatenation strategy
    ckpt_delete_input_dim = state_dict["delete_net.net.0.weight"].shape[1]
    # delete_input = k * h_dim, where k depends on the forward() implementation
    ckpt_delete_k = ckpt_delete_input_dim // ckpt_h_dim

    print(f"  Inferred from checkpoint: h_dim={ckpt_h_dim}, z_dim={ckpt_z_dim}, "
          f"delete_net input={ckpt_delete_k}*h_dim")

    # Build model args matching the checkpoint
    model_args = types.SimpleNamespace()
    model_args.device = device
    model_args.x_dim = x_dim
    model_args.h_dim = ckpt_h_dim
    model_args.z_dim = ckpt_z_dim
    model_args.num_classes = 2
    model_args.dataset = args.dataset

    # If checkpoint DeleteNet uses 3*h_dim but code uses 2*h_dim, patch it
    from models.myexplainerV2 import DeleteNet, AddVGAENet
    import torch.nn as nn

    explainer = MyExplainerV2(model_args, gnn).to(device)

    if ckpt_delete_k != 2:
        # Replace DeleteNet with one matching the checkpoint architecture
        class PatchedDeleteNet(nn.Module):
            def __init__(self, h_dim, k):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(k * h_dim, h_dim),
                    nn.ReLU(),
                    nn.Linear(h_dim, 1)
                )
            def forward(self, node_rep, edge_index):
                src, dst = edge_index
                if ckpt_delete_k == 3:
                    e_feat = torch.cat([node_rep[src], node_rep[dst],
                                        node_rep[src] * node_rep[dst]], dim=-1)
                else:
                    e_feat = torch.cat([node_rep[src], node_rep[dst]], dim=-1)
                logit = self.net(e_feat).view(-1)
                prob = torch.sigmoid(logit)
                return prob, logit

        explainer.delete_net = PatchedDeleteNet(ckpt_h_dim, ckpt_delete_k).to(device)

    explainer.load_state_dict(state_dict, strict=False)
    explainer.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")

    # ------------------------------------------------------------------
    # 6. Sample 10% and generate CF graphs
    # ------------------------------------------------------------------
    n_total = len(mapped_dataset)
    n_sample = max(int(n_total * args.sample_ratio), 1)
    sample_indices = sorted(random.sample(range(n_total), n_sample))
    print(f"\n[6] Sampling {n_sample}/{n_total} graphs ({args.sample_ratio:.0%})...")

    # Build sampled items manually (avoid Subset + custom collate_fn issues)
    sampled_items = [mapped_dataset[i] for i in sample_indices]

    # Collate manually in batches
    cf_embeddings = []
    cf_target_labels = []
    n_valid_cf = 0
    n_flip_success = 0

    print("  Generating CF graphs...")
    with torch.no_grad():
        for batch_start in range(0, len(sampled_items), args.batch_size):
            batch_items = sampled_items[batch_start:batch_start + args.batch_size]
            batch_data = train_collate_fn(batch_items)

            graphs = batch_data["graphs"].to(device)
            subgraphs = batch_data["subgraphs"]

            # Get actual predictions for this batch
            probs, _ = gnn.get_pred(graphs.x, graphs.edge_index, graphs.batch)
            ori_preds = probs.argmax(dim=1)
            y_desired = 1 - ori_preds  # flip

            # Generate CF graphs
            outputs = explainer(
                graphs=graphs, subgraphs=subgraphs,
            )

            # Convert to PyG Batch
            cf_batch = output_to_batch(graphs, outputs, use_hard=True, thresh=0.5)

            if cf_batch is not None and cf_batch.num_nodes > 0:
                # Get embeddings for CF graphs
                cf_emb = gnn.get_graph_rep(
                    cf_batch.x, cf_batch.edge_index, cf_batch.batch
                )
                cf_embeddings.append(cf_emb.cpu())

                # Check flip validity: does GNN predict the desired class on CF graph?
                cf_probs, _ = gnn.get_pred(cf_batch.x, cf_batch.edge_index, cf_batch.batch)
                cf_preds = cf_probs.argmax(dim=1)

                cf_batch_size = int(cf_batch.batch.max().item()) + 1
                for g_idx in range(cf_batch_size):
                    orig_idx = g_idx
                    if orig_idx < len(ori_preds):
                        target = 1 - ori_preds[orig_idx].item()
                    else:
                        target = 0
                    cf_target_labels.append(target)
                    # Check if flip succeeded
                    if g_idx < len(cf_preds) and cf_preds[g_idx].item() == target:
                        n_flip_success += 1
                    n_valid_cf += 1

    if len(cf_embeddings) == 0:
        print("  No valid CF graphs generated. Exiting.")
        return

    cf_embeddings = torch.cat(cf_embeddings, dim=0).numpy()
    cf_target_labels = np.array(cf_target_labels)
    flip_rate = n_flip_success / max(n_valid_cf, 1) * 100
    print(f"  Generated {n_valid_cf} valid CF graphs")
    print(f"  Flip success rate: {n_flip_success}/{n_valid_cf} ({flip_rate:.2f}%)")

    # ------------------------------------------------------------------
    # 7. Get embeddings for ALL original graphs
    # ------------------------------------------------------------------
    print("\n[7] Extracting embeddings for all original graphs...")
    orig_loader = PyGDataLoader(dataset, batch_size=256, shuffle=False)
    orig_embeddings = []
    with torch.no_grad():
        for batch in orig_loader:
            batch = batch.to(device)
            emb = gnn.get_graph_rep(batch.x, batch.edge_index, batch.batch)
            orig_embeddings.append(emb.cpu())
    orig_embeddings = torch.cat(orig_embeddings, dim=0).numpy()

    # ------------------------------------------------------------------
    # 8. Combine and run t-SNE
    # ------------------------------------------------------------------
    print("\n[8] Running t-SNE...")
    # Normalize
    orig_emb_norm = normalize(orig_embeddings, norm='l2', axis=1)
    cf_emb_norm = normalize(cf_embeddings, norm='l2', axis=1)

    # Stack all embeddings
    all_emb = np.vstack([orig_emb_norm, cf_emb_norm])

    # Group labels: 0=orig_cls0, 1=orig_cls1, 2=cf_cls0, 3=cf_cls1
    orig_group_labels = all_pred_labels.copy()  # 0 or 1
    cf_group_labels = cf_target_labels.copy() + 2  # 2 or 3
    all_group_labels = np.concatenate([orig_group_labels, cf_group_labels])

    # Run t-SNE
    all_2d = run_tsne(all_emb, perplexity=args.perplexity,
                      random_state=args.seed, use_pca_pre=True)

    orig_2d = all_2d[:len(orig_embeddings)]
    cf_2d = all_2d[len(orig_embeddings):]

    print(f"  Original graphs: {len(orig_embeddings)}")
    print(f"  CF graphs: {len(cf_embeddings)}")

    # Accuracy
    accuracy = (all_pred_labels == np.array([
        dataset[i].y.item() if hasattr(dataset[i].y, 'item') else int(dataset[i].y)
        for i in range(len(dataset))
    ])).mean() * 100
    print(f"  GNN Accuracy: {accuracy:.2f}%")

    # ------------------------------------------------------------------
    # 9. Plot
    # ------------------------------------------------------------------
    combined_2d = np.vstack([orig_2d, cf_2d])

    title = (f"t-SNE: Original vs Generated CF Graphs "
             f"({args.dataset.upper()}, {args.sample_ratio:.0%} sampled)")

    if args.output:
        output_path = args.output
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "outputs", "indistribution"
        )
        output_path = os.path.join(
            output_dir, f"tsne_indistribution_{args.dataset}.png"
        )

    plot_tsne_4groups(
        combined_2d, all_group_labels,
        title=title, save_path=output_path,
        flip_rate=flip_rate,
        gnn_accuracy=accuracy,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
