"""Evaluation utilities for MyExplainer.

Currently this module provides a validity metric implementation which is
responsible for measuring the proportion of generated counterfactual graphs
that obtain the desired labels when evaluated by a pre-trained GNN model.
"""

from typing import Dict

import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from tqdm import tqdm


def compute_validity(args, model, gnn, data_loader, edge_threshold: float = 0.5) -> Dict[str, float]:
    """Evaluate the validity of counterfactual explanations produced by MyExplainer.

    The validity metric is defined as the proportion of generated
    counterfactual graphs that obtain the desired labels when passed through
    the target GNN classifier.

    Args:
        args: Namespace of runtime arguments. The function expects attributes
            such as ``device``, ``max_num_nodes`` and ``x_dim`` to be present.
        model: The trained MyExplainer model that generates counterfactuals.
        gnn: The pre-trained GNN classifier used to validate counterfactuals.
        data_loader: DataLoader that yields original graphs (without pairing).
        edge_threshold: Threshold applied to the reconstructed adjacency matrix
            to decide whether an edge exists between two nodes.

    Returns:
        A dictionary containing:

        * ``validity`` - the proportion of successful counterfactuals.
        * ``successful`` - number of counterfactuals that achieved the desired
          label.
        * ``total`` - total number of evaluated counterfactual graphs.
    """

    model.eval()
    gnn.eval()

    successful_cf = 0
    total_cf = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating validity"):
            batch = batch.to(args.device)
            batch_size = batch.num_graphs

            # Obtain the original GNN predictions.
            ori_pred_logits = gnn.get_pred(batch.x, batch.edge_index, batch.batch)[0]
            ori_pred = ori_pred_logits.argmax(dim=1)

            # Construct desired counterfactual labels by flipping predictions.
            cf_pred = 1 - ori_pred

            # Convert graph batch to dense representation for the explainer.
            ori_x, ori_mask = to_dense_batch(
                batch.x, batch.batch, max_num_nodes=args.max_num_nodes
            )
            ori_adj = to_dense_adj(
                batch.edge_index, batch.batch, max_num_nodes=args.max_num_nodes
            )

            # Prepare labels for the explainer model.
            y_cf = cf_pred.float().unsqueeze(1)
            y = ori_pred.float().unsqueeze(1)

            outputs = model(
                features=ori_x,
                adj=ori_adj,
                y_cf=y_cf,
                features_tgt=ori_x,
                adj_tgt=ori_adj,
                y=y,
            )

            x_recon = outputs["x_recon"].view(batch_size, args.max_num_nodes, args.x_dim)
            adj_recon = outputs["adj_recon"].view(
                batch_size, args.max_num_nodes, args.max_num_nodes
            )

            recon_graphs = []
            recon_batch_indices = []

            for b in range(batch_size):
                num_nodes = int(ori_mask[b].sum().item())
                if num_nodes == 0:
                    continue

                # Discretise node features with argmax over categories.
                x_sample = x_recon[b, :num_nodes, :]
                x_one_hot = torch.zeros_like(x_sample)
                atom_idx = torch.argmax(x_sample, dim=-1)
                x_one_hot.scatter_(1, atom_idx.unsqueeze(-1), 1.0)

                # Threshold adjacency to obtain sparse representation.
                adj_sample = adj_recon[b, :num_nodes, :num_nodes]
                edge_indices = (adj_sample > edge_threshold).nonzero(as_tuple=False)
                if edge_indices.size(0) > 0:
                    edge_index = edge_indices.t()
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long, device=args.device)

                recon_graphs.append({
                    "x": x_one_hot,
                    "edge_index": edge_index,
                    "num_nodes": num_nodes,
                })
                recon_batch_indices.extend([b] * num_nodes)

            if not recon_graphs:
                continue

            recon_x = torch.cat([g["x"] for g in recon_graphs], dim=0)

            recon_edge_indices = []
            node_offset = 0
            for g in recon_graphs:
                if g["edge_index"].size(1) > 0:
                    recon_edge_indices.append(g["edge_index"] + node_offset)
                node_offset += g["num_nodes"]

            if recon_edge_indices:
                recon_edge_index = torch.cat(recon_edge_indices, dim=1)
            else:
                recon_edge_index = torch.empty((2, 0), dtype=torch.long, device=args.device)

            recon_batch = torch.tensor(recon_batch_indices, dtype=torch.long, device=args.device)

            pred_logits_recon = gnn.get_pred(recon_x, recon_edge_index, recon_batch)[0]
            pred_labels_recon = pred_logits_recon.argmax(dim=1)

            valid_indices = torch.arange(len(recon_graphs), device=args.device)
            desired_labels = cf_pred[valid_indices]

            successful_cf += (pred_labels_recon == desired_labels).sum().item()
            total_cf += len(recon_graphs)

    validity = successful_cf / total_cf if total_cf > 0 else 0.0

    return {
        "validity": validity,
        "successful": successful_cf,
        "total": total_cf,
    }
