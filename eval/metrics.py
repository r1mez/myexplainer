"""Consolidated evaluation metrics for counterfactual explanations.

All metric functions accept config object for device resolution.
"""
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Batch

from utils.graph_utils import extract_explanatory_subgraph


def proximity(config, cf_graphs, ori_graphs):
    """Compute adjacency matrix distance (L1 Norm) between original and CF graphs.

    Args:
        config: ExplainerConfig with device attribute
        cf_graphs: Batch of counterfactual graphs
        ori_graphs: Batch of original graphs

    Returns:
        float: sum of per-graph proximity distances
    """
    rho = 1.0

    ori_list = ori_graphs.to_data_list()
    cf_list = cf_graphs.to_data_list()
    batch_size = len(ori_list)
    distances = torch.zeros(batch_size, device=config.device)

    for i in range(batch_size):
        orig_data = ori_list[i]
        cf_data = cf_list[i]

        # Determine unified node count N
        if getattr(orig_data, 'num_nodes', None) is not None:
            N = orig_data.num_nodes
        elif getattr(orig_data, 'x', None) is not None:
            N = orig_data.x.size(0)
        else:
            max_idx = 0
            if orig_data.edge_index.numel() > 0:
                max_idx = int(orig_data.edge_index.max())
            if cf_data.edge_index.numel() > 0:
                max_idx = max(max_idx, int(cf_data.edge_index.max()))
            N = max_idx + 1

        orig_adj = to_dense_adj(orig_data.edge_index, max_num_nodes=N).squeeze(0)
        cf_adj = to_dense_adj(cf_data.edge_index, max_num_nodes=N).squeeze(0)

        d_adj_entries = torch.norm(orig_adj - cf_adj, p=1)

        m_orig = orig_data.num_edges // 2 if orig_data.is_undirected() else orig_data.num_edges
        m_cf = cf_data.num_edges // 2 if cf_data.is_undirected() else cf_data.num_edges
        max_m = max(m_orig, m_cf)

        normalization = 2.0 * max_m if max_m > 0 else 1.0
        distances[i] = rho * (d_adj_entries / normalization)

    return distances.sum().item()


def fidelity(config, ori_graphs, cf_graphs, ori_prob, gnn):
    """Compute prediction probability drop for original class.

    Args:
        config: ExplainerConfig with device attribute
        ori_graphs: Batch of original graphs
        cf_graphs: Batch of counterfactual graphs
        ori_prob: Original prediction probabilities [N, num_classes]
        gnn: Pre-trained GNN classifier

    Returns:
        float: sum of per-graph fidelity values
    """
    ori_pred = ori_prob.argmax(dim=1)

    cf_pred_logits = gnn.get_pred(
        cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch
    )[0]
    cf_prob = F.softmax(cf_pred_logits, dim=1)

    fidelity_sum = 0.0
    for i in range(len(ori_pred)):
        ori_prob_single = ori_prob[i, ori_pred[i]].item()
        cf_prob_single = cf_prob[i, ori_pred[i]].item()
        fidelity_sum += (ori_prob_single - cf_prob_single)

    return fidelity_sum


def sparsity(config, ori_graphs, cf_graphs):
    """Compute 1 - (explanatory_edges / original_edges).

    Args:
        config: ExplainerConfig (unused but kept for interface consistency)
        ori_graphs: Batch of original graphs
        cf_graphs: Batch of counterfactual graphs

    Returns:
        float: sum of per-graph sparsity values
    """
    ori_list = ori_graphs.to_data_list()
    cf_list = cf_graphs.to_data_list()
    exp_graphs = [extract_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_list, cf_list)]

    sparsity_sum = 0.0
    for ori, exp in zip(ori_list, exp_graphs):
        ori_e = ori.num_edges
        exp_e = exp.num_edges
        sparsity_sum += 1 - (exp_e / ori_e) if ori_e > 0 else 0.0

    return sparsity_sum
