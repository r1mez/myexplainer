"""Evaluation utilities for MyExplainer.

Currently this module provides a validity metric implementation which is
responsible for measuring the proportion of generated counterfactual graphs
that obtain the desired labels when evaluated by a pre-trained GNN model.
"""

from typing import Dict

import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from tqdm import tqdm

from utils import concat_graphs

import torch.nn.functional as F


def evaluate(args, model, gnn, data_loader):
    model.eval()
    gnn.eval()

    device = args.device
    max_sub_nodes = args.max_subgraph_nodes

    proximity = 0.0
    valid_cf = 0
    total = data_loader.dataset.__len__()
    num_batches = 0  # 添加batch计数

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating validity"):
            graphs_batch = batch["graphs"].to(device)
            subgraphs_batch = batch["subgraphs"].to(device)
            batch["graphs"] = graphs_batch
            batch["subgraphs"] = subgraphs_batch

            batch_size = graphs_batch.num_graphs
            if batch_size == 0:
                continue

            ori_pred_logits = gnn.get_pred(
                graphs_batch.x, graphs_batch.edge_index, graphs_batch.batch
            )[0]
            ori_pred = ori_pred_logits.argmax(dim=1)

            cf_pred = 1 - ori_pred
            y_cf = cf_pred.float().unsqueeze(1)

            zero_template = torch.zeros(max_sub_nodes, args.x_dim, device=device)
            subgraph_x_list = []

            for b in range(batch_size):
                mask = subgraphs_batch.batch == b
                num_nodes = int(mask.sum().item())

                padded = zero_template.clone()
                if num_nodes > 0:
                    padded[:num_nodes] = subgraphs_batch.x[mask]

                subgraph_x_list.append(padded)

            subgraph_x = torch.stack(subgraph_x_list, dim=0)
            subgraph_adj = to_dense_adj(
                subgraphs_batch.edge_index,
                batch=subgraphs_batch.batch,
                max_num_nodes=max_sub_nodes,
            )

            outputs = model(features=subgraph_x, adj=subgraph_adj, y_cf=y_cf)

            concated_graphs = concat_graphs(args, outputs, batch)

            valid_cf += count_valid(cf_pred, concated_graphs, gnn)
            proximity += compute_proximity(args, concated_graphs, graphs_batch)
            num_batches += 1


    validity = valid_cf / total if total > 0 else 0.0
    # 计算平均proximity
    avg_proximity = proximity / num_batches if num_batches > 0 else 0.0

    return {
        "validity": validity,
        "proximity": avg_proximity,  # 返回平均值
        "successful": valid_cf,
        "total": total,
    }


def count_valid(target_lables, cf_graphs, gnn):
    gnn.eval()

    pred_logits_cf = gnn.get_pred(cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch)[0]
    pred_labels_cf = pred_logits_cf.argmax(dim=1)

    flipped_lables = (pred_labels_cf == target_lables).sum().item()

    return flipped_lables

def compute_proximity(args, cf_graphs, ori_graphs):
    rho, beta, gamma = (1.0, 1.0, 0.0)

    ori_graphs = ori_graphs.to_data_list()
    cf_graphs = cf_graphs.to_data_list()
    batch_size = len(ori_graphs)
    distances = torch.zeros(batch_size, device=args.device)

    for i in range(batch_size):
        orig_data = ori_graphs[i]
        cf_data = cf_graphs[i]

        orig_adj = to_dense_adj(orig_data.edge_index, max_num_nodes=args.max_num_nodes).squeeze(0)
        cf_adj = to_dense_adj(cf_data.edge_index, max_num_nodes=args.max_num_nodes).squeeze(0)

        orig_node_feat = orig_data.x
        cf_node_feat = cf_data.x
        orig_edge_feat = orig_data.edge_attr if hasattr(orig_data, 'edge_attr') else None
        cf_edge_feat = cf_data.edge_attr if hasattr(cf_data, 'edge_attr') else None

        # 计算原始距离（用 sum 匹配总范数）
        # d_adj = F.frobenius_norm(orig_adj - cf_adj)
        d_adj = torch.norm(orig_adj - cf_adj, p='fro')
        d_node = F.mse_loss(orig_node_feat, cf_node_feat, reduction='sum')
        d_edge = 0.0
        if orig_edge_feat is not None and cf_edge_feat is not None:
            # 假设 edge_attr 形状匹配（需 pad 如果 num_edges 不同）
            min_edges = min(len(orig_edge_feat), len(cf_edge_feat))
            d_edge = F.mse_loss(orig_edge_feat[:min_edges], cf_edge_feat[:min_edges], reduction='sum')

        # 获取节点/边数
        n_orig, n_cf = orig_node_feat.size(0), cf_node_feat.size(0)
        m_orig = orig_data.num_edges // 2 if orig_data.is_undirected() else orig_data.num_edges  # 假设无向图
        m_cf = cf_data.num_edges // 2 if cf_data.is_undirected() else cf_data.num_edges

        max_n = max(n_orig, n_cf)
        max_m = max(m_orig, m_cf)

        # 归一化
        norm_d_adj = d_adj / max_m if max_m > 0 else 0.0
        norm_d_node = d_node / max_n if max_n > 0 else 0.0
        norm_d_edge = d_edge / max_m if max_m > 0 else 0.0

        # 加权求和
        distances[i] = rho * norm_d_adj + beta * norm_d_node + gamma * norm_d_edge

    return distances.mean().item()


# def compute_validity(args, model, gnn, data_loader) -> Dict[str, float]:
#     """Compute counterfactual validity on a preprocessed evaluation loader.
#
#     The ``data_loader`` must yield batches identical to those consumed during
#     training, i.e. dictionaries containing ``graphs`` (original graphs) and
#     ``subgraphs`` (the corresponding frequent-subgraph reconstructions produced
#     from the SMILES vocabulary). Each batch is processed by:
#
#     1. Running the frozen GNN to obtain the original predictions.
#     2. Flipping the predictions to create desired counterfactual labels.
#     3. Passing the frequent subgraphs through MyExplainer to reconstruct a
#        counterfactual subgraph.
#     4. Stitching the reconstructed subgraph back into the original graph via
#        :func:`utils.concat_graphs`.
#     5. Re-evaluating the stitched graphs with the GNN to determine how many
#        achieve the desired label.
#
#     Args:
#         args: Runtime configuration. Must expose ``device``, ``x_dim`` and
#             ``max_subgraph_nodes`` that match the data loader's preprocessing.
#         model: Trained MyExplainer instance.
#         gnn: Frozen task GNN used for validation.
#         data_loader: ``DataLoader`` created from ``GraphTrainData`` with
#             ``train_collate_fn`` so that each batch mirrors the training
#             pipeline's preprocessing.
#
#     Returns:
#         Dict[str, float]: ``validity`` (success rate), ``successful`` counter-
#         factuals, and ``total`` evaluated samples.
#     """
#
#     model.eval()
#     gnn.eval()
#
#     device = args.device
#     max_sub_nodes = args.max_subgraph_nodes
#
#     ged_weights = (1.0, 1.0, 0.0)
#
#     proximity = 0.0
#     successful_cf = 0
#     total_cf = 0
#
#     with torch.no_grad():
#         for batch in tqdm(data_loader, desc="Evaluating validity"):
#             graphs_batch = batch["graphs"].to(device)
#             subgraphs_batch = batch["subgraphs"].to(device)
#             batch["graphs"] = graphs_batch
#             batch["subgraphs"] = subgraphs_batch
#
#             batch_size = graphs_batch.num_graphs
#             if batch_size == 0:
#                 continue
#
#             ori_pred_logits = gnn.get_pred(
#                 graphs_batch.x, graphs_batch.edge_index, graphs_batch.batch
#             )[0]
#             ori_pred = ori_pred_logits.argmax(dim=1)
#
#             cf_pred = 1 - ori_pred
#             y_cf = cf_pred.float().unsqueeze(1)
#
#             zero_template = torch.zeros(max_sub_nodes, args.x_dim, device=device)
#             subgraph_x_list = []
#
#             for b in range(batch_size):
#                 mask = subgraphs_batch.batch == b
#                 num_nodes = int(mask.sum().item())
#
#                 padded = zero_template.clone()
#                 if num_nodes > 0:
#                     padded[:num_nodes] = subgraphs_batch.x[mask]
#
#                 subgraph_x_list.append(padded)
#
#             subgraph_x = torch.stack(subgraph_x_list, dim=0)
#             subgraph_adj = to_dense_adj(
#                 subgraphs_batch.edge_index,
#                 batch=subgraphs_batch.batch,
#                 max_num_nodes=max_sub_nodes,
#             )
#
#             outputs = model(features=subgraph_x, adj=subgraph_adj, y_cf=y_cf)
#
#             concated_graphs = concat_graphs(args, outputs, batch)
#
#             pred_logits_cf = gnn.get_pred(
#                 concated_graphs.x,
#                 concated_graphs.edge_index,
#                 concated_graphs.batch,
#             )[0]
#             pred_labels_cf = pred_logits_cf.argmax(dim=1)
#
#             successful_cf += (pred_labels_cf == cf_pred).sum().item()
#             total_cf += batch_size
#
#             proximity += model.compute_proximity(
#                 outputs,
#                 ged_weights,
#                 batch,
#                 subgraphs_batch,
#                 graphs_batch,
#             )
#
#
#     validity = successful_cf / total_cf if total_cf > 0 else 0.0
#
#     return {
#         "validity": validity,
#         "successful": successful_cf,
#         "total": total_cf,
#     }


def compute_ged(original_data, cf_data, weights=(1.0, 1.0, 0.0)):
    rho, beta, gamma = weights

    # 提取矩阵
    orig_adj = original_data.adj  # 假设稀疏或稠密邻接矩阵
    cf_adj = cf_data.adj
    orig_node_feat = original_data.x
    cf_node_feat = cf_data.x
    orig_edge_feat = original_data.edge_attr if hasattr(original_data, 'edge_attr') else None
    cf_edge_feat = cf_data.edge_attr if hasattr(cf_data, 'edge_attr') else None

    # 计算原始距离
    d_adj = F.frobenius_norm(orig_adj - cf_adj)
    d_node = F.mse_loss(orig_node_feat, cf_node_feat, reduction='sum')  # 用 sum 以匹配范数
    d_edge = 0.0
    if orig_edge_feat is not None and cf_edge_feat is not None:
        d_edge = F.mse_loss(orig_edge_feat, cf_edge_feat, reduction='sum')

    # 获取节点/边数（PyG Data 对象直接可用）
    n_orig, n_cf = orig_node_feat.size(0), cf_node_feat.size(0)
    m_orig, m_cf = orig_adj.sum().item() / 2 if orig_adj.is_sparse else orig_adj.sum().item()  # 假设无向图，边数 = 非零/2
    m_orig_cf = cf_adj.sum().item() / 2 if cf_adj.is_sparse else cf_adj.sum().item()

    max_n = max(n_orig, n_cf)
    max_m = max(m_orig, m_cf)

    # 归一化
    norm_d_adj = d_adj / max_m if max_m > 0 else 0.0
    norm_d_node = d_node / max_n if max_n > 0 else 0.0
    norm_d_edge = d_edge / max_m if max_m > 0 else 0.0

    # 加权求和
    return rho * norm_d_adj + beta * norm_d_node + gamma * norm_d_edge