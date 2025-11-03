"""Evaluation utilities for MyExplainer.

Currently this module provides a validity metric implementation which is
responsible for measuring the proportion of generated counterfactual graphs
that obtain the desired labels when evaluated by a pre-trained GNN model.
"""

from typing import Dict

import torch
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Batch
from tqdm import tqdm

from utils import concat_graphs
from utils.graph_utils import extract_explanatory_subgraph, exclude_explanatory_subgraph
import torch.nn.functional as F


def evaluate(args, model, gnn, data_loader):
    model.eval()
    gnn.eval()
    args.train_mode = False

    device = args.device
    max_sub_nodes = args.max_subgraph_nodes

    proximity = 0.0
    valid_cf = 0
    fidel_plus_count = 0
    fidel_minus_count = 0
    fidel_sum = 0.00
    sparsity_sum = 0.00


    total = data_loader.dataset.__len__()
    total_0 = 0
    total_1 = 0
    num_batches = 0  # 添加batch计数

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating:"):
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
            ori_prob = F.softmax(ori_pred_logits, dim=1)
            ori_pred = ori_pred_logits.argmax(dim=1)
            # 计算有多少个0类样本
            total_0 += (ori_pred == 0).sum().item()
            total_1 += (ori_pred == 1).sum().item()

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
            fidel_plus, fidel_minus = compute_fidelity(args, graphs_batch, concated_graphs, ori_pred, gnn)
            fidel_sum += compute_fidelity_prob(args, graphs_batch, concated_graphs, ori_prob, gnn)
            sparsity_sum += compute_sparsity(args, graphs_batch, concated_graphs)
            fidel_plus_count += fidel_plus
            fidel_minus_count += fidel_minus

            num_batches += 1

    validity = valid_cf / total if total > 0 else 0.0
    sparsity = sparsity_sum / total if total > 0 else 0.0
    avg_proximity = proximity / num_batches if num_batches > 0 else 0.0
    fidelity_plus = 1 - fidel_plus_count / total if total > 0 else 0.0
    fidelity_minus = 1 - fidel_minus_count / total if total > 0 else 0.0
    fidelity = fidel_sum / total if total > 0 else 0.0


    args.train_mode = True

    return {
        "validity": validity,
        "proximity": avg_proximity,  # 返回平均值
        "fidelity+": fidelity_plus,
        "fidelity-": fidelity_minus,
        "fidelity": fidelity,
        "sparsity": sparsity,

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


def compute_fidelity(args, ori_graphs, cf_graphs, ori_pred, gnn):
    ori_graphs, cf_graphs = ori_graphs.to_data_list(), cf_graphs.to_data_list()

    # extract_explanatory_subgraph now returns both explain and non-explain graphs
    results = [extract_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_graphs, cf_graphs)]
    exp_graphs = [r[0] for r in results]  # Explanation subgraphs
    # Use exclude_explanatory_subgraph to get non-explanation subgraphs
    exp_excluded_graphs = [exclude_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_graphs, cf_graphs)]

    fidel_plus_count = 0
    fidel_minus_count = 0

    # Process each graph individually to handle empty graphs properly
    for i in range(len(ori_graphs)):
        exp_graph = exp_graphs[i]
        exp_excluded_graph = exp_excluded_graphs[i]

        # Fidelity+: explanation subgraph should preserve original prediction
        if exp_graph.num_nodes > 0:
            exp_batch = Batch.from_data_list([exp_graph]).to(args.device)
            exp_pred_logits = gnn.get_pred(
                exp_batch.x, exp_batch.edge_index, exp_batch.batch
            )[0]
            exp_pred = exp_pred_logits.argmax(dim=1).item()

            if exp_pred == ori_pred[i]:
                fidel_plus_count += 1
        # else: empty explanation graph, skip (cannot compute fidelity)

        # Fidelity-: non-explanation subgraph should flip to counterfactual prediction
        if exp_excluded_graph.num_nodes > 0:
            exp_excluded_batch = Batch.from_data_list([exp_excluded_graph]).to(args.device)
            exp_excluded_logits = gnn.get_pred(
                exp_excluded_batch.x, exp_excluded_batch.edge_index, exp_excluded_batch.batch
            )[0]
            exp_excluded_pred = exp_excluded_logits.argmax(dim=1).item()

            if exp_excluded_pred == (1 - ori_pred[i]):
                fidel_minus_count += 1
        # else: empty non-explanation graph, skip (cannot compute fidelity)

    return fidel_plus_count, fidel_minus_count


def compute_fidelity_prob(args, ori_graphs, cf_graphs, ori_prob, gnn):
    ori_graphs, cf_graphs = ori_graphs.to_data_list(), cf_graphs.to_data_list()
    # exp_graphs = [extract_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_graphs, cf_graphs)]
    exp_excluded_graphs = [exclude_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_graphs, cf_graphs)]

    excluded_graphs_batch = Batch.from_data_list(exp_excluded_graphs).to(args.device)


    fidelity_sum = 0.00
    ori_pred = ori_prob.argmax(dim=1)

    excluded_pred_logits = gnn.get_pred(
        excluded_graphs_batch.x, excluded_graphs_batch.edge_index, excluded_graphs_batch.batch
    )[0]
    excluded_prob = F.softmax(excluded_pred_logits, dim=1)

    for i in range(len(ori_pred)):
        excluded_prob_single = excluded_prob[i, ori_pred[i]].item()
        ori_prob_single = ori_prob[i, ori_pred[i]].item()
        fidelity_sum += (ori_prob_single - excluded_prob_single)

    return fidelity_sum

def compute_sparsity(args, ori_graphs, cf_graphs):
    ori_graphs, cf_graphs = ori_graphs.to_data_list(), cf_graphs.to_data_list()
    # extract_explanatory_subgraph now returns both explain and non-explain graphs
    results = [extract_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_graphs, cf_graphs)]
    exp_graphs = [r[0] for r in results]  # Explanation subgraphs

    exp_num_edges = [exp.num_edges for exp in exp_graphs]
    ori_num_edges = [ori.num_edges for ori in ori_graphs]

    sparsity = 0.0
    for ori_e, exp_e in zip(ori_num_edges, exp_num_edges):
        if ori_e > 0:
            sparsity += 1 - (exp_e / ori_e)
        # else: original graph has no edges, skip

    return sparsity
