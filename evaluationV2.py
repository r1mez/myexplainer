"""Evaluation utilities for MyExplainer.

Currently this module provides a validity metric implementation which is
responsible for measuring the proportion of generated counterfactual graphs
that obtain the desired labels when evaluated by a pre-trained GNN model.
"""

from typing import Dict, Tuple

import time
import numpy as np
import torch
from networkx.classes import subgraph
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from utils.batch_utils import core_data_from_batch, output_to_batch
from utils.graph_utils import extract_explanatory_subgraph, exclude_explanatory_subgraph
import torch.nn.functional as F

from utils.vis_utils import visualize_explainer_graph


class _OracleWrappedGNN:
    """
    简单的 GNN 包装器，用于在评估时统计 oracle 调用次数。
    所有对 gnn.get_pred 的调用都会累加到 oracle_calls 计数器中。
    """

    def __init__(self, gnn: torch.nn.Module):
        self.gnn = gnn
        self.oracle_calls = 0

    def get_pred(self, *args, **kwargs):
        self.oracle_calls += 1
        return self.gnn.get_pred(*args, **kwargs)

    def eval(self):
        self.gnn.eval()


def evaluate(args, model, gnn, data_loader):
    model.eval()

    wrapped_gnn = _OracleWrappedGNN(gnn)
    wrapped_gnn.eval()
    args.train_mode = False

    # 1. 预计算原始预测（不计入 runtime 和 oracle_calls）
    y_desired_all = []
    ori_prob_all = []
    ori_pred_all = []
    with torch.no_grad():
        for batch in data_loader:
            origraphs = batch['graphs'].to(args.device)
            _, ori_pred_logits = gnn.get_pred(origraphs.x, origraphs.edge_index, origraphs.batch)
            ori_prob = F.softmax(ori_pred_logits, dim=1)
            ori_pred = ori_pred_logits.argmax(dim=1)
            y_desired = (1 - ori_pred).float().unsqueeze(1)
            y_desired_all.append(y_desired.cpu())
            ori_prob_all.append(ori_prob.cpu())
            ori_pred_all.append(ori_pred.cpu())
    device = args.device



    proximity = 0.0
    valid_cf = 0
    fidel_sum = 0.00
    sparsity_sum = 0.00
    class_total = {0: 0, 1: 0}
    class_success = {0: 0, 1: 0}


    total = data_loader.dataset.__len__()
    num_batches = 0
    total_cf_time = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating:")):
            origraphs = batch['graphs'].to(args.device)
            subgraphs = batch['subgraphs']

            x = origraphs.x
            edge_index = origraphs.edge_index
            batch_vec = origraphs.batch

            # ✅ 使用预计算的y_desired，确保每个epoch一致
            y_desired = y_desired_all[batch_idx].to(args.device)
            ori_pred = ori_pred_all[batch_idx].to(args.device).view(-1).long()
            y_hat = (1 - y_desired).float()

            # 2. CF 生成 —— 仅此阶段计入 runtime 和 oracle_calls
            calls_before = wrapped_gnn.oracle_calls
            t0 = time.time()
            outputs = model(
                graphs=origraphs,
                subgraphs=subgraphs
            )
            total_cf_time += time.time() - t0

            visualize_explainer_graph(origraphs, y_desired, outputs)

            cf_graphs = output_to_batch(origraphs, outputs)

            # 🔍 调试：在第一个batch打印统计信息
            if batch_idx == 0:
                ori_graphs_list = origraphs.to_data_list()
                cf_graphs_list = cf_graphs.to_data_list()
                exp_graphs_list = [extract_explanatory_subgraph(o, c) for o, c in zip(ori_graphs_list, cf_graphs_list)]

                print(f"\n[DEBUG] Batch {batch_idx} - First 3 graphs:")
                for i in range(min(3, len(ori_graphs_list))):
                    ori_edges = ori_graphs_list[i].num_edges
                    cf_edges = cf_graphs_list[i].num_edges
                    exp_edges = exp_graphs_list[i].num_edges
                    print(f"  Graph {i}: ori_edges={ori_edges}, cf_edges={cf_edges}, exp_edges={exp_edges}")
                    print(f"            sparsity = 1 - ({exp_edges}/{ori_edges}) = {1 - exp_edges/ori_edges:.4f}")

            # 3. 验证与指标计算（不计入 oracle_calls，使用原始 gnn）
            success_mask = get_flip_success_mask(y_desired, cf_graphs, gnn)
            valid_cf += int(success_mask.sum().item())
            for class_idx in class_total:
                class_mask = (ori_pred == class_idx)
                class_total[class_idx] += int(class_mask.sum().item())
                if class_mask.any():
                    class_success[class_idx] += int(success_mask[class_mask].sum().item())
            proximity += compute_proximity(args, cf_graphs, origraphs)
            fidel_sum += compute_fidelity_prob(args, origraphs, cf_graphs, ori_prob_all[batch_idx], gnn)
            sparsity_sum += compute_sparsity(args, origraphs, cf_graphs)


            num_batches += 1

    validity = valid_cf / total if total > 0 else 0.0
    sparsity = sparsity_sum / total if total > 0 else 0.0
    avg_proximity = proximity / total if total > 0 else 0.0
    fidelity = fidel_sum / total if total > 0 else 0.0




    avg_runtime_per_graph = total_cf_time / total if total > 0 else 0.0
    avg_oracle_calls_per_graph = wrapped_gnn.oracle_calls / total if total > 0 else 0.0
    per_class_flip = {}
    for class_idx in class_total:
        total_class = class_total[class_idx]
        success_class = class_success[class_idx]
        per_class_flip[class_idx] = {
            "successful": success_class,
            "total": total_class,
            "ratio": success_class / total_class if total_class > 0 else 0.0,
        }

    args.train_mode = True

    return {
        "validity": validity,
        "proximity": avg_proximity,  # 返回平均值
        "fidelity": fidelity,
        "sparsity": sparsity,
        "successful": valid_cf,
        "total": total,
        # 下面两个为“平均每张图”的耗时与调用次数
        "runtime": avg_runtime_per_graph,
        "oracle_calls": avg_oracle_calls_per_graph,
        "per_class_flip": per_class_flip,
    }


def get_flip_success_mask(target_lables, cf_graphs, gnn):
    gnn.eval()

    pred_logits_cf = gnn.get_pred(cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch)[0]
    pred_labels_cf = pred_logits_cf.argmax(dim=1).view(-1, 1)
    return (pred_labels_cf == target_lables).view(-1)


def count_valid(target_lables, cf_graphs, gnn):
    flipped_lables = get_flip_success_mask(target_lables, cf_graphs, gnn).sum().item()

    return flipped_lables


def compute_proximity(args, cf_graphs, ori_graphs):
    """
    计算原始图与反事实图的邻接矩阵距离 (L1 Norm / Graph Edit Distance Approximation)
    修复了维度对齐问题，并解决了 Frobenius 范数导致的量纲不匹配问题。
    """
    rho = 1.0

    ori_graphs = ori_graphs.to_data_list()
    cf_graphs = cf_graphs.to_data_list()
    batch_size = len(ori_graphs)
    distances = torch.zeros(batch_size, device=args.device)

    for i in range(batch_size):
        orig_data = ori_graphs[i]
        cf_data = cf_graphs[i]

        # ---------------------------------------------------------
        # 步骤 1: 确定统一的节点数 N
        # 即使 cf_data 删除了边导致孤立点，矩阵维度仍需保持与原图一致
        # ---------------------------------------------------------
        if getattr(orig_data, 'num_nodes', None) is not None:
            N = orig_data.num_nodes
        elif getattr(orig_data, 'x', None) is not None:
            N = orig_data.x.size(0)
        else:
            # 兜底逻辑：取最大的索引值
            max_idx = 0
            if orig_data.edge_index.numel() > 0:
                max_idx = int(orig_data.edge_index.max())
            if cf_data.edge_index.numel() > 0:
                max_idx = max(max_idx, int(cf_data.edge_index.max()))
            N = max_idx + 1

        # ---------------------------------------------------------
        # 步骤 2: 转换为稠密矩阵 (强制指定 max_num_nodes=N)
        # 这确保了 orig_adj 和 cf_adj 形状严格一致 [N, N]
        # ---------------------------------------------------------
        orig_adj = to_dense_adj(orig_data.edge_index, max_num_nodes=N).squeeze(0) \
            if orig_data.edge_index.numel() > 0 else torch.zeros(N, N, device=args.device)
        cf_adj = to_dense_adj(cf_data.edge_index, max_num_nodes=N).squeeze(0) \
            if cf_data.edge_index.numel() > 0 else torch.zeros(N, N, device=args.device)

        # ---------------------------------------------------------
        # 步骤 3: 计算差异 (使用 L1 范数)
        # p=1 代表绝对值之和。对于无向图，删 1 条边，这里的值是 2。
        # ---------------------------------------------------------
        d_adj_entries = torch.norm(orig_adj - cf_adj, p=1)

        # ---------------------------------------------------------
        # 步骤 4: 归一化
        # 分子是矩阵条目的变化量，分母也应是矩阵条目的最大容量 (2 * max_edges)
        # ---------------------------------------------------------
        m_orig = orig_data.num_edges // 2 if orig_data.is_undirected() else orig_data.num_edges
        m_cf = cf_data.num_edges // 2 if cf_data.is_undirected() else cf_data.num_edges
        max_m = max(m_orig, m_cf)

        # 乘以 2.0 是为了匹配无向图邻接矩阵的对称性 (每条边占 2 个坑位)
        normalization = 2.0 * max_m if max_m > 0 else 1.0

        distances[i] = rho * (d_adj_entries / normalization)

    return distances.sum().item()

def compute_fidelity_prob(args, ori_graphs, cf_graphs, ori_prob, gnn):
    """
    计算将原始图替换为反事实图后，原始预测类别的概率下降值（保真度）。

    Args:
        args: 包含 device 等配置的参数对象
        ori_graphs: 原始图 Batch 对象
        cf_graphs: 反事实图 Batch 对象（已修改的图）
        ori_prob: 原始图的预测概率 [N, num_classes]
        gnn: 图神经网络模型，需支持 get_pred(x, edge_index, batch)

    Returns:
        fidelity_sum: 所有样本上原始类别概率的下降总和
    """
    # 获取原始预测类别（每个图最可能的类别）
    ori_pred = ori_prob.argmax(dim=1)  # shape: [N]

    # 在反事实图上进行预测
    cf_pred_logits = gnn.get_pred(
        cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch
    )[0]  # 假设返回 (logits, ...)
    cf_prob = F.softmax(cf_pred_logits, dim=1)  # shape: [N, num_classes]

    fidelity_sum = 0.0
    for i in range(len(ori_pred)):
        ori_prob_single = ori_prob[i, ori_pred[i]].item()  # 原图对原始预测类的概率
        cf_prob_single = cf_prob[i, ori_pred[i]].item()  # 反事实图对同一类的概率
        fidelity_sum += (ori_prob_single - cf_prob_single)  # 下降量（越大说明解释越有效）

    return fidelity_sum

def compute_sparsity(args, ori_graphs, cf_graphs):
    ori_graphs, cf_graphs = ori_graphs.to_data_list(), cf_graphs.to_data_list()
    exp_graphs = [extract_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_graphs, cf_graphs)]

    exp_num_edges = [exp.num_edges for exp in exp_graphs]
    ori_num_edges = [ori.num_edges for ori in ori_graphs]

    sparsity = 0.0
    for ori_e, exp_e in zip(ori_num_edges, exp_num_edges):
        sparsity += 1 - (exp_e / ori_e)

    return sparsity

