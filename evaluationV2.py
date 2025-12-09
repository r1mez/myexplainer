"""Evaluation utilities for MyExplainer.

Currently this module provides a validity metric implementation which is
responsible for measuring the proportion of generated counterfactual graphs
that obtain the desired labels when evaluated by a pre-trained GNN model.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from networkx.classes import subgraph
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from utils import concat_graphs
from utils.batch_utils import core_data_from_batch, output_to_batch
from utils.graph_utils import extract_explanatory_subgraph, exclude_explanatory_subgraph
import torch.nn.functional as F

from utils.vis_utils import visualize_explainer_graph


def evaluate(args, model, gnn, data_loader):
    model.eval()
    gnn.eval()
    args.train_mode = False


    y_desired_all = []
    ori_prob_all = []
    with torch.no_grad():
        for batch in data_loader:
            origraphs = batch['graphs'].to(args.device)
            _, ori_pred_logits = gnn.get_pred(origraphs.x, origraphs.edge_index, origraphs.batch)
            ori_prob = F.softmax(ori_pred_logits, dim=1)
            ori_pred = ori_pred_logits.argmax(dim=1)
            y_desired = (1 - ori_pred).float().unsqueeze(1)
            y_desired_all.append(y_desired.cpu())
            ori_prob_all.append(ori_prob.cpu())
    device = args.device



    proximity = 0.0
    valid_cf = 0
    fidel_sum = 0.00
    sparsity_sum = 0.00


    total = data_loader.dataset.__len__()
    num_batches = 0  # 添加batch计数

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating:")):
            origraphs = batch['graphs'].to(args.device)
            subgraphs = batch['subgraphs']

            x = origraphs.x
            edge_index = origraphs.edge_index
            batch_vec = origraphs.batch

            # ✅ 使用预计算的y_desired，确保每个epoch一致
            y_desired = y_desired_all[batch_idx].to(args.device)
            y_hat = (1 - y_desired).float()  # 原始预测 = 1 - 反事实标签


            outputs = model(
                graphs=origraphs,
                subgraphs=subgraphs
            )

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

            valid_cf += count_valid(y_desired, cf_graphs, gnn)
            proximity += compute_proximity(args, cf_graphs, origraphs)
            fidel_sum += compute_fidelity_prob(args, origraphs, cf_graphs, ori_prob_all[batch_idx], gnn)
            sparsity_sum += compute_sparsity(args, origraphs, cf_graphs)


            num_batches += 1

    validity = valid_cf / total if total > 0 else 0.0
    sparsity = sparsity_sum / total if total > 0 else 0.0
    avg_proximity = proximity / total if total > 0 else 0.0
    fidelity = fidel_sum / total if total > 0 else 0.0




    args.train_mode = True

    return {
        "validity": validity,
        "proximity": avg_proximity,  # 返回平均值
        "fidelity": fidelity,
        "sparsity": sparsity,
        "successful": valid_cf,
        "total": total,
    }


def count_valid(target_lables, cf_graphs, gnn):
    gnn.eval()

    pred_logits_cf = gnn.get_pred(cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch)[0]
    print(pred_logits_cf)
    pred_labels_cf = pred_logits_cf.argmax(dim=1).view(-1,1)

    flipped_lables = (pred_labels_cf == target_lables).sum().item()

    return flipped_lables

def compute_proximity(args, cf_graphs, ori_graphs):
    # 现在只用邻接矩阵距离，rho=1即可
    rho = 1.0

    ori_graphs = ori_graphs.to_data_list()
    cf_graphs = cf_graphs.to_data_list()
    batch_size = len(ori_graphs)
    distances = torch.zeros(batch_size, device=args.device)

    # 安全版本的 to_dense_adj，支持空 edge_index
    def safe_to_dense_adj(data):
        edge_index = data.edge_index

        # 确定节点数：
        # 1) 优先用 num_nodes
        # 2) 再用 x.size(0)
        # 3) 最后从 edge_index 推
        if getattr(data, 'num_nodes', None) is not None and data.num_nodes is not None:
            num_nodes = data.num_nodes
        elif getattr(data, 'x', None) is not None and data.x is not None:
            num_nodes = data.x.size(0)
        else:
            if edge_index.numel() == 0:
                # 没任何信息，只能返回 0x0
                return torch.zeros(0, 0, device=args.device)
            num_nodes = int(edge_index.max().item()) + 1

        # 确定 device
        if edge_index.numel() > 0:
            device = edge_index.device
        elif getattr(data, 'x', None) is not None and data.x is not None:
            device = data.x.device
        else:
            device = args.device

        # 如果没有边：返回全 0 邻接矩阵 [num_nodes, num_nodes]
        if edge_index.numel() == 0:
            return torch.zeros(num_nodes, num_nodes, device=device)

        # 正常情况：确保形状是 [num_nodes, num_nodes]
        dense = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
        return dense

    for i in range(batch_size):
        orig_data = ori_graphs[i]
        cf_data = cf_graphs[i]

        # 邻接矩阵（已经处理空图）
        orig_adj = safe_to_dense_adj(orig_data)  # [n1, n1]
        cf_adj = safe_to_dense_adj(cf_data)      # [n2, n2]

        # 若节点数不同，做零填充对齐
        n_orig, n_cf = orig_adj.size(0), cf_adj.size(0)

        # 邻接矩阵差异（Frobenius 范数）
        d_adj = torch.norm(orig_adj - cf_adj, p='fro')

        # 用边数归一化
        m_orig = orig_data.num_edges // 2 if orig_data.is_undirected() else orig_data.num_edges
        m_cf = cf_data.num_edges // 2 if cf_data.is_undirected() else cf_data.num_edges
        max_m = max(m_orig, m_cf)

        norm_d_adj = d_adj / max_m if max_m > 0 else 0.0

        # 现在距离只由邻接矩阵决定
        distances[i] = rho * norm_d_adj

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




def compute_robust_fidelity(
        args,
        ori_graphs: Batch,
        cf_graphs: Batch,
        ori_pred: torch.Tensor,
        gnn,
        alpha1=0.1,
        alpha2=0.9,
        sample_num=50,
        undirect=True,
        use_gt_label=False
) -> Tuple[float, float, float, float, float, float]:
    r"""
    计算 Robust Fidelity 指标，适配当前框架。

    该指标来自论文 "Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks"
    (https://arxiv.org/abs/2310.01820)

    核心思想：通过随机采样来缓解 fidelity 指标的 Out-of-Distribution 问题

    概率版本的计算公式：
        Fid_{alpha_1,+} = f(G)_y - E[f(G - E_{alpha_1}(G^{exp}))_y]
        Fid_{alpha_2,-} = f(G)_y - E[f(G^{exp} + E_{alpha_2}(G - G^{exp}))_y]
        Fid_{alpha_1,alpha_2,Δ} = Fid_{alpha_1,+} - Fid_{alpha_2,-}

    准确率版本的计算公式：
        Fid_{alpha_1,+} = 1(y_hat == y) - E[1(y_hat_masked == y)]
        Fid_{alpha_2,-} = 1(y_hat == y) - E[1(y_hat_masked == y)]
        Fid_{alpha_1,alpha_2,Δ} = Fid_{alpha_1,+} - Fid_{alpha_2,-}

    Args:
        args: 参数对象（包含device等配置）
        ori_graphs: 原始图的批次数据
        cf_graphs: 反事实图的批次数据
        ori_pred: 原始图的预测标签 (batch_size,)
        gnn: 预训练的GNN模型
        alpha1: Fidelity+ 中每次移除解释子图边的比例（默认0.1）
        alpha2: Fidelity- 中每次保持非解释子图边的比例（默认0.9）
        sample_num: 采样次数（默认50）
        undirect: 是否为无向图（默认True）
        use_gt_label: 是否使用ground truth标签（默认False）

    Returns:
        Tuple: (fid_plus_mean, fid_minus_mean, fid_delta,
                fid_plus_label_mean, fid_minus_label_mean, fid_delta_label)
    """
    max_length = sample_num
    alpha2 = 1 - alpha2  # 转换为移除比例

    gnn.eval()

    # 转换为数据列表进行逐个处理
    ori_graphs_list = ori_graphs.to_data_list()
    cf_graphs_list = cf_graphs.to_data_list()

    # 汇总所有图的fidelity结果
    all_fid_plus_prob = []
    all_fid_minus_prob = []
    all_fid_plus_acc = []
    all_fid_minus_acc = []

    # 对每个图单独计算robust fidelity
    for idx, (ori_graph, cf_graph) in enumerate(zip(ori_graphs_list, cf_graphs_list)):
        ori_pred_i = ori_pred[idx].item() if torch.is_tensor(ori_pred[idx]) else ori_pred[idx]

        # 1. 提取解释子图（对应edge_mask中的explanation edges）
        exp_graph = extract_explanatory_subgraph(ori_graph, cf_graph)

        # 2. 获取原始预测概率
        ori_graph_batch = Batch.from_data_list([ori_graph]).to(args.device)
        with torch.no_grad():
            y_hat = gnn.get_pred(ori_graph_batch.x, ori_graph_batch.edge_index, ori_graph_batch.batch)[0]
            y_hat_prob = F.softmax(y_hat, dim=1)[0]  # (num_classes,)
            y_label = y_hat.argmax(dim=1).item()

        # 使用ground truth或预测标签
        if use_gt_label and hasattr(ori_graph, 'y'):
            label = int(ori_graph.y)
        else:
            label = y_label

        # 3. 创建边掩码：哪些边属于解释子图
        # 构建原图和解释子图的边集合
        def get_canonical_edge_set(edge_index: torch.Tensor):
            edge_set = set()
            for i in range(edge_index.size(1)):
                u = edge_index[0, i].item()
                v = edge_index[1, i].item()
                edge_set.add((min(u, v), max(u, v)))
            return edge_set

        ori_edge_set = get_canonical_edge_set(ori_graph.edge_index)
        cf_edge_set = get_canonical_edge_set(cf_graph.edge_index)
        exp_edge_set = get_canonical_edge_set(exp_graph.edge_index)

        # 4. 分离explanation edges和non-explanation edges
        # explanation edges: 原图中存在但反事实图中不存在的边
        # non-explanation edges: 原图和反事实图中都存在的边

        # 创建edge_index到索引的映射（处理无向图）
        if undirect:
            maps = {}  # (min_node, max_node) -> [edge_indices]
            explain_list = []  # 解释边的canonical形式
            non_explain_list = []  # 非解释边的canonical形式

            for i in range(ori_graph.edge_index.size(1)):
                u = ori_graph.edge_index[0, i].item()
                v = ori_graph.edge_index[1, i].item()
                edge_key = (min(u, v), max(u, v))

                if edge_key not in maps:
                    maps[edge_key] = []
                maps[edge_key].append(i)

            # 分类边
            for edge_key in maps.keys():
                if edge_key not in cf_edge_set:
                    # 解释边：原图有但反事实图没有
                    if edge_key not in [e for e in explain_list]:
                        explain_list.append(edge_key)
                else:
                    # 非解释边：两图都有
                    if edge_key not in [e for e in non_explain_list]:
                        non_explain_list.append(edge_key)
        else:
            # 有向图情况（直接使用边索引）
            explain_list = []
            non_explain_list = []
            for i in range(ori_graph.edge_index.size(1)):
                u = ori_graph.edge_index[0, i].item()
                v = ori_graph.edge_index[1, i].item()
                edge_key = (min(u, v), max(u, v))
                if edge_key not in cf_edge_set:
                    explain_list.append(i)
                else:
                    non_explain_list.append(i)

        # 如果没有解释边或非解释边，跳过该图
        if len(explain_list) == 0 or len(non_explain_list) == 0:
            continue

        # 5. 生成随机采样的边移除方案（保留原始逻辑）
        # 对于Fidelity+：随机移除alpha1比例的explanation edges
        explaine_ratio = np.ones(len(explain_list))
        explaine_ratio = alpha1 * explaine_ratio.sum() * (explaine_ratio / explaine_ratio.sum())
        explaine_ratio_remove = np.random.binomial(1, explaine_ratio,
                                                     size=(max_length, explaine_ratio.shape[0]))

        # 对于Fidelity-：随机移除alpha2比例的non-explanation edges
        non_explaine_ratio = np.ones(len(non_explain_list))
        non_explaine_ratio = alpha2 * non_explaine_ratio.sum() * (non_explaine_ratio / non_explaine_ratio.sum())
        non_explaine_ratio_remove = np.random.binomial(1, non_explaine_ratio,
                                                        size=(max_length, non_explaine_ratio.shape[0]))

        # 6. 计算 Fidelity+ (移除部分解释边)
        fid_plus_prob_list = []
        fid_plus_acc_list = []

        for i in range(max_length):
            remove_edges_mask = explaine_ratio_remove[i]

            # 构建移除指定边后的图
            edges_to_remove_indices = set()
            for idx_edge, edge in enumerate(explain_list):
                if remove_edges_mask[idx_edge] == 1:
                    if undirect:
                        # 获取该边对应的所有边索引（双向）
                        edge_indices = maps[edge]
                        edges_to_remove_indices.update(edge_indices)
                    else:
                        edges_to_remove_indices.add(edge)

            # 保留不被移除的边
            keep_mask = torch.ones(ori_graph.edge_index.size(1), dtype=torch.bool)
            for edge_idx in edges_to_remove_indices:
                keep_mask[edge_idx] = False

            new_edge_index = ori_graph.edge_index[:, keep_mask]

            # 如果有边特征，也需要过滤
            if hasattr(ori_graph, 'edge_attr') and ori_graph.edge_attr is not None:
                new_edge_attr = ori_graph.edge_attr[keep_mask]
            else:
                new_edge_attr = None

            # 创建新的图数据
            masked_graph = Data(
                x=ori_graph.x,
                edge_index=new_edge_index,
                edge_attr=new_edge_attr
            )

            # 预测掩码后的图
            if masked_graph.num_nodes > 0:
                masked_batch = Batch.from_data_list([masked_graph]).to(args.device)
                with torch.no_grad():
                    mask_pred_plus = gnn.get_pred(masked_batch.x, masked_batch.edge_index, masked_batch.batch)[0]
                    mask_pred_plus_prob = F.softmax(mask_pred_plus, dim=1)[0]
                    mask_pred_plus_label = mask_pred_plus.argmax(dim=1).item()

                # 计算Fidelity+ (概率版本和准确率版本)
                fid_plus_prob = y_hat_prob[label].item() - mask_pred_plus_prob[label].item()
                fid_plus_acc = int(y_label == label) - int(mask_pred_plus_label == label)

                fid_plus_prob_list.append(fid_plus_prob)
                fid_plus_acc_list.append(fid_plus_acc)

        # 7. 计算 Fidelity- (只保留解释子图 + 部分非解释边)
        fid_minus_prob_list = []
        fid_minus_acc_list = []

        for i in range(max_length):
            keep_edges_mask = non_explaine_ratio_remove[i]

            # 构建只保留解释边和部分非解释边的图
            # 首先获取所有解释边的索引
            explain_edge_indices = set()
            for edge in explain_list:
                if undirect:
                    edge_indices = maps[edge]
                    explain_edge_indices.update(edge_indices)
                else:
                    explain_edge_indices.add(edge)

            # 然后添加要保留的非解释边
            non_explain_edge_indices = set()
            for idx_edge, edge in enumerate(non_explain_list):
                if keep_edges_mask[idx_edge] == 1:
                    if undirect:
                        edge_indices = maps[edge]
                        non_explain_edge_indices.update(edge_indices)
                    else:
                        non_explain_edge_indices.add(edge)

            # 合并要保留的边
            keep_indices = explain_edge_indices | non_explain_edge_indices
            keep_mask = torch.zeros(ori_graph.edge_index.size(1), dtype=torch.bool)
            for edge_idx in keep_indices:
                keep_mask[edge_idx] = True

            new_edge_index = ori_graph.edge_index[:, keep_mask]

            # 如果有边特征，也需要过滤
            if hasattr(ori_graph, 'edge_attr') and ori_graph.edge_attr is not None:
                new_edge_attr = ori_graph.edge_attr[keep_mask]
            else:
                new_edge_attr = None

            # 创建新的图数据
            masked_graph = Data(
                x=ori_graph.x,
                edge_index=new_edge_index,
                edge_attr=new_edge_attr
            )

            # 预测掩码后的图
            if masked_graph.num_nodes > 0:
                masked_batch = Batch.from_data_list([masked_graph]).to(args.device)
                with torch.no_grad():
                    mask_pred_minus = gnn.get_pred(masked_batch.x, masked_batch.edge_index, masked_batch.batch)[0]
                    mask_pred_minus_prob = F.softmax(mask_pred_minus, dim=1)[0]
                    mask_pred_minus_label = mask_pred_minus.argmax(dim=1).item()

                # 计算Fidelity- (概率版本和准确率版本)
                fid_minus_prob = y_hat_prob[label].item() - mask_pred_minus_prob[label].item()
                fid_minus_acc = int(y_label == label) - int(mask_pred_minus_label == label)

                fid_minus_prob_list.append(fid_minus_prob)
                fid_minus_acc_list.append(fid_minus_acc)

        # 8. 计算该图的平均Fidelity
        if len(fid_plus_prob_list) > 0:
            fid_plus_mean = np.mean(fid_plus_prob_list)
            fid_plus_label_mean = np.mean(fid_plus_acc_list)
        else:
            fid_plus_mean = 0.0
            fid_plus_label_mean = 0.0

        if len(fid_minus_prob_list) > 0:
            fid_minus_mean = np.mean(fid_minus_prob_list)
            fid_minus_label_mean = np.mean(fid_minus_acc_list)
        else:
            fid_minus_mean = 0.0
            fid_minus_label_mean = 0.0

        # 添加到总列表
        all_fid_plus_prob.append(fid_plus_mean)
        all_fid_minus_prob.append(fid_minus_mean)
        all_fid_plus_acc.append(fid_plus_label_mean)
        all_fid_minus_acc.append(fid_minus_label_mean)

    # 9. 计算所有图的平均Fidelity
    if len(all_fid_plus_prob) > 0:
        final_fid_plus_prob = np.mean(all_fid_plus_prob)
        final_fid_minus_prob = np.mean(all_fid_minus_prob)
        final_fid_delta_prob = final_fid_plus_prob - final_fid_minus_prob

        final_fid_plus_acc = np.mean(all_fid_plus_acc)
        final_fid_minus_acc = np.mean(all_fid_minus_acc)
        final_fid_delta_acc = final_fid_plus_acc - final_fid_minus_acc
    else:
        final_fid_plus_prob = 0.0
        final_fid_minus_prob = 0.0
        final_fid_delta_prob = 0.0
        final_fid_plus_acc = 0.0
        final_fid_minus_acc = 0.0
        final_fid_delta_acc = 0.0

    return (final_fid_plus_prob, final_fid_minus_prob, final_fid_delta_prob,
            final_fid_plus_acc, final_fid_minus_acc, final_fid_delta_acc)


