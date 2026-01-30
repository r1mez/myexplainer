"""Evaluation utilities for MyExplainer.

Currently this module provides a validity metric implementation which is
responsible for measuring the proportion of generated counterfactual graphs
that obtain the desired labels when evaluated by a pre-trained GNN model.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from scipy.sparse import coo_matrix
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.data import Batch, Data
from tqdm import tqdm

from utils import concat_graphs
from utils.batch_utils import core_data_from_batch
from utils.graph_utils import extract_explanatory_subgraph, exclude_explanatory_subgraph
import torch.nn.functional as F




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
    """
    计算fidelity+和fidelity-：
    - fidelity+：保留解释子图时，预测与原始一致的比例（越高越好）；
    - fidelity-：移除解释子图时，预测与原始一致的比例（越低越好）。

    使用批量处理+索引映射的方式，既避免空图问题又保持高速度。
    """
    ori_graphs_list = ori_graphs.to_data_list()
    cf_graphs_list = cf_graphs.to_data_list()

    fidel_plus_count = 0
    fidel_minus_count = 0

    # 收集所有非空的解释子图和非解释子图，并记录它们对应的原始图索引
    exp_graphs_list = []
    exp_excluded_graphs_list = []
    exp_graph_indices = []  # 记录解释子图对应的原始图索引
    exp_excluded_indices = []  # 记录非解释子图对应的原始图索引

    for i, (ori_graph, cf_graph) in enumerate(zip(ori_graphs_list, cf_graphs_list)):
        exp_graph = extract_explanatory_subgraph(ori_graph, cf_graph)
        exp_excluded_graph = exclude_explanatory_subgraph(ori_graph, cf_graph)

        # 只添加非空图
        if exp_graph.num_nodes > 0:
            exp_graphs_list.append(exp_graph)
            exp_graph_indices.append(i)

        if exp_excluded_graph.num_nodes > 0:
            exp_excluded_graphs_list.append(exp_excluded_graph)
            exp_excluded_indices.append(i)

    # 批量预测所有非空的解释子图
    if len(exp_graphs_list) > 0:
        exp_graphs_batch = Batch.from_data_list(exp_graphs_list).to(args.device)
        exp_pred_logits = gnn.get_pred(
            exp_graphs_batch.x, exp_graphs_batch.edge_index, exp_graphs_batch.batch
        )[0]
        exp_preds = exp_pred_logits.argmax(dim=1)

        # 根据索引映射计算fidelity+
        for batch_idx, orig_idx in enumerate(exp_graph_indices):
            ori_pred_i = ori_pred[orig_idx].item() if torch.is_tensor(ori_pred[orig_idx]) else ori_pred[orig_idx]
            exp_pred = exp_preds[batch_idx].item()

            if exp_pred == ori_pred_i:
                fidel_plus_count += 1

    # 批量预测所有非空的非解释子图
    if len(exp_excluded_graphs_list) > 0:
        exp_excluded_batch = Batch.from_data_list(exp_excluded_graphs_list).to(args.device)
        exp_excluded_logits = gnn.get_pred(
            exp_excluded_batch.x, exp_excluded_batch.edge_index, exp_excluded_batch.batch
        )[0]
        exp_excluded_preds = exp_excluded_logits.argmax(dim=1)

        # 根据索引映射计算fidelity-
        for batch_idx, orig_idx in enumerate(exp_excluded_indices):
            ori_pred_i = ori_pred[orig_idx].item() if torch.is_tensor(ori_pred[orig_idx]) else ori_pred[orig_idx]
            exp_excluded_pred = exp_excluded_preds[batch_idx].item()

            if exp_excluded_pred == ori_pred_i:
                fidel_minus_count += 1

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
    exp_graphs = [extract_explanatory_subgraph(ori, cf) for ori, cf in zip(ori_graphs, cf_graphs)]

    exp_num_edges = [exp.num_edges for exp in exp_graphs]
    ori_num_edges = [ori.num_edges for ori in ori_graphs]

    sparsity = 0.0
    for ori_e, exp_e in zip(ori_num_edges, exp_num_edges):
        sparsity += 1 - (exp_e / ori_e)

    return sparsity


# def robust_fidelity(
#         explainer: Explainer,
#         explanation: Explanation,
#         alpha1=0.1,
#         alpha2=0.9,
#         sample_num=50,
#         top_k=-1,
#         k_hop=3,
#         undirect=True,
#         use_gt_label=True
# ) -> Tuple[float, float, float, float, float, float]:
#     r"""Calculate the robust fidelity  metric, given an
#     :class:`~torch_geometric.explain.Explainer`  and
#     :class:`~torch_geometric.explain.Explanation`, as described in the
#     `"Towards Robust Fidelity for Evaluating Explainability of Graph
#     Neural Networks" <https://arxiv.org/abs/2310.01820>`_ paper.
#
#     Fidelity is a metric that evaluates the contribution of the given
#     explanation subgraph to the original prediction. However, due to
#     the prediction function may not be trained in the distribution
#     of explanation subgraphs, the fidelity might be not accurate.
#     Robust Fidelity allievate the Out-of-Distbution problem in the
#     fidelity metric. Similar to fidelity, this function return two
#     scores  by giving only the subgraph to the model (fidelity-) or
#     by removing it from the entire graph (fidelity+).
#
#     the probability-based robust fidelity scores are given by:
#
#     .. math::
#                 Fid_{\alpha_1,+} &= f(\overline{G})_y -
#                 \mathbb{E}f(\overline{G}-
#                 E_{\alpha_1}(\overline{G}^{(exp)}))_y
#
#                 Fid_{\alpha_2,-} &=  f(\overline{G})_y -
#                 \mathbb{E}f(\overline{G}^{(exp)}+
#                 E_{\alpha_2}(\overline{G}-\overline{G}^{(exp)}))_y
#
#                 Fid_{\alpha_1,\alpha_2,\Delta} &=
#                 Fid_{\alpha_1,+} - Fid_{\alpha_2,-}
#
#     the accuracy-based robust fidelity scores are given by:
#
#     .. math::
#                 Fid_{\alpha_1,+} &= \mathbb{1}(
#                 \widehat{y}_{\overline{G}} == y ) -
#                 \mathbb{E}( \mathbb{1}( \widehat{y}_
#                 {\overline{G}-E_{\alpha_1}
#                 (\overline{G}^{(exp)})} == y))
#
#                 Fid_{\alpha_2,-} &=  \mathbb{1}(
#                 \widehat{y}_{\overline{G}} == y ) -
#                 \mathbb{E}( \mathbb{1}( \widehat{y}_
#                 {\overline{G}^{(exp)}+E_{\alpha_2}(
#                 \overline{G}-\overline{G}^{(exp)})}==y))
#
#                 Fid_{\alpha_1,\alpha_2,\Delta} &=
#                 Fid_{\alpha_1,+} - Fid_{\alpha_2,-}
#
#     this method is designed for edge-based explanation subgraphs,
#     node-based explanation subgraphs should convert into
#     edge-based explanation subgraphs.
#
#     Args:
#         explainer (Explainer): The explainer to evaluate.
#         explanation (Explanation): The explanation to evaluate.
#         alpha1: the ratio of remove explanation subgraph each
#                 time in fid+ calculation
#         alpha2: the ratio of maintain non-explanation subgraph
#                 each time in fid- calculation
#         sample_num: how many samples will be used to estimate
#                 the fidelity
#         k_hop: the number of hop for node classification
#         undirect: if the graph is undirected graph (default:
#                 graph task is true, node task is false)
#         use_gt_label: use gt_label to calculate the fid
#
#     """
#     max_length = sample_num
#     alpha2 = 1 - alpha2
#     task_type = 'node' if explanation.get('index') is not None else 'graph'
#     if explainer.model_config.mode == ModelMode.regression:
#         raise ValueError("Fidelity not defined for 'regression' models")
#
#     node_mask = explanation.get('node_mask')
#     edge_mask = explanation.get('edge_mask')
#     edge_mask_np = explanation.get('edge_mask').cpu().detach().numpy()
#     if top_k != -1:
#         idx = np.argpartition(edge_mask_np, top_k)
#         edge_mask_np = np.where(edge_mask_np > edge_mask_np[idx],
#                                 np.ones_like(edge_mask_np),
#                                 np.zeros_like(edge_mask_np))
#
#     kwargs = {key: explanation[key] for key in explanation._model_args}
#
#     y = explanation.target
#     y_hat = explainer.get_prediction(
#         explanation.x,
#         explanation.edge_index,
#         **kwargs,
#     )
#     y_label = explainer.get_target(y_hat)  # original label
#
#     features = explanation.x
#     graphs = explanation.edge_index
#
#     matrix_0 = graphs[0].cpu().numpy()
#     matrix_1 = graphs[1].cpu().numpy()
#     exp_graph_matrix = coo_matrix(
#         (edge_mask_np,
#          (matrix_0, matrix_1)),
#         shape=(features.shape[0], features.shape[0])).tocsr()
#
#     if task_type == 'node':
#         index = explanation.index
#
#         y = y[index].view(-1)
#         y_hat = y_hat[index].view(-1)
#         y_label = y_label[index].view(-1)
#
#         subset, edge_index, mapping, edge_mask_ = \
#             k_hop_subgraph(index, k_hop, graphs, relabel_nodes=False)
#         edge_index_np = edge_index.cpu().detach().numpy()
#         sample_matrix = coo_matrix(
#             (np.ones_like(edge_index_np[0]),
#              (edge_index_np[0], edge_index_np[1])),
#             shape=(features.shape[0], features.shape[0])).tocsr()
#
#         graph_matrix = sample_matrix.multiply(exp_graph_matrix)
#         non_graph_matrix = sample_matrix - graph_matrix
#         weights = graph_matrix[edge_index_np[0], edge_index_np[1]].A[0]
#         explain = torch.tensor(weights).float().to(graphs.device)
#         weights = non_graph_matrix[edge_index_np[0], edge_index_np[1]].A[0]
#         non_explain = torch.tensor(weights).float().to(graphs.device)
#     else:
#         weights = edge_mask_np
#         explain = torch.tensor(weights).float().to(graphs.device)
#         non_explain = torch.tensor(1 - weights).float().to(graphs.device)
#
#     if undirect:
#         maps = {}
#         explain_list = []
#         non_explain_list = []
#         for i, (nodeid0, nodeid1, ex) in \
#                 enumerate(zip(matrix_0, matrix_1, edge_mask_np)):
#             max_node = max(nodeid0, nodeid1)
#             min_node = min(nodeid0, nodeid1)
#             if (min_node, max_node) in maps.keys():
#                 maps[(min_node, max_node)].append(i)
#                 if ex > 0.5:
#                     explain_list.append((min_node, max_node))
#                 else:
#                     non_explain_list.append((min_node, max_node))
#             else:
#                 maps[(min_node, max_node)] = [i]
#
#     else:
#         explain_list = \
#             torch.nonzero(explain).cpu().detach().numpy().tolist()
#         non_explain_list = \
#             torch.nonzero(non_explain).cpu().detach().numpy().tolist()
#
#     if use_gt_label:
#         label = int(y)
#     else:
#         label = int(y_label)
#
#     explaine_ratio = np.ones(len(explain_list))
#     explaine_ratio = \
#         alpha1 * explaine_ratio.sum() * \
#         (explaine_ratio / explaine_ratio.sum())
#     explaine_ratio_remove = \
#         np.random.binomial(1, explaine_ratio,
#                            size=(max_length, explaine_ratio.shape[0]))
#
#     non_explaine_ratio = np.ones(len(non_explain_list))
#     non_explaine_ratio = \
#         alpha2 * non_explaine_ratio.sum() * \
#         (non_explaine_ratio / non_explaine_ratio.sum())
#     non_explaine_ratio_remove = \
#         np.random.binomial(1, non_explaine_ratio,
#                            size=(max_length, non_explaine_ratio.shape[0]))
#
#     def cal_fid_embedding_plus():
#         list_explain = torch.zeros([max_length, explain.shape[0]])
#         for i in range(max_length):
#             remove_edges = explaine_ratio_remove[i]
#             for idx, edge in enumerate(explain_list):
#                 if remove_edges[idx] == 1:
#                     if undirect:
#                         id_lists = maps[edge]
#                         for id in id_lists:
#                             list_explain[i, id] = 1.0
#                     else:
#                         list_explain[i, idx] = 1.0
#
#         fid_plus_prob_list = []
#         fid_plus_acc_list = []
#
#         for i in range(max_length):
#             if task_type == 'node':
#                 with torch.no_grad():
#                     mask_pred_plus = explainer.get_masked_prediction(
#                         features,
#                         edge_index,
#                         1. - node_mask if node_mask is not None else None,
#                         1. - list_explain[i].to(features.device)
#                         if edge_mask is not None else None,
#                         **kwargs,
#                     )
#                     mask_pred_plus_label = explainer.get_target(mask_pred_plus)
#                     mask_pred_plus = mask_pred_plus[index].view(-1)
#
#                     mask_label_plus = mask_pred_plus_label[index].view(-1)
#
#                     fid_plus = y_hat[label] - mask_pred_plus[label]
#                     fid_plus_label = \
#                         int(y_label == label) - int(mask_label_plus == label)
#
#             else:
#                 with torch.no_grad():
#                     mask_pred_plus = explainer.get_masked_prediction(
#                         features,
#                         graphs,
#                         1. - node_mask if node_mask is not None else None,
#                         1. - list_explain[i].to(features.device)
#                         if edge_mask is not None else None,
#                         **kwargs,
#                     )
#
#                     mask_pred_plus_label = explainer.get_target(mask_pred_plus)
#
#                     mask_label_plus = mask_pred_plus_label
#
#                     fid_plus = y_hat[:, label] - mask_pred_plus[:, label]
#                     fid_plus_label = \
#                         int(y_label == label) - int(mask_label_plus == label)
#
#             fid_plus_prob_list.append(fid_plus)
#             fid_plus_acc_list.append(fid_plus_label)
#         if len(fid_plus_prob_list) < 1:
#             return 0, 0
#         else:
#             fid_plus_mean = \
#                 torch.stack(fid_plus_prob_list).mean().cpu().detach().numpy()
#             fid_plus_label_mean = np.stack(fid_plus_acc_list).mean()
#         return fid_plus_mean, fid_plus_label_mean
#
#     def cal_fid_embedding_minus():
#         # global non_explain_indexs_combin
#         list_explain = torch.zeros([max_length, non_explain.shape[0]])
#         for i in range(max_length):
#             remove_edges = non_explaine_ratio_remove[i]
#             for idx, edge in enumerate(non_explain_list):
#                 if remove_edges[idx] == 1:
#                     if undirect:
#                         id_lists = maps[edge]  # get two edges id
#                         for id in id_lists:
#                             list_explain[i, id] = 1.0
#                     else:
#                         list_explain[i, idx] = 1.0
#
#         fid_minus_prob_list = []
#         fid_minus_acc_list = []
#         # fid_minus_embedding_distance_list = []
#
#         for i in range(max_length):
#             if task_type == 'node':
#                 with torch.no_grad():
#                     mask_pred_minus = explainer.get_masked_prediction(
#                         features,
#                         edge_index,
#                         node_mask,
#                         list_explain[i].to(features.device),
#                         **kwargs, )
#                     mask_pred_minus_label = \
#                         explainer.get_target(mask_pred_minus)
#
#                     mask_pred_minus = mask_pred_minus[index].view(-1)
#                     mask_label_minus = mask_pred_minus_label[index].view(-1)
#
#                     fid_minus = y_hat[label] - mask_pred_minus[label]
#                     fid_minus_label = \
#                         int(y_label == label) - int(mask_label_minus == label)
#
#             else:
#                 with torch.no_grad():
#                     mask_pred_minus = explainer.get_masked_prediction(
#                         features,
#                         graphs,
#                         node_mask,
#                         list_explain[i].to(features.device),
#                         **kwargs, )
#                     mask_pred_minus_label = \
#                         explainer.get_target(mask_pred_minus)
#                     mask_label_minus = mask_pred_minus_label
#
#                     fid_minus = y_hat[:, label] - mask_pred_minus[:, label]
#                     fid_minus_label = \
#                         int(y_label == label) - int(mask_label_minus == label)
#
#             fid_minus_prob_list.append(fid_minus)
#             fid_minus_acc_list.append(fid_minus_label)
#
#         if len(fid_minus_prob_list) < 1:
#             return 1, 1
#         else:
#             fid_minus_mean = \
#                 torch.stack(fid_minus_prob_list).mean().cpu().detach().numpy()
#             fid_minus_label_mean = np.stack(fid_minus_acc_list).mean()
#         return fid_minus_mean, fid_minus_label_mean
#
#     fid_plus_mean, fid_plus_label_mean = cal_fid_embedding_plus()
#     fid_minus_mean, fid_minus_label_mean = cal_fid_embedding_minus()
#     fid_delta = fid_plus_mean-fid_minus_mean
#     fid_delta_label = fid_plus_label_mean - fid_minus_label_mean
#
#     return \
#         fid_plus_mean, fid_minus_mean, fid_delta, \
#         fid_plus_label_mean, fid_minus_label_mean, fid_delta_label


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

