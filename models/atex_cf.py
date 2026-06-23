import os
import random
import time

import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_dense_adj, to_undirected, sort_edge_index, to_networkx
from tqdm import tqdm
import math

# 假设 utils.py 和 gnns.py 在当前目录下
from utils import get_datasets
from utils.baseline_eval_metrics import (
    compute_proximity_from_edge_index,
    compute_fidelity_prob_from_probs,
    compute_sparsity_from_edge_index,
    OracleWrappedModel,
)
from gnns import *

# ==========================================
# 1. 核心算法: WL-based 加边候选集生成算法
# ==========================================
def get_wl_candidates(data, L=2, K=10, device='cuda:0'):
    """
    根据 WL 算法生成高危加边候选集 S^+
    """
    G = to_networkx(data, to_undirected=True)
    num_nodes = data.x.size(0)

    # 1. 初始化节点颜色 (由于某些数据集没有离散特征，这里默认使用度数作为初始颜色)
    colors = {v: str(G.degree[v]) for v in G.nodes()}

    # 2. WL 邻域聚合与哈希
    for _ in range(L):
        new_colors = {}
        for v in G.nodes():
            neighbor_colors = sorted([colors[u] for u in G.neighbors(v)])
            # 拼接当前颜色与邻居颜色，并进行哈希
            hash_str = colors[v] + "_" + "".join(neighbor_colors)
            new_colors[v] = str(hash(hash_str))
        colors = new_colors

    # 3. 计算所有节点对的最短路径距离
    # 返回的是字典的字典: dict[source][target] = distance
    try:
        lengths = dict(nx.all_pairs_shortest_path_length(G))
    except:
        lengths = {}

    candidates_list = []

    # 获取原图已有的边集合(无向)
    existing_edges = set()
    for u, v in data.edge_index.t().tolist():
        existing_edges.add((min(u, v), max(u, v)))

    # 4. 筛选非边并打分
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):  # 保证 u < v，无向图不重复
            if (u, v) not in existing_edges:
                # 检查第 L 轮的 WL 颜色是否相同
                if colors[u] == colors[v]:
                    # 获取最短路径距离 (如果两点不连通，设为一个很大的惩罚值)
                    if u in lengths and v in lengths[u]:
                        dist = lengths[u][v]
                    else:
                        dist = 999
                    candidates_list.append((u, v, dist))

    # 5. 按距离得分从大到小排序 (取捷径跨度最大的边)
    candidates_list.sort(key=lambda x: x[2], reverse=True)

    # 6. 输出 Top-K 候选边
    S_plus = []
    for i in range(min(K, len(candidates_list))):
        S_plus.append([candidates_list[i][0], candidates_list[i][1]])

    if S_plus:
        return torch.tensor(S_plus, device=device).t()
    else:
        return torch.empty((2, 0), dtype=torch.long, device=device)

def get_random_candidates(data, L=2, K=10, device='cuda:0'):
    """
    随机采样 K 条原图中不存在的边 (无向，保证 u < v)。
    注: 保留 L=2 参数仅为了兼容原有的调用接口，实际内部不再使用。
    """
    num_nodes = data.x.size(0)

    # 1. 获取原图已有的边集合(无向，统一为小索引在前，大索引在后)
    existing_edges = set()
    for u, v in data.edge_index.t().tolist():
        existing_edges.add((min(u, v), max(u, v)))

    # 2. 收集所有不存在的边 (非边候选池)
    non_edges = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):  # 保证 u < v，无向图不重复
            if (u, v) not in existing_edges:
                non_edges.append([u, v])

    # 3. 处理图中已经全连通（无边可加）的极端情况
    if not non_edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # 4. 随机采样 K 条边 (如果不足 K 条，则取全部)
    num_samples = min(K, len(non_edges))
    sampled_edges = random.sample(non_edges, num_samples)

    # 5. 转换为 Tensor 并转置为 [2, num_samples] 的形状返回
    return torch.tensor(sampled_edges, device=device).t()

# ==========================================
# 2. ATEX-CF 图分类核心扰动层
# ==========================================
class ATEXCFExplainer(nn.Module):
    def __init__(self, model, epochs=100, lr=0.01, top_k=5, lambda_dist=0.5, lambda_plau=1.0, C=1.0, tau_plus=0.5, tau_minus=-0.5,
                 pruning=False):
        """
        ATEX-CF 参数:
        - top_k: 扰动预算 (\kappa)，最多允许修改几条边
        - lambda_dist: 稀疏性损失的权重
        - C: 非对称成本，加边的惩罚权重 (相对于删边权重为1.0)
        - tau_plus / tau_minus: 加边和删边的阈值
        - pruning: 是否开启事后剪枝 (Phase III: Explanation Refinement)
        """
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.top_k = top_k
        self.lambda_dist = lambda_dist
        self.lambda_plau = lambda_plau
        self.C = C
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.pruning = pruning  # 新增：剪枝开关

        params = list(model.parameters())
        self.device = params[0].device if len(params) > 0 else 'cpu'

    def minimality_pruning(self, x, batch, S_minus, S_plus, mask_del, mask_add, M_soft, desired_y):
        """
        事后极小性剪枝 (Post-hoc Minimality Pruning)
        按重要性(掩码绝对值)从小到大排序，尝试逐一撤销扰动，如果预测仍保持翻转，则永久撤销该扰动。
        """
        active_del = torch.where(mask_del)[0].tolist()
        active_add = torch.where(mask_add)[0].tolist()
        len_minus = S_minus.size(1)

        # 收集所有生效的扰动: (扰动类型, 对应索引, 重要性得分)
        perturbations = []
        for idx in active_del:
            score = abs(M_soft[idx].item())
            perturbations.append(('del', idx, score))
        for idx in active_add:
            score = abs(M_soft[len_minus + idx].item())
            perturbations.append(('add', idx, score))

        # 按重要性得分升序排序 (优先尝试去除最不重要的扰动)
        perturbations.sort(key=lambda item: item[2])

        current_mask_del = mask_del.clone()
        current_mask_add = mask_add.clone()

        # 贪心试探
        for p_type, idx, score in perturbations:
            temp_mask_del = current_mask_del.clone()
            temp_mask_add = current_mask_add.clone()

            # 尝试撤销扰动
            if p_type == 'del':
                temp_mask_del[idx] = False  # 撤销删边 -> 恢复原边
            else:
                temp_mask_add[idx] = False  # 撤销加边 -> 不加新边

            # 组装临时测试图
            kept_unique = S_minus[:, ~temp_mask_del]
            added_unique = S_plus[:, temp_mask_add]
            combined_unique = torch.cat([kept_unique, added_unique], dim=1)
            temp_edge_index = to_undirected(combined_unique)

            temp_edge_weight = torch.ones(temp_edge_index.size(1), device=self.device)

            # 前向传播验证
            with torch.no_grad():
                _, logits = self.model.get_pred_explain(
                    x, temp_edge_index, edge_weight=temp_edge_weight, batch=batch
                )
                if logits.dim() > 1: logits = logits[0]
                pred = logits.argmax().item()

            # 如果预测仍然保持翻转 (满足反事实目标)
            if pred == desired_y:
                current_mask_del = temp_mask_del  # 确认该扰动是冗余的，保留撤销操作
                current_mask_add = temp_mask_add

        # 基于最终精简后的 Mask 构建图
        kept_unique = S_minus[:, ~current_mask_del]
        added_unique = S_plus[:, current_mask_add]
        combined_unique = torch.cat([kept_unique, added_unique], dim=1)
        final_edge_index = to_undirected(combined_unique)

        return final_edge_index

    def explain_graph(self, x, edge_index, batch, data):
        self.model.eval()
        device = self.device

        # --- 1. 获取删边候选集 S^- (原图的所有无向边，u < v) ---
        edge_index = sort_edge_index(edge_index)
        row, col = edge_index
        mask_unique = row < col
        S_minus = edge_index[:, mask_unique]
        len_minus = S_minus.size(1)

        # --- 2. 获取加边候选集 S^+ (通过 WL 算法) ---
        # S_plus = get_wl_candidates(data, L=2, K=20, device=device)
        S_plus = get_random_candidates(data, L=2, K=20, device=device)
        len_plus = S_plus.size(1)

        candidate_edges = torch.cat([S_minus, S_plus], dim=1)

        # --- 3. 初始化符号掩码 (Signed Mask) M ---
        M_init = torch.cat([
            torch.full((len_minus,), -0.5) + 0.01 * torch.randn(len_minus),
            torch.full((len_plus,), 0.5) + 0.01 * torch.randn(len_plus)
        ]).to(device)

        M = Parameter(M_init)
        optimizer = torch.optim.Adam([M], lr=self.lr)

        # --- 4. 确定原始预测与反事实目标 ---
        with torch.no_grad():
            _, orig_logits = self.model.get_pred_explain(
                x, edge_index, edge_weight=torch.ones(edge_index.size(1), device=device), batch=batch
            )
            if orig_logits.dim() > 1: orig_logits = orig_logits[0]
            orig_pred = orig_logits.argmax().item()
            probs = F.softmax(orig_logits, dim=-1)
            probs[orig_pred] = -1.0
            desired_y = probs.argmax().item()
        # === [新增] 预计算原图的结构特征 (用于计算 Plausibility Loss) ===
        num_nodes = data.x.size(0)
        # 1. 原图邻接矩阵
        ori_adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
        # 2. 原图度数向量
        ori_degree = ori_adj.sum(dim=1)
        # 3. 原图三角形期望数量: Trace(A^3) / 6
        ori_A3 = torch.matmul(torch.matmul(ori_adj, ori_adj), ori_adj)
        ori_triangles = torch.trace(ori_A3) / 6.0
        # ==============================================================

        target_tensor = torch.tensor([desired_y], device=device)

        best_loss = float('inf')
        best_cf_edge_index = edge_index

        # 记录用于剪枝的最优掩码状态
        best_mask_del = None
        best_mask_add = None
        best_M_soft = None

        # --- 5. 优化循环 ---
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            M_soft = torch.tanh(M)

            m_del_soft = torch.relu(-M_soft[:len_minus])
            m_add_soft = torch.relu(M_soft[len_minus:])

            m_del_hard = (M_soft[:len_minus] < self.tau_minus).float()
            m_add_hard = (M_soft[len_minus:] > self.tau_plus).float()

            with torch.no_grad():
                scores = torch.abs(M_soft)
                costs = torch.cat([
                    torch.ones(len_minus, device=device),
                    torch.full((len_plus,), self.C, device=device)
                ])

                _, sorted_idx = torch.sort(scores, descending=True)
                budget = 0.0
                mask_topk = torch.zeros_like(M_soft)

                for idx in sorted_idx:
                    cost = costs[idx].item()
                    if budget + cost > self.top_k:
                        continue
                    mask_topk[idx] = 1.0
                    budget += cost
                    if budget == self.top_k:
                        break

                m_del_hard = m_del_hard * mask_topk[:len_minus]
                m_add_hard = m_add_hard * mask_topk[len_minus:]

            m_del = (m_del_hard - m_del_soft).detach() + m_del_soft
            m_add = (m_add_hard - m_add_soft).detach() + m_add_soft

            weight_minus = 1.0 - m_del
            weight_plus = m_add
            candidate_weights = torch.cat([weight_minus, weight_plus])

            src = torch.cat([candidate_edges[0], candidate_edges[1]])
            dst = torch.cat([candidate_edges[1], candidate_edges[0]])
            full_edge_index = torch.stack([src, dst], dim=0)
            full_edge_weight = torch.cat([candidate_weights, candidate_weights])

            # === [新增] 动态计算 Plausibility Loss ===
            # 1. 构建反事实图的软邻接矩阵 (Soft Adjacency Matrix)
            cf_adj_soft = torch.zeros((num_nodes, num_nodes), device=device)
            cf_adj_soft[full_edge_index[0], full_edge_index[1]] = full_edge_weight

            # 2. 计算度数异常损失 (MSE)
            cf_degree = cf_adj_soft.sum(dim=1)
            loss_degree = F.mse_loss(cf_degree, ori_degree)

            # 3. 计算模体破坏损失 (三角形数量差异)
            cf_A3 = torch.matmul(torch.matmul(cf_adj_soft, cf_adj_soft), cf_adj_soft)
            cf_triangles = torch.trace(cf_A3) / 6.0
            loss_motif = torch.abs(cf_triangles - ori_triangles)

            # 合并合理性损失 (可根据需求调整两者相对权重)
            loss_plau = loss_degree + 0.1 * loss_motif  # motif数值通常较大，可用0.1缩放
            # ==============================================================

            _, logits = self.model.get_pred_explain(
                x, full_edge_index, edge_weight=full_edge_weight, batch=batch
            )
            if logits.dim() > 1: logits = logits[0]
            log_probs = F.log_softmax(logits, dim=-1)

            loss_pred = F.nll_loss(log_probs.unsqueeze(0), target_tensor)
            loss_dist = self.C * torch.sum(m_add_soft) + 1.0 * torch.sum(m_del_soft)

            loss = loss_pred + self.lambda_dist * loss_dist + self.lambda_plau * loss_plau

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                pred_label = logits.argmax().item()
                if pred_label == desired_y or True:
                    best_loss = loss.item()
                    with torch.no_grad():
                        # 记录当前最优的掩码分布，供后续剪枝使用
                        mask_del_final = m_del_hard > 0.5
                        mask_add_final = m_add_hard > 0.5

                        best_mask_del = mask_del_final.clone()
                        best_mask_add = mask_add_final.clone()
                        best_M_soft = M_soft.clone()

                        kept_unique = S_minus[:, ~mask_del_final]
                        added_unique = S_plus[:, mask_add_final]
                        combined_unique = torch.cat([kept_unique, added_unique], dim=1)
                        best_cf_edge_index = to_undirected(combined_unique)

        # --- Phase III: Explanation Refinement (事后极小性剪枝) ---
        # 仅在优化循环找到至少一个成功的反事实样本时执行剪枝
        if self.pruning and best_loss != float('inf'):
            best_cf_edge_index = self.minimality_pruning(
                x, batch, S_minus, S_plus,
                best_mask_del, best_mask_add, best_M_soft, desired_y
            )

        return best_cf_edge_index, x


# ==========================================
# 3. 评估函数 (与 C2Explainer 和 CF-GNNExplainer 对齐)
# ==========================================
def evaluate_atex_cf_graph(pred_model, dataset, device, epochs=100, lr=0.01, top_k=5, C=1.0):
    print("\n" + "=" * 60)
    print("Evaluating ATEX-CF (Graph Classification) - ALL Samples Mode")
    print(f"Hyperparameters: Budget(K)={top_k}, Asymmetric Cost(C)={C}")
    print("=" * 60)

    wrapped_model = OracleWrappedModel(pred_model)
    explainer = ATEXCFExplainer(wrapped_model, epochs=epochs, lr=lr, top_k=top_k, C=C)

    valid_cf = 0
    proximity_sum = 0.0
    fidelity_prob_sum = 0.0
    sparsity_sum = 0.0
    processed_graphs = 0

    print(f"Processing {len(dataset)} graphs...")

    total_cf_time = 0.0
    total_cf_oracle_calls = 0

    for idx in tqdm(range(len(dataset))):
        data = dataset[idx].to(device)
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        # 1. 原始预测（不计入 runtime 和 oracle_calls）
        with torch.no_grad():
            _, ori_logits = pred_model.get_pred_explain(
                data.x, data.edge_index, edge_weight=torch.ones(data.edge_index.size(1), device=device), batch=batch
            )
            if ori_logits.dim() > 1: ori_logits = ori_logits[0]
            ori_probs = F.softmax(ori_logits, dim=-1)
            ori_pred = ori_logits.argmax().item()

        # 2. 生成解释 (ATEX-CF) —— 仅此阶段计入 runtime 和 oracle_calls
        calls_before = wrapped_model.oracle_calls
        t0 = time.time()
        cf_edge_index, cf_x = explainer.explain_graph(data.x, data.edge_index, batch, data)
        total_cf_time += time.time() - t0
        total_cf_oracle_calls += wrapped_model.oracle_calls - calls_before

        # if cf_edge_index is None:
        #     continue
        processed_graphs += 1

        # 3. 验证 CF（不计入 oracle_calls）
        with torch.no_grad():
            _, cf_logits = pred_model.get_pred_explain(
                cf_x, cf_edge_index, edge_weight=torch.ones(cf_edge_index.size(1), device=device), batch=batch
            )
            if cf_logits.dim() > 1: cf_logits = cf_logits[0]
            cf_probs = F.softmax(cf_logits, dim=-1)
            cf_pred = cf_logits.argmax().item()

        # --- A. Validity ---
        if cf_pred != ori_pred:
            valid_cf += 1

        # --- B. 计算评价指标（与 MyExplainer 一致） ---
        num_nodes = data.x.size(0)
        proximity_sum += compute_proximity_from_edge_index(
            ori_edge_index=data.edge_index,
            cf_edge_index=cf_edge_index,
            num_nodes=num_nodes,
            device=device,
        )

        fidelity_prob_sum += compute_fidelity_prob_from_probs(
            ori_probs=ori_probs,
            cf_probs=cf_probs,
        )

        sparsity_sum += compute_sparsity_from_edge_index(
            ori_edge_index=data.edge_index,
            cf_edge_index=cf_edge_index,
        )

        if idx < 10:
            visualize_comparison(
                data=data.cpu(),  # 转回 CPU 绘图
                cf_edge_index=cf_edge_index.cpu(),
                ori_pred=ori_pred,
                cf_pred=cf_pred,
                idx=idx,
                save_dir='cf_results_vis'  # 图片保存在这个文件夹
            )

    total_graphs = len(dataset)
    avg_runtime_per_graph = total_cf_time / total_graphs if total_graphs > 0 else 0.0
    avg_oracle_calls_per_graph = total_cf_oracle_calls / total_graphs if total_graphs > 0 else 0.0
    print("processed_graphs: ", processed_graphs)
    # 汇总
    metrics = {
        "validity": valid_cf / processed_graphs if processed_graphs > 0 else 0,
        "proximity": proximity_sum / processed_graphs if processed_graphs > 0 else 0,
        "fidelity_prob": fidelity_prob_sum / processed_graphs if processed_graphs > 0 else 0,
        "sparsity": sparsity_sum / processed_graphs if processed_graphs > 0 else 0,
        "successful_count": valid_cf,
        "total_processed": processed_graphs,
        "runtime": avg_runtime_per_graph,
        "oracle_calls": avg_oracle_calls_per_graph,
    }

    print("\n" + "=" * 60)
    print("Evaluation Results (Calculated on ALL processed graphs):")
    print("=" * 60)
    print(f"  Validity ↑: {metrics['validity']:.4f} ({metrics['successful_count']}/{metrics['total_processed']})")
    print(f"  Proximity (Adj Diff) ↓: {metrics['proximity']:.4f}")
    print(f"  Fidelity (Prob Drop) ↑: {metrics['fidelity_prob']:.4f}")
    print(f"  Sparsity (Structure) ↑: {metrics['sparsity']:.4f}")
    print(f"  Runtime per graph (s) ↓: {avg_runtime_per_graph:.6f}")
    print(f"  Oracle calls per graph ↓: {avg_oracle_calls_per_graph:.4f}")
    print("=" * 60 + "\n")

    return metrics

def visualize_comparison(data, cf_edge_index, ori_pred, cf_pred, idx, save_dir='results'):
    """
    绘制原图与反事实图的对比。
    - data: 原始 PyG Data 对象
    - cf_edge_index: 解释器生成的反事实边索引
    - ori_pred: 原始预测类别
    - cf_pred: 反事实预测类别
    - idx: 图片编号
    """
    # 1. 转换为 NetworkX 对象
    # 原始图
    G_ori = to_networkx(data, to_undirected=True)

    # 反事实图 (注意：需要手动构建，以确保节点数一致)
    G_cf = nx.Graph()
    G_cf.add_nodes_from(range(data.num_nodes))  # 确保包含孤立点
    if cf_edge_index.size(1) > 0:
        edges = cf_edge_index.t().tolist()
        G_cf.add_edges_from(edges)

    # 2. 计算固定布局 (以原图为基准，保证两张图节点位置不动)
    # seed 保证每次运行位置一样，spring_layout 适合一般图结构
    pos = nx.spring_layout(G_ori, seed=42)

    # 3. 开始绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # --- 左图：Original ---
    nx.draw_networkx_nodes(G_ori, pos, ax=axes[0], node_size=50, node_color='#3498db')
    nx.draw_networkx_edges(G_ori, pos, ax=axes[0], alpha=0.5)
    axes[0].set_title(f"Original Graph\nPred: {ori_pred}", fontsize=14)
    axes[0].axis('off')

    # --- 右图：Counterfactual ---
    # 1. 区分边
    ori_edges = set(to_networkx(data, to_undirected=True).edges())
    cf_edges = set(G_cf.edges())

    kept_edges = list(cf_edges.intersection(ori_edges))  # 保留的边
    added_edges = list(cf_edges - ori_edges)  # 新增的边 (加边)
    # removed_edges = list(ori_edges - cf_edges)        # 被删的边 (如果想画虚线表示)

    # 2. 分别绘制
    # 绘制节点
    nx.draw_networkx_nodes(G_cf, pos, ax=axes[1], node_size=50, node_color='#e74c3c')

    # 绘制保留的边 (黑色/灰色)
    nx.draw_networkx_edges(G_cf, pos, edgelist=kept_edges, ax=axes[1], alpha=0.5, edge_color='black')

    # 绘制新增的边 (绿色，加粗)
    if added_edges:
        nx.draw_networkx_edges(G_cf, pos, edgelist=added_edges, ax=axes[1], alpha=1.0, edge_color='green', width=2)

    axes[1].set_title(f"Counterfactual Graph\nPred: {cf_pred}\n(Green=Added)", fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()

    # 4. 保存或显示
    plt.show()
# ==========================================
# 4. 运行入口
# ==========================================
if __name__ == "__main__":
    dataset_name = 'fluoride_carbonyl'  # 或者 nci1, ba2motif
    dataset_name = os.environ.get("MYEXPLAINER_DATASET", dataset_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using dataset: {dataset_name}")

    try:
        # 加载数据集和预训练模型
        train_dataset, val_dataset, test_dataset = get_datasets(name=dataset_name, root='data/')
        model_path = f'../param/gnns/{dataset_name}_gcn.pt'

        if os.path.exists(model_path):
            gnn = torch.load(model_path, map_location=device)
            gnn.eval()

            # 运行评估
            evaluate_atex_cf_graph(
                pred_model=gnn,
                dataset=test_dataset,
                device=device,
                epochs=100,  # 单个图搜索周期
                lr=0.001,  # 优化学习率
                top_k=5,  # 最大扰动预算(限制修改几条边)
                C=0.5  # 惩罚系数，如果觉得加边太多，可以将 C 调高 (如 1.5 - 3.0)
            )
        else:
            print(f"Model path {model_path} not found.")

    except ImportError:
        print("Please ensure 'utils.py' and 'gnns.py' are in the path.")
