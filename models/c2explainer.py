import os
import time

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_undirected, sort_edge_index, k_hop_subgraph, to_networkx
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
# 1. 核心算法: C2Explainer (仅结构扰动版)
# ==========================================
class C2ExplainerStructuralOnly(nn.Module):
    def __init__(self, model, epochs=100, lr=0.05, lambda_sim=1.0, max_candidates=1000, **kwargs):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.lambda_sim = lambda_sim
        self.max_candidates = max_candidates

        params = list(model.parameters())
        self.device = params[0].device if len(params) > 0 else 'cpu'
        self.undirected = True

    def _get_candidate_edges(self, edge_index, num_nodes, device):
        """
        生成候选边。仅针对无向图的上三角部分 (u < v) 生成，避免重复。
        """
        # 1. 识别已有的边 (只存 u < v)
        edge_index = sort_edge_index(edge_index)
        row, col = edge_index
        mask = row < col
        existing_edges_set = set(zip(row[mask].tolist(), col[mask].tolist()))

        # 2. 生成候选
        if num_nodes < 200:
            # 全连接上三角
            r, c = torch.triu_indices(num_nodes, num_nodes, offset=1, device=device)
            candidates = []
            for i in range(r.size(0)):
                u, v = r[i].item(), c[i].item()
                if (u, v) not in existing_edges_set:
                    candidates.append([u, v])

            if candidates:
                candidates = torch.tensor(candidates, device=device).t()
            else:
                candidates = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            # 随机采样
            candidates_list = []
            count = 0
            while count < self.max_candidates:
                idx = torch.randint(0, num_nodes, (2, self.max_candidates), device=device)
                idx = idx[:, idx[0] < idx[1]]  # 强制 u < v
                for i in range(idx.size(1)):
                    u, v = idx[0, i].item(), idx[1, i].item()
                    if (u, v) not in existing_edges_set:
                        candidates_list.append([u, v])
                        existing_edges_set.add((u, v))
                        count += 1
                        if count >= self.max_candidates: break
                if count >= self.max_candidates: break

            if candidates_list:
                candidates = torch.tensor(candidates_list, device=device).t()
            else:
                candidates = torch.empty((2, 0), dtype=torch.long, device=device)

        return candidates

    def explain_graph(self, x, edge_index, **kwargs):
        self.model.eval()
        num_nodes = x.size(0)
        device = self.device

        # 获取 Batch 信息
        batch = kwargs.get('batch', torch.zeros(x.size(0), dtype=torch.long, device=device))

        # --- 1. 处理原始边 (提取单向 u < v) ---
        edge_index = sort_edge_index(edge_index)
        row, col = edge_index
        mask_unique = row < col
        edge_index_del_unique = edge_index[:, mask_unique]

        # --- 2. 准备候选边 (u < v) ---
        edge_index_add_unique = self._get_candidate_edges(edge_index, num_nodes, device)

        E_del = edge_index_del_unique.size(1)
        E_add = edge_index_add_unique.size(1)

        # --- 3. 初始化参数 ---
        edge_mask_delete = Parameter(torch.randn(E_del, device=device) + 2.0)
        edge_mask_add = Parameter(torch.randn(E_add, device=device) - 2.0)

        optimizer = torch.optim.Adam([edge_mask_delete, edge_mask_add], lr=self.lr)

        # --- 4. 确定目标 ---
        with torch.no_grad():
            # 原始预测直接用 forward
            logits = self.model(x, edge_index, batch=batch)
            if logits.dim() > 1: logits = logits[0]
            orig_pred = logits.argmax().item()
            probs = F.softmax(logits, dim=-1)
            probs[orig_pred] = -1.0
            desired_y = probs.argmax().item()

        target_tensor = torch.tensor([desired_y], device=device)

        # 辅助函数：将单向 mask 扩展为双向 mask
        def expand_to_undirected(u_edges, u_mask):
            u, v = u_edges
            # 构造双向边: [u, v] 和 [v, u]
            full_src = torch.cat([u, v])
            full_dst = torch.cat([v, u])
            full_edge_index = torch.stack([full_src, full_dst], dim=0)
            # 权重对称复制
            full_mask = torch.cat([u_mask, u_mask])
            return full_edge_index, full_mask

        # --- 5. 优化循环 ---
        best_loss = float('inf')
        best_cf_edge_index = edge_index

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            m_del_soft = torch.sigmoid(edge_mask_delete)
            m_add_soft = torch.sigmoid(edge_mask_add)

            # STE (Straight-Through Estimator)
            m_del_hard = (m_del_soft > 0.5).float()
            m_add_hard = (m_add_soft > 0.5).float()

            m_del = (m_del_hard - m_del_soft).detach() + m_del_soft
            m_add = (m_add_hard - m_add_soft).detach() + m_add_soft

            # --- 构建完整的无向图输入 ---
            full_idx_del, full_weight_del = expand_to_undirected(edge_index_del_unique, m_del)
            full_idx_add, full_weight_add = expand_to_undirected(edge_index_add_unique, m_add)

            final_edge_index = torch.cat([full_idx_del, full_idx_add], dim=1)
            final_edge_weight = torch.cat([full_weight_del, full_weight_add])

            # ============================================================
            # 修改点：适配 BA2MotifGCN，调用 get_pred_explain
            # ============================================================
            if hasattr(self.model, 'get_pred_explain'):
                # 你的模型返回 (probs, logits)，我们取 logits
                _, logits = self.model.get_pred_explain(
                    x,
                    final_edge_index,
                    edge_mask=final_edge_weight,
                    batch=batch,
                )
            else:
                # 兜底逻辑：普通模型通常支持 edge_weight
                logits = self.model(x, final_edge_index, edge_weight=final_edge_weight, batch=batch)
            # ============================================================

            if logits.dim() > 1: logits = logits[0]
            log_probs = F.log_softmax(logits, dim=-1)

            loss_cf = F.nll_loss(log_probs.unsqueeze(0), target_tensor)
            loss_struct = torch.sum(torch.abs(m_del_soft - 1.0)) + torch.sum(torch.abs(m_add_soft - 0.0))
            loss = loss_cf + self.lambda_sim * loss_struct

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                pred_label = logits.argmax().item()
                if pred_label == desired_y:
                    best_loss = loss.item()
                    with torch.no_grad():
                        # 生成最终的 Hard CF 图
                        mask_del_final = m_del_hard > 0.5
                        mask_add_final = m_add_hard > 0.5

                        kept_unique = edge_index_del_unique[:, mask_del_final]
                        added_unique = edge_index_add_unique[:, mask_add_final]

                        combined_unique = torch.cat([kept_unique, added_unique], dim=1)
                        best_cf_edge_index = to_undirected(combined_unique)

        return best_cf_edge_index, x


# ==========================================
# 2. 评估函数 (保持指标计算逻辑不变)
# ==========================================
def evaluate_c2_structural(pred_model, dataset, device, epochs=100, lr=0.05):
    print("\n" + "=" * 60)
    print("Evaluating C2Explainer (Structural Only) - ALL Samples Mode")
    print("=" * 60)

    wrapped_model = OracleWrappedModel(pred_model)
    explainer = C2ExplainerStructuralOnly(wrapped_model, epochs=epochs, lr=lr, max_candidates=500)

    valid_cf = 0
    proximity_sum = 0.0
    fidelity_prob_sum = 0.0
    sparsity_sum = 0.0

    # 这里的 total_graphs 指的是有效处理的图数量（排除掉 explainer 报错返回 None 的情况）
    processed_graphs = 0

    print(f"Processing {len(dataset)} graphs...")

    total_cf_time = 0.0
    total_cf_oracle_calls = 0

    for idx in tqdm(range(len(dataset))):
        data = dataset[idx].to(device)

        # 1. 原始预测（不计入 runtime 和 oracle_calls）
        with torch.no_grad():
            ori_logits = pred_model(data.x, data.edge_index, data.batch)
            if ori_logits.dim() > 1: ori_logits = ori_logits[0]
            ori_probs = F.softmax(ori_logits, dim=-1)
            ori_pred = ori_logits.argmax().item()

        # 2. 生成解释 —— 仅此阶段计入 runtime 和 oracle_calls
        batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
        calls_before = wrapped_model.oracle_calls
        t0 = time.time()
        cf_edge_index, cf_x = explainer.explain_graph(data.x, data.edge_index, batch=batch)
        total_cf_time += time.time() - t0
        total_cf_oracle_calls += wrapped_model.oracle_calls - calls_before

        if cf_edge_index is None:
            continue

        processed_graphs += 1

        # 3. 验证 CF（不计入 oracle_calls）
        with torch.no_grad():
            cf_logits = pred_model(cf_x, cf_edge_index, batch)
            if cf_logits.dim() > 1: cf_logits = cf_logits[0]
            cf_probs = F.softmax(cf_logits, dim=-1)
            cf_pred = cf_logits.argmax().item()

        # --- 这里的逻辑变了 ---

        # A. 计算 Validity (仅作为计数，不影响下面指标的计算)
        if cf_pred != ori_pred:
            valid_cf += 1

        # B. 无论是否翻转成功，都计算指标（与 MyExplainer 保持一致）
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

    # 汇总 (分母改为 processed_graphs)
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


# ==========================================
# 3. 运行入口
# ==========================================
if __name__ == "__main__":
    dataset_name = os.environ.get("MYEXPLAINER_DATASET", "fluoride_carbonyl")
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        train_dataset, val_dataset, test_dataset = get_datasets(name=dataset_name, root='data/')

        model_path = f'../param/gnns/{dataset_name}_gcn.pt'
        if os.path.exists(model_path):
            gnn = torch.load(model_path, map_location=device)
            gnn.eval()

            # 运行评估 (仅结构)
            evaluate_c2_structural(
                pred_model=gnn,
                dataset=test_dataset,
                device=device,
                epochs=100,
                lr=0.05
            )
        else:
            print(f"Model path {model_path} not found.")

    except ImportError:
        print("Please ensure 'utils.py' and 'gnns.py' are in the path.")
