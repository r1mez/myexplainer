import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.utils import to_undirected, sort_edge_index
from tqdm import tqdm


from utils import get_datasets
from eval.baseline_eval_metrics import OracleWrappedModel
from gnns import *
from models.base import BaseExplainer, CFResult


# ==========================================
# 1. 核心算法: C2Explainer (仅结构扰动版)
# ==========================================
class C2ExplainerStructuralOnly(BaseExplainer):
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

    def _explain_graph_core(self, x, edge_index, **kwargs):
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
                    edge_weight=final_edge_weight,
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

    def explain_graph(self, data, device="cpu"):
        """Generate a counterfactual explanation for a single graph.

        Args:
            data: PyG Data object with x, edge_index.
            device: Device string for computation.

        Returns:
            CFResult with cf_edge_index and cf_edge_weight.
        """
        data = data.to(device)
        x = data.x
        edge_index = data.edge_index

        cf_edge_index, cf_x = self._explain_graph_core(x, edge_index)

        return CFResult(
            cf_edge_index=cf_edge_index,
            cf_edge_weight=torch.ones(cf_edge_index.size(1), device=device),
        )


# ==========================================
# 2. 运行入口
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

            wrapped_model = OracleWrappedModel(gnn)
            explainer = C2ExplainerStructuralOnly(wrapped_model, epochs=100, lr=0.05, max_candidates=500)

            for idx in range(min(10, len(test_dataset))):
                data = test_dataset[idx]
                result = explainer.explain_graph(data, device=device)
                print(f"Graph {idx}: CF edges={result.cf_edge_index.size(1)}")
        else:
            print(f"Model path {model_path} not found.")

    except ImportError:
        print("Please ensure 'utils.py' and 'gnns.py' are in the path.")
