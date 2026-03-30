import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.dense import dense_diff_pool
from torch_geometric.nn import DenseGCNConv, GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn.inits import glorot, zeros

# from graph_conv import DenseGATConv
from torch_geometric.utils import to_dense_adj, to_dense_batch

from typing import Optional

from utils.graph_utils import process_outputs





class DeleteNet(nn.Module):
    """
    专门负责预测现有边保留概率的模块
    """

    def __init__(self, h_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self, node_rep, edge_index):
        src, dst = edge_index
        # 拼接源节点和目标节点特征: [E, 2*h_dim]
        e_feat = torch.cat([node_rep[src], node_rep[dst]], dim=-1)
        logit = self.net(e_feat).view(-1)
        prob = torch.sigmoid(logit)
        return prob, logit


class AddVGAENet(nn.Module):
    """
    专门负责 VGAE 逻辑的模块：Encoder -> Reparam -> Decoder
    """

    def __init__(self, h_dim, z_dim):
        super().__init__()
        self.encoder_mu = nn.Linear(h_dim, z_dim)
        self.encoder_logvar = nn.Linear(h_dim, z_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, h):
        """
        输入: 子图节点特征 h [N_sub, h_dim]
        输出: 重构概率矩阵 prob_adj [N_sub, N_sub], mu, logvar
        """
        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        z = self.reparameterize(mu, logvar)
        logits = torch.matmul(z, z.t())
        prob_adj = torch.sigmoid(logits)
        return prob_adj, mu, logvar


class MyExplainerV2(nn.Module):
    def __init__(self, args, gnn):
        super().__init__()
        self.args = args
        self.gnn = gnn
        self.device = args.device

        # 1. Graph Encoder
        self.conv1 = GCNConv(args.x_dim, args.h_dim)
        self.conv2 = GCNConv(args.h_dim, args.h_dim)

        # 2. Functional Modules
        self.delete_net = DeleteNet(args.h_dim)
        self.add_net = AddVGAENet(args.h_dim, args.z_dim)

    # ================= 核心逻辑：Pipeline =================

    def forward(self, graphs, subgraphs=None, use_subgraph_gt=True, max_cand_per_graph=15):
        x, edge_index, batch = graphs.x, graphs.edge_index, graphs.batch
        N = x.size(0)

        # Step 1: Encode Nodes
        node_rep = self._encode_graph(x, edge_index)

        # Step 2: Determine Candidate Region (FS Nodes)
        fs_nodes_bool = self._get_fs_mask(batch, subgraphs, N, use_subgraph_gt)

        # Step 3: Delete Logic (Predict p_keep for existing edges)
        p_keep, logit_keep = self.delete_net(node_rep, edge_index)

        # Step 4: Add Logic (VGAE inside FS nodes)
        add_results = self._sample_add_candidates(
            node_rep, edge_index, batch, fs_nodes_bool, max_cand_per_graph
        )
        cand_src, cand_dst, p_add = add_results['src'], add_results['dst'], add_results['probs']

        # Step 5: Construct CF Graph
        edge_index_cf, edge_weight_cf = self._build_cf_graph(
            edge_index, p_keep, cand_src, cand_dst, p_add
        )

        return {
            "node_rep": node_rep,
            "fs_nodes_bool": fs_nodes_bool,
            "p_keep": p_keep,
            "logit_keep": logit_keep,
            "cand_src": cand_src,
            "cand_dst": cand_dst,
            "p_add": p_add,
            "add_recon_loss": add_results['recon_loss'],
            "add_kl_loss": add_results['kl_loss'],
            "edge_index_cf": edge_index_cf,
            "edge_weight_cf": edge_weight_cf,
        }

    # ================= 辅助函数：具体的实现细节 =================

    def _encode_graph(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return h

    def _get_fs_mask(self, batch, subgraphs, N, use_subgraph_gt):
        """获取频繁子图节点的 bool mask"""
        if not (use_subgraph_gt and subgraphs is not None):
            # Fallback 逻辑，如果没有提供 GT，默认全 False 或基于其他逻辑
            return torch.zeros(N, dtype=torch.bool, device=self.device)

        fs_nodes_bool = torch.zeros(N, dtype=torch.bool, device=self.device)
        B = int(batch.max().item()) + 1

        for g, sub_g in enumerate(subgraphs):
            if not hasattr(sub_g, "node_mappings"): continue

            # 获取当前图 g 的全局索引
            idx_g = (batch == g).nonzero(as_tuple=False).view(-1)
            local_idx = sub_g.node_mappings.to(self.device)

            if local_idx.numel() > 0:
                global_idx = idx_g[local_idx]
                fs_nodes_bool[global_idx] = True

        return fs_nodes_bool

    def _sample_add_candidates(self, node_rep, edge_index, batch, fs_nodes_bool, max_cand):
        """
        运行 VGAE 并根据策略筛选候选边
        """
        device = node_rep.device
        row, col = edge_index
        N = node_rep.size(0)
        B = int(batch.max().item()) + 1

        # 构造全局邻接矩阵 (用于快速切片子图)
        # 注意：对于超大图，这里可能需要稀疏矩阵优化，但在 Explainer 场景通常图不大
        adj_global = torch.zeros(N, N, device=device)
        adj_global[row, col] = 1
        adj_global[col, row] = 1

        cand_src_list, cand_dst_list, p_add_list = [], [], []
        recon_losses, kl_losses = [], []

        for g in range(B):
            # 1. 提取子图数据
            idx_g = (batch == g).nonzero(as_tuple=False).view(-1)
            fs_idx_g = idx_g[fs_nodes_bool[idx_g]]  # 仅取 FS 节点

            if fs_idx_g.size(0) <= 1: continue

            h_g = node_rep[fs_idx_g]
            A_g = adj_global[fs_idx_g][:, fs_idx_g]  # 子邻接矩阵

            # 2. VGAE Forward
            prob_A, mu, logvar = self.add_net(h_g)

            # 3. 计算 Loss (Recon + KL)
            recon_loss_g = self._compute_vgae_recon_loss(prob_A, A_g)
            kl_loss_g = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            recon_losses.append(recon_loss_g)
            kl_losses.append(kl_loss_g)

            # 4. 筛选候选边 (非边 & 上三角)
            triu_mask = torch.triu(torch.ones_like(A_g, dtype=torch.bool), diagonal=1)
            cand_mask = (A_g == 0) & triu_mask

            if not cand_mask.any(): continue

            scores = prob_A[cand_mask]
            cand_idx_local = cand_mask.nonzero(as_tuple=False)

            # 5. Top-K 策略 (核心修改逻辑)
            kept_indices = self._apply_topk_strategy(scores, max_cand)
            if kept_indices is None: continue  # 推理阶段被截断了

            scores = scores[kept_indices]
            cand_idx_local = cand_idx_local[kept_indices]

            # 映射回全局索引
            cand_src_list.append(fs_idx_g[cand_idx_local[:, 0]])
            cand_dst_list.append(fs_idx_g[cand_idx_local[:, 1]])
            p_add_list.append(scores)

        # 聚合结果
        return {
            'src': torch.cat(cand_src_list) if cand_src_list else None,
            'dst': torch.cat(cand_dst_list) if cand_dst_list else None,
            'probs': torch.cat(p_add_list) if p_add_list else None,
            'recon_loss': torch.stack(recon_losses).mean() if recon_losses else torch.tensor(0.0, device=device),
            'kl_loss': torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0, device=device)
        }

    def _compute_vgae_recon_loss(self, prob_A, target_A):
        """加权二元交叉熵重构损失"""
        triu_mask = torch.triu(torch.ones_like(target_A, dtype=torch.bool), diagonal=1)
        A_pos = target_A[triu_mask]
        P_pos = prob_A[triu_mask]

        if A_pos.numel() == 0: return torch.tensor(0.0, device=prob_A.device)

        # 动态平衡正负样本权重
        num_pos = (A_pos == 1).sum().item()
        num_neg = (A_pos == 0).sum().item()
        pos_weight = num_neg / max(num_pos, 1) if num_pos > 0 else 1.0

        weight = torch.ones_like(A_pos)
        weight[A_pos == 1] = pos_weight

        loss = F.binary_cross_entropy(P_pos, A_pos, weight=weight)
        return loss

    def _apply_topk_strategy(self, scores, max_cand):
        """
        根据训练/推理模式决定保留哪些边。
        返回: 保留的边的索引 (LongTensor) 或 None
        """
        if self.training:
            # 训练模式：保留所有边或 TopK，不做阈值截断，保证梯度回传
            if max_cand is not None and scores.numel() > max_cand:
                _, top_idx = torch.topk(scores, max_cand)
                return top_idx
            return torch.arange(scores.numel(), device=scores.device)
        else:
            # 推理模式：必须 > 0.5 且 TopK
            keep_mask = scores > 0.5
            if keep_mask.sum() == 0: return None

            # 先滤掉 < 0.5 的
            valid_indices = keep_mask.nonzero(as_tuple=False).view(-1)
            valid_scores = scores[valid_indices]

            if max_cand is not None and valid_scores.numel() > max_cand:
                _, topk_sub_idx = torch.topk(valid_scores, max_cand)
                return valid_indices[topk_sub_idx]  # 映射回原索引

            return valid_indices

    def _build_cf_graph(self, edge_index, p_keep, cand_src, cand_dst, p_add):
        """组合原有边和新增边，构建 CF 图"""
        if cand_src is None or p_add is None:
            return edge_index, p_keep

        edge_index_add = torch.stack([cand_src, cand_dst], dim=0)

        edge_index_cf = torch.cat([edge_index, edge_index_add], dim=1)
        edge_weight_cf = torch.cat([p_keep, p_add], dim=0)

        return edge_index_cf, edge_weight_cf

    # ================= Loss 计算 =================

    def compute_loss(self, args, graphs, y_desired, outputs):
        # 提取变量，使公式更干净
        cf_probs, cf_logits = self.gnn.get_pred_explain(
            graphs.x, outputs["edge_index_cf"], outputs["edge_weight_cf"], graphs.batch
        )

        # 1. CF Prediction Loss (Classification + Margin)
        y_target = y_desired.to(self.device).view(-1).long()
        cf_loss = self._compute_cf_loss(cf_logits, y_target, args)

        # 2. Regularization (L1 Sparsity)
        l1_add = outputs["p_add"].mean() if outputs["p_add"] is not None else 0.0
        l1_del = (1 - outputs["p_keep"]).mean()

        # 3. VGAE Structure Loss
        recon_loss = outputs["add_recon_loss"]
        kl_loss = outputs["add_kl_loss"]

        # 总损失聚合
        total_loss = (
                getattr(args, "w_cf", 5.0) * cf_loss +
                getattr(args, "w_l1_add", 10.0) * l1_add +
                getattr(args, "w_l1_del", 0.5) * l1_del +
                getattr(args, "w_vgae_recon", 5.0) * recon_loss +
                getattr(args, "w_vgae_kl", 1.0) * kl_loss
        )

        # 4. (Optional) Budget Loss - 如有需要可在此处恢复

        return {
            "total": total_loss,
            "cf": cf_loss.detach(),
            "recon": recon_loss.detach(),
            "kl": kl_loss.detach(),
        }

    def _compute_cf_loss(self, logits, y_target, args):
        """计算 CrossEntropy 和 Margin Loss"""
        ce_loss = F.cross_entropy(logits, y_target)

        # Margin Loss
        B = logits.size(0)
        idx = torch.arange(B, device=self.device)
        logits_t = logits[idx, y_target]
        logits_o = logits[idx, 1 - y_target]  # 假设是二分类，多分类需修改取最大非target逻辑

        margin = getattr(args, "cf_margin", 0.5)
        margin_loss = F.relu(margin + logits_o - logits_t).mean()

        lambda_margin = getattr(args, "lambda_cf_margin", 1.0)
        return ce_loss + lambda_margin * margin_loss






class FrequentSubgraphMiner(nn.Module):
    def __init__(self, node_emb_dim):
        super(FrequentSubgraphMiner, self).__init__()
        self.node_emb_dim = node_emb_dim
        self.mlp_mask = nn.Sequential(
            nn.Linear(self.node_emb_dim, self.node_emb_dim * 2),
            nn.ReLU(),
            nn.Linear(self.node_emb_dim * 2, 1)
        )

    def forward(self, x):
        return self.mlp_mask(x)



class DenseGATConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_attr_dim, aggr='add', bias=True):
        super(DenseGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_attr_dim = edge_attr_dim
        self.aggr = aggr
        # Linear transformations for node features
        self.lin_rel = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=False)

        # Additional transformation for edge attributes
        self.lin_edge = nn.Linear(edge_attr_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()
        self.lin_edge.reset_parameters()

    def forward(self, x, adj, edge_attr, mask=None):
        B, N, _ = x.size()
        # Expand edge attributes to match the adjacency matrix
        full_edge_attr = torch.zeros(B, N, N, self.edge_attr_dim, device=edge_attr.device)
        tril_indices = torch.tril_indices(N, N, offset=-1)
        full_edge_attr[:, tril_indices[0], tril_indices[1]] = edge_attr
        full_edge_attr[:, tril_indices[1], tril_indices[0]] = edge_attr
        # Transform edge attributes
        edge_attr_transformed = self.lin_edge(full_edge_attr).squeeze(-1)  # Shape: [B, N, N]

        # Modify adjacency matrix with edge attributes
        edge_adj = adj * edge_attr_transformed  # Shape: [B, N, N]

        # Perform graph convolution
        out = torch.matmul(edge_adj, x)  # Shape: [B, N, out_channels]
        if self.aggr == 'mean':
            out = out / edge_adj.sum(dim=-1, keepdim=True).clamp_(min=1)
        out = self.lin_rel(out)
        out += self.lin_root(x)
        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, edge_attr_dim={self.edge_attr_dim})')