import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.data import Batch
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
        return prob_adj, logits, mu, logvar


class MyExplainerV2(nn.Module):
    def __init__(self, args, gnn):
        super().__init__()
        self.args = args
        self.gnn = gnn
        self.device = args.device
        self.num_proto_classes = getattr(args, "num_classes", 2)

        # 1. Graph Encoder
        self.conv1 = GCNConv(args.x_dim, args.h_dim)
        self.conv2 = GCNConv(args.h_dim, args.h_dim)

        # 2. Functional Modules
        self.delete_net = DeleteNet(args.h_dim)
        self.add_net = AddVGAENet(args.h_dim, args.z_dim)
        self.register_buffer("class_prototypes", torch.zeros(self.num_proto_classes, args.h_dim))
        self.register_buffer("prototype_available", torch.zeros(self.num_proto_classes, dtype=torch.bool))

    # ================= 核心逻辑：Pipeline =================

    def forward(self, graphs, subgraphs=None, use_subgraph_gt=True, max_cand_per_graph=15):
        x, edge_index, batch = graphs.x, graphs.edge_index, graphs.batch
        N = x.size(0)

        # Step 1: Encode Nodes
        node_rep = self._encode_graph(x, edge_index)

        # Step 2: Determine Candidate Region (FS Nodes)
        fs_nodes_bool = self._get_fs_mask(batch, subgraphs, N, use_subgraph_gt)

        # Step 3: DeleteNet manages edge deletion over all original edges.
        p_keep, logit_keep = self.delete_net(node_rep, edge_index)

        # Step 4: AddVGAE only manages candidate additions inside FS.
        add_results = self._sample_add_candidates(
            node_rep, edge_index, batch, fs_nodes_bool, max_cand_per_graph
        )
        cand_src, cand_dst, p_add = add_results['src'], add_results['dst'], add_results['probs']

        # Step 5: Construct CF Graph
        edge_index_cf, edge_weight_cf = self._build_cf_graph(
            edge_index, p_keep, cand_src, cand_dst, p_add
        )
        cf_fs_embed, cf_fs_graph_ids = self._encode_reconstructed_fs_subgraphs(
            x,
            batch,
            fs_nodes_bool,
            edge_index_cf,
            edge_weight_cf,
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
            "cf_fs_embed": cf_fs_embed,
            "cf_fs_graph_ids": cf_fs_graph_ids,
        }

    # ================= 辅助函数：具体的实现细节 =================

    def _encode_graph(self, x, edge_index, edge_weight=None):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.relu(self.conv2(h, edge_index, edge_weight=edge_weight))
        return h

    def _encode_graph_embedding(self, x, edge_index, batch, edge_weight=None):
        if x.numel() == 0:
            return torch.empty((0, self.args.h_dim), device=x.device)
        node_rep = self._encode_graph(x, edge_index, edge_weight=edge_weight)
        return global_mean_pool(node_rep, batch)

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

    def _get_prototype_graphs_for_class(self, prototype_bank, class_idx):
        if prototype_bank is None:
            return None
        if isinstance(prototype_bank, dict):
            return prototype_bank.get(class_idx)
        if isinstance(prototype_bank, (list, tuple)):
            if 0 <= class_idx < len(prototype_bank):
                return prototype_bank[class_idx]
            return None
        return None

    def refresh_class_prototypes(self, prototype_bank):
        prototype_values = torch.zeros_like(self.class_prototypes)
        availability = torch.zeros_like(self.prototype_available)

        if prototype_bank is None:
            self.class_prototypes.copy_(prototype_values)
            self.prototype_available.copy_(availability)
            return

        was_training = self.training
        self.eval()
        with torch.no_grad():
            for class_idx in range(self.num_proto_classes):
                class_graphs = self._get_prototype_graphs_for_class(prototype_bank, class_idx)
                if class_graphs is None:
                    continue

                if isinstance(class_graphs, list):
                    if len(class_graphs) == 0:
                        continue
                    class_batch = Batch.from_data_list(class_graphs).to(self.device)
                elif isinstance(class_graphs, Batch):
                    class_batch = class_graphs.to(self.device)
                else:
                    class_batch = Batch.from_data_list([class_graphs]).to(self.device)

                if not hasattr(class_batch, "x") or class_batch.x is None or class_batch.x.numel() == 0:
                    continue

                class_embed = self._encode_graph_embedding(
                    class_batch.x.float(),
                    class_batch.edge_index,
                    class_batch.batch,
                )
                if class_embed.numel() == 0:
                    continue

                prototype_values[class_idx] = class_embed.mean(dim=0)
                availability[class_idx] = True

        if was_training:
            self.train()

        self.class_prototypes.copy_(prototype_values)
        self.prototype_available.copy_(availability)

    def _encode_reconstructed_fs_subgraphs(self, x, batch, fs_nodes_bool, edge_index_cf, edge_weight_cf):
        if edge_index_cf is None or edge_weight_cf is None:
            return None, None

        device = x.device
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if B == 0:
            return None, None

        x_parts, batch_parts = [], []
        edge_index_parts, edge_weight_parts = [], []
        graph_ids = []
        node_offset = 0
        N = x.size(0)

        for g in range(B):
            idx_g = (batch == g).nonzero(as_tuple=False).view(-1)
            fs_idx_g = idx_g[fs_nodes_bool[idx_g]]
            if fs_idx_g.numel() <= 1:
                continue

            global_to_local = torch.full((N,), -1, dtype=torch.long, device=device)
            global_to_local[fs_idx_g] = torch.arange(fs_idx_g.size(0), device=device)

            x_parts.append(x[fs_idx_g])
            batch_parts.append(torch.full((fs_idx_g.size(0),), len(graph_ids), dtype=torch.long, device=device))
            graph_ids.append(g)

            cf_src, cf_dst = edge_index_cf
            edge_mask_g = (
                (batch[cf_src] == g) &
                (batch[cf_dst] == g) &
                fs_nodes_bool[cf_src] &
                fs_nodes_bool[cf_dst]
            )

            if edge_mask_g.any():
                src_local = global_to_local[cf_src[edge_mask_g]] + node_offset
                dst_local = global_to_local[cf_dst[edge_mask_g]] + node_offset
                edge_index_parts.append(torch.stack([src_local, dst_local], dim=0))
                edge_weight_parts.append(edge_weight_cf[edge_mask_g])
            else:
                edge_index_parts.append(torch.empty((2, 0), dtype=torch.long, device=device))
                edge_weight_parts.append(torch.empty((0,), dtype=x.dtype, device=device))

            node_offset += fs_idx_g.size(0)

        if not graph_ids:
            return None, None

        x_fs = torch.cat(x_parts, dim=0).float()
        batch_fs = torch.cat(batch_parts, dim=0)
        edge_index_fs = torch.cat(edge_index_parts, dim=1)
        edge_weight_fs = torch.cat(edge_weight_parts, dim=0)

        fs_embed = self._encode_graph_embedding(
            x_fs,
            edge_index_fs,
            batch_fs,
            edge_weight=edge_weight_fs if edge_weight_fs.numel() > 0 else None,
        )
        graph_ids = torch.tensor(graph_ids, dtype=torch.long, device=device)
        return fs_embed, graph_ids

    def _sample_add_candidates_legacy(self, node_rep, edge_index, batch, fs_nodes_bool, max_cand):
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
            prob_A, _, mu, logvar = self.add_net(h_g)

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

    def _sample_add_candidates(self, node_rep, edge_index, batch, fs_nodes_bool, max_cand):
        return self._sample_add_candidates_legacy(
            node_rep, edge_index, batch, fs_nodes_bool, max_cand
        )

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

    def _build_cf_graph_legacy(self, edge_index, p_keep, cand_src, cand_dst, p_add):
        """组合原有边和新增边，构建 CF 图"""
        if cand_src is None or p_add is None:
            return edge_index, p_keep

        edge_index_add = torch.stack([cand_src, cand_dst], dim=0)

        edge_index_cf = torch.cat([edge_index, edge_index_add], dim=1)
        edge_weight_cf = torch.cat([p_keep, p_add], dim=0)

        return edge_index_cf, edge_weight_cf

    # ================= Loss 计算 =================

    def _build_cf_graph(self, edge_index, p_keep, cand_src, cand_dst, p_add):
        return self._build_cf_graph_legacy(edge_index, p_keep, cand_src, cand_dst, p_add)

    # ================= Loss 璁＄畻 =================
    def _compute_proto_loss(self, outputs, y_target):
        cf_fs_embed = outputs.get("cf_fs_embed")
        cf_fs_graph_ids = outputs.get("cf_fs_graph_ids")
        if cf_fs_embed is None or cf_fs_graph_ids is None or cf_fs_embed.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        target_classes = y_target[cf_fs_graph_ids]
        in_range = (target_classes >= 0) & (target_classes < self.num_proto_classes)
        if in_range.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        cf_fs_embed = cf_fs_embed[in_range]
        target_classes = target_classes[in_range]
        available = self.prototype_available[target_classes]
        if available.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        cf_fs_embed = cf_fs_embed[available]
        target_classes = target_classes[available]
        target_proto = self.class_prototypes[target_classes].detach()

        cf_fs_embed = F.normalize(cf_fs_embed, p=2, dim=-1)
        target_proto = F.normalize(target_proto, p=2, dim=-1)
        cosine = (cf_fs_embed * target_proto).sum(dim=-1)
        return (1 - cosine).mean()

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
        proto_loss = self._compute_proto_loss(outputs, y_target)

        # 总损失聚合
        total_loss = (
                getattr(args, "w_cf", 10.0) * cf_loss +
                getattr(args, "w_l1_add", 10.0) * l1_add +
                getattr(args, "w_l1_del", 5) * l1_del +
                getattr(args, "w_vgae_recon", 5.0) * recon_loss +
                getattr(args, "w_vgae_kl", 1.0) * kl_loss +
                getattr(args, "w_proto", 1.0) * proto_loss
        )

        # 4. (Optional) Budget Loss - 如有需要可在此处恢复

        return {
            "total": total_loss,
            "cf": cf_loss.detach(),
            "recon": recon_loss.detach(),
            "kl": kl_loss.detach(),
            "proto": proto_loss.detach(),
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
