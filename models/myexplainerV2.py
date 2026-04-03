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

from torch_geometric.utils import to_dense_adj, to_dense_batch
from torchdiffeq import odeint

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
        e_feat = torch.cat([node_rep[src], node_rep[dst]], dim=-1)
        logit = self.net(e_feat).view(-1)
        prob = torch.sigmoid(logit)
        return prob, logit


class SubgraphEncoder(nn.Module):
    """
    子图编码器：融合节点特征 h_g 与稠密邻接 A_g，输出隐表示 H1 [N_sub, z_dim]。
    与 Explainer 的 GCN 表征一起端到端训练。
    """

    def __init__(self, h_dim: int, z_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(h_dim, z_dim)
        self.lin_nei = nn.Linear(h_dim, z_dim)

    def forward(self, h_g: Tensor, A_g: Tensor) -> Tensor:
        deg = A_g.sum(dim=1, keepdim=True).clamp(min=1.0)
        agg = (A_g @ h_g) / deg
        return torch.relu(self.lin_self(h_g) + self.lin_nei(agg))


class SubgraphDecoder(nn.Module):
    """
    子图解码器：隐表示 H1 -> 连续邻接概率 [N_sub, N_sub]（sigmoid）。
    """

    def __init__(self, z_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hid = hidden_dim or z_dim
        self.mlp = nn.Sequential(
            nn.Linear(z_dim, hid),
            nn.ReLU(),
            nn.Linear(hid, z_dim),
        )

    def forward(self, H1: Tensor) -> Tensor:
        Z = self.mlp(H1)
        logits = Z @ Z.t()
        return torch.sigmoid(logits)


class FlowVectorField(nn.Module):
    """
    流匹配向量场 v_theta(t, H)，供 torchdiffeq.odeint 调用。
    t 为标量 0 维张量，H 为 [N_sub, z_dim]。
    """

    def __init__(self, z_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.t_emb = nn.Sequential(nn.Linear(1, hidden_dim), nn.Tanh())
        self.net = nn.Sequential(
            nn.Linear(z_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, z_dim),
        )

    def forward(self, t: Tensor, H: Tensor) -> Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=H.device, dtype=H.dtype)
        t = t.to(device=H.device, dtype=H.dtype).reshape(-1)
        t_scalar = t.reshape(1, 1).expand(H.size(0), 1)
        te = self.t_emb(t_scalar)
        return self.net(torch.cat([H, te], dim=-1))


class MyExplainerV2(nn.Module):
    def __init__(self, args, gnn):
        """
        Args:
            args: 超参数配置对象
            gnn: 被解释的预训练 GNN 分类器（冻结参数）
        """
        super().__init__()
        self.args = args
        self.gnn = gnn
        self.device = args.device
        z_dim = args.z_dim

        self.conv1 = GCNConv(args.x_dim, args.h_dim)
        self.conv2 = GCNConv(args.h_dim, args.h_dim)
        self.delete_net = DeleteNet(args.h_dim)

        # 子图编解码与向量场：内建于模型中，与 DeleteNet/GCN 一起端到端训练
        self.subgraph_encoder = SubgraphEncoder(args.h_dim, z_dim)
        self.subgraph_decoder = SubgraphDecoder(z_dim)
        self.vector_field = FlowVectorField(z_dim, hidden_dim=getattr(args, "sgfm_vf_hidden", 128))

        self.sgfm_opt_steps = getattr(args, 'sgfm_opt_steps', 50)
        self.sgfm_lr = getattr(args, 'sgfm_lr', 0.01)
        self.lambda_cf = getattr(args, 'lambda_cf', 5.0)
        self.lambda_dist = getattr(args, 'lambda_dist', 1.0)
        # ODE：训练默认 euler + 少量步数，比 dopri5 快 orders of magnitude
        self.sgfm_ode_method = getattr(args, 'sgfm_ode_method', 'euler')
        self.sgfm_ode_n_steps = int(getattr(args, 'sgfm_ode_n_steps', 8))

    def forward(self, graphs, subgraphs=None, y_desired=None, use_subgraph_gt=True, max_cand_per_graph=15):
        x, edge_index, batch = graphs.x, graphs.edge_index, graphs.batch
        N = x.size(0)

        node_rep = self._encode_graph(x, edge_index)
        fs_nodes_bool = self._get_fs_mask(batch, subgraphs, N, use_subgraph_gt)
        p_keep, logit_keep = self.delete_net(node_rep, edge_index)

        add_results = self._sample_add_candidates(
            x, node_rep, edge_index, batch, fs_nodes_bool, max_cand_per_graph,
            y_desired=y_desired, p_keep=p_keep
        )
        cand_src, cand_dst, p_add = add_results['src'], add_results['dst'], add_results['probs']

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
            "edge_index_cf": edge_index_cf,
            "edge_weight_cf": edge_weight_cf,
            "sgfm_recon_loss": add_results.get("sgfm_recon_loss"),
            "sgfm_cycle_loss": add_results.get("sgfm_cycle_loss"),
        }

    def _encode_graph(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return h

    def _get_fs_mask(self, batch, subgraphs, N, use_subgraph_gt):
        if not (use_subgraph_gt and subgraphs is not None):
            return torch.zeros(N, dtype=torch.bool, device=self.device)

        fs_nodes_bool = torch.zeros(N, dtype=torch.bool, device=self.device)
        B = int(batch.max().item()) + 1

        for g, sub_g in enumerate(subgraphs):
            if not hasattr(sub_g, "node_mappings"): continue

            idx_g = (batch == g).nonzero(as_tuple=False).view(-1)
            local_idx = sub_g.node_mappings.to(self.device)

            if local_idx.numel() > 0:
                global_idx = idx_g[local_idx]
                fs_nodes_bool[global_idx] = True

        return fs_nodes_bool

    def _dense_adj_for_fs(self, edge_index: Tensor, fs_idx_g: Tensor, num_nodes: int) -> Tensor:
        """仅构造频繁子图节点上的稠密邻接 [n_sub, n_sub]，避免 O(N^2) 全局矩阵。"""
        n_sub = fs_idx_g.numel()
        device = edge_index.device
        glob_to_local = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        glob_to_local[fs_idx_g] = torch.arange(n_sub, device=device, dtype=torch.long)
        row, col = edge_index
        gi = glob_to_local[row]
        gj = glob_to_local[col]
        mask = (gi >= 0) & (gj >= 0)
        A = torch.zeros(n_sub, n_sub, device=device, dtype=torch.float32)
        A[gi[mask], gj[mask]] = 1.0
        return A

    def _single_graph_explain_tensors(
        self, x_raw: Tensor, edge_index_full: Tensor, edge_weight_full: Tensor,
        batch_vec: Tensor, g: int,
    ):
        """
        将「全局 batch 图」上的边与权重裁剪为第 g 张图上的子图，供 get_pred_explain 单独前向。
        避免 SGFM 内循环每一步都对整 batch（上万节点）跑被解释 GNN。
        """
        device = x_raw.device
        idx_g = (batch_vec == g).nonzero(as_tuple=False).view(-1)
        n_loc = idx_g.numel()
        if n_loc == 0:
            return None
        global_to_local = torch.full((x_raw.size(0),), -1, dtype=torch.long, device=device)
        global_to_local[idx_g] = torch.arange(n_loc, device=device, dtype=torch.long)
        src, dst = edge_index_full
        mask = (global_to_local[src] >= 0) & (global_to_local[dst] >= 0)
        ei_loc = torch.stack([global_to_local[src[mask]], global_to_local[dst[mask]]], dim=0)
        ew_loc = edge_weight_full[mask]
        x_loc = x_raw[idx_g]
        batch_loc = torch.zeros(n_loc, dtype=torch.long, device=device)
        return x_loc, ei_loc, ew_loc, batch_loc

    def _ode_forward(self, H, t_start, t_end, reverse=False):
        sign = -1.0 if reverse else 1.0
        if self.sgfm_ode_method == 'euler':
            # 显式 Euler：每步一次 vector_field，适合训练且远快于自适应 dopri5
            n = max(self.sgfm_ode_n_steps, 1)
            dt = (t_end - t_start) / float(n)
            H_t = H
            for i in range(n):
                t_i = t_start + (i + 0.5) * dt
                t_tensor = torch.as_tensor(t_i, device=H.device, dtype=H.dtype)
                H_t = H_t + dt * sign * self.vector_field(t_tensor, H_t)
            return H_t
        def ode_func(t, H_t):
            return sign * self.vector_field(t, H_t)

        t_span = torch.tensor([t_start, t_end], dtype=H.dtype, device=H.device)
        H_trajectory = odeint(ode_func, H, t_span, method='dopri5', atol=1e-5, rtol=1e-5)
        return H_trajectory[-1]

    def _sample_add_candidates(self, x_raw, node_rep, edge_index, batch, fs_nodes_bool, max_cand,
                                y_desired=None, p_keep=None):
        """
        x_raw: 数据集原始节点特征 [N, x_dim]，须传入被解释 GNN 的 get_pred_explain（与 compute_loss 一致）。
        node_rep: Explainer GCN 隐层 [N, h_dim]，用于子图编码与 DeleteNet。
        """
        device = node_rep.device
        row, col = edge_index
        N = node_rep.size(0)
        B = int(batch.max().item()) + 1

        p_keep_detached = p_keep.detach() if p_keep is not None else None

        cand_src_list, cand_dst_list, p_add_list = [], [], []
        sgfm_recon_list, sgfm_cycle_list = [], []

        for g in range(B):
            idx_g = (batch == g).nonzero(as_tuple=False).view(-1)
            fs_idx_g = idx_g[fs_nodes_bool[idx_g]]

            if fs_idx_g.size(0) <= 1:
                continue

            h_g = node_rep[fs_idx_g]
            A_g = self._dense_adj_for_fs(edge_index, fs_idx_g, N)

            # 端到端结构损失：重构 + ODE 往返一致性（仅训练模式）
            if self.training:
                H1_struct = self.subgraph_encoder(h_g, A_g)
                A_rec = self.subgraph_decoder(H1_struct)
                triu_m = torch.triu(torch.ones_like(A_g, dtype=torch.bool), diagonal=1)
                if triu_m.any():
                    sgfm_recon_list.append(
                        F.binary_cross_entropy(A_rec[triu_m], A_g[triu_m].float())
                    )
                H0_c = self._ode_forward(H1_struct, 1.0, 0.0, reverse=True)
                H1_c = self._ode_forward(H0_c, 0.0, 1.0, reverse=False)
                sgfm_cycle_list.append(F.mse_loss(H1_c, H1_struct))
                H1_for_inv = H1_struct.detach()
            else:
                with torch.no_grad():
                    H1_for_inv = self.subgraph_encoder(h_g, A_g)

            if y_desired is None:
                continue

            with torch.no_grad():
                H0_orig = self._ode_forward(H1_for_inv, t_start=1.0, t_end=0.0, reverse=True)
            H0_orig = H0_orig.detach()

            with torch.enable_grad():
                H0_param = nn.Parameter(H0_orig.clone())
                inner_optimizer = torch.optim.Adam([H0_param], lr=self.sgfm_lr)

                best_loss_val = float('inf')
                H0_best = H0_orig.clone()

                y_target_g = y_desired[g].long().view(-1)

                for step in range(self.sgfm_opt_steps):
                    inner_optimizer.zero_grad(set_to_none=True)

                    H1_pred = self._ode_forward(H0_param, t_start=0.0, t_end=1.0, reverse=False)
                    A_cf_prob = self.subgraph_decoder(H1_pred)

                    triu_mask = torch.triu(torch.ones_like(A_g, dtype=torch.bool), diagonal=1)
                    new_edge_mask = (A_g == 0) & triu_mask

                    if not new_edge_mask.any():
                        break

                    new_edge_probs = A_cf_prob[new_edge_mask]
                    new_edge_idx_local = new_edge_mask.nonzero(as_tuple=False)

                    new_src_global = fs_idx_g[new_edge_idx_local[:, 0]]
                    new_dst_global = fs_idx_g[new_edge_idx_local[:, 1]]
                    edge_index_new = torch.stack([new_src_global, new_dst_global], dim=0)

                    edge_index_temp = torch.cat([edge_index, edge_index_new], dim=1)
                    edge_weight_temp = torch.cat([p_keep_detached, new_edge_probs], dim=0)

                    # 仅对第 g 张图做子图前向：与整图 batch 上 logits[g] 等价（图级池化），复杂度 O(|V_g|+|E_g|)
                    sub = self._single_graph_explain_tensors(
                        x_raw.detach(), edge_index_temp, edge_weight_temp, batch, g
                    )
                    if sub is None:
                        break
                    x_loc, ei_loc, ew_loc, b_loc = sub
                    _, logits = self.gnn.get_pred_explain(x_loc, ei_loc, ew_loc, b_loc)
                    logits_g = logits

                    prior_loss = 0.5 * torch.sum(H0_param ** 2)
                    cf_loss = F.cross_entropy(logits_g, y_target_g)
                    dist_loss = F.mse_loss(H1_pred, H1_for_inv)
                    total_energy = prior_loss + self.lambda_cf * cf_loss + self.lambda_dist * dist_loss

                    grad_h0, = torch.autograd.grad(
                        total_energy, H0_param, retain_graph=False, create_graph=False
                    )
                    H0_param.grad = grad_h0
                    inner_optimizer.step()

                    if total_energy.item() < best_loss_val:
                        best_loss_val = total_energy.item()
                        H0_best = H0_param.data.clone()

            with torch.no_grad():
                H1_best = self._ode_forward(H0_best, t_start=0.0, t_end=1.0, reverse=False)
                A_best_prob = self.subgraph_decoder(H1_best)

                triu_mask = torch.triu(torch.ones_like(A_g, dtype=torch.bool), diagonal=1)
                cand_mask = (A_g == 0) & triu_mask

                if not cand_mask.any():
                    continue

                scores = A_best_prob[cand_mask]
                cand_idx_local = cand_mask.nonzero(as_tuple=False)

                kept_indices = self._apply_topk_strategy(scores, max_cand)
                if kept_indices is None:
                    continue

                scores = scores[kept_indices]
                cand_idx_local = cand_idx_local[kept_indices]

                cand_src_list.append(fs_idx_g[cand_idx_local[:, 0]])
                cand_dst_list.append(fs_idx_g[cand_idx_local[:, 1]])
                p_add_list.append(scores)

        z = torch.tensor(0.0, device=device)
        sgfm_recon_loss = torch.stack(sgfm_recon_list).mean() if sgfm_recon_list else z
        sgfm_cycle_loss = torch.stack(sgfm_cycle_list).mean() if sgfm_cycle_list else z

        return {
            'src': torch.cat(cand_src_list) if cand_src_list else None,
            'dst': torch.cat(cand_dst_list) if cand_dst_list else None,
            'probs': torch.cat(p_add_list) if p_add_list else None,
            'sgfm_recon_loss': sgfm_recon_loss,
            'sgfm_cycle_loss': sgfm_cycle_loss,
        }

    def _apply_topk_strategy(self, scores, max_cand):
        if self.training:
            if max_cand is not None and scores.numel() > max_cand:
                _, top_idx = torch.topk(scores, max_cand)
                return top_idx
            return torch.arange(scores.numel(), device=scores.device)
        else:
            keep_mask = scores > 0.5
            if keep_mask.sum() == 0: return None

            valid_indices = keep_mask.nonzero(as_tuple=False).view(-1)
            valid_scores = scores[valid_indices]

            if max_cand is not None and valid_scores.numel() > max_cand:
                _, topk_sub_idx = torch.topk(valid_scores, max_cand)
                return valid_indices[topk_sub_idx]

            return valid_indices

    def _build_cf_graph(self, edge_index, p_keep, cand_src, cand_dst, p_add):
        if cand_src is None or p_add is None:
            return edge_index, p_keep

        edge_index_add = torch.stack([cand_src, cand_dst], dim=0)

        edge_index_cf = torch.cat([edge_index, edge_index_add], dim=1)
        edge_weight_cf = torch.cat([p_keep, p_add], dim=0)

        return edge_index_cf, edge_weight_cf

    def compute_loss(self, args, graphs, y_desired, outputs):
        cf_probs, cf_logits = self.gnn.get_pred_explain(
            graphs.x, outputs["edge_index_cf"], outputs["edge_weight_cf"], graphs.batch
        )

        y_target = y_desired.to(self.device).view(-1).long()
        cf_loss = self._compute_cf_loss(cf_logits, y_target, args)

        l1_add = outputs["p_add"].mean() if outputs["p_add"] is not None else 0.0
        l1_del = (1 - outputs["p_keep"]).mean()

        sgfm_recon = outputs.get("sgfm_recon_loss")
        sgfm_cycle = outputs.get("sgfm_cycle_loss")
        if not isinstance(sgfm_recon, torch.Tensor):
            sgfm_recon = torch.tensor(0.0, device=self.device)
        if not isinstance(sgfm_cycle, torch.Tensor):
            sgfm_cycle = torch.tensor(0.0, device=self.device)

        w_rec = getattr(args, "w_sgfm_recon", 1.0)
        w_cyc = getattr(args, "w_sgfm_cycle", 0.5)

        total_loss = (
                getattr(args, "w_cf", 5.0) * cf_loss +
                getattr(args, "w_l1_add", 10.0) * l1_add +
                getattr(args, "w_l1_del", 0.5) * l1_del +
                w_rec * sgfm_recon +
                w_cyc * sgfm_cycle
        )

        return {
            "total": total_loss,
            "cf": cf_loss.detach(),
            "sgfm_recon": sgfm_recon.detach() if isinstance(sgfm_recon, torch.Tensor) else sgfm_recon,
            "sgfm_cycle": sgfm_cycle.detach() if isinstance(sgfm_cycle, torch.Tensor) else sgfm_cycle,
        }

    def _compute_cf_loss(self, logits, y_target, args):
        ce_loss = F.cross_entropy(logits, y_target)

        B = logits.size(0)
        idx = torch.arange(B, device=self.device)
        logits_t = logits[idx, y_target]
        logits_o = logits[idx, 1 - y_target]

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
        self.lin_rel = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_root = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_edge = nn.Linear(edge_attr_dim, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rel.reset_parameters()
        self.lin_root.reset_parameters()
        self.lin_edge.reset_parameters()

    def forward(self, x, adj, edge_attr, mask=None):
        B, N, _ = x.size()
        full_edge_attr = torch.zeros(B, N, N, self.edge_attr_dim, device=edge_attr.device)
        tril_indices = torch.tril_indices(N, N, offset=-1)
        full_edge_attr[:, tril_indices[0], tril_indices[1]] = edge_attr
        full_edge_attr[:, tril_indices[1], tril_indices[0]] = edge_attr
        edge_attr_transformed = self.lin_edge(full_edge_attr).squeeze(-1)

        edge_adj = adj * edge_attr_transformed

        out = torch.matmul(edge_adj, x)
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
