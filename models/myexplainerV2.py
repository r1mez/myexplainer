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

from typing import Dict, List, Optional

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
        if z_dim < 2:
            raise ValueError(f"Conditional AddVGAENet requires z_dim >= 2, got {z_dim}.")
        self.latent_dim = z_dim
        # Preserve the old stochastic latent size; inject class information in the decoder.
        self.stochastic_dim = z_dim - 1
        self.encoder_mu = nn.Linear(h_dim, self.stochastic_dim)
        self.encoder_logvar = nn.Linear(h_dim, self.stochastic_dim)
        self.label_embedding = nn.Embedding(2, self.stochastic_dim)
        self.decoder = nn.Sequential(
            nn.Linear(4 * self.stochastic_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, h, cond_labels):
        """
        输入: 子图节点特征 h [N_sub, h_dim]
        输出: 重构概率矩阵 prob_adj [N_sub, N_sub], mu, logvar
        """
        cond_labels = torch.as_tensor(cond_labels, device=h.device)
        if cond_labels.dim() == 0:
            cond_labels = cond_labels.view(1)
        elif cond_labels.dim() == 2 and cond_labels.size(-1) == 1:
            cond_labels = cond_labels.reshape(-1)
        elif cond_labels.dim() != 1:
            raise ValueError(
                f"cond_labels must have shape [N_sub], [N_sub, 1], or scalar, got {tuple(cond_labels.shape)}."
            )

        if cond_labels.numel() == 1:
            cond_labels = cond_labels.expand(h.size(0))

        if cond_labels.size(0) != h.size(0):
            raise ValueError(
                f"cond_labels must align with h on the node dimension, got {cond_labels.size(0)} vs {h.size(0)}."
            )

        cond_labels = cond_labels.to(dtype=torch.long)
        valid_mask = (cond_labels == 0) | (cond_labels == 1)
        if not valid_mask.all().item():
            raise ValueError("Conditional AddVGAENet expects binary labels encoded as 0/1.")

        graph_cond = cond_labels[0]
        if not torch.all(cond_labels == graph_cond).item():
            raise ValueError(
                "Conditional AddVGAENet expects one graph-level label broadcast to all FS nodes."
            )

        mu = self.encoder_mu(h)
        logvar = self.encoder_logvar(h)
        z = self.reparameterize(mu, logvar)

        cond_embed = self.label_embedding(graph_cond).to(dtype=h.dtype)
        num_nodes = z.size(0)
        z_i = z.unsqueeze(1).expand(num_nodes, num_nodes, -1)
        z_j = z.unsqueeze(0).expand(num_nodes, num_nodes, -1)
        cond_pair = cond_embed.view(1, 1, -1).expand(num_nodes, num_nodes, -1)
        pair_feat = torch.cat(
            [
                z_i + z_j,
                torch.abs(z_i - z_j),
                z_i * z_j,
                cond_pair,
            ],
            dim=-1,
        )
        logits = self.decoder(pair_feat).squeeze(-1)
        logits = 0.5 * (logits + logits.transpose(0, 1))
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

    def forward(self, graphs, subgraphs=None, cond_labels=None, use_subgraph_gt=True, max_cand_per_graph=15):
        x, edge_index, batch = graphs.x, graphs.edge_index, graphs.batch
        N = x.size(0)

        # Step 1: Encode Nodes
        node_rep = self._encode_graph(x, edge_index)

        # Step 2: Determine Candidate Region (FS Nodes)
        fs_nodes_bool = self._get_fs_mask(batch, subgraphs, N, use_subgraph_gt)

        # Step 3: DeleteNet manages edge deletion over all original edges.
        p_keep, logit_keep = self.delete_net(node_rep, edge_index)

        # Step 4: AddVGAE only manages candidate additions inside FS.
        cond_labels = self._prepare_cond_labels(cond_labels, batch)
        add_results = self._sample_add_candidates(
            node_rep, edge_index, batch, fs_nodes_bool, cond_labels, max_cand_per_graph
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

    def _prepare_cond_labels(self, cond_labels, batch):
        if cond_labels is None:
            raise ValueError("cond_labels must be provided for conditional AddVGAE.")

        cond_labels = torch.as_tensor(cond_labels, device=self.device)
        if cond_labels.dim() == 2 and cond_labels.size(-1) == 1:
            cond_labels = cond_labels.reshape(-1)
        elif cond_labels.dim() != 1:
            raise ValueError(
                f"cond_labels must have shape [B] or [B, 1], got {tuple(cond_labels.shape)}."
            )

        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        if cond_labels.numel() != B:
            raise ValueError(
                f"cond_labels must provide one label per graph, got {cond_labels.numel()} labels for {B} graphs."
            )

        cond_labels = cond_labels.to(dtype=torch.float)
        valid_mask = (cond_labels == 0) | (cond_labels == 1)
        if not valid_mask.all().item():
            raise ValueError("Conditional AddVGAE currently expects binary labels encoded as 0/1.")

        return cond_labels

    def _collect_undirected_bonds(self, edge_index, batch):
        device = edge_index.device
        if edge_index.numel() == 0:
            empty_long = torch.empty((0,), dtype=torch.long, device=device)
            return {
                "graph_ids": empty_long,
                "src": empty_long,
                "dst": empty_long,
                "edge_indices": [],
                "graph_to_bond_indices": {},
            }

        src_all = edge_index[0].detach().cpu().tolist()
        dst_all = edge_index[1].detach().cpu().tolist()
        graph_ids_all = batch[edge_index[0]].detach().cpu().tolist()

        bond_to_idx: Dict[tuple, int] = {}
        bond_graph_ids: List[int] = []
        bond_src: List[int] = []
        bond_dst: List[int] = []
        bond_edge_indices: List[List[int]] = []

        for edge_idx, (src, dst, graph_id) in enumerate(zip(src_all, dst_all, graph_ids_all)):
            if src == dst:
                continue

            u, v = (src, dst) if src < dst else (dst, src)
            key = (int(graph_id), int(u), int(v))
            bond_idx = bond_to_idx.get(key)
            if bond_idx is None:
                bond_idx = len(bond_graph_ids)
                bond_to_idx[key] = bond_idx
                bond_graph_ids.append(int(graph_id))
                bond_src.append(int(u))
                bond_dst.append(int(v))
                bond_edge_indices.append([edge_idx])
            else:
                bond_edge_indices[bond_idx].append(edge_idx)

        graph_to_bond_indices: Dict[int, torch.Tensor] = {}
        for bond_idx, graph_id in enumerate(bond_graph_ids):
            graph_to_bond_indices.setdefault(int(graph_id), []).append(bond_idx)

        graph_to_bond_indices = {
            graph_id: torch.tensor(indices, dtype=torch.long, device=device)
            for graph_id, indices in graph_to_bond_indices.items()
        }

        return {
            "graph_ids": torch.tensor(bond_graph_ids, dtype=torch.long, device=device),
            "src": torch.tensor(bond_src, dtype=torch.long, device=device),
            "dst": torch.tensor(bond_dst, dtype=torch.long, device=device),
            "edge_indices": [
                torch.tensor(indices, dtype=torch.long, device=device)
                for indices in bond_edge_indices
            ],
            "graph_to_bond_indices": graph_to_bond_indices,
        }

    def _aggregate_bond_keep_logits(self, edge_logits, bond_edge_indices):
        if edge_logits.numel() == 0 or not bond_edge_indices:
            return torch.empty((0,), dtype=edge_logits.dtype, device=edge_logits.device)
        return torch.stack([edge_logits[edge_ids].mean() for edge_ids in bond_edge_indices], dim=0)

    def _select_oracle_probe_graph_ids(self, graph_to_bond_indices, max_graphs):
        eligible_graph_ids = sorted(
            graph_id for graph_id, bond_indices in graph_to_bond_indices.items() if bond_indices.numel() > 0
        )
        if max_graphs is None or max_graphs <= 0 or len(eligible_graph_ids) <= max_graphs:
            return eligible_graph_ids

        if self.training:
            sampled = torch.randperm(len(eligible_graph_ids), device=self.device)[:max_graphs].cpu().tolist()
            return [eligible_graph_ids[idx] for idx in sampled]

        return eligible_graph_ids[:max_graphs]

    def _select_delete_candidates_for_graph(self, graph_id, bond_data, delete_scores, fs_nodes_bool, args):
        graph_bond_indices = bond_data["graph_to_bond_indices"].get(int(graph_id))
        if graph_bond_indices is None or graph_bond_indices.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        graph_bond_list = graph_bond_indices.detach().cpu().tolist()

        topk = max(0, int(getattr(args, "oracle_del_topk", 6)))
        num_random = max(0, int(getattr(args, "oracle_del_random_negatives", 2)))

        top_candidate_list: List[int] = []
        if graph_bond_list and topk > 0:
            top_count = min(topk, len(graph_bond_list))
            top_scores = delete_scores[graph_bond_indices]
            top_pos = torch.topk(top_scores, k=top_count).indices.detach().cpu().tolist()
            top_candidate_list = [graph_bond_list[idx] for idx in top_pos]

        selected_set = set(top_candidate_list)
        remaining_list = [bond_idx for bond_idx in graph_bond_list if bond_idx not in selected_set]
        random_candidate_list: List[int] = []
        if remaining_list and num_random > 0:
            sample_size = min(num_random, len(remaining_list))
            if self.training:
                perm = torch.randperm(len(remaining_list), device=self.device)[:sample_size].cpu().tolist()
                random_candidate_list = [remaining_list[idx] for idx in perm]
            else:
                random_candidate_list = remaining_list[:sample_size]

        candidate_list = top_candidate_list + random_candidate_list
        if not candidate_list:
            return torch.empty((0,), dtype=torch.long, device=self.device)

        return torch.tensor(candidate_list, dtype=torch.long, device=self.device)

    def _extract_single_graph_inputs(self, x, edge_index, batch, graph_id):
        device = x.device
        node_idx = (batch == int(graph_id)).nonzero(as_tuple=False).view(-1)
        if node_idx.numel() == 0:
            return None

        src_all, dst_all = edge_index
        edge_mask = (batch[src_all] == int(graph_id)) & (batch[dst_all] == int(graph_id))
        edge_ids = edge_mask.nonzero(as_tuple=False).view(-1)

        global_to_local = torch.full((x.size(0),), -1, dtype=torch.long, device=device)
        global_to_local[node_idx] = torch.arange(node_idx.size(0), device=device)

        if edge_ids.numel() == 0:
            edge_index_local = torch.empty((2, 0), dtype=torch.long, device=device)
        else:
            edge_index_local = torch.stack(
                [global_to_local[src_all[edge_ids]], global_to_local[dst_all[edge_ids]]],
                dim=0,
            )

        edge_id_to_local = torch.full((edge_index.size(1),), -1, dtype=torch.long, device=device)
        edge_id_to_local[edge_ids] = torch.arange(edge_ids.size(0), device=device)

        x_local = x[node_idx].float()
        batch_local = torch.zeros(node_idx.size(0), dtype=torch.long, device=device)
        return x_local, edge_index_local, batch_local, edge_id_to_local

    def _probe_graph_delete_rewards(self, graphs, graph_id, target_label, bond_data, candidate_bond_indices):
        extracted = self._extract_single_graph_inputs(graphs.x, graphs.edge_index, graphs.batch, graph_id)
        if extracted is None:
            empty_long = torch.empty((0,), dtype=torch.long, device=self.device)
            empty_float = torch.empty((0,), dtype=graphs.x.dtype, device=self.device)
            return empty_long, empty_float

        x_local, edge_index_local, batch_local, edge_id_to_local = extracted
        if edge_index_local.size(1) == 0:
            empty_long = torch.empty((0,), dtype=torch.long, device=self.device)
            empty_float = torch.empty((0,), dtype=graphs.x.dtype, device=self.device)
            return empty_long, empty_float

        source_label = 1 - int(target_label.item())
        base_mask = torch.ones(edge_index_local.size(1), dtype=x_local.dtype, device=self.device)

        valid_bond_indices: List[int] = []
        rewards: List[torch.Tensor] = []
        with torch.no_grad():
            _, baseline_logits = self.gnn.get_pred_explain(x_local, edge_index_local, base_mask, batch_local)
            baseline_margin = baseline_logits[0, int(target_label.item())] - baseline_logits[0, source_label]

            for bond_idx in candidate_bond_indices.detach().cpu().tolist():
                local_edge_ids = edge_id_to_local[bond_data["edge_indices"][bond_idx]]
                local_edge_ids = local_edge_ids[local_edge_ids >= 0]
                if local_edge_ids.numel() == 0:
                    continue

                edge_mask = base_mask.clone()
                edge_mask[local_edge_ids] = 0.0
                _, probe_logits = self.gnn.get_pred_explain(x_local, edge_index_local, edge_mask, batch_local)
                margin_after = probe_logits[0, int(target_label.item())] - probe_logits[0, source_label]
                rewards.append((margin_after - baseline_margin).detach())
                valid_bond_indices.append(int(bond_idx))

        if not rewards:
            empty_long = torch.empty((0,), dtype=torch.long, device=self.device)
            empty_float = torch.empty((0,), dtype=graphs.x.dtype, device=self.device)
            return empty_long, empty_float

        return (
            torch.tensor(valid_bond_indices, dtype=torch.long, device=self.device),
            torch.stack(rewards, dim=0).to(device=self.device, dtype=x_local.dtype),
        )

    def _compute_pairwise_delete_rank_loss(self, delete_scores, rewards, tie_eps):
        if delete_scores.numel() < 2 or rewards.numel() < 2:
            return None

        reward_diff = rewards.unsqueeze(1) - rewards.unsqueeze(0)
        valid_pairs = torch.triu(torch.abs(reward_diff) > tie_eps, diagonal=1)
        if not valid_pairs.any():
            return None

        score_diff = delete_scores.unsqueeze(1) - delete_scores.unsqueeze(0)
        reward_sign = torch.sign(reward_diff)
        return F.softplus(-score_diff[valid_pairs] * reward_sign[valid_pairs]).mean()

    def _compute_oracle_delete_rank_loss(self, args, graphs, y_target, outputs):
        zero = torch.tensor(0.0, device=self.device)
        bond_data = self._collect_undirected_bonds(graphs.edge_index, graphs.batch)
        if bond_data["graph_ids"].numel() == 0:
            return zero

        bond_keep_logit = self._aggregate_bond_keep_logits(outputs["logit_keep"], bond_data["edge_indices"])
        if bond_keep_logit.numel() == 0:
            return zero

        delete_scores = -bond_keep_logit
        probe_graph_ids = self._select_oracle_probe_graph_ids(
            bond_data["graph_to_bond_indices"],
            int(getattr(args, "oracle_del_probe_graphs_per_batch", 4)),
        )
        if not probe_graph_ids:
            return zero

        graph_losses = []
        tie_eps = float(getattr(args, "oracle_del_reward_tie_eps", 1e-6))
        for graph_id in probe_graph_ids:
            candidate_bond_indices = self._select_delete_candidates_for_graph(
                graph_id,
                bond_data,
                delete_scores,
                outputs["fs_nodes_bool"],
                args,
            )
            if candidate_bond_indices.numel() < 2:
                continue

            valid_bond_indices, rewards = self._probe_graph_delete_rewards(
                graphs,
                graph_id,
                y_target[int(graph_id)],
                bond_data,
                candidate_bond_indices,
            )
            if valid_bond_indices.numel() < 2:
                continue

            graph_loss = self._compute_pairwise_delete_rank_loss(
                delete_scores[valid_bond_indices],
                rewards,
                tie_eps,
            )
            if graph_loss is not None:
                graph_losses.append(graph_loss)

        if not graph_losses:
            return zero

        return torch.stack(graph_losses, dim=0).mean()

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

                weights = None
                if isinstance(class_graphs, dict):
                    weights = class_graphs.get("weights")
                    class_graphs = class_graphs.get("graphs")
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

                if weights is not None:
                    weights = torch.as_tensor(weights, dtype=class_embed.dtype, device=self.device).view(-1)
                    if weights.numel() != class_embed.size(0):
                        min_len = min(weights.numel(), class_embed.size(0))
                        weights = weights[:min_len]
                        class_embed = class_embed[:min_len]
                    if weights.numel() == 0:
                        continue
                    weight_sum = weights.sum()
                    if weight_sum.item() <= 0:
                        weights = torch.ones_like(weights) / max(weights.numel(), 1)
                    else:
                        weights = weights / weight_sum
                    prototype_values[class_idx] = torch.sum(class_embed * weights.unsqueeze(-1), dim=0)
                else:
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

    def _sample_add_candidates_legacy(self, node_rep, edge_index, batch, fs_nodes_bool, cond_labels, max_cand):
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
            graph_cond = cond_labels[g].to(dtype=h_g.dtype)
            cond_node = graph_cond.view(1, 1).expand(fs_idx_g.size(0), 1)
            prob_A, _, mu, logvar = self.add_net(h_g, cond_node)

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

    def _sample_add_candidates(self, node_rep, edge_index, batch, fs_nodes_bool, cond_labels, max_cand):
        return self._sample_add_candidates_legacy(
            node_rep, edge_index, batch, fs_nodes_bool, cond_labels, max_cand
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
    def _get_loss_hparam(self, args, name):
        if not hasattr(args, name):
            raise AttributeError(
                f"Missing loss hyperparameter '{name}'. "
                "Call apply_loss_hparams(args) before training/evaluation."
            )
        return getattr(args, name)

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
        oracle_del_rank = self._compute_oracle_delete_rank_loss(args, graphs, y_target, outputs)
        w_cf = self._get_loss_hparam(args, "w_cf")
        w_l1_add = self._get_loss_hparam(args, "w_l1_add")
        w_l1_del = self._get_loss_hparam(args, "w_l1_del")
        w_oracle_del_rank = self._get_loss_hparam(args, "w_oracle_del_rank")
        w_vgae_recon = self._get_loss_hparam(args, "w_vgae_recon")
        w_vgae_kl = self._get_loss_hparam(args, "w_vgae_kl")
        w_proto = self._get_loss_hparam(args, "w_proto")

        # 总损失聚合
        total_loss = (
                w_cf * cf_loss +
                w_l1_add * l1_add +
                w_l1_del * l1_del +
                w_oracle_del_rank * oracle_del_rank +
                w_vgae_recon * recon_loss +
                w_vgae_kl * kl_loss +
                w_proto * proto_loss
        )

        # 4. (Optional) Budget Loss - 如有需要可在此处恢复

        return {
            "total": total_loss,
            "cf": cf_loss.detach(),
            "recon": recon_loss.detach(),
            "kl": kl_loss.detach(),
            "proto": proto_loss.detach(),
            "oracle_del_rank": oracle_del_rank.detach(),
        }

    def _compute_cf_loss(self, logits, y_target, args):
        """计算 CrossEntropy 和 Margin Loss"""
        ce_loss = F.cross_entropy(logits, y_target)

        # Margin Loss
        B = logits.size(0)
        idx = torch.arange(B, device=self.device)
        logits_t = logits[idx, y_target]
        logits_o = logits[idx, 1 - y_target]  # 假设是二分类，多分类需修改取最大非target逻辑

        margin = self._get_loss_hparam(args, "cf_margin")
        margin_loss = F.relu(margin + logits_o - logits_t).mean()

        lambda_margin = self._get_loss_hparam(args, "lambda_cf_margin")
        return ce_loss + lambda_margin * margin_loss
