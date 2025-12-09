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


# class MyExplainerV2(nn.Module):
#     def __init__(self, args, gnn):
#         super(MyExplainerV2, self).__init__()
#         self.x_dim = args.x_dim
#         self.h_dim = args.h_dim
#         self.z_dim = args.z_dim
#         self.edge_attr_dim = args.edge_attr_dim
#
#         # self.fsm = FrequentSubgraphMiner(self.h_dim)
#         self.gnn = gnn
#
#         self.edge_attr_dim = args.edge_attr_dim
#         self.max_num_nodes = args.max_num_nodes
#         self.dropout = args.dropout if hasattr(args, 'dropout') else 0.1
#         self.graph_pool_type = 'mean'
#         self.device = args.device
#         self.model = gnn
#
#         if self.edge_attr_dim != 0:
#             self.conv1 = GATConv(self.x_dim, self.h_dim, edge_dim=self.edge_attr_dim)
#             self.conv2 = GATConv(self.h_dim, self.h_dim, edge_dim=self.edge_attr_dim)
#             # self.conv3 = GATConv(self.h_dim, self.h_dim, edge_dim=self.edge_attr_dim)
#         else:
#             self.conv1 = GCNConv(self.x_dim, self.h_dim)
#             self.conv2 = GCNConv(self.h_dim, self.h_dim)
#             # self.conv3 = GCNConv(self.h_dim, self.h_dim)
#
#
#
#         # ===== encoder: graph_rep + y_cf -> z =====
#         self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + 1, self.z_dim), nn.BatchNorm1d(self.z_dim), nn.ReLU())
#         self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + 1, self.z_dim))
#
#         # ===== decoder：重构整张邻接矩阵（每个图 N_g × N_g）=====
#         # 对每个节点对 (i, j)：
#         #   输入: [h_i, h_j, z_g, y_cf_g] -> 概率 p(A_ij=1)
#         self.decoder_a = nn.Sequential(nn.Linear(2 * self.h_dim + 1, self.h_dim), nn.BatchNorm1d(self.h_dim),
#                                        nn.ReLU(),nn.Dropout(self.dropout),
#                                        nn.Linear(self.h_dim, self.h_dim), nn.BatchNorm1d(self.h_dim),
#                                        nn.ReLU(), nn.Dropout(self.dropout),
#                                        nn.Linear(self.h_dim, 1), nn.Sigmoid())
#
#     def reparameterize(self, mu, logvar):
#         '''
#         compute z = mu + std * epsilon
#         '''
#         if self.training:
#             # compute the standard deviation from logvar
#             std = torch.exp(0.5 * logvar)
#             # sample epsilon from a normal distribution with mean 0 and
#             # variance 1
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
#
#     def graph_pooling(self, node_rep, batch, type='mean'):
#         """
#         node_rep: [num_nodes, h_dim]
#         batch:    [num_nodes]
#         """
#         if type == 'max':
#             return global_max_pool(node_rep, batch)
#         elif type == 'sum':
#             return global_add_pool(node_rep, batch)
#         else:  # mean
#             return global_mean_pool(node_rep, batch)
#
#     def encoder(self, x, edge_index, batch, y_cf, edge_attr=None):
#         # 节点编码
#         if self.edge_attr_dim > 0 and edge_attr is not None:
#             h = self.conv1(x, edge_index, edge_attr)   # GATConv 有 edge_attr
#             h = F.relu(h)
#             node_rep = self.conv2(h, edge_index, edge_attr)
#             # h = F.relu(h)
#             # node_rep = self.conv3(h, edge_index, edge_attr)
#         else:
#             h = self.conv1(x, edge_index)              # GCNConv 没 edge_attr
#             h = F.relu(h)
#             node_rep = self.conv2(h, edge_index)
#             # h = F.relu(h)
#             # node_rep = self.conv3(h, edge_index)
#
#         graph_rep = self.graph_pooling(node_rep, batch, self.graph_pool_type)  # [B, h_dim]
#
#         z_mu = self.encoder_mean(torch.cat([graph_rep, y_cf], dim=-1))
#         z_logvar = self.encoder_var(torch.cat([graph_rep, y_cf], dim=-1))
#
#         return z_mu, z_logvar, node_rep, graph_rep
#
#     def decoder(self, node_rep, batch, z, y_desired):
#         """
#         按图重构完整邻接矩阵（包括原图中不存在的边）
#
#         node_rep: [num_nodes, h_dim]
#         batch:    [num_nodes] (图 id)
#         z:        [B, z_dim]
#         y_cf:     [B, 1]
#
#         返回:
#             adj_recon: [num_nodes, num_nodes]
#             -> block-diagonal，每个图一块 N_g × N_g 概率矩阵
#         """
#         num_nodes = node_rep.size(0)
#         B = z.size(0)
#
#         adj_recon_prob = node_rep.new_zeros(num_nodes, num_nodes)
#
#         # 对每个图 g，枚举该图所有节点对 (i, j)，跑一遍 edge_decoder
#         for g in range(B):
#             idx = (batch == g).nonzero(as_tuple=False).view(-1)  # 该图的所有节点索引
#             if idx.numel() == 0:
#                 continue
#
#             node_embs = node_rep[idx]  # 当前图的节点特征矩阵，[n_g, h_dim]
#             num_nodes = node_embs.size(0)   # 这张图的节点数
#
#             # 构造所有节点对 (i, j)
#             # src_idx: 重复每个 i，长度 num_nodes * num_nodes
#             # dst_idx: 遍历所有 j
#             src_idx = idx.repeat_interleave(num_nodes)  # [num_nodes * num_nodes]
#             dst_idx = idx.repeat(num_nodes)  # [num_nodes * num_nodes]
#
#             node_emb_src = node_rep[src_idx]  # 源节点embedding列表， [num_nodes * num_nodes, h_dim]
#             node_emb_dst = node_rep[dst_idx]  # 目标节点embedding列表，[num_nodes * num_nodes, h_dim]
#
#             z_g = z[g].unsqueeze(0).expand(num_nodes * num_nodes, -1)  # 每一行都是同一个"图embedding"， [num_nodes * num_nodes, z_dim]
#             y_g = y_desired[g].unsqueeze(0).expand(num_nodes * num_nodes, -1)  # 每一行都是同一个"y_desired"，[num_nodes * num_nodes, 1]
#
#             dec_input = torch.cat([node_emb_src, node_emb_dst, y_g], dim=-1)  # [num_nodes * num_nodes, 2*h_dim+z_dim+1]
#             edge_logits = self.decoder_a(dec_input).view(num_nodes, num_nodes)  # [num_nodes, num_nodes]
#
#             # 写回到总的 [num_nodes, num_nodes] 里，形成 block-diagonal
#             adj_recon_prob[idx.unsqueeze(1), idx.unsqueeze(0)] = edge_logits
#
#         return adj_recon_prob
#
#     def get_fs_node_mask(self, node_rep):
#
#         mask_logits = self.fsm(node_rep)        # [num_nodes, 1]
#         fs_node_mask = torch.sigmoid(mask_logits)
#         return fs_node_mask
#
#     def forward(self, x, edge_index, batch, y_desired, edge_attr=None):
#
#         z_mu, z_logvar, node_rep, graph_rep = self.encoder(x, edge_index, batch, y_desired, edge_attr)
#
#         z = self.reparameterize(z_mu, z_logvar)
#
#         # fs_node_mask = self.get_fs_node_mask(node_rep)
#
#         adj_recon = self.decoder(node_rep, batch, z, y_desired)
#
#         return {
#             'adj_recon': adj_recon, # [num_nodes, num_nodes]
#             'z_mu': z_mu, # [B, z_dim]
#             'z_logvar': z_logvar, # [B, z_dim]
#             # 'fs_node_mask': fs_node_mask, # [num_nodes, 1]
#         }
#
#     def compute_loss(self, args, origraphs, subgraphs, gnn, y_desired, outputs):
#         device = args.device
#         adj_recon = outputs['adj_recon'].to(device)  # [N, N]
#         z_mu = outputs['z_mu'].to(device)  # [B, z_dim]
#         z_logvar = outputs['z_logvar'].to(device)  # [B, z_dim]
#         # fs_node_mask = outputs['fs_node_mask'].to(device)  # [N, 1]
#
#         x = origraphs.x  # [N, x_dim]
#         edge_index = origraphs.edge_index  # [2, E]
#         batch = origraphs.batch  # [N]
#         edge_attr = getattr(origraphs, 'edge_attr', None)
#
#         N = x.size(0)
#         B = int(batch.max().item()) + 1
#
#         # ---- 0. loss 权重超参 ----
#         w_recon = getattr(args, 'lambda_recon', 20.0)  # 邻接重构
#         w_mask = getattr(args, 'lambda_mask', 0.0)  # 节点 mask 监督
#         w_edit_in = getattr(args, 'lambda_edit_in', 0.1)  # 子图内部鼓励修改
#         w_edit_out = getattr(args, 'lambda_edit_out', 1.0)  # 子图外部惩罚修改
#         w_edit_in_add = getattr(args, "lambda_edit_in_add", 0.1)
#         w_edit_in_keep = getattr(args, "lambda_edit_in_keep", 1.0)
#         w_cf = getattr(args, 'lambda_cf', 5.0)  # 反事实预测
#         w_kl = getattr(args, 'lambda_kl', 1)  # VAE KL 正则
#
#         # ============================================================
#         # 1. 邻接矩阵重构损失：adj_label vs adj_recon
#         # ============================================================
#         adj_label = torch.zeros(N, N, device=device)
#         row, col = edge_index
#         adj_label[row, col] = 1.0
#         adj_label[col, row] = 1.0
#
#         # 只在同一图内部算重构损失（跨图本来就是 0）
#         intra_mask = batch.unsqueeze(1).eq(batch.unsqueeze(0))  # [N, N] bool
#
#         triu_mask = torch.triu(torch.ones_like(adj_label, dtype=torch.bool), diagonal=1)
#         nonedge_mask = (adj_label == 0) & intra_mask & triu_mask
#
#         adj_pred_intra = adj_recon[intra_mask]
#         adj_label_intra = adj_label[intra_mask]
#
#         bce_raw = F.binary_cross_entropy(
#             adj_pred_intra,  # 预测概率
#             adj_label_intra,  # 0/1 标签
#             reduction='none'  # 不平均，保留每个元素的 loss
#         )  # [M]
#
#         # 2) 使用 Focal Loss 来处理极度不平衡
#         # FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
#         # 其中 p_t = p (if y=1) else (1-p)
#
#         alpha = 0.75  # 正样本的权重（0.5-0.75之间）
#         gamma = 2.0   # focusing参数，越大越关注难样本
#
#         # 计算 p_t: 预测"正确类别"的概率
#         p_t = adj_pred_intra * adj_label_intra + (1 - adj_pred_intra) * (1 - adj_label_intra)
#
#         # Focal weight: (1 - p_t)^gamma
#         focal_weight = (1 - p_t) ** gamma
#
#         # Alpha weight: 正样本用alpha，负样本用(1-alpha)
#         alpha_weight = adj_label_intra * alpha + (1 - adj_label_intra) * (1 - alpha)
#
#         # Focal Loss = alpha * focal_weight * BCE
#         focal_loss = alpha_weight * focal_weight * bce_raw
#
#         recon_loss = focal_loss.mean()
#
#         # 🔍 调试：检查recon_loss和adj_recon的统计
#         if torch.rand(1).item() < 0.01:  # 1%的概率打印，避免刷屏
#             num_pos = (adj_label_intra == 1).sum().item()
#             num_neg = (adj_label_intra == 0).sum().item()
#             print(f"\n[DEBUG recon_loss with Focal Loss]")
#             print(f"  adj_recon: min={adj_recon.min():.4f}, max={adj_recon.max():.4f}, mean={adj_recon.mean():.4f}")
#             print(f"  adj_label: #edges={adj_label.sum().item()}, #non-edges={(1-adj_label).sum().item()}")
#             print(f"  num_pos={num_pos}, num_neg={num_neg}, ratio={num_neg/max(num_pos,1):.2f}")
#             print(f"  focal_loss: min={focal_loss.min():.4f}, max={focal_loss.max():.4f}, mean={focal_loss.mean():.4f}")
#             print(f"  recon_loss={recon_loss.item():.4f}")
#
#
#         # ============================================================
#         # 2. 用 ground-truth 子图 (subgraphs.node_mappings) 做节点监督
#         #    & 子图内/外边编辑区域约束
#         # ============================================================
#
#         # 2.1 节点 ground-truth mask：在频繁子图里的节点 = 1，其它 = 0
#         #    注意：这里 node_gt 是 bool 版，再转成 float 用于 BCE
#         node_gt_bool = torch.zeros(N, dtype=torch.bool, device=device)
#
#         assert len(subgraphs) == B, \
#             f"len(subgraphs)={len(subgraphs)} 与 batch 中图个数 B={B} 不一致"
#
#         # inside_mask：哪个 (i,j) 属于“ground-truth 子图内部区域”
#         inside_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
#
#         for g, sub_g in enumerate(subgraphs):
#             # 当前原图 g 在 batch 里的所有节点索引（全局）
#             orig_nodes_g = (batch == g).nonzero(as_tuple=False).view(-1)  # [n_g]
#
#             if not hasattr(sub_g, "node_mappings"):
#                 continue
#
#             # sub_g.node_mappings: 子图节点在“原图 g 的局部节点编号” (0..n_g-1)
#             local_idx_g = sub_g.node_mappings.to(device)  # [n_sub_g]
#
#             if local_idx_g.numel() == 0:
#                 continue
#
#             # 映射到 Batch 里的全局节点索引
#             global_idx_g = orig_nodes_g[local_idx_g]  # [n_sub_g]
#
#             # 这些节点就是 ground-truth 频繁子图的节点
#             node_gt_bool[global_idx_g] = True
#
#             # 在 (i,j) 级别上，构造这个子图的“内部区域”掩码
#             # 对于这一图中，这些节点之间的所有 (i,j) 都认为属于“子图内部”
#             gi = global_idx_g
#             inside_mask[gi.unsqueeze(1), gi.unsqueeze(0)] = True
#
#         # 强制只在同一原图内部
#         inside_mask = inside_mask & intra_mask
#         outside_mask = (~inside_mask) & intra_mask
#
#         # 2.2 节点 mask 监督：fs_node_mask ≈ node_gt
#         # node_gt = node_gt_bool.float().view(-1, 1)  # [N,1] float
#         # mask_loss = F.binary_cross_entropy(fs_node_mask, node_gt)
#
#         # 2.3 边修改区域约束：鼓励“修改集中在 ground-truth 子图内部”
#         edge_change = torch.abs(adj_recon - adj_label)
#
#         inside_edge_mask = inside_mask & (adj_label == 1)
#         inside_nonedge_mask = inside_mask & (adj_label == 0)
#
#         outside_mask = (~inside_mask) & intra_mask
#
#         # 子图内部：
#         #   对原本没有边(0)的地方，鼓励改动大（即鼓励加边）
#         edit_inside_add = -edge_change[inside_nonedge_mask].mean() if inside_nonedge_mask.any() else 0.0
#
#         #   对原本有边(1)的地方，鼓励改动小（别乱删边）
#         edit_inside_keep = edge_change[inside_edge_mask].mean() if inside_edge_mask.any() else 0.0
#
#         # 子图外部：尽量别乱动
#         edit_outside_loss = edge_change[outside_mask].mean() if outside_mask.any() else 0.0
#
#         edit_loss = (
#                 w_edit_in_add * edit_inside_add +
#                 w_edit_in_keep * edit_inside_keep +
#                 w_edit_out * edit_outside_loss
#         )
#
#         # =========================================================
#         # 3. 反事实预测：用 adj_recon 给边加权，让 GNN 输出接近 y_desired
#         # =========================================================
#
#         # ====== 3. 反事实预测：用 adj_recon 给边加权 + 新增边 ======
#
#         # 超参：每图最多新增 K 条边，新增边阈值
#         cf_add_threshold = getattr(args, "cf_add_threshold", 0.5)
#
#         # 构造“原始边 + 新增边”的反事实图
#         edge_index_cf, edge_weights_cf = self._build_cf_graph(
#             adj_recon=adj_recon,
#             adj_label=adj_label,
#             batch=batch,
#             edge_index=edge_index,
#             threshold=cf_add_threshold,
#         )
#         # print(edge_weights_cf)
#         # 用这张"扩展后的图"做反事实预测
#         cf_probs, cf_logits = gnn.get_pred_explain(
#             x,
#             edge_index_cf,
#             edge_weights_cf,
#             batch,
#         )
#         # print(f"x shape = {x.shape}")
#         # print(f"edge_weights_cf shape = {edge_weights_cf.shape}")
#         # print(f"edge_index shape = {edge_index.shape}")
#         # print(f"batch shape = {batch.shape}")
#
#         y_cf_target = y_desired.to(device).view(-1).long()  # [B], 0/1
#
#         # --- (1) 交叉熵部分：保持原来的 cf 信号 ---
#         cf_ce_loss = F.cross_entropy(cf_logits, y_cf_target)
#
#         # # --- (2) margin 部分：要求目标类 logit 至少比另一类大 margin ---
#         # logits = cf_logits  # [B, 2]
#         # B = logits.size(0)
#         #
#         # # 目标类 logit
#         # logits_t = logits[torch.arange(B, device=device), y_cf_target]  # [B]
#         #
#         # # “对手”类 logit（BA2Motif 是二分类，直接 1 - y）
#         # other_class = 1 - y_cf_target
#         # logits_o = logits[torch.arange(B, device=device), other_class]  # [B]
#         #
#         # # margin 超参数（可以放 args 里）
#         # margin = getattr(args, "cf_margin", 0.8)   # 比如 0.3~1.0 之间试
#         # alpha_margin = getattr(args, "lambda_cf_margin", 1.0)  # margin loss 的权重
#         #
#         # # hinge-style：max(0, margin + logit_other - logit_target)
#         # cf_margin_loss = F.relu(margin + logits_o - logits_t).mean()
#
#         # --- (3) 对“新增边 >0.5 的数量”做 soft 约束 =====
#         lambda_cf_budget_add = getattr(args, "lambda_cf_budget_add", 0.1)
#         lambda_cf_budget_del = getattr(args, "lambda_cf_budget_del", 0.1)
#         K_max_add = getattr(args, "cf_max_added_edges", 10)
#         K_max_del = getattr(args, "cf_max_deleted_edges", 10)
#         tau = getattr(args, "cf_thresh_temperature", 0.1)
#
#         budget_add_losses = []
#         budget_del_losses = []
#
#         for g in range(B):
#             idx = (batch == g).nonzero(as_tuple=False).view(-1)
#             if idx.numel() == 0:
#                 continue
#
#             p_g = adj_recon[idx][:, idx]
#             lab_g = adj_label[idx][:, idx]
#             n_g = idx.size(0)
#
#             triu_mask = torch.triu(torch.ones(n_g, n_g, dtype=torch.bool, device=device), diagonal=1)
#
#             # ------- 新增边预算（原本没边） -------
#             cand_add_mask = (lab_g == 0) & triu_mask
#             if cand_add_mask.any():
#                 p_add = p_g[cand_add_mask]
#                 ind_add = torch.sigmoid((p_add - 0.5) / tau)
#                 num_add_soft = ind_add.sum()
#                 budget_add_losses.append(F.relu(num_add_soft - K_max_add))
#
#             # ------- 删边预算（原本有边） -------
#             cand_del_mask = (lab_g == 1) & triu_mask
#             if cand_del_mask.any():
#                 p_edge = p_g[cand_del_mask]
#                 # p_edge < 0.5 视为“删边倾向”
#                 ind_del = torch.sigmoid((0.5 - p_edge) / tau)
#                 num_del_soft = ind_del.sum()
#                 budget_del_losses.append(F.relu(num_del_soft - K_max_del))
#
#         cf_budget_add_loss = torch.stack(budget_add_losses).mean() if budget_add_losses else adj_recon.new_tensor(0.0)
#         cf_budget_del_loss = torch.stack(budget_del_losses).mean() if budget_del_losses else adj_recon.new_tensor(0.0)
#
#         cf_budget_loss = (lambda_cf_budget_add * cf_budget_add_loss +
#                           lambda_cf_budget_del * cf_budget_del_loss)
#
#         cf_loss = cf_ce_loss + cf_budget_loss
#
#         # =========================================================
#         # 4. VAE KL 正则
#         # =========================================================
#         kl_loss = -0.5 * torch.mean(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#
#         # =========================================================
#         # 5. 总损失
#         # =========================================================
#         total_loss = (
#                 w_recon * recon_loss +
#                 # w_mask * mask_loss +
#                 edit_loss +
#                 w_cf * cf_loss +
#                 w_kl * kl_loss
#         )
#
#         return {
#             "total": total_loss,
#             "recon": recon_loss.detach(),
#             # "mask": mask_loss.detach(),
#             "edit_inside": (w_edit_in_add * edit_inside_add + w_edit_in_keep * edit_inside_keep).detach(),
#             "edit_outside": edit_outside_loss.detach(),
#             "cf": cf_loss.detach(),
#             "kl": kl_loss.detach(),
#         }
#
#     def _build_cf_graph(self, adj_recon, adj_label, batch, edge_index,
#                         K=None, threshold=0.5):
#         """
#         根据 adj_recon 构造反事实图：
#         - 原始边：来自 edge_index，边权 = adj_recon[row, col]
#         - 新增边：在原来不存在的边中，从 adj_recon 里挑分数高的，
#                   过滤阈值 threshold。
#           ⚠️ 注意：这里的 K 是「新增有向边数」的上限，
#                    因为无向边在 PyG 中会加成 (i,j) 和 (j,i) 两条。
#
#         返回:
#             edge_index_cf: [2, E_orig + E_add]
#             edge_weight_cf: [E_orig + E_add]
#         """
#         device = adj_recon.device
#         row, col = edge_index
#
#         # 原来的边：直接用 adj_recon 上对应位置当权重
#         edge_weight_orig = adj_recon[row, col]  # [E_orig]
#
#         B = int(batch.max().item()) + 1
#
#         row_add_all = []
#         col_add_all = []
#         w_add_all = []
#
#         for g in range(B):
#             # 本图所有节点的全局索引
#             idx = (batch == g).nonzero(as_tuple=False).view(-1)
#             if idx.numel() == 0:
#                 continue
#
#             # 取本图的局部邻接矩阵
#             adj_rec_g = adj_recon[idx][:, idx]  # [n_g, n_g]
#             adj_lab_g = adj_label[idx][:, idx]  # [n_g, n_g]
#             n_g = idx.size(0)
#
#             # 只考虑“原来没边”的地方，并且只看上三角，防止重复
#             triu_mask = torch.triu(
#                 torch.ones(n_g, n_g, dtype=torch.bool, device=device),
#                 diagonal=1
#             )
#             cand_mask = (adj_lab_g == 0) & triu_mask  # 候选新增边（无向）
#
#             if not cand_mask.any():
#                 continue
#
#             # 候选边得分、坐标（无向边）
#             cand_scores = adj_rec_g[cand_mask]  # [M_g]
#             cand_idx = cand_mask.nonzero(as_tuple=False)  # [M_g, 2], (i_local, j_local)
#
#             # 先按阈值过滤一轮
#             if threshold is not None:
#                 keep = cand_scores > threshold
#                 if keep.sum() == 0:
#                     continue
#                 cand_scores = cand_scores[keep]
#                 cand_idx = cand_idx[keep]
#
#             # ===== 关键修改：利用 K，控制「新增有向边数」上限 =====
#             if K is not None:
#                 # 每条无向边会变成两条有向边，所以允许的无向边数:
#                 max_pairs = K // 2  # floor(K/2)
#                 if max_pairs <= 0:
#                     # 这个图就不加边了
#                     continue
#
#                 # 按得分做 top-K（这里是对“无向边对数”做 top-K）
#                 if cand_scores.numel() > max_pairs:
#                     topv, topind = torch.topk(cand_scores, max_pairs)
#                     cand_scores = topv
#                     cand_idx = cand_idx[topind]
#             # ====================================================
#
#             if cand_scores.numel() == 0:
#                 continue
#
#             row_local = cand_idx[:, 0]
#             col_local = cand_idx[:, 1]
#
#             # 映射回全局索引
#             src = idx[row_local]  # [m]
#             dst = idx[col_local]  # [m]
#
#             # 无向图：加双向边 (i,j) & (j,i)
#             row_add = torch.cat([src, dst], dim=0)  # 2m 条有向边
#             col_add = torch.cat([dst, src], dim=0)
#             w_add = torch.cat([cand_scores, cand_scores], dim=0)
#
#             row_add_all.append(row_add)
#             col_add_all.append(col_add)
#             w_add_all.append(w_add)
#
#         if len(row_add_all) > 0:
#             row_add_all = torch.cat(row_add_all, dim=0)
#             col_add_all = torch.cat(col_add_all, dim=0)
#             w_add_all = torch.cat(w_add_all, dim=0)
#
#             edge_index_add = torch.stack([row_add_all, col_add_all], dim=0)  # [2, E_add]
#
#             # 拼接“原始边 + 新增边”
#             edge_index_cf = torch.cat([edge_index, edge_index_add], dim=1)  # [2, E_orig+E_add]
#             edge_weight_cf = torch.cat([edge_weight_orig, w_add_all], dim=0)  # [E_orig+E_add]
#         else:
#             # 没有新增边，就和原图一样
#             edge_index_cf = edge_index
#             edge_weight_cf = edge_weight_orig
#
#         return edge_index_cf, edge_weight_cf

class MyExplainerV2(nn.Module):
    def __init__(self, args, gnn):
        super().__init__()
        self.gnn = gnn
        self.x_dim = args.x_dim
        self.h_dim = args.h_dim
        self.z_dim = args.z_dim # VGAE 的 latent 维度
        self.device = args.device

        # -------- 1. encoder：简单 GCN，当然你可以换成你的 conv1/2/3 --------
        self.conv1 = GCNConv(self.x_dim, self.h_dim)
        self.conv2 = GCNConv(self.h_dim, self.h_dim)

        # -------- 2. DeleteHead：在已有边上预测 p_keep --------
        self.delete_head = nn.Sequential(
            nn.Linear(2 * self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, 1)
        )

        # -------- 3. AddHead = VGAE：在 FS 内做节点级 VAE --------
        # encoder: node_rep -> μ, logvar
        self.vgae_mu = nn.Linear(self.h_dim, self.z_dim)
        self.vgae_logvar = nn.Linear(self.h_dim, self.z_dim)
        # decoder: 内积解码，不额外参数

        # # 可选：节点 mask 模块
        # self.fsm = FrequentSubgraphMiner(self.h_dim)

    # ================= encoder & FS 节点集合 =================

    def encode_nodes(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        return h  # [N, h_dim]

    # def get_fs_node_mask(self, node_rep):
    #     logits = self.fsm(node_rep)      # [N,1]
    #     mask = torch.sigmoid(logits).view(-1)
    #     return mask

    def get_fs_nodes_bool_from_subgraphs(self, batch, subgraphs, N):
        """
        用 ground-truth 频繁子图：subgraphs[g].node_mappings
        构造 bool 向量 fs_nodes_bool
        """
        device = batch.device
        fs_nodes_bool = torch.zeros(N, dtype=torch.bool, device=device)
        B = int(batch.max().item()) + 1
        assert len(subgraphs) == B

        for g, sub_g in enumerate(subgraphs):
            idx_g = (batch == g).nonzero(as_tuple=False).view(-1)
            if not hasattr(sub_g, "node_mappings"):
                continue
            local_idx = sub_g.node_mappings.to(device)
            if local_idx.numel() == 0:
                continue
            global_idx = idx_g[local_idx]
            fs_nodes_bool[global_idx] = True

        return fs_nodes_bool

    # ================= DeleteHead：已有边的 keep 概率 =================

    def compute_p_keep(self, node_rep, edge_index):
        src, dst = edge_index
        e_feat = torch.cat([node_rep[src], node_rep[dst]], dim=-1)  # [E, 2h]
        logit_keep = self.delete_head(e_feat).view(-1)              # [E]
        p_keep = torch.sigmoid(logit_keep)
        return p_keep, logit_keep

    # ================= AddHead = VGAE（在 FS 内） =================

    def vgae_reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def run_fs_vgae(
        self, node_rep, edge_index, batch, fs_nodes_bool,
        max_cand_per_graph=None
    ):
        """
        在每个图 g 的 FS 节点集合 S_g 上跑一个小 VGAE：
          - 输入：node_rep[S_g]
          - encoder: μ, logvar
          - decoder: 内积 -> p_ij
          - 监督：recon A_fs (FS 内真实邻接)
          - 输出：
              cand_src, cand_dst（全局 idx）
              p_add（对应候选加边概率）
              add_recon_loss, add_kl_loss
        """
        device = node_rep.device
        row, col = edge_index
        N = node_rep.size(0)
        B = int(batch.max().item()) + 1

        # 构造全局邻接，方便 FS 内取子矩阵
        adj_global = torch.zeros(N, N, device=device)
        adj_global[row, col] = 1
        adj_global[col, row] = 1

        cand_src_all = []
        cand_dst_all = []
        p_add_all = []

        recon_losses = []
        kl_losses = []

        for g in range(B):
            idx_g = (batch == g).nonzero(as_tuple=False).view(-1)
            if idx_g.numel() == 0:
                continue

            fs_idx_g = idx_g[fs_nodes_bool[idx_g]]
            n_fg = fs_idx_g.size(0)
            if n_fg <= 1:
                continue

            # 1) 取 FS 节点的表示和子图邻接
            h_g = node_rep[fs_idx_g]                   # [n_fg, h_dim]
            A_g = adj_global[fs_idx_g][:, fs_idx_g]    # [n_fg, n_fg]

            # 2) VGAE encoder: μ, logvar
            mu_g = self.vgae_mu(h_g)                   # [n_fg, z_dim]
            logvar_g = self.vgae_logvar(h_g)           # [n_fg, z_dim]
            z_g = self.vgae_reparameterize(mu_g, logvar_g)  # [n_fg, z_dim]

            # 3) VGAE decoder: 内积解码，得到完整邻接概率矩阵
            #    logits: [n_fg, n_fg]
            logits_A = torch.matmul(z_g, z_g.t())
            prob_A = torch.sigmoid(logits_A)

            # 4) 重构损失 (简单版本：对上三角所有位置做 BCE)
            triu_mask = torch.triu(torch.ones_like(A_g, dtype=torch.bool), diagonal=1)
            A_pos = A_g[triu_mask]             # [M_triu]
            P_pos = prob_A[triu_mask]         # [M_triu]

            # class imbalance 可能很严重，可以加 pos_weight
            if A_pos.numel() > 0:
                # 统计正负样本
                num_pos = (A_pos == 1).sum().item()
                num_neg = (A_pos == 0).sum().item()
                if num_pos == 0:
                    pos_weight = 1.0
                else:
                    pos_weight = num_neg / max(num_pos, 1)
                weight = torch.ones_like(A_pos)
                weight[A_pos == 1] = pos_weight

                bce_raw = F.binary_cross_entropy(P_pos, A_pos, reduction="none")
                recon_loss_g = (bce_raw * weight).mean()
            else:
                recon_loss_g = torch.tensor(0.0, device=device)

            # 5) KL loss（节点级 VAE）
            kl_g = -0.5 * torch.mean(1 + logvar_g - mu_g.pow(2) - logvar_g.exp())

            recon_losses.append(recon_loss_g)
            kl_losses.append(kl_g)

            # 6) 从 FS 内部的非边里提取候选加边
            cand_mask = (A_g == 0) & triu_mask   # 非边 & 上三角
            if not cand_mask.any():
                continue

            scores = prob_A[cand_mask]                  # [M_cand]
            cand_idx = cand_mask.nonzero(as_tuple=False)  # [M_cand, 2]

            # ★ 第一步：按阈值筛选，只保留 scores > 0.5 的候选边
            keep = scores > 0.5
            if keep.sum() == 0:
                # 没有任何>0.5的非边，不在这张图中添加边
                continue

            scores = scores[keep]  # 只剩下 >0.5 的
            cand_idx = cand_idx[keep]  # 对应的 (i,j)

            # ★ 第二步：如果候选还很多，再按分数取 top-K（最多 max_cand_per_graph 条）
            if max_cand_per_graph is not None and scores.numel() > max_cand_per_graph:
                topv, topind = torch.topk(scores, max_cand_per_graph)
                scores = topv
                cand_idx = cand_idx[topind]

            # 映射回全局 index
            row_local = cand_idx[:, 0]
            col_local = cand_idx[:, 1]
            src_global = fs_idx_g[row_local]   # [m]
            dst_global = fs_idx_g[col_local]   # [m]

            cand_src_all.append(src_global)
            cand_dst_all.append(dst_global)
            p_add_all.append(scores)

        # 聚合所有图的结果
        if len(recon_losses) > 0:
            add_recon_loss = torch.stack(recon_losses).mean()
            add_kl_loss = torch.stack(kl_losses).mean()
        else:
            add_recon_loss = node_rep.new_tensor(0.0)
            add_kl_loss = node_rep.new_tensor(0.0)

        if len(cand_src_all) > 0:
            cand_src_all = torch.cat(cand_src_all, dim=0)
            cand_dst_all = torch.cat(cand_dst_all, dim=0)
            p_add_all = torch.cat(p_add_all, dim=0)
        else:
            cand_src_all = None
            cand_dst_all = None
            p_add_all = None

        return cand_src_all, cand_dst_all, p_add_all, add_recon_loss, add_kl_loss

    # ================= CF 图构造 =================

    def build_cf_graph_soft(self, edge_index, p_keep, cand_src, cand_dst, p_add):
        """
        已有边权重 = p_keep
        新增边权重 = p_add
        """
        edge_index_del = edge_index
        edge_weight_del = p_keep

        if cand_src is None or p_add is None or p_add.numel() == 0:
            return edge_index_del, edge_weight_del

        edge_index_add = torch.stack([cand_src, cand_dst], dim=0)
        edge_weight_add = p_add

        edge_index_cf = torch.cat([edge_index_del, edge_index_add], dim=1)
        edge_weight_cf = torch.cat([edge_weight_del, edge_weight_add], dim=0)
        return edge_index_cf, edge_weight_cf

    # ================= forward：构造 CF 图 =================

    def forward(self, graphs, subgraphs=None, use_subgraph_gt=True,
                max_cand_per_graph=5):

        x = graphs.x
        edge_index = graphs.edge_index
        batch = graphs.batch
        N = x.size(0)

        # 1) 节点编码
        node_rep = self.encode_nodes(x, edge_index)

        # # 2) 节点级 mask（可选）
        # fs_node_mask = self.get_fs_node_mask(node_rep)

        # 3) 频繁子图节点集合
        if use_subgraph_gt and subgraphs is not None:
            fs_nodes_bool = self.get_fs_nodes_bool_from_subgraphs(batch, subgraphs, N)
        else:
            tau_node = 0.5
            # fs_nodes_bool = (fs_node_mask > tau_node)

        # 4) DeleteHead：已有边的 p_keep
        p_keep, logit_keep = self.compute_p_keep(node_rep, edge_index)
        # if torch.rand(1).item()<0.1:
        #     print(p_keep)

        # 5) AddHead(VGAE)：FS 内部非边的 p_add + VGAE losses
        cand_src, cand_dst, p_add, add_recon_loss, add_kl_loss = self.run_fs_vgae(
            node_rep, edge_index, batch, fs_nodes_bool,
            max_cand_per_graph=max_cand_per_graph
        )

        # 6) CF 图
        edge_index_cf, edge_weight_cf = self.build_cf_graph_soft(
            edge_index, p_keep, cand_src, cand_dst, p_add
        )

        return {
            "node_rep": node_rep,
            # "fs_node_mask": fs_node_mask,
            "fs_nodes_bool": fs_nodes_bool,
            "p_keep": p_keep,
            "logit_keep": logit_keep,
            "cand_src": cand_src,
            "cand_dst": cand_dst,
            "p_add": p_add,
            "add_recon_loss": add_recon_loss,
            "add_kl_loss": add_kl_loss,
            "edge_index_cf": edge_index_cf,
            "edge_weight_cf": edge_weight_cf,
        }

    # ================= compute_loss：CF + VGAE + 预算 =================

    def compute_loss(self, args, graphs, y_desired, outputs):
        device = self.device

        x = graphs.x
        batch = graphs.batch

        edge_index_cf = outputs["edge_index_cf"]
        edge_weight_cf = outputs["edge_weight_cf"]
        p_keep = outputs["p_keep"]
        p_add = outputs["p_add"]

        add_recon_loss = outputs["add_recon_loss"]
        add_kl_loss = outputs["add_kl_loss"]

        # ---- 1. CF 主损失 ----
        cf_probs, cf_logits = self.gnn.get_pred_explain(
            x, edge_index_cf, edge_weight_cf, batch
        )
        y_target = y_desired.to(device).view(-1).long()

        cf_ce = F.cross_entropy(cf_logits, y_target)

        margin = getattr(args, "cf_margin", 0.5)
        lambda_cf_margin = getattr(args, "lambda_cf_margin", 1.0)

        logits = cf_logits
        B = logits.size(0)
        logits_t = logits[torch.arange(B), y_target]
        other = 1 - y_target
        logits_o = logits[torch.arange(B), other]

        cf_margin = F.relu(margin + logits_o - logits_t).mean()
        cf_main_loss = cf_ce + lambda_cf_margin * cf_margin

        # ---- 2. 加/删边预算 ----
        tau = getattr(args, "cf_tau", 0.1)

        # add budget
        K_add = getattr(args, "cf_max_added_edges", 5)
        if p_add is not None and p_add.numel() > 0:
            indicator_add = torch.sigmoid((p_add - 0.5) / tau)
            num_add_soft = indicator_add.sum()
            add_budget_loss = F.relu(num_add_soft - K_add)
        else:
            add_budget_loss = cf_main_loss.new_tensor(0.0)

        # delete budget
        K_del = getattr(args, "cf_max_deleted_edges", 5)
        indicator_del = torch.sigmoid((0.5 - p_keep) / tau)
        num_del_soft = indicator_del.sum()
        del_budget_loss = F.relu(num_del_soft - K_del)

        # sparsity 正则
        l1_add = p_add.mean() if p_add is not None and p_add.numel() > 0 else 0.0
        l1_del = (1 - p_keep).mean()

        # ---- 3. VGAE loss 权重 ----
        w_cf = getattr(args, "w_cf", 5.0)
        w_add_budget = getattr(args, "w_add_budget", 0.1)
        w_del_budget = getattr(args, "w_del_budget", 0.0)
        w_l1_add = getattr(args, "w_l1_add", 0.1)
        w_l1_del = getattr(args, "w_l1_del", 0.5)

        w_vgae_recon = getattr(args, "w_vgae_recon", 5.0)
        w_vgae_kl = getattr(args, "w_vgae_kl", 1)

        total_loss = (
            w_cf * cf_main_loss +
                # w_add_budget * add_budget_loss +
                # w_del_budget * del_budget_loss +
            w_l1_add * l1_add +
            w_l1_del * l1_del +
            w_vgae_recon * add_recon_loss +
            w_vgae_kl * add_kl_loss
        )

        return {
            "total": total_loss,
            "cf": cf_main_loss.detach(),
            # "cf_ce": cf_ce.detach(),
            # "cf_margin": cf_margin.detach(),
            # "add_budget": add_budget_loss.detach(),
            # "del_budget": del_budget_loss.detach(),
            # "l1_add": (l1_add if isinstance(l1_add, torch.Tensor) else torch.tensor(l1_add)).detach(),
            # "l1_del": l1_del.detach(),
            "recon": add_recon_loss.detach(),
            "kl": add_kl_loss.detach(),
        }






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