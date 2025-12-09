import os
import time
import random
import numpy as np
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch_geometric.loader import DataLoader
from torch_geometric.nn import DenseGCNConv

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import to_dense_adj
from tqdm import tqdm

from utils import get_datasets
from gnns import *


##############################################
# 辅助函数：稠密格式 → PyG 格式 → GNN 预测
##############################################

def predict_with_dense_format(
    pred_model: nn.Module,
    feat: torch.Tensor,
    adj: torch.Tensor,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用稠密格式的数据调用 PyG 格式的 GNN。

    Args:
        pred_model: GNN 分类器（期望 PyG 格式：x, edge_index, batch）
        feat: [B, N, x_dim] 稠密节点特征
        adj: [B, N, N] 稠密邻接矩阵
        device: 设备

    Returns:
        probs: [B, num_classes] softmax 概率
        logits: [B, num_classes] 原始 logits
    """
    from torch_geometric.utils import dense_to_sparse

    feat = feat.to(device)
    adj = adj.to(device)

    B = feat.size(0)
    N = feat.size(1)

    # 展平为 PyG 格式
    x_sparse = feat.view(B * N, -1)  # [B*N, x_dim]

    # 转换邻接矩阵为 edge_index
    edge_index_list = []
    batch_vec_list = []
    offset = 0

    for b in range(B):
        adj_b = adj[b]  # [N, N]
        # 二值化
        adj_binary = (adj_b > 0.5).float()
        edge_index_b, _ = dense_to_sparse(adj_binary)  # [2, E_b]

        edge_index_list.append(edge_index_b + offset)
        batch_vec_list.extend([b] * N)
        offset += N

    if len(edge_index_list) > 0:
        edge_index = torch.cat(edge_index_list, dim=1).to(device)  # [2, total_edges]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    batch_vec = torch.tensor(batch_vec_list, dtype=torch.long, device=device)  # [B*N]

    # 调用 GNN
    # ✅ 修复：forward 直接返回 logits [B, num_classes]，不要加 [1]
    logits = pred_model(x_sparse, edge_index, batch_vec)  # [B, num_classes]
    probs = F.softmax(logits, dim=-1)

    return probs, logits


##############################################
# 1. GraphCFE 模型
##############################################

class GraphCFE(nn.Module):
    def __init__(
        self,
        pred_model: nn.Module,
        x_dim: int,
        edge_attr_dim: int,
        h_dim: int,
        z_dim: int,
        max_num_nodes: int,
        dropout: float,
        device: str
    ):
        super(GraphCFE, self).__init__()
        self.pred_model = pred_model
        self.x_dim = x_dim
        self.edge_attr_dim = edge_attr_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.max_num_nodes = max_num_nodes
        self.dropout = dropout
        self.device = device

        # GCN 图编码层（Dense 格式）
        self.graph_model = DenseGCNConv(x_dim, h_dim)

        # encoder: 输出均值和 log 方差
        self.encoder_mean = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU()
        )
        # 用更标准的 logvar，直接 Linear 即可（不再用 Sigmoid）
        self.encoder_logvar = nn.Linear(h_dim, z_dim)

        # 用于图级表示的 BN
        self.graph_norm = nn.BatchNorm1d(h_dim)

        # decoder for node features
        self.decoder_x = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_dim, max_num_nodes * x_dim)
        )

        # decoder for adjacency (upper triangle vector)
        self.decoder_a = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_dim, int(max_num_nodes * (max_num_nodes - 1) / 2)),
            nn.Sigmoid()
        )

        # decoder for edge attributes（可选，目前不参与 loss，仅保留结构）
        if edge_attr_dim != 0:
            self.decoder_edge_attr = nn.Sequential(
                nn.Linear(z_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(h_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(
                    h_dim,
                    int((max_num_nodes - 1) * max_num_nodes / 2) * edge_attr_dim
                )
            )
        else:
            self.decoder_edge_attr = None

        self.initialize_parameters()

    def initialize_parameters(self):
        """参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def encoder(self, features: torch.Tensor, adj: torch.Tensor):
        """
        features: [B, N, x_dim]
        adj:      [B, N, N]
        """
        # DenseGCNConv: (x: [B, N, F], adj: [B, N, N]) -> [B, N, h_dim]
        graph_rep = self.graph_model(features, adj)      # [B, N, h_dim]
        graph_rep = torch.sum(graph_rep, dim=1)          # [B, h_dim]
        graph_rep = self.graph_norm(graph_rep)           # [B, h_dim]

        z_mu = self.encoder_mean(graph_rep)              # [B, z_dim]
        z_logvar = self.encoder_logvar(graph_rep)        # [B, z_dim] 真实 logvar
        return z_mu, z_logvar

    def convert_to_symmetric_tensor(self, num_nodes: int, adj_vec: torch.Tensor):
        """
        adj_vec: [B, num_edges], num_edges = N*(N-1)/2 (上三角不含对角线)
        返回对称矩阵 [B, N, N]
        """
        upper_triangular = torch.zeros(
            (adj_vec.shape[0], num_nodes, num_nodes),
            device=self.device
        )
        # 上三角索引（offset=1 表示不包含对角线）
        mask = torch.triu_indices(num_nodes, num_nodes, offset=1).to(self.device)
        upper_triangular[:, mask[0], mask[1]] = adj_vec
        # 对称
        symm = upper_triangular + torch.transpose(upper_triangular, 1, 2)
        return symm

    def decoder(self, z: torch.Tensor):
        """
        z: [B, z_dim]
        """
        # 邻接
        adj_reconst_half = self.decoder_a(z)      # [B, num_edges]
        adj_reconst = self.convert_to_symmetric_tensor(
            self.max_num_nodes, adj_reconst_half
        )                                         # [B, N, N]

        # 节点特征
        features_reconst = self.decoder_x(z).view(
            -1, self.max_num_nodes, self.x_dim
        )                                         # [B, N, x_dim]

        # 边属性（目前没有用于 loss，仅返回）
        if self.decoder_edge_attr is not None:
            edge_attrs_reconst = self.decoder_edge_attr(z).view(
                -1,
                int(self.max_num_nodes * (self.max_num_nodes - 1) / 2),
                self.edge_attr_dim
            )
        else:
            edge_attrs_reconst = None

        return features_reconst, adj_reconst, edge_attrs_reconst

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        z = mu + std * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        # encoder
        z_mu, z_logvar = self.encoder(features, adj)
        # 采样
        z_sample = self.reparameterize(z_mu, z_logvar)
        # decoder
        features_reconst, adj_reconst, edge_attrs_reconst = self.decoder(z_sample)
        return {
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'adj_reconst': adj_reconst,
            'feat_reconst': features_reconst,
            'edge_attr_reconst': edge_attrs_reconst
        }

    def loss(
        self,
        feat: torch.Tensor,
        adj: torch.Tensor,
        explainer_output: Dict[str, torch.Tensor],
        cf_label: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        feat: 原始节点特征 [B, N, x_dim]
        adj:  原始邻接        [B, N, N]
        cf_label: 目标反事实标签 [B]，目前假定为二分类中的目标类（0/1）
        """
        B = feat.shape[0]

        # ========== 1) KL loss（标准 VAE KL）==========
        z_mu = explainer_output['z_mu']          # [B, z_dim]
        z_logvar = explainer_output['z_logvar']  # [B, z_dim]

        # KL( N(mu, sigma^2) || N(0, I) )
        # = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        loss_kl = -0.5 * torch.sum(
            1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar),
            dim=1
        ).mean()

        # ========== 2) similarity loss ==========
        # 特征距离（对所有实际节点 & pad 节点统一处理）
        dist_x = F.pairwise_distance(
            feat.view(B, -1),
            explainer_output['feat_reconst'].view(B, -1),
            p=2
        ).mean()
        # 邻接 BCE
        dist_a = F.binary_cross_entropy(
            explainer_output['adj_reconst'],
            adj
        )
        loss_sim = 0.0 * dist_x + 10.0 * dist_a

        # ========== 3) CFE loss（用 pred_model 在 CF 图上的预测）==========
        from torch_geometric.utils import dense_to_sparse

        feat_reconst = explainer_output['feat_reconst']  # [B, N, x_dim]
        adj_reconst = explainer_output['adj_reconst']  # [B, N, N]

        B = feat_reconst.size(0)
        N = feat_reconst.size(1)

        # [B*N, x_dim]
        x_sparse = feat_reconst.view(B * N, -1)

        edge_index_list = []
        edge_weight_list = []
        batch_vec_list = []
        offset = 0

        for b in range(B):
            adj_b = adj_reconst[b]  # [N, N]，范围在 (0,1)（decoder_a 最后一层是 Sigmoid）
            # 不再硬阈值，直接当成连续权重
            edge_index_b, edge_weight_b = dense_to_sparse(adj_b)  # [2, E_b], [E_b]

            edge_index_list.append(edge_index_b + offset)
            edge_weight_list.append(edge_weight_b)
            batch_vec_list.extend([b] * N)
            offset += N

        if len(edge_index_list) > 0:
            edge_index = torch.cat(edge_index_list, dim=1).to(self.device)
            edge_weight = torch.cat(edge_weight_list, dim=0).to(self.device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            edge_weight = torch.empty((0,), dtype=torch.float, device=self.device)

        batch_vec = torch.tensor(batch_vec_list, dtype=torch.long, device=self.device)  # [B*N]

        # 使用 get_pred_explain，让 edge_weight 作为 edge_mask 参与预测
        # ✅ 参照 CF-GNNExplainer：使用位置参数，edge_weight 已在 [0,1] 范围，无需设置 mask_is_logit
        probs_cf, y_pred_logits = self.pred_model.get_pred_explain(
            x_sparse.to(self.device),
            edge_index,
            edge_weight,  # 第3个位置参数：edge_mask（范围 [0,1]，来自 sigmoid 后的 adj_reconst）
            batch_vec     # 第4个位置参数：batch
        )

        loss_cfe = F.nll_loss(
            F.log_softmax(y_pred_logits, dim=-1),
            cf_label.view(-1).long()
        )

        loss = loss_sim + loss_kl + loss_cfe
        loss_results = {
            'loss': loss,
            'loss_kl': loss_kl,
            'loss_sim': loss_sim,
            'loss_cfe': loss_cfe
        }
        return loss_results

    def run_one_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from torch_geometric.utils import to_dense_batch

        # PyG Batch 格式：x is [B*N, F], edge_index is [2, E], batch is [B*N]
        x_sparse = batch['x'].to(self.device)   # [B*N, F]
        edge_index = batch['edge_index'].to(self.device)
        batch_vec = batch['batch'].to(self.device)

        # 转换为稠密格式
        x_dense, node_mask = to_dense_batch(
            x_sparse, batch_vec, max_num_nodes=self.max_num_nodes
        )  # [B, N, F]
        adj = to_dense_adj(
            edge_index=edge_index, batch=batch_vec, max_num_nodes=self.max_num_nodes
        )  # [B, N, N]

        # 使用 PyG 格式计算原始预测（这里只用来生成目标标签，不需要梯度）
        with torch.no_grad():
            ori_logits = self.pred_model(x_sparse, edge_index, batch_vec)  # [B, num_classes]
            ori_pred = ori_logits.argmax(dim=-1)                           # [B]

        # 这里假定是二分类任务：目标类 = 1 - 原预测（0↔1 翻转）
        y_desired = 1 - ori_pred

        # 使用稠密格式传给 explainer
        explainer_output = self(x_dense, adj)
        loss = self.loss(x_dense, adj, explainer_output, y_desired)
        return loss


##############################################
# 2. 训练 & 验证用的辅助函数
##############################################


def train_explainer_inner(
    epochs: int,
    model: GraphCFE,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    model_path: str
) -> GraphCFE:
    print("Start training GraphCFE explainer...")
    time_begin = time.time()
    best_loss, best_model_dict = 1e10, None

    for epoch in tqdm(range(epochs)):
        random.seed(42 + epoch)
        model.train()

        tot_loss_epoch = 0.0
        loss_kl_epoch = 0.0
        loss_sim_epoch = 0.0
        loss_cfe_epoch = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = model.run_one_batch(batch)
            loss_batch = batch_loss['loss']
            loss_kl_batch = batch_loss['loss_kl']
            loss_sim_batch = batch_loss['loss_sim']
            loss_cfe_batch = batch_loss['loss_cfe']

            # 关键：对每个 batch 单独 backward
            if epoch < 70:
                loss_cfe_batch.backward()
            else:
                loss_batch.backward()
            optimizer.step()

            # 统计损失（只做 logging）
            tot_loss_epoch += loss_batch.item() / len(train_loader)
            loss_kl_epoch += loss_kl_batch.item() / len(train_loader)
            loss_sim_epoch += loss_sim_batch.item() / len(train_loader)
            loss_cfe_epoch += loss_cfe_batch.item() / len(train_loader)

        # 保存 best model（这里用总 loss）
        if tot_loss_epoch < best_loss:
            best_loss = tot_loss_epoch
            best_model_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        # 你可以在这里 print 一下
        print(
            f"Epoch {epoch:03d} | "
            f"loss: {tot_loss_epoch:.4f} | "
            f"KL: {loss_kl_epoch:.4f} | "
            f"sim: {loss_sim_epoch:.4f} | "
            f"cfe: {loss_cfe_epoch:.4f}"
        )

    if best_model_dict is not None:
        torch.save(best_model_dict, model_path)
        model.load_state_dict(best_model_dict)

    print(f"Training done in {(time.time() - time_begin):.1f}s, best loss={best_loss:.4f}")
    return model


##############################################
# 3. 对外接口：在“训练集”训练 GraphCFE
##############################################

def train_graphcfe(
    pred_model: nn.Module,
    train_dataset: InMemoryDataset,
    epochs: int,
    device: str,
    lr: float,
    model_path: str,
    max_num_nodes: int
) -> GraphCFE:
    """
    只在 train_dataset 上训练 GraphCFE explainer
    """
    # 冻结预测模型参数，只作为固定黑盒
    pred_model.eval()
    for p in pred_model.parameters():
        p.requires_grad = False

    explainer = GraphCFE(
        pred_model=pred_model,
        x_dim=train_dataset[0].x.size(1),
        edge_attr_dim=(
            train_dataset[0].edge_attr.size(1)
            if hasattr(train_dataset[0], "edge_attr")
            and train_dataset[0].edge_attr is not None
            else 0
        ),
        h_dim=16,
        z_dim=16,
        max_num_nodes=max_num_nodes,
        dropout=0.1,
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(
        explainer.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True  # 训练集建议打乱
    )

    explainer = train_explainer_inner(
        epochs=epochs,
        model=explainer,
        optimizer=optimizer,
        train_loader=train_loader,
        model_path=model_path
    )
    return explainer


##############################################
# 4. 用训练好的 GraphCFE 在任意数据集上生成 CF
##############################################

@torch.no_grad()
def generate_cfs_with_graphcfe(
    pred_model: nn.Module,
    explainer: GraphCFE,
    eval_dataset: InMemoryDataset,
    device: str
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[int]]:
    """
    用训练好的 explainer 在 eval_dataset 上生成 CF（逐图处理）。

    ✅ 参照 CF-GNNExplainer：
      - 直接遍历 dataset（不使用 DataLoader）
      - 使用实际节点数（不使用固定的 max_num_nodes）
      - 为所有图生成 CF（不筛选）
      - 添加 tqdm 进度条

    返回：
      - cf_feat_list:   每个图的反事实节点特征 [N, x_dim]
      - cf_adj_list:    每个图的反事实邻接 [N, N]
      - graph_idx_list: 每个 CF 对应的原图索引
    """
    from tqdm import tqdm

    explainer.eval()
    pred_model.eval()

    cf_feat_list = []
    cf_adj_list = []
    graph_idx_list = []

    for idx in tqdm(range(len(eval_dataset)), desc="Generating CFs"):
        data = eval_dataset[idx]
        x = data.x.to(device)  # [N, F]
        edge_index = data.edge_index.to(device)

        # ✅ 使用实际节点数而不是固定的 max_num_nodes
        num_nodes = x.size(0)

        # 转换为稠密格式
        x_dense = x.unsqueeze(0)  # [1, N, F]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).to(device)  # [1, N, N]

        # 如果节点数小于 max_num_nodes，需要padding
        if num_nodes < explainer.max_num_nodes:
            pad_size = explainer.max_num_nodes - num_nodes
            x_dense = F.pad(x_dense, (0, 0, 0, pad_size))  # [1, max_N, F]
            adj = F.pad(adj, (0, pad_size, 0, pad_size))  # [1, max_N, max_N]

        # 生成 CF
        output = explainer(x_dense, adj)

        # ✅ 保留所有生成的CF，不进行筛选（和 CF-GNNExplainer 一致）
        # 去除 padding，只保留实际节点
        cf_feat_list.append(output['feat_reconst'][0, :num_nodes, :].detach().cpu())  # [N, x_dim]
        cf_adj_list.append(output['adj_reconst'][0, :num_nodes, :num_nodes].detach().cpu())  # [N, N]
        graph_idx_list.append(idx)

    return cf_feat_list, cf_adj_list, graph_idx_list


##############################################
# 5. 评估：参照 CF-GNNExplainer 的正确模式
##############################################

@torch.no_grad()
def evaluate_graphcfe(
    pred_model: nn.Module,
    explainer: GraphCFE,
    eval_dataset: InMemoryDataset,
    device: str
) -> Dict[str, float]:
    """
    在 eval_dataset 上评估 GraphCFE

    ✅ 参照 CF-GNNExplainer 的正确实现：
      - 直接遍历 dataset（不使用 DataLoader）
      - 对所有图计算指标（分母是 total_graphs，不是 successful_cf）
      - proximity 使用实际边数归一化（不是 adj.sum()）
      - 添加 tqdm 进度条
      - 使用实际节点数（不是固定的 max_num_nodes）

    评估指标：
      - validity:   CF 是否预测为目标类（y_desired = 1 - ori_pred）
      - proximity:  原图/CF 图邻接的 Frobenius 距离（用边数归一化）
      - fidelity_prob: 原类概率下降量
      - sparsity:   边修改的稀疏性
    """
    from torch_geometric.utils import dense_to_sparse

    print("\n" + "="*60)
    print("Evaluating GraphCFE")
    print("="*60)

    pred_model.eval()
    explainer.eval()

    # 1. 预计算所有图的原始预测和概率
    ori_probs = []
    ori_preds = []
    ori_edge_indices = []

    print("\n1. Computing original predictions...")
    for data in tqdm(eval_dataset, desc="Original predictions"):
        data = data.to(device)
        ori_pred_logits = pred_model(data.x, data.edge_index, data.batch)
        ori_prob = F.softmax(ori_pred_logits, dim=1)[0]  # [num_classes]
        ori_pred = ori_pred_logits.argmax(dim=1).item()

        ori_probs.append(ori_prob.cpu())
        ori_preds.append(ori_pred)
        ori_edge_indices.append(data.edge_index.cpu())

    # 2. 生成 CF
    print("\n2. Generating counterfactuals...")
    cf_feat_list, cf_adj_list, graph_idx_list = \
        generate_cfs_with_graphcfe(pred_model, explainer, eval_dataset, device)

    # 3. 计算指标
    valid_cf = 0
    proximity_sum = 0.0
    fidelity_prob_sum = 0.0
    sparsity_sum = 0.0
    total_graphs = len(eval_dataset)  # ✅ 关键：分母是所有图

    print("\n3. Computing evaluation metrics...")
    for idx in tqdm(range(len(eval_dataset)), desc="Computing metrics"):
        ori_edge_index = ori_edge_indices[idx].to(device)
        ori_prob = ori_probs[idx].to(device)
        ori_pred = ori_preds[idx]

        # 获取对应的 CF
        cf_feat = cf_feat_list[idx].to(device)  # [N, x_dim]
        cf_adj = cf_adj_list[idx].to(device)    # [N, N]
        cf_adj = (cf_adj > 0.5).float()  # 二值化

        # 将 cf_adj 转换为 edge_index
        cf_edge_index, _ = dense_to_sparse(cf_adj)

        # CF 的预测
        num_nodes = cf_feat.size(0)
        batch_vec = torch.zeros(num_nodes, dtype=torch.long, device=device)
        cf_pred_logits = pred_model(cf_feat, cf_edge_index, batch_vec)
        cf_prob = F.softmax(cf_pred_logits, dim=1)[0]  # [num_classes]
        cf_pred = cf_pred_logits.argmax(dim=1).item()

        # 1) Validity：CF 是否预测为目标类
        y_desired = 1 - ori_pred  # ✅ 反事实标签
        if cf_pred == y_desired:
            valid_cf += 1

        # 2) Proximity：原邻接 / CF 邻接的 Frobenius 距离（用边数归一化）
        # ✅ 关键：使用实际边数，不是 adj.sum()
        ori_adj = to_dense_adj(ori_edge_index, max_num_nodes=num_nodes).squeeze(0)
        ori_adj_np = ori_adj.cpu().numpy()
        cf_adj_np = cf_adj.cpu().numpy()

        adj_diff = np.linalg.norm(ori_adj_np - cf_adj_np, ord='fro')

        # ✅ 使用实际边数（参照 CF-GNNExplainer）
        m_ori = ori_edge_index.size(1) // 2  # 无向图
        m_cf = cf_edge_index.size(1) // 2
        max_m = max(m_ori, m_cf, 1)

        proximity = adj_diff / max_m
        proximity_sum += proximity

        # 3) Fidelity_prob：原类概率下降量
        ori_prob_on_ori_class = ori_prob[ori_pred].item()
        cf_prob_on_ori_class = cf_prob[ori_pred].item()
        fidelity_prob = ori_prob_on_ori_class - cf_prob_on_ori_class
        fidelity_prob_sum += fidelity_prob

        # 4) Sparsity：边修改的比例
        ori_edge_set = set()
        for i in range(ori_edge_index.size(1)):
            u, v = ori_edge_index[0, i].item(), ori_edge_index[1, i].item()
            ori_edge_set.add((min(u, v), max(u, v)))

        cf_edge_set = set()
        for i in range(cf_edge_index.size(1)):
            u, v = cf_edge_index[0, i].item(), cf_edge_index[1, i].item()
            cf_edge_set.add((min(u, v), max(u, v)))

        exp_edges = ori_edge_set.symmetric_difference(cf_edge_set)
        num_exp_edges = len(exp_edges)
        num_ori_edges = len(ori_edge_set)
        sparsity = 1 - (num_exp_edges / max(num_ori_edges, 1))
        sparsity_sum += sparsity

    # ✅ 关键：validity 的分母是 total_graphs（所有图），不是 len(cf_feat_list)
    validity = valid_cf / total_graphs
    avg_proximity = proximity_sum / total_graphs
    avg_fidelity_prob = fidelity_prob_sum / total_graphs
    avg_sparsity = sparsity_sum / total_graphs

    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"  Validity ↑: {validity:.4f} (successful: {valid_cf}/{total_graphs})")
    print(f"  Proximity ↓: {avg_proximity:.4f}")
    print(f"  Fidelity (Prob Drop) ↑: {avg_fidelity_prob:.4f}")
    print(f"  Sparsity ↑: {avg_sparsity:.4f}")
    print(f"  CF Generation Rate: {len(cf_feat_list)}/{total_graphs} ({len(cf_feat_list)/max(total_graphs,1):.2%})")
    print("="*60 + "\n")

    return {
        "validity": validity,
        "proximity": avg_proximity,
        "fidelity_prob": avg_fidelity_prob,
        "sparsity": avg_sparsity,
        "successful": valid_cf,
        "total": total_graphs,
    }


##############################################
# 6. 示例 main：流程类似 CF-GNNExplainer
##############################################

if __name__ == "__main__":

    dataset_name = "mutag"
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # ===== 1. 加载/构造 GraphDataset =====
    train_dataset, val_dataset, test_dataset = get_datasets(
        name=dataset_name, root="../data/"
    )
    # 删除数据集中节点数大于300个的数据
    train_dataset = [data for data in train_dataset if data.num_nodes <= 300]
    val_dataset = [data for data in val_dataset if data.num_nodes <= 300]
    test_dataset = [data for data in test_dataset if data.num_nodes <= 300]

    # ===== 2. 加载预训练 GNN 分类器 =====
    pred_model = torch.load(
        f"../param/gnns/{dataset_name}_gcn.pt",
        map_location=device
    ).to(device)
    pred_model.eval()
    print("GNN classifier loaded.")

    # ===== 3. 在训练集上训练 GraphCFE explainer =====
    explainer_model_path = f"../param/explainers/{dataset_name}_graphcfe.pt"
    os.makedirs(os.path.dirname(explainer_model_path), exist_ok=True)

    max_num_nodes = max(
        len(sample.x)
        for dataset in (train_dataset, val_dataset, test_dataset)
        for sample in dataset
    )

    explainer = train_graphcfe(
        pred_model=pred_model,
        train_dataset=train_dataset,
        epochs=100,
        device=device,
        lr=1e-3,
        model_path=explainer_model_path,
        max_num_nodes=max_num_nodes
    )

    # ===== 4. 在验证集上评估 GraphCFE =====
    metrics = evaluate_graphcfe(
        pred_model=pred_model,
        explainer=explainer,
        eval_dataset=val_dataset,
        device=device
    )

    print("Final GraphCFE metrics on val:", metrics)
