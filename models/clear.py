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

        # ============================================================
        # Encoder: 依然需要同时看 Features 和 Adjacency 来理解图语义
        # ============================================================
        self.graph_model = DenseGCNConv(x_dim, h_dim)
        self.graph_norm = nn.BatchNorm1d(h_dim)

        self.encoder_mean = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.BatchNorm1d(z_dim),
            nn.ReLU()
        )
        self.encoder_logvar = nn.Linear(h_dim, z_dim)

        # ============================================================
        # Decoder: 【修改】只保留 decoder_a (邻接矩阵解码器)
        # ============================================================
        # 注意：这里移除了 self.decoder_x

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

        # 如果有边属性需要重构，保留 decoder_edge_attr，否则忽略
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
                nn.Linear(h_dim, int((max_num_nodes - 1) * max_num_nodes / 2) * edge_attr_dim)
            )
        else:
            self.decoder_edge_attr = None

        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def encoder(self, features: torch.Tensor, adj: torch.Tensor):
        # Encoder 保持不变，需要利用图结构和节点特征生成 Latent Z
        graph_rep = self.graph_model(features, adj)  # [B, N, h_dim]
        graph_rep = torch.sum(graph_rep, dim=1)  # [B, h_dim]
        graph_rep = self.graph_norm(graph_rep)
        z_mu = self.encoder_mean(graph_rep)
        z_logvar = self.encoder_logvar(graph_rep)
        return z_mu, z_logvar

    def convert_to_symmetric_tensor(self, num_nodes: int, adj_vec: torch.Tensor):
        upper_triangular = torch.zeros((adj_vec.shape[0], num_nodes, num_nodes), device=self.device)
        mask = torch.triu_indices(num_nodes, num_nodes, offset=1).to(self.device)
        upper_triangular[:, mask[0], mask[1]] = adj_vec
        symm = upper_triangular + torch.transpose(upper_triangular, 1, 2)
        return symm

    def decoder(self, z: torch.Tensor):
        # 【修改】Decoder 只解码 adj，不再解码 features
        adj_reconst_half = self.decoder_a(z)
        adj_reconst = self.convert_to_symmetric_tensor(self.max_num_nodes, adj_reconst_half)

        if self.decoder_edge_attr is not None:
            edge_attrs_reconst = self.decoder_edge_attr(z).view(
                -1, int(self.max_num_nodes * (self.max_num_nodes - 1) / 2), self.edge_attr_dim
            )
        else:
            edge_attrs_reconst = None

        return adj_reconst, edge_attrs_reconst

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_mu, z_logvar = self.encoder(features, adj)
        z_sample = self.reparameterize(z_mu, z_logvar)

        # Decoder 仅返回结构
        adj_reconst, edge_attrs_reconst = self.decoder(z_sample)

        return {
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'adj_reconst': adj_reconst,
            'feat_reconst': features,  # 【关键】直接返回原始特征，不修改
            'edge_attr_reconst': edge_attrs_reconst
        }

    def loss(
            self,
            feat: torch.Tensor,
            adj: torch.Tensor,
            explainer_output: Dict[str, torch.Tensor],
            cf_label: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        B = feat.shape[0]

        # 1. KL Loss
        z_mu = explainer_output['z_mu']
        z_logvar = explainer_output['z_logvar']
        loss_kl = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - torch.exp(z_logvar), dim=1).mean()

        # 2. Similarity Loss (仅针对 Adjacency)
        # 【修改】彻底移除 dist_x，只计算 dist_a
        dist_a = F.binary_cross_entropy(explainer_output['adj_reconst'], adj)
        loss_sim = 10.0 * dist_a  # 这里的系数可以根据需要调整

        # 3. CFE Loss (Counterfactual Prediction Loss)
        # 使用原始特征 (feat) 和 重构的邻接 (adj_reconst)

        # 准备数据给 pred_model
        # 注意：这里 explainer_output['feat_reconst'] 已经是原始 feat 了，所以逻辑是自洽的
        feat_for_pred = explainer_output['feat_reconst']
        adj_reconst = explainer_output['adj_reconst']

        # 将 adj_reconst (Dense) 转为 PyG 格式
        from torch_geometric.utils import dense_to_sparse
        x_sparse = feat_for_pred.view(B * feat.size(1), -1)

        edge_index_list = []
        edge_weight_list = []
        batch_vec_list = []
        offset = 0
        N = feat.size(1)

        for b in range(B):
            adj_b = adj_reconst[b]
            edge_index_b, edge_weight_b = dense_to_sparse(adj_b)
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

        batch_vec = torch.tensor(batch_vec_list, dtype=torch.long, device=self.device)

        # 预测
        _, y_pred_logits = self.pred_model.get_pred_explain(
            x_sparse.to(self.device),
            edge_index,
            edge_weight,
            batch_vec
        )

        loss_cfe = 10 * F.nll_loss(
            F.log_softmax(y_pred_logits, dim=-1),
            cf_label.view(-1).long()
        )

        loss = loss_sim + loss_kl + loss_cfe

        return {
            'loss': loss,
            'loss_kl': loss_kl,
            'loss_sim': loss_sim,
            'loss_cfe': loss_cfe
        }

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
    print("\n" + "=" * 60)
    print("Evaluating GraphCFE (Topology-Only)")
    print("=" * 60)

    pred_model.eval()
    explainer.eval()

    # 1. 预计算原始预测
    ori_probs = []
    ori_preds = []
    ori_edge_indices = []

    # 我们仍然保存一份原始特征，双重保险
    ori_features = []

    for data in tqdm(eval_dataset, desc="Original predictions"):
        data = data.to(device)
        # 手动构建 batch 防止 data.batch 为 None
        batch_vec = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        ori_pred_logits = pred_model(data.x, data.edge_index, batch_vec)
        ori_prob = F.softmax(ori_pred_logits, dim=1)[0]
        ori_pred = ori_pred_logits.argmax(dim=1).item()

        ori_probs.append(ori_prob.cpu())
        ori_preds.append(ori_pred)
        ori_edge_indices.append(data.edge_index.cpu())
        ori_features.append(data.x.cpu())

        # 2. 生成 CF
    # 注意：现在 explainer 内部只生成 adj_reconst，feat_reconst 只是 copy
    cf_feat_list, cf_adj_list, graph_idx_list = \
        generate_cfs_with_graphcfe(pred_model, explainer, eval_dataset, device)

    # 3. 计算指标
    valid_cf = 0
    proximity_sum = 0.0
    fidelity_prob_sum = 0.0
    sparsity_sum = 0.0
    total_graphs = len(eval_dataset)

    for idx in tqdm(range(len(eval_dataset)), desc="Computing metrics"):
        # 基础数据准备
        ori_edge_index = ori_edge_indices[idx].to(device)
        ori_prob = ori_probs[idx].to(device)
        ori_pred = ori_preds[idx]

        # 强制使用原始特征 (虽然 cf_feat_list[idx] 现在也应该是原始特征，但这样写更清晰)
        cf_feat = ori_features[idx].to(device)

        # CF 边结构
        cf_adj = cf_adj_list[idx].to(device)
        cf_adj = (cf_adj > 0.5).float()  # 这里的阈值可以调整
        cf_edge_index, _ = to_dense_adj_sparse_format_helper(cf_adj)  # 下面提供个helper防止import报错

        # CF 预测
        batch_vec = torch.zeros(cf_feat.size(0), dtype=torch.long, device=device)
        cf_pred_logits = pred_model(cf_feat, cf_edge_index, batch_vec)
        cf_prob = F.softmax(cf_pred_logits, dim=1)[0]
        cf_pred = cf_pred_logits.argmax(dim=1).item()

        # --- 指标计算 (保持不变) ---
        y_desired = 1 - ori_pred
        if cf_pred == y_desired:
            valid_cf += 1

        # Proximity
        ori_adj = to_dense_adj(ori_edge_index, max_num_nodes=cf_feat.size(0)).squeeze(0)
        adj_diff = torch.norm(ori_adj - cf_adj, p='fro').item()
        max_m = max(ori_edge_index.size(1) // 2, cf_edge_index.size(1) // 2, 1)
        proximity_sum += (adj_diff / max_m)

        # Fidelity
        fidelity_prob_sum += (ori_prob[ori_pred].item() - cf_prob[ori_pred].item())

        # Sparsity
        sparsity_sum += calculate_sparsity(ori_edge_index, cf_edge_index)  # 封装一下之前的逻辑

    # 汇总
    results = {
        "validity": valid_cf / total_graphs,
        "proximity": proximity_sum / total_graphs,
        "fidelity_prob": fidelity_prob_sum / total_graphs,
        "sparsity": sparsity_sum / total_graphs
    }

    print(f"Results: {results}")
    return results


# 辅助函数：防止 import 错误
def to_dense_adj_sparse_format_helper(adj_binary):
    from torch_geometric.utils import dense_to_sparse
    return dense_to_sparse(adj_binary)


def calculate_sparsity(ori_edge_index, cf_edge_index):
    # 原先的 Sparsity 计算逻辑
    ori_edge_set = set()
    for i in range(ori_edge_index.size(1)):
        u, v = ori_edge_index[0, i].item(), ori_edge_index[1, i].item()
        ori_edge_set.add((min(u, v), max(u, v)))

    cf_edge_set = set()
    for i in range(cf_edge_index.size(1)):
        u, v = cf_edge_index[0, i].item(), cf_edge_index[1, i].item()
        cf_edge_set.add((min(u, v), max(u, v)))

    num_ori = len(ori_edge_set)
    if num_ori == 0: return 0.0

    # 编辑距离 / 原边数
    diff = ori_edge_set.symmetric_difference(cf_edge_set)
    return 1.0 - (len(diff) / num_ori)


##############################################
# 6. 示例 main：流程类似 CF-GNNExplainer
##############################################

if __name__ == "__main__":

    dataset_name = "nci1"
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