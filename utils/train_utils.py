import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch


def Gtrain(train_loader, model, optimizer, device, criterion=nn.MSELoss()):
    """
    General training function for graph classification
    :param train_loader: DataLoader
    :param model: model
    :param optimizer: optimizer
    :param device: device
    :param criterion: loss function (default: MSELoss)
    """
    model.train()
    loss_all = 0
    criterion = criterion

    for data in train_loader:
        data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)


def Gtest(test_loader, model, device, criterion=nn.L1Loss(reduction="mean")):
    """
    General test function for graph classification
    :param test_loader: DataLoader
    :param model: model
    :param device: device
    :param criterion: loss function (default: L1Loss)
    :return: error, accuracy
    """
    model.eval()
    error = 0
    correct = 0

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(
                data.x,
                data.edge_index,
                data.batch,
            )
            readout_output = model.readout
            error += criterion(output, data.y) * data.num_graphs
            correct += float(output.argmax(dim=1).eq(data.y).sum().item())

        return error / len(test_loader.dataset), correct / len(test_loader.dataset)


def Etrain(args, train_loader, model, exp_model, optimizer):
    """
    General training function for graph classification with explanation
    :param args: arguments
    :param train_loader: DataLoader
    :param model: prediction model
    :param exp_model: explanation model
    :param optimizer: optimizer
    :return: loss
    """

    model.eval()
    exp_model.train()
    loss_all = 0
    criterion = nn.CrossEntropyLoss()

    for data in train_loader:
        data.to(args.device)

        optimizer.zero_grad()
        out, _ = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)

        edge_mask = exp_model(
            x=data.x,
            edge_index=data.edge_index,
            batch=data.batch,
        )
        edge_mask = (edge_mask * args.EPS).sigmoid()

        out_exp, _ = model.get_pred_explain(data.x, data.edge_index, edge_mask, data.batch)
        loss_exp = criterion(out_exp, data.y)

        loss_total = loss + args.alpha * loss_exp
        loss_total.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_loader.dataset)


# def compute_loss(args, outputs, batch, gnn, y_cf, concated_graphs, epoch_losses):
#     device = args.device
#     batch['subgraphs'].x = batch['subgraphs'].x.to(device)
#     concated_graphs.x = concated_graphs.x.to(device)
#     concated_graphs.edge_index = concated_graphs.edge_index.to(device)
#     concated_graphs.batch = concated_graphs.batch.to(device)
#     y_cf = y_cf.to(device)
#
#     max_sub_nodes = args.max_subgraph_nodes
#     ori_adj = to_dense_adj(batch['subgraphs']['edge_index'],batch['subgraphs']['batch']).to(device)
#     total_loss, kl_loss, recon_loss, pred_loss = 0.0, 0.0, 0.0, 0.0
#
#
#     #计算KL散度损失
#     z_mu = outputs['z_mu'].to(device)
#     z_logvar = outputs['z_logvar'].to(device)
#     kl_loss += -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#
#     #计算重构损失 - 节点特征
#     # 使用交叉熵损失而不是L2距离，因为节点特征是分类数据（one-hot编码的原子类型）
#     recon_x = outputs['x_recon'].to(device)  # (batch_size, max_subgraph_nodes * x_dim)
#     batch_size = recon_x.size(0)
#
#     # Reshape为(batch_size, max_subgraph_nodes, x_dim)用于交叉熵计算
#     recon_x_reshaped = recon_x.view(batch_size, max_sub_nodes, args.x_dim)
#
#     # 准备目标标签：将batch['subgraphs'].x转换为dense格式
#     # batch['subgraphs'].x: (total_nodes, x_dim) one-hot encoded
#     # 需要转换为(batch_size, max_subgraph_nodes, x_dim)
#     batch['subgraphs'].x, batch['subgraphs'].batch = batch['subgraphs'].x.to(device), batch['subgraphs'].batch.to(device)
#     target_x_dense, node_mask = to_dense_batch(
#         batch['subgraphs'].x,
#         batch['subgraphs'].batch,
#         max_num_nodes=max_sub_nodes
#     )
#     target_x_dense = target_x_dense.to(device)  # (batch_size, max_subgraph_nodes, x_dim)
#
#     # 重要：正确计算真实节点的mask
#     # to_dense_batch返回的mask可能不准确，需要手动计算
#     # 统计每个图在batch中的实际节点数
#     node_mask = batch['subgraphs']['real_mask'].reshape(batch_size, max_sub_nodes).to(device)
#
#     # 将one-hot编码转换为类别索引
#     target_classes = target_x_dense.argmax(dim=-1)  # (batch_size, max_subgraph_nodes)
#
#     # 计算交叉熵损失（仅对有效节点）
#     # Reshape recon_x_reshaped: (batch_size, max_subgraph_nodes, x_dim) -> (batch_size * max_subgraph_nodes, x_dim)
#     recon_x_flat = recon_x_reshaped.view(-1, args.x_dim)
#     target_classes_flat = target_classes.view(-1)
#     node_mask_flat = node_mask.view(-1)
#
#     # 只计算有效节点的损失
#     valid_indices = node_mask_flat.bool()
#     if valid_indices.sum() > 0:
#         recon_x_loss = F.cross_entropy(
#             recon_x_flat[valid_indices],
#             target_classes_flat[valid_indices],
#             reduction='mean'
#         )
#         # recon_x_loss = F.pairwise_distance(recon_x_flat[valid_indices].flatten(-2), batch['subgraphs'].x[node_mask_flat].flatten(-2)).mean()
#
#         # 添加原子类型多样性损失 - 鼓励模型预测多种原子类型
#         # 计算预测的原子类型分布（在batch内）
#         pred_probs = F.softmax(recon_x_flat[valid_indices], dim=-1)  # (num_valid_nodes, x_dim)
#
#         # 计算每种原子类型的平均预测概率
#         atom_type_distribution = pred_probs.mean(dim=0)  # (x_dim,)
#
#         # 熵正则化：鼓励输出分布更均匀（不要只输出碳原子）
#         # 最大熵 = log(x_dim)，当所有原子类型概率相等时达到
#         epsilon = 1e-10  # 防止log(0)
#         entropy = -torch.sum(atom_type_distribution * torch.log(atom_type_distribution + epsilon))
#         max_entropy = torch.log(torch.tensor(args.x_dim, dtype=torch.float, device=device))
#
#         # 多样性损失：越接近最大熵越好（损失越低）
#         diversity_loss = max_entropy - entropy
#
#         # 添加原子类型频率匹配损失：鼓励生成的原子类型分布与目标分布匹配
#         target_atom_distribution = torch.zeros(args.x_dim, device=device)
#         for atom_idx in target_classes_flat[valid_indices]:
#             target_atom_distribution[atom_idx] += 1
#         target_atom_distribution = target_atom_distribution / valid_indices.sum().float()
#
#         # KL散度：衡量预测分布与目标分布的差异
#         distribution_loss = F.kl_div(
#             torch.log(atom_type_distribution + epsilon),
#             target_atom_distribution,
#             reduction='sum'
#         )
#     else:
#         recon_x_loss = torch.tensor(0.0, device=device)
#         diversity_loss = torch.tensor(0.0, device=device)
#         distribution_loss = torch.tensor(0.0, device=device)
#
#     # 计算重构损失 - 邻接矩阵
#     recon_adj = outputs['adj_recon'].to(device)
#     recon_adj_loss = F.binary_cross_entropy(recon_adj.flatten(-2), ori_adj.flatten(-3))
#
#     # 重新平衡损失权重：
#     # - 大幅增加节点特征重构权重（20.0）
#     # - 添加多样性损失（1.0）来鼓励不同原子类型
#     # - 添加分布匹配损失（5.0）来匹配真实原子类型分布
#     recon_loss += 20.0 * recon_x_loss + 1.0 * recon_adj_loss + 0.0 * diversity_loss + 1.0 * distribution_loss
#
#     #计算预测损失
#     pred_probs, _ = gnn.get_pred(concated_graphs.x, concated_graphs.edge_index, concated_graphs.batch)
#     pred = torch.argmax(pred_probs,dim=-1).to(device)
#     pred_loss += F.nll_loss(F.log_softmax(pred.float(), dim=-1), y_cf.view(-1).long())
#
#     # 调整后的总损失权重
#     total_loss = recon_loss + 5.0 * pred_loss + 0.01 * kl_loss
#
#     return {
#         'total': total_loss,
#         'recon_x': recon_x_loss * 20.0,
#         'recon_adj': recon_adj_loss * 1.0,
#         'diversity': diversity_loss * 0.0,
#         'distribution': distribution_loss * 1.0,
#         'kl': kl_loss * 0.01,
#         'pred': pred_loss * 5.0
#     }

def compute_loss(args, outputs, batch, gnn, y_cf, concated_graphs, epoch_losses):
    loss_proportion = args.loss_proportion

    device = args.device
    batch['subgraphs'].x = batch['subgraphs'].x.to(device)
    concated_graphs.x = concated_graphs.x.to(device)
    concated_graphs.edge_index = concated_graphs.edge_index.to(device)
    concated_graphs.batch = concated_graphs.batch.to(device)
    y_cf = y_cf.to(device)

    max_sub_nodes = args.max_subgraph_nodes
    ori_adj = to_dense_adj(batch['subgraphs']['edge_index'], batch['subgraphs']['batch']).to(device)
    total_loss, kl_loss, recon_loss, pred_loss = 0.0, 0.0, 0.0, 0.0

    # 计算KL散度损失
    z_mu = outputs['z_mu'].to(device)
    z_logvar = outputs['z_logvar'].to(device)
    kl_raw = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    batch_size = z_mu.size(0)  # 假设 z_mu: (batch_size, latent_dim)
    total_latent_elements = batch_size * z_mu.size(1)  # 总元素数
    kl_loss = kl_raw / total_latent_elements if total_latent_elements > 0 else torch.tensor(0.0, device=device)

    # 计算重构损失 - 节点特征
    # 使用交叉熵损失而不是L2距离，因为节点特征是分类数据（one-hot编码的原子类型）
    recon_x = outputs['x_recon'].to(device)  # (batch_size, max_subgraph_nodes * x_dim)
    batch_size = recon_x.size(0)

    # Reshape为(batch_size, max_subgraph_nodes, x_dim)用于交叉熵计算
    recon_x_reshaped = recon_x.view(batch_size, max_sub_nodes, args.x_dim)

    # 准备目标标签：将batch['subgraphs'].x转换为dense格式
    # batch['subgraphs'].x: (total_nodes, x_dim) one-hot encoded
    # 需要转换为(batch_size, max_subgraph_nodes, x_dim)
    batch['subgraphs'].x, batch['subgraphs'].batch = batch['subgraphs'].x.to(device), batch['subgraphs'].batch.to(device)
    target_x_dense, node_mask = to_dense_batch(
        batch['subgraphs'].x,
        batch['subgraphs'].batch,
        max_num_nodes=max_sub_nodes
    )
    target_x_dense = target_x_dense.to(device)  # (batch_size, max_subgraph_nodes, x_dim)

    # 重要：正确计算真实节点的mask

    node_mask = batch['subgraphs']['real_mask'].reshape(batch_size, max_sub_nodes).to(device)

    # 将one-hot编码转换为类别索引
    target_classes = target_x_dense.argmax(dim=-1)  # (batch_size, max_subgraph_nodes)

    # 计算交叉熵损失（仅对有效节点）
    # Reshape recon_x_reshaped: (batch_size, max_subgraph_nodes, x_dim) -> (batch_size * max_subgraph_nodes, x_dim)
    recon_x_flat = recon_x_reshaped.view(-1, args.x_dim)
    target_classes_flat = target_classes.view(-1)
    node_mask_flat = node_mask.view(-1)

    # 只计算有效节点的损失
    valid_indices = node_mask_flat.bool()
    num_valid_nodes = valid_indices.sum().item()
    if num_valid_nodes > 0:
        recon_x_loss = F.cross_entropy(
            recon_x_flat[valid_indices],
            target_classes_flat[valid_indices],
            reduction='mean'
        )
        # recon_x_loss = F.pairwise_distance(recon_x_flat[valid_indices].flatten(-2), batch['subgraphs'].x[node_mask_flat].flatten(-2)).mean()

        # 添加原子类型多样性损失 - 鼓励模型预测多种原子类型
        # 计算预测的原子类型分布（在batch内）
        pred_probs = F.softmax(recon_x_flat[valid_indices], dim=-1)  # (num_valid_nodes, x_dim)

        # 计算每种原子类型的平均预测概率
        atom_type_distribution = pred_probs.mean(dim=0)  # (x_dim,)

        # 熵正则化：鼓励输出分布更均匀（不要只输出碳原子）
        # 最大熵 = log(x_dim)，当所有原子类型概率相等时达到
        epsilon = 1e-10  # 防止log(0)
        entropy = -torch.sum(atom_type_distribution * torch.log(atom_type_distribution + epsilon))
        max_entropy = torch.log(torch.tensor(args.x_dim, dtype=torch.float, device=device))

        # 多样性损失：越接近最大熵越好（损失越低）
        diversity_loss = max_entropy - entropy

        # 添加原子类型频率匹配损失：鼓励生成的原子类型分布与目标分布匹配
        target_atom_distribution = torch.zeros(args.x_dim, device=device)
        for atom_idx in target_classes_flat[valid_indices]:
            target_atom_distribution[atom_idx] += 1
        target_atom_distribution = target_atom_distribution / num_valid_nodes

        # KL散度：衡量预测分布与目标分布的差异
        distribution_loss = F.kl_div(
            torch.log(atom_type_distribution + epsilon),
            target_atom_distribution,
            reduction='sum'
        ) / args.x_dim
    else:
        recon_x_loss = torch.tensor(0.0, device=device)
        diversity_loss = torch.tensor(0.0, device=device)
        distribution_loss = torch.tensor(0.0, device=device)

    # 计算重构损失 - 邻接矩阵
    recon_adj = outputs['adj_recon'].to(device)
    recon_adj_flat = recon_adj.flatten(-2)
    ori_adj_flat = ori_adj.flatten(-3)
    total_edges = recon_adj_flat.numel()
    recon_adj_loss = F.binary_cross_entropy(recon_adj_flat, ori_adj_flat, reduction='mean')

    # 重新平衡损失权重：
    # - 大幅增加节点特征重构权重（20.0）
    # - 添加多样性损失（1.0）来鼓励不同原子类型
    # - 添加分布匹配损失（5.0）来匹配真实原子类型分布
    recon_loss += (loss_proportion['recon_x'] * recon_x_loss +
                   loss_proportion['recon_adj'] * recon_adj_loss +
                   loss_proportion['diversity'] * diversity_loss +
                   loss_proportion['distribution'] * distribution_loss)

    # 计算预测损失
    pred_probs, _ = gnn.get_pred(concated_graphs.x, concated_graphs.edge_index, concated_graphs.batch)
    # pred = torch.argmax(pred_probs, dim=-1).to(device)
    pred = pred_probs
    # print("---")
    # print(f"Pred:{pred[:15]}")
    # print(f"CF:  {y_cf[:15].view(-1).int()}")
    pred_loss += F.nll_loss(F.log_softmax(pred.float(), dim=-1), y_cf.view(-1).long())

    # 调整后的总损失权重
    total_loss = (recon_loss +
                  loss_proportion['pred'] * pred_loss
                  + loss_proportion['kl'] * kl_loss)

    return {
        'total': total_loss,
        'recon_x': recon_x_loss * loss_proportion['recon_x'],
        'recon_adj': recon_adj_loss * loss_proportion['recon_adj'],
        'diversity': diversity_loss * loss_proportion['diversity'],
        'distribution': distribution_loss * loss_proportion['distribution'],
        'kl': kl_loss * loss_proportion['kl'],
        'pred': pred_loss * loss_proportion['pred']
    }

