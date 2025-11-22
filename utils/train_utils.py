import networkx as nx
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_dense_batch
from utils.graph_utils import extract_explanatory_subgraph, exclude_explanatory_subgraph
import time

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
        data = data.to(device)
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
            # readout_output = model.readout
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
        data = data.to(args.device)

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


def compute_loss(args, outputs, batch, gnn, y_cf, concated_graphs):
    loss_proportion = args.loss_proportion

    device = args.device
    batch['subgraphs'].x = batch['subgraphs'].x.to(device)
    cf_graphs_x = concated_graphs.x.to(device)
    cf_graphs_edge_index = concated_graphs.edge_index.to(device)
    cf_graphs_batch = concated_graphs.batch.to(device)
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
    else:
        recon_x_loss = torch.tensor(0.0, device=device)

    # 计算重构损失 - 邻接矩阵
    recon_adj = outputs['adj_recon'].to(device)
    recon_adj_flat = recon_adj.flatten(-2)
    ori_adj_flat = ori_adj.flatten(-3)
    total_edges = recon_adj_flat.numel()
    recon_adj_loss = F.binary_cross_entropy(recon_adj_flat, ori_adj_flat, reduction='mean')

    # 重新平衡损失权重：
    # - 大幅增加节点特征重构权重（20.0）
    recon_loss += (loss_proportion['recon_x'] * recon_x_loss +
                   loss_proportion['recon_adj'] * recon_adj_loss)

    # 计算预测损失
    pred_probs, _ = gnn.get_pred(concated_graphs.x, concated_graphs.edge_index, concated_graphs.batch)
    # pred = torch.argmax(pred_probs, dim=-1).to(device)
    pred = pred_probs
    # print("---")
    # print(f"Pred:{pred[:15]}")
    # print(f"CF:  {y_cf[:15].view(-1).int()}")
    pred_loss += F.nll_loss(F.log_softmax(pred.float(), dim=-1), y_cf.view(-1).long())

    # Fidelity 作为损失

    # L_Fp, L_Fm, _ = compute_fidelity_proxy(
    #     args=args,
    #     ori_graphs=batch['graphs'],         # 原始图
    #     cf_graphs=concated_graphs,          # 拼接后的图（用来“定位”解释子图）
    #     gnn=gnn,
    #     extract_explanatory_subgraph=extract_explanatory_subgraph,
    #     exclude_explanatory_subgraph=exclude_explanatory_subgraph
    # )
    #
    # fid_loss = L_Fp + L_Fm


    # 调整后的总损失权重
    total_loss = (recon_loss +
                  loss_proportion['pred'] * pred_loss
                  + loss_proportion['kl'] * kl_loss
                  # + loss_proportion['fid'] * fid_loss
                  )

    return {
        'total': total_loss,
        'recon_x': recon_x_loss * loss_proportion['recon_x'],
        'recon_adj': recon_adj_loss * loss_proportion['recon_adj'],
        # 'fid': fid_loss * loss_proportion['fid'],
        'kl': kl_loss * loss_proportion['kl'],
        'pred': pred_loss * loss_proportion['pred']
    }


def compute_loss_causality(args, outputs, batch, gnn, y_cf, concated_graphs):
    loss_proportion = args.loss_proportion

    device = args.device
    batch['subgraphs'].x = batch['subgraphs'].x.to(device)
    max_nodes = args.max_num_nodes
    cf_graphs_x = concated_graphs.x.to(device)
    cf_graphs_edge_index = concated_graphs.edge_index.to(device)
    cf_graphs_batch = concated_graphs.batch.to(device)
    y_cf = y_cf.to(device)

    max_sub_nodes = args.max_subgraph_nodes
    # 重要：指定max_num_nodes确保ori_adj尺寸固定，与模型输出adj_recon尺寸匹配
    ori_adj = to_dense_adj(
        batch['graphs']['edge_index'],
        batch['graphs']['batch'],
        max_num_nodes=args.max_num_nodes  # 使用原图的max_num_nodes (25)
    ).to(device)
    total_loss, kl_loss, recon_loss, pred_loss = 0.0, 0.0, 0.0, 0.0

    # 计算KL散度损失
    z_mu = outputs['z_mu'].to(device)
    z_logvar = outputs['z_logvar'].to(device)
    kl_raw = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    batch_size = z_mu.size(0)  # 假设 z_mu: (batch_size, latent_dim)
    total_latent_elements = batch_size * z_mu.size(1)  # 总元素数
    kl_loss = kl_raw / total_latent_elements if total_latent_elements > 0 else torch.tensor(0.0, device=device)

    # 计算重构损失 - 节点特征
    # 使用MSE损失，因为节点特征是连续的
    recon_x = outputs['x_recon'].to(device)  # (batch_size, max_num_nodes * x_dim)
    batch_size = recon_x.size(0)

    # Reshape为(batch_size, max_num_nodes, x_dim)用于MSE计算
    recon_x_reshaped = recon_x.view(batch_size, max_nodes, args.x_dim)

    # 准备目标标签：将batch['graphs'].x转换为dense格式
    # batch['graphs'].x: (total_nodes, x_dim) 连续特征
    # 需要转换为(batch_size, max_num_nodes, x_dim)
    batch['graphs'].x, batch['graphs'].batch = batch['graphs'].x.to(device), batch['graphs'].batch.to(device)
    target_x_dense, _ = to_dense_batch(
        batch['graphs'].x,
        batch['graphs'].batch,
        max_num_nodes=max_nodes
    )
    target_x_dense = target_x_dense.to(device)  # (batch_size, max_num_nodes, x_dim)

    # 计算MSE损失（全图所有位置，包括填充节点）
    recon_x_loss = F.mse_loss(recon_x_reshaped, target_x_dense, reduction='mean')

    # 计算重构损失 - 邻接矩阵
    recon_adj = outputs['adj_recon'].to(device)
    recon_adj_flat = recon_adj.flatten(1)
    ori_adj_flat = ori_adj.flatten(1)
    # if args.tmp:
    #     adj_matrix = recon_adj[3].reshape(args.max_num_nodes, args.max_num_nodes).cpu().detach().numpy()
    #     # 使用0.5作为阈值进行二值化
    #     adj_binary = (adj_matrix > 0.5).astype(int)
    #     G = nx.from_numpy_array(adj_binary)
    #     nx.draw(G, with_labels=True)
    #     plt.show()
    #     print("this is recon")
    #     time.sleep(5)
    #     G = nx.from_numpy_array(ori_adj[3].reshape(args.max_num_nodes, args.max_num_nodes).cpu().detach().numpy())
    #     nx.draw(G, with_labels=True)
    #     plt.show()
    #     print("this is ori")
    #     time.sleep(5)


    total_edges = recon_adj_flat.numel()
    recon_adj_loss = F.binary_cross_entropy(recon_adj_flat, ori_adj_flat, reduction='mean')

    # 重新平衡损失权重：
    # - 大幅增加节点特征重构权重（20.0）
    recon_loss += (loss_proportion['recon_x'] * recon_x_loss +
                   loss_proportion['recon_adj'] * recon_adj_loss)

    # 计算预测损失
    pred_probs, _ = gnn.get_pred(concated_graphs.x, concated_graphs.edge_index, concated_graphs.batch)
    # pred = torch.argmax(pred_probs, dim=-1).to(device)
    pred = pred_probs
    # print("---")
    # print(f"Pred:{pred[:15]}")
    # print(f"CF:  {y_cf[:15].view(-1).int()}")
    pred_loss += F.nll_loss(F.log_softmax(pred.float(), dim=-1), y_cf.view(-1).long())

    # 独立性约束


    # 调整后的总损失权重
    total_loss = (recon_loss +
                  loss_proportion['pred'] * pred_loss
                  + loss_proportion['kl'] * kl_loss
                  )

    return {
        'total': total_loss,
        'recon_x': recon_x_loss * loss_proportion['recon_x'],
        'recon_adj': recon_adj_loss * loss_proportion['recon_adj'],
        'kl': kl_loss * loss_proportion['kl'],
        'pred': pred_loss * loss_proportion['pred']
    }


# def compute_loss_causality(args, outputs, batch, gnn, y_cf, concated_graphs):
#     loss_proportion = args.loss_proportion
#
#     device = args.device
#     batch['subgraphs'].x = batch['subgraphs'].x.to(device)
#     max_nodes = args.max_num_nodes
#     cf_graphs_x = concated_graphs.x.to(device)
#     cf_graphs_edge_index = concated_graphs.edge_index.to(device)
#     cf_graphs_batch = concated_graphs.batch.to(device)
#     y_cf = y_cf.to(device)
#
#     max_sub_nodes = args.max_subgraph_nodes
#     # 重要：指定max_num_nodes确保ori_adj尺寸固定，与模型输出adj_recon尺寸匹配
#     ori_adj = to_dense_adj(
#         batch['graphs']['edge_index'],
#         batch['graphs']['batch'],
#         max_num_nodes=args.max_num_nodes  # 使用原图的max_num_nodes (25)
#     ).to(device)
#     total_loss, kl_loss, recon_loss, pred_loss = 0.0, 0.0, 0.0, 0.0
#
#     # 计算KL散度损失
#     z_mu = outputs['z_mu'].to(device)
#     z_logvar = outputs['z_logvar'].to(device)
#     kl_raw = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
#     batch_size = z_mu.size(0)  # 假设 z_mu: (batch_size, latent_dim)
#     total_latent_elements = batch_size * z_mu.size(1)  # 总元素数
#     kl_loss = kl_raw / total_latent_elements if total_latent_elements > 0 else torch.tensor(0.0, device=device)
#
#     # 计算重构损失 - 节点特征
#     # 使用MSE损失，因为节点特征是连续的
#     recon_x = outputs['x_recon'].to(device)  # (batch_size, max_num_nodes * x_dim)
#     batch_size = recon_x.size(0)
#
#     # Reshape为(batch_size, max_num_nodes, x_dim)用于MSE计算
#     recon_x_reshaped = recon_x.view(batch_size, max_nodes, args.x_dim)
#
#     # 准备目标标签：将batch['graphs'].x转换为dense格式
#     # batch['graphs'].x: (total_nodes, x_dim) 连续特征
#     # 需要转换为(batch_size, max_num_nodes, x_dim)
#     batch['graphs'].x, batch['graphs'].batch = batch['graphs'].x.to(device), batch['graphs'].batch.to(device)
#     target_x_dense, _ = to_dense_batch(
#         batch['graphs'].x,
#         batch['graphs'].batch,
#         max_num_nodes=max_nodes
#     )
#     target_x_dense = target_x_dense.to(device)  # (batch_size, max_num_nodes, x_dim)
#
#     # 计算MSE损失（全图所有位置，包括填充节点）
#     recon_x_loss = F.mse_loss(recon_x_reshaped, target_x_dense, reduction='mean')
#
#     # 计算重构损失 - 邻接矩阵
#     recon_adj = outputs['adj_recon'].to(device)
#     recon_adj_flat = recon_adj.flatten(1)
#     ori_adj_flat = ori_adj.flatten(1)
#     # if args.tmp:
#     #     adj_matrix = recon_adj[3].reshape(args.max_num_nodes, args.max_num_nodes).cpu().detach().numpy()
#     #     # 使用0.5作为阈值进行二值化
#     #     adj_binary = (adj_matrix > 0.5).astype(int)
#     #     G = nx.from_numpy_array(adj_binary)
#     #     nx.draw(G, with_labels=True)
#     #     plt.show()
#     #     print("this is recon")
#     #     time.sleep(5)
#     #     G = nx.from_numpy_array(ori_adj[3].reshape(args.max_num_nodes, args.max_num_nodes).cpu().detach().numpy())
#     #     nx.draw(G, with_labels=True)
#     #     plt.show()
#     #     print("this is ori")
#     #     time.sleep(5)
#
#
#     total_edges = recon_adj_flat.numel()
#     recon_adj_loss = F.binary_cross_entropy(recon_adj_flat, ori_adj_flat, reduction='mean')
#
#     # 重新平衡损失权重：
#     # - 大幅增加节点特征重构权重（20.0）
#     recon_loss += (loss_proportion['recon_x'] * recon_x_loss +
#                    loss_proportion['recon_adj'] * recon_adj_loss)
#
#     # 计算预测损失
#     pred_probs, _ = gnn.get_pred(concated_graphs.x, concated_graphs.edge_index, concated_graphs.batch)
#     # pred = torch.argmax(pred_probs, dim=-1).to(device)
#     pred = pred_probs
#     # print("---")
#     # print(f"Pred:{pred[:15]}")
#     # print(f"CF:  {y_cf[:15].view(-1).int()}")
#     pred_loss += F.nll_loss(F.log_softmax(pred.float(), dim=-1), y_cf.view(-1).long())
#
#     # 独立性约束
#
#     u = outputs['u'].to(device)  # (batch_size, z_dim)
#     u_cov = u.T @ u / batch_size  # (z_dim, z_dim)：U的协方差矩阵
#     eye = torch.eye(args.z_dim).to(args.device)  # 单位矩阵
#     L_ortho = torch.norm(u_cov - eye, p='fro')
#
#
#     # 调整后的总损失权重
#     total_loss = (recon_loss +
#                   loss_proportion['pred'] * pred_loss
#                   + loss_proportion['kl'] * kl_loss
#                   + loss_proportion['ortho'] * L_ortho
#                   )
#
#     return {
#         'total': total_loss,
#         'recon_x': recon_x_loss * loss_proportion['recon_x'],
#         'recon_adj': recon_adj_loss * loss_proportion['recon_adj'],
#         'ortho': L_ortho * loss_proportion['ortho'],
#         'kl': kl_loss * loss_proportion['kl'],
#         'pred': pred_loss * loss_proportion['pred']
#     }


def compute_fidelity_proxy(
        args,
        ori_graphs,  # Batch (原始图，未拼 CF)
        cf_graphs,  # Batch (拼接了生成子图后的图；只用于抽取解释/非解释子图)
        gnn,  # 预测模型，参数应冻结：for p in gnn.parameters(): p.requires_grad=False
        extract_explanatory_subgraph,  # Single Data -> Data (original version)
        exclude_explanatory_subgraph  # Single Data -> Data (original version)
):
    """
    返回：
      L_Fp:  可导的 F+ 代理（越小越好）
      L_Fm:  可导的 F- 代理（越小越好）
      stats: 统计信息（命中/样本数，用于日志）
    协议：
      F+ 代理 = CE( f(S), y_ori )            —— 让解释子图 S 单独能支持原预测 y_ori（充分性）
      F- 代理 = margin(p_yori(G\\S))         —— 让去掉 S 后，原标签概率下降到 margin 以下（必要性）

    实现：使用 to_data_list() 逐图提取子图（高效 O(sum(V+E))），然后 Batch.from_data_list() 批量前向。
    无需修改 extract/exclude 函数，支持原版单图输入。
    """
    device = args.device

    # 1) 原图预测作为“解释目标标签” y_ori（只前向，不反传到 gnn 参数；但允许对输入梯度）
    # 注意：不要用 torch.no_grad() 包住前向；否则会阻断对 VGAE 可微掩码的梯度。
    gnn.eval()
    ori_graphs = ori_graphs.to(device)
    ori_logits, _ = gnn.get_pred(ori_graphs.x, ori_graphs.edge_index, ori_graphs.batch)
    y_ori = ori_logits.argmax(dim=-1)  # (B,)

    # 2) 构建解释子图与非解释子图（逐样本）
    ori_list = ori_graphs.to_data_list()
    cf_list = cf_graphs.to_data_list()

    exp_graphs, exp_indices = [], []
    exc_graphs, exc_indices = [], []

    for i, (g_ori, g_cf) in enumerate(zip(ori_list, cf_list)):
        S = extract_explanatory_subgraph(g_ori, g_cf)  # Data
        R = exclude_explanatory_subgraph(g_ori, g_cf)  # Data
        if S.num_nodes > 0:
            exp_graphs.append(S)
            exp_indices.append(i)
        if R.num_nodes > 0:  # Always true, but for consistency
            exc_graphs.append(R)
            exc_indices.append(i)

    # 3) F+ 代理：-log p(y_ori | S)
    L_Fp = torch.tensor(0.0, device=device)
    fp_cnt, n_plus = 0, 0
    if len(exp_graphs) > 0:
        exp_batch = Batch.from_data_list(exp_graphs).to(device)
        exp_logits, _ = gnn.get_pred(exp_batch.x, exp_batch.edge_index, exp_batch.batch)  # (n_plus, C)

        y_local = y_ori[torch.tensor(exp_indices, device=device, dtype=torch.long)]  # (n_plus,)
        # Cross-Entropy：越小越好，等价于让 S 单独预测 y_ori
        L_Fp = F.cross_entropy(exp_logits, y_local, reduction='mean')

        # 统计 F+（非必须，仅用于日志/可视化）
        with torch.no_grad():
            exp_pred = exp_logits.argmax(dim=-1)
            fp_cnt = (exp_pred == y_local).sum().item()
            n_plus = y_local.numel()

    # 4) F- 代理：让 p(y_ori | G\\S) 压到 margin 以下
    #    使用 hinge 风格：mean(max(0, p_yori - margin))
    margin = getattr(args, 'fminus_margin', 0.2)
    L_Fm = torch.tensor(0.0, device=device)
    fm_cnt, n_minus = 0, 0
    if len(exc_graphs) > 0:
        exc_batch = Batch.from_data_list(exc_graphs).to(device)
        exc_logits, _ = gnn.get_pred(exc_batch.x, exc_batch.edge_index, exc_batch.batch)  # (n_minus, C)

        y_local = y_ori[torch.tensor(exc_indices, device=device, dtype=torch.long)]  # (n_minus,)
        prob_ori = F.softmax(exc_logits, dim=-1).gather(1, y_local.view(-1, 1)).squeeze(1)  # (n_minus,)

        logits_y_ori = exc_logits.gather(1, y_local.view(-1, 1)).squeeze(1)  # (n_minus,)
        # 可设目标 logits < log(margin / (1 - margin)) 或直接设阈值如 0
        target_logit = getattr(args, 'fminus_logit_threshold', 0.0)
        L_Fm = torch.clamp(logits_y_ori - target_logit, min=0.0).mean()

        # L_Fm = torch.clamp(prob_ori - margin, min=0.0).mean()

        # 统计 F-（“一致”越少越好，这里仅为日志）
        with torch.no_grad():
            exc_pred = exc_logits.argmax(dim=-1)
            fm_cnt = (exc_pred == y_local).sum().item()
            n_minus = y_local.numel()

    stats = {
        'F+_hits': fp_cnt, 'F+_den': max(n_plus, 1),
        'F-_hits': fm_cnt, 'F-_den': max(n_minus, 1),
        'F+': (fp_cnt / max(n_plus, 1)),
        'F-': (fm_cnt / max(n_minus, 1)),
    }


    return L_Fp, L_Fm, stats





