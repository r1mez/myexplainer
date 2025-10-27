import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_dense_batch
from tqdm import tqdm
import os
from datetime import datetime
from utils import compute_loss, concat_graphs
import time






def train_myexplainer(args, model, gnn, pair_loader, optimizer, epochs=200, log_dir='logs'):
    """
    训练MyExplainer模型的主函数

    流程:
    1. 对每个batch的图对 (原始图, 目标图):
       - 原始图 -> 编码器 -> 潜在表示 z (使用反事实标签 y_cf)
       - z + y_cf -> 解码器 -> 重构图
       - 目标图 -> 编码器 -> 目标潜在表示 z_tgt (使用原标签 y)
    2. 计算损失:
       - 重构损失: 节点特征重构 + 邻接矩阵重构
       - KL散度: 使原始图编码分布接近目标图编码分布
       - 预测损失: 重构图经过GNN后应预测为y_cf
    3. 反向传播优化

    参数:
        args: 参数对象，包含device, max_num_nodes等
        model: MyExplainer模型
        gnn: 预训练的GNN分类器
        pair_loader: GraphPairData的DataLoader
        optimizer: 优化器
        epochs: 训练轮数
        log_dir: 日志保存目录

    返回:
        model: 训练好的模型
        losses: 损失历史字典
        log_file: 日志文件路径
    """
    model.train()
    gnn.eval()  # GNN保持评估模式

    # 验证max_num_nodes设置
    if not hasattr(args, 'max_num_nodes') or args.max_num_nodes is None:
        raise ValueError("args.max_num_nodes must be set before training!")

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_myexplainer_{timestamp}.txt")

    # 写入日志头部信息
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MyExplainer Training Log\n")
        f.write("=" * 80 + "\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")

        # 记录超参数
        f.write("-" * 80 + "\n")
        f.write("Hyperparameters:\n")
        f.write("-" * 80 + "\n")

        # 遍历args的所有属性
        args_dict = vars(args)
        for key in sorted(args_dict.keys()):
            value = args_dict[key]
            # 跳过device对象（它的字符串表示很长）
            if key == 'device':
                f.write(f"  {key}: {str(value)}\n")
            else:
                f.write(f"  {key}: {value}\n")

        f.write("\n")
        f.write("-" * 80 + "\n")
        f.write("Training Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Epochs: {epochs}\n")
        f.write(f"  Max Num Nodes: {args.max_num_nodes}\n")
        f.write(f"  Optimizer: {type(optimizer).__name__}\n")
        f.write(f"  Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"  Weight Decay: {optimizer.param_groups[0]['weight_decay']}\n")
        f.write("\n")

        # 记录损失权重
        f.write("-" * 80 + "\n")
        f.write("Loss Weights:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  alpha_recon_x: 1.0\n")
        f.write(f"  alpha_recon_adj: 1.0\n")
        f.write(f"  alpha_kl: 0.01\n")
        f.write(f"  alpha_pred: 0.1\n")
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Training Progress:\n")
        f.write("=" * 80 + "\n\n")

    print(f"Training with max_num_nodes={args.max_num_nodes}")
    print(f"Log file: {log_file}")

    # 记录损失历史
    losses = {
        'total': [],
        'recon_x': [],
        'recon_adj': [],
        'kl': [],
        'pred': []
    }

    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        epoch_losses = {
            'total': 0.0,
            'recon_x': 0.0,
            'recon_adj': 0.0,
            'kl': 0.0,
            'pred': 0.0
        }

        num_batches = 0

        progress_bar = tqdm(pair_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for batch in progress_bar:
            # 1. 准备数据
            ori_graph = batch['ori_graph'].to(args.device)
            tgt_graph = batch['tgt_graph'].to(args.device)
            ori_pred = batch['ori_pred'].to(args.device)  # 原始预测标签
            tgt_pred = batch['tgt_pred'].to(args.device)  # 目标预测标签

            batch_size = ori_graph.num_graphs

            # 将图转换为dense格式 (batch_size, max_num_nodes, feature_dim)
            ori_x, ori_mask = to_dense_batch(ori_graph.x, ori_graph.batch,
                                             max_num_nodes=args.max_num_nodes)
            ori_adj = to_dense_adj(ori_graph.edge_index, ori_graph.batch,
                                   # edge_attr=ori_graph.edge_attr,
                                   max_num_nodes=args.max_num_nodes)

            tgt_x, tgt_mask = to_dense_batch(tgt_graph.x, tgt_graph.batch,
                                             max_num_nodes=args.max_num_nodes)
            tgt_adj = to_dense_adj(tgt_graph.edge_index, tgt_graph.batch,
                                   # edge_attr=tgt_graph.edge_attr,
                                   max_num_nodes=args.max_num_nodes)

            # 准备反事实标签 (batch_size, 1)
            # y_cf是目标标签(我们想生成的标签), y是原始标签
            y_cf = tgt_pred.float().unsqueeze(1)  # 反事实标签
            y = ori_pred.float().unsqueeze(1)     # 原始标签

            # 2. 前向传播
            optimizer.zero_grad()

            outputs = model(
                features=ori_x,
                adj=ori_adj,
                y_cf=y_cf,
                features_tgt=tgt_x,
                adj_tgt=tgt_adj,
                y=y
            )

            x_recon = outputs['x_recon']
            adj_recon = outputs['adj_recon']
            z_mu = outputs['z_mu']
            z_logvar = outputs['z_logvar']
            z_mu_tgt = outputs['z_mu_tgt']
            z_logvar_tgt = outputs['z_logvar_tgt']

            # 3. 计算损失
            # 3.1 重构损失 - 节点特征
            # Reshape: (batch_size, max_num_nodes * x_dim) -> (batch_size, max_num_nodes, x_dim)
            x_recon_reshaped = x_recon.view(batch_size, args.max_num_nodes, args.x_dim)
            loss_recon_x = F.mse_loss(x_recon_reshaped, tgt_x, reduction='none')
            # 只计算有效节点的损失
            loss_recon_x = (loss_recon_x * tgt_mask.unsqueeze(-1)).sum() / tgt_mask.sum()

            # 3.2 重构损失 - 邻接矩阵
            # Reshape: (batch_size, max_num_nodes * max_num_nodes) -> (batch_size, max_num_nodes, max_num_nodes)
            adj_recon_reshaped = adj_recon.view(batch_size, args.max_num_nodes, args.max_num_nodes)

            # 处理边特征: 如果有边特征，取第一维度；否则使用二值邻接矩阵
            if tgt_adj.dim() == 4:  # (batch_size, max_num_nodes, max_num_nodes, edge_attr_dim)
                # 简化：将边属性转换为二值邻接矩阵 (存在边=1)
                tgt_adj_binary = (tgt_adj.sum(dim=-1) > 0).float()
            else:
                tgt_adj_binary = tgt_adj

            loss_recon_adj = F.binary_cross_entropy(adj_recon_reshaped, tgt_adj_binary, reduction='none')
            # 创建邻接矩阵掩码 (只计算有效节点对的损失)
            adj_mask = tgt_mask.unsqueeze(-1) * tgt_mask.unsqueeze(-2)  # (batch_size, max_num_nodes, max_num_nodes)
            loss_recon_adj = (loss_recon_adj * adj_mask).sum() / adj_mask.sum()

            # 3.3 KL散度 - 使原始图的潜在分布接近目标图的潜在分布
            # KL(q(z|G_ori, y_cf) || q(z|G_tgt, y))
            # kl_loss = -0.5 * torch.sum(
            #     1 + z_logvar - z_logvar_tgt
            #     - ((z_mu - z_mu_tgt).pow(2) + z_logvar.exp()) / z_logvar_tgt.exp()
            # )
            # kl_loss = kl_loss / batch_size

            # 3.3 KL散度 - 标准VAE KL散度损失
            # KL(q(z|G_ori, y_cf) || p(z)) 其中 p(z) = N(0, I)
            kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
            kl_loss = kl_loss / batch_size


            # 3.4 预测损失 - 重构图应该被GNN预测为y_cf
            # 必须将重构的dense图转换回sparse格式，然后交给GNN预测

            # 将重构的dense图转换为sparse格式
            recon_graphs_list = []
            recon_batch_indices = []

            for b in range(batch_size):
                # 提取该样本的节点特征和邻接矩阵
                num_nodes = tgt_mask[b].sum().item()
                if num_nodes == 0:
                    continue

                # 节点特征: 将连续值转换为one-hot (取argmax)
                x_sample = x_recon_reshaped[b, :num_nodes, :]  # (num_nodes, x_dim)
                x_sample_onehot = torch.zeros_like(x_sample)
                atom_indices = torch.argmax(x_sample, dim=-1)
                x_sample_onehot.scatter_(1, atom_indices.unsqueeze(-1), 1.0)

                # 邻接矩阵: 使用阈值二值化
                adj_sample = adj_recon_reshaped[b, :num_nodes, :num_nodes]  # (num_nodes, num_nodes)

                # 构建edge_index (sparse格式)
                edge_threshold = 0.5
                edge_indices = (adj_sample > edge_threshold).nonzero(as_tuple=False)  # (num_edges, 2)

                if edge_indices.size(0) > 0:
                    edge_index_sample = edge_indices.t()  # (2, num_edges)
                else:
                    # 如果没有边，创建空的edge_index
                    edge_index_sample = torch.empty((2, 0), dtype=torch.long, device=args.device)

                # 保存重构的图数据
                recon_graphs_list.append({
                    'x': x_sample_onehot,
                    'edge_index': edge_index_sample,
                    'num_nodes': num_nodes
                })
                recon_batch_indices.extend([b] * num_nodes)

            # 合并所有重构的图为一个大batch
            if len(recon_graphs_list) > 0:
                # 拼接所有节点特征
                recon_x_all = torch.cat([g['x'] for g in recon_graphs_list], dim=0)

                # 拼接所有edge_index，注意需要偏移节点索引
                recon_edge_indices = []
                node_offset = 0
                for g in recon_graphs_list:
                    if g['edge_index'].size(1) > 0:
                        edge_index_offset = g['edge_index'] + node_offset
                        recon_edge_indices.append(edge_index_offset)
                    node_offset += g['num_nodes']

                if len(recon_edge_indices) > 0:
                    recon_edge_index_all = torch.cat(recon_edge_indices, dim=1)
                else:
                    recon_edge_index_all = torch.empty((2, 0), dtype=torch.long, device=args.device)

                # 创建batch索引
                recon_batch = torch.tensor(recon_batch_indices, dtype=torch.long, device=args.device)

                # 使用GNN进行预测
                pred_logits_recon = gnn.get_pred(recon_x_all, recon_edge_index_all, recon_batch)[0]

                # 计算预测损失: 希望重构图被预测为y_cf
                # 只对成功重构的图计算损失
                valid_indices = torch.tensor([g_idx for g_idx, g in enumerate(recon_graphs_list)],
                                             dtype=torch.long, device=args.device)
                y_cf_valid = tgt_pred[valid_indices]

                loss_pred = F.cross_entropy(pred_logits_recon, y_cf_valid.long(), reduction='mean')
            else:
                # 如果没有有效的重构图，损失为0
                loss_pred = torch.tensor(0.0, device=args.device)

            # 3.5 总损失
            # 权重可以调整
            alpha_recon_x = 1.0
            alpha_recon_adj = 1.0
            alpha_kl = 0.01  # KL散度权重通常较小
            alpha_pred = 0.1

            total_loss = (
                alpha_recon_x * loss_recon_x +
                alpha_recon_adj * loss_recon_adj +
                alpha_kl * kl_loss +
                alpha_pred * loss_pred
            )

            # 4. 反向传播
            total_loss.backward()

            # 梯度裁剪 (可选)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # 5. 记录损失
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon_x'] += loss_recon_x.item()
            epoch_losses['recon_adj'] += loss_recon_adj.item()
            epoch_losses['kl'] += kl_loss.item()
            epoch_losses['pred'] += loss_pred.item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'recon_x': f'{loss_recon_x.item():.4f}',
                'recon_adj': f'{loss_recon_adj.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })

        # 计算epoch平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            losses[key].append(epoch_losses[key])

        # 打印epoch总结
        epoch_summary = f"\nEpoch {epoch+1}/{epochs} Summary:"
        print(epoch_summary)
        print(f"  Total Loss: {epoch_losses['total']:.4f}")
        print(f"  Recon X Loss: {epoch_losses['recon_x']:.4f}")
        print(f"  Recon Adj Loss: {epoch_losses['recon_adj']:.4f}")
        print(f"  KL Loss: {epoch_losses['kl']:.4f}")
        print(f"  Pred Loss: {epoch_losses['pred']:.4f}")

        # 写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{epochs}:\n")
            f.write(f"  Total Loss:     {epoch_losses['total']:.6f}\n")
            f.write(f"  Recon X Loss:   {epoch_losses['recon_x']:.6f}\n")
            f.write(f"  Recon Adj Loss: {epoch_losses['recon_adj']:.6f}\n")
            f.write(f"  KL Loss:        {epoch_losses['kl']:.6f}\n")
            f.write(f"  Pred Loss:      {epoch_losses['pred']:.6f}\n")

        # 保存最佳模型
        is_best = False
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            best_epoch = epoch + 1
            is_best = True
            torch.save(model.state_dict(), f'param/myexplainer_best.pt')
            print(f"  *** Saved best model with loss {best_loss:.4f} ***")

            # 记录到日志
            with open(log_file, 'a') as f:
                f.write(f"  >>> Best model saved! (loss: {best_loss:.6f})\n")

        with open(log_file, 'a') as f:
            f.write("\n")

        # 定期保存checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f'param/myexplainer_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total'],
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

            # 记录到日志
            with open(log_file, 'a') as f:
                f.write(f"  Checkpoint saved: {checkpoint_path}\n\n")

    # 训练结束，写入总结
    end_time = datetime.now()

    with open(log_file, 'a') as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Training Summary:\n")
        f.write("=" * 80 + "\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs: {epochs}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Loss: {best_loss:.6f}\n")
        f.write("\n")

        f.write("Final Loss Values:\n")
        f.write(f"  Total Loss:     {losses['total'][-1]:.6f}\n")
        f.write(f"  Recon X Loss:   {losses['recon_x'][-1]:.6f}\n")
        f.write(f"  Recon Adj Loss: {losses['recon_adj'][-1]:.6f}\n")
        f.write(f"  KL Loss:        {losses['kl'][-1]:.6f}\n")
        f.write(f"  Pred Loss:      {losses['pred'][-1]:.6f}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

    # 加载最佳模型
    model.load_state_dict(torch.load(f'param/myexplainer_best.pt'))
    print("\nTraining completed! Loaded best model.")
    print(f"Best model from epoch {best_epoch} with loss {best_loss:.4f}")
    print(f"Training log saved to: {log_file}")

    return model, losses


def evaluate_myexplainer(args, model, gnn, val_loader):
    """
    评估MyExplainer模型 - 类似测试阶段，只需要原图即可生成反事实解释

    评估流程：
    1. 对验证集中的每个图，使用GNN获取原始预测
    2. 自动翻转预测标签作为反事实目标
    3. 使用MyExplainer生成反事实解释图
    4. 将生成的图转换回sparse格式，通过GNN预测
    5. 计算成功率（生成图被GNN预测为反事实标签）

    参数:
        args: 参数对象
        model: 训练好的MyExplainer模型
        gnn: 预训练的GNN分类器
        val_loader: 验证数据的DataLoader (原始数据，非配对)

    返回:
        metrics: 评估指标字典
    """
    model.eval()
    gnn.eval()

    total_samples = 0
    successful_cf = 0  # 成功生成反事实的数量
    total_recon_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            batch = batch.to(args.device)
            batch_size = batch.num_graphs

            # 1. 使用GNN获取原始预测
            ori_pred_logits = gnn.get_pred(batch.x, batch.edge_index, batch.batch)[0]
            ori_pred = ori_pred_logits.argmax(dim=1)  # (batch_size,)

            # 2. 反事实标签：翻转预测（0->1, 1->0）
            cf_pred = 1 - ori_pred  # (batch_size,)

            # 3. 转换为dense格式
            ori_x, ori_mask = to_dense_batch(batch.x, batch.batch,
                                             max_num_nodes=args.max_num_nodes)
            ori_adj = to_dense_adj(batch.edge_index, batch.batch,
                                   max_num_nodes=args.max_num_nodes)

            # 4. 准备标签
            y_cf = cf_pred.float().unsqueeze(1)  # Target label (counterfactual)
            y = ori_pred.float().unsqueeze(1)     # Original label

            # 5. 前向传播 - 生成反事实图
            # 使用原始图作为tgt_graph的占位符
            outputs = model(
                features=ori_x,
                adj=ori_adj,
                y_cf=y_cf,
                features_tgt=ori_x,  # 占位符
                adj_tgt=ori_adj,      # 占位符
                y=y
            )

            x_recon = outputs['x_recon']
            adj_recon = outputs['adj_recon']

            # 6. Reshape重构结果
            x_recon_reshaped = x_recon.view(batch_size, args.max_num_nodes, args.x_dim)
            adj_recon_reshaped = adj_recon.view(batch_size, args.max_num_nodes, args.max_num_nodes)

            # 7. 计算重构损失（可选，用于监控）
            loss_x = F.mse_loss(x_recon_reshaped, ori_x, reduction='none')
            loss_x = (loss_x * ori_mask.unsqueeze(-1)).sum() / ori_mask.sum()

            if ori_adj.dim() == 4:
                ori_adj_binary = (ori_adj.sum(dim=-1) > 0).float()
            else:
                ori_adj_binary = ori_adj

            loss_adj = F.binary_cross_entropy(adj_recon_reshaped, ori_adj_binary, reduction='none')
            adj_mask = ori_mask.unsqueeze(-1) * ori_mask.unsqueeze(-2)
            loss_adj = (loss_adj * adj_mask).sum() / adj_mask.sum()

            total_recon_loss += (loss_x + loss_adj).item()

            # 8. 将重构的dense图转换为sparse格式
            recon_graphs_list = []
            recon_batch_indices = []

            for b in range(batch_size):
                num_nodes = ori_mask[b].sum().item()
                if num_nodes == 0:
                    continue

                # 节点特征: 离散化为one-hot
                x_sample = x_recon_reshaped[b, :num_nodes, :]
                x_sample_onehot = torch.zeros_like(x_sample)
                atom_indices = torch.argmax(x_sample, dim=-1)
                x_sample_onehot.scatter_(1, atom_indices.unsqueeze(-1), 1.0)

                # 邻接矩阵: 阈值化
                adj_sample = adj_recon_reshaped[b, :num_nodes, :num_nodes]
                edge_threshold = 0.5
                edge_indices = (adj_sample > edge_threshold).nonzero(as_tuple=False)

                if edge_indices.size(0) > 0:
                    edge_index_sample = edge_indices.t()
                else:
                    edge_index_sample = torch.empty((2, 0), dtype=torch.long, device=args.device)

                recon_graphs_list.append({
                    'x': x_sample_onehot,
                    'edge_index': edge_index_sample,
                    'num_nodes': num_nodes
                })
                recon_batch_indices.extend([b] * num_nodes)

            # 9. 合并所有重构的图并用GNN预测
            if len(recon_graphs_list) > 0:
                recon_x_all = torch.cat([g['x'] for g in recon_graphs_list], dim=0)

                recon_edge_indices = []
                node_offset = 0
                for g in recon_graphs_list:
                    if g['edge_index'].size(1) > 0:
                        edge_index_offset = g['edge_index'] + node_offset
                        recon_edge_indices.append(edge_index_offset)
                    node_offset += g['num_nodes']

                if len(recon_edge_indices) > 0:
                    recon_edge_index_all = torch.cat(recon_edge_indices, dim=1)
                else:
                    recon_edge_index_all = torch.empty((2, 0), dtype=torch.long, device=args.device)

                recon_batch = torch.tensor(recon_batch_indices, dtype=torch.long, device=args.device)

                # 使用GNN预测重构图
                try:
                    pred_logits_recon = gnn.get_pred(recon_x_all, recon_edge_index_all, recon_batch)[0]
                    pred_labels_recon = pred_logits_recon.argmax(dim=1)

                    # 检查成功率
                    valid_indices = torch.arange(len(recon_graphs_list), device=args.device)
                    cf_pred_valid = cf_pred[valid_indices]

                    successful_cf += (pred_labels_recon == cf_pred_valid).sum().item()
                    total_samples += len(recon_graphs_list)
                except Exception as e:
                    # 如果预测失败，记录但继续
                    print(f"Warning: Failed to predict reconstructed graphs: {e}")
                    total_samples += len(recon_graphs_list)

            num_batches += 1

    # 计算指标
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
    success_rate = successful_cf / total_samples if total_samples > 0 else 0.0

    metrics = {
        'avg_recon_loss': avg_recon_loss,
        'success_rate': success_rate,
        'successful_cf': successful_cf,
        'total_samples': total_samples
    }

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"  Total samples: {total_samples}")
    print(f"  Successful counterfactuals: {successful_cf}")
    print(f"  Success rate: {success_rate*100:.2f}%")
    print(f"  Average reconstruction loss: {avg_recon_loss:.4f}")
    print(f"{'='*60}\n")

    return metrics








def train_myexplainer_with_subgraph(args, model, gnn, train_loader, optimizer, epochs=200, log_dir='logs'):
    """
    使用频繁子图掩码训练MyExplainer模型

    训练流程:
    1. 对每个图，提取掩码指定的子图部分（若无掩码则使用全图）
    2. 将子图送入VGAE进行重构
    3. 将重构的子图拼回到原图中，形成完整的反事实图
    4. 计算损失：重构损失 + KL散度 + 预测损失

    参数:
        args: 参数对象
        model: MyExplainer模型
        gnn: 预训练的GNN分类器
        train_loader: GraphTrainData的DataLoader (使用train_collate_fn)
        optimizer: 优化器
        epochs: 训练轮数
        log_dir: 日志保存目录

    返回:
        model: 训练好的模型
        losses: 损失历史字典
    """



    model.train()
    gnn.eval()

    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_subgraph_{timestamp}.txt")

    # 写入日志头部
    with open(log_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MyExplainer Subgraph Training Log\n")
        f.write("=" * 80 + "\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 记录超参数
        f.write("-" * 80 + "\n")
        f.write("Hyperparameters:\n")
        f.write("-" * 80 + "\n")
        args_dict = vars(args)
        for key in sorted(args_dict.keys()):
            value = args_dict[key]
            if key == 'device':
                f.write(f"  {key}: {str(value)}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("Training Configuration:\n")
        f.write("-" * 80 + "\n")
        f.write(f"  Total Epochs: {epochs}\n")
        f.write(f"  Max Num Nodes: {args.max_num_nodes}\n")
        f.write(f"  Optimizer: {type(optimizer).__name__}\n")
        f.write(f"  Learning Rate: {optimizer.param_groups[0]['lr']}\n")
        f.write(f"  Weight Decay: {optimizer.param_groups[0]['weight_decay']}\n")
        #将损失函数的权重写入日志
        f.write(f"  Loss Weights: {args.loss_proportion}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("Training Progress:\n")
        f.write("=" * 80 + "\n\n")

    print(f"Training with subgraph masks, max_num_nodes={args.max_num_nodes}")
    print(f"Log file: {log_file}")

    # 记录损失历史
    losses = {
        'total': [],
        'recon_x': [],
        'recon_adj': [],
        'diversity': [],
        'distribution': [],
        'kl': [],
        'pred': []
    }

    best_loss = float('inf')
    best_epoch = 0

    # 可视化配置：选择训练集中第一个batch的第一个图作为示例图，并在每个epoch结束时可视化其变换
    viz_graph_idx = 0  # 固定选择第一个图
    viz_enabled = False  # 可切换是否启用可视化
    viz_dir = os.path.join(log_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    original_graph_smiles = None  # 将存储原始SMILES用于比较

    for epoch in range(epochs):
        epoch_losses = {
            'total': 0.0,
            'recon_x': 0.0,
            'recon_adj': 0.0,
            'diversity': 0.0,
            'distribution': 0.0,
            'kl': 0.0,
            'pred': 0.0
        }

        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')


        for batch_idx, batch in enumerate(progress_bar):
            batch_losses = {"recon": 0.0, "kl": 0.0, "pred": 0.0}
            # 1. 准备数据
            graphs_batch = batch['graphs'].to(args.device)
            # freq_sub_masks = batch['freq_sub_masks']  # list of edge masks
            batch_size = graphs_batch.num_graphs


            # 2. 使用GNN获取原始预测
            with torch.no_grad():
                ori_pred_logits = gnn.get_pred(graphs_batch.x, graphs_batch.edge_index, graphs_batch.batch)[0]
                ori_pred = ori_pred_logits.argmax(dim=1)  # (batch_size,)

            # 3. 反事实标签：翻转预测
            cf_pred = 1 - ori_pred
            y_cf = cf_pred.float().unsqueeze(1)
            y = ori_pred.float().unsqueeze(1)

            # 4. 处理每个图：提取子图 -> 准备批量数据
            optimizer.zero_grad()




            subgraph_x_list = []
            zero_x_template = torch.zeros(args.max_subgraph_nodes, args.x_dim, device=args.device)

            # 第一步：提取所有子图节点特征（不调用model）
            for b in range(batch_size):
                # 提取单个子图
                mask = (batch['subgraphs'].batch == b)
                num_nodes_tensor = mask.sum()  # 保持为张量，避免 .item() 过早转换

                subgraph_x = batch['subgraphs'].x[mask]  # 提取该子图的节点特征

                # 使用clone而不是zeros创建新张量
                subgraph_x_padded = zero_x_template.clone()
                num_nodes_i = num_nodes_tensor.item()  # 只在赋值时转换
                if num_nodes_i > 0:  # 避免空图
                    subgraph_x_padded[:num_nodes_i] = subgraph_x

                # 存储子图数据
                subgraph_x_list.append(subgraph_x_padded)

            # 第二步：堆叠所有子图节点特征
            all_subgraph_x = torch.stack(subgraph_x_list, dim=0)  # (B, max_num_nodes, x_dim)

            all_subgraph_adj = to_dense_adj(
                batch["subgraphs"].edge_index,
                batch=batch["subgraphs"].batch,
                max_num_nodes=args.max_subgraph_nodes
            )
            all_subgraph_x, all_subgraph_adj = all_subgraph_x.to(args.device), all_subgraph_adj.to(args.device)

            outputs = model(
                features=all_subgraph_x,
                adj=all_subgraph_adj,
                y_cf=y_cf
            )

            # 第三步：使用批量输出结果重构每个图
            concated_graphs = concat_graphs(args, outputs, batch)

            # 5. 计算损失
            batch_losses = compute_loss(args, outputs, batch, gnn, y_cf, concated_graphs, epoch_losses)

            # 提取总损失用于反向传播
            total_loss = batch_losses['total']

            # 6. 反向传播
            total_loss.backward()

            # 7. 梯度裁剪（可选，防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 8. 优化器更新参数
            optimizer.step()

            # 9. 累积损失到epoch_losses
            epoch_losses['total'] += batch_losses['total'].item()
            epoch_losses['recon_x'] += batch_losses['recon_x'].item()
            epoch_losses['recon_adj'] += batch_losses['recon_adj'].item()
            epoch_losses['diversity'] += batch_losses['diversity'].item()
            epoch_losses['distribution'] += batch_losses['distribution'].item()
            epoch_losses['kl'] += batch_losses['kl'].item()
            epoch_losses['pred'] += batch_losses['pred'].item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'recon_x': f'{batch_losses["recon_x"].item():.4f}',
                'recon_adj': f'{batch_losses["recon_adj"].item():.4f}',
                'diversity': f'{batch_losses["diversity"].item():.4f}',
                'distribution': f'{batch_losses["distribution"].item():.4f}',
                'kl': f'{batch_losses["kl"].item():.4f}',
                'pred': f'{batch_losses["pred"].item():.4f}',
            })

        # 计算epoch平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            losses[key].append(epoch_losses[key])

        # 可视化：每个epoch结束时，绘制示例图的原始和重构版本

        # 打印epoch总结
        print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
        print(f"  Total Loss: {epoch_losses['total']:.4f}")
        print(f"  Recon X Loss: {epoch_losses['recon_x']:.4f}")
        print(f"  Recon Adj Loss: {epoch_losses['recon_adj']:.4f}")
        print(f"  Diversity Loss: {epoch_losses['diversity']:.4f}")
        print(f"  Distribution Loss: {epoch_losses['distribution']:.4f}")
        print(f"  KL Loss: {epoch_losses['kl']:.4f}")
        print(f"  Pred Loss: {epoch_losses['pred']:.4f}")

        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{epochs}:\n")
            f.write(f"  Total Loss:     {epoch_losses['total']:.6f}\n")
            f.write(f"  Recon X Loss:   {epoch_losses['recon_x']:.6f}\n")
            f.write(f"  Recon Adj Loss: {epoch_losses['recon_adj']:.6f}\n")
            f.write(f"  Diversity Loss: {epoch_losses['diversity']:.6f}\n")
            f.write(f"  Distribution Loss: {epoch_losses['distribution']:.6f}\n")
            f.write(f"  KL Loss:        {epoch_losses['kl']:.6f}\n")
            f.write(f"  Pred Loss:      {epoch_losses['pred']:.6f}\n")

        # 保存最佳模型
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'param/myexplainer_subgraph_best.pt')
            print(f"  *** Saved best model with loss {best_loss:.4f} ***")

            with open(log_file, 'a') as f:
                f.write(f"  >>> Best model saved! (loss: {best_loss:.6f})\n")

        with open(log_file, 'a') as f:
            f.write("\n")

        # 定期保存checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = f'param/myexplainer_subgraph_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total'],
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

            with open(log_file, 'a') as f:
                f.write(f"  Checkpoint saved: {checkpoint_path}\n\n")

    # 训练结束
    end_time = datetime.now()

    with open(log_file, 'a') as f:
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("Training Summary:\n")
        f.write("=" * 80 + "\n")
        f.write(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs: {epochs}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best Loss: {best_loss:.6f}\n")
        f.write("\n")
        f.write("Final Loss Values:\n")
        f.write(f"  Total Loss:     {losses['total'][-1]:.6f}\n")
        f.write(f"  Recon X Loss:   {losses['recon_x'][-1]:.6f}\n")
        f.write(f"  Recon Adj Loss: {losses['recon_adj'][-1]:.6f}\n")
        f.write(f"  KL Loss:        {losses['kl'][-1]:.6f}\n")
        f.write(f"  Pred Loss:      {losses['pred'][-1]:.6f}\n")
        f.write("\n")
        f.write("=" * 80 + "\n")

    # 加载最佳模型
    model.load_state_dict(torch.load(f'param/myexplainer_subgraph_best.pt'))
    print("\nTraining completed! Loaded best model.")
    print(f"Best model from epoch {best_epoch} with loss {best_loss:.4f}")
    print(f"Training log saved to: {log_file}")
    if viz_enabled:
        print(f"Visualizations saved to: {viz_dir}")

    return model, losses