from typing import Dict

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_networkx
from tqdm import tqdm
import os
from datetime import datetime

from evaluation import evaluate
from utils import compute_loss, concat_graphs
import time

import matplotlib.pyplot as plt

from utils.batch_utils import core_data_from_batch
from utils.graph_utils import process_outputs
from utils.train_utils import compute_loss_causality
from utils.vis_utils import visualize_explainer_graph


def train_myexplainer_with_subgraph(args, model, gnn, train_loader, eval_loader, optimizer, epochs=200, log_dir='logs'):
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
        'kl': [],
        'pred': [],
        # 'fid': []
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
        model.train()
        gnn.eval()
        epoch_losses = {
            'total': 0.0,
            'recon_x': 0.0,
            'recon_adj': 0.0,
            'kl': 0.0,
            'pred': 0.0,
            # 'fid': 0.0
        }

        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')


        for batch_idx, batch in enumerate(progress_bar):
            # 1. 准备数据
            graphs_batch = batch['graphs'].to(args.device)


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



            # 第一步：提取批量信息
            if args.edge_attr_dim != 0:
                all_subgraph_x, all_subgraph_adj, all_subgraph_edge_attr = core_data_from_batch(args, batch)

                outputs = model(
                    x=all_subgraph_x,
                    adj=all_subgraph_adj,
                    y_cf=y_cf,
                    edge_attr = all_subgraph_edge_attr
                )
            else:
                all_subgraph_x, all_subgraph_adj, _ = core_data_from_batch(args, batch)

                outputs = model(
                    x=all_subgraph_x,
                    adj=all_subgraph_adj,
                    y_cf=y_cf
                )

            # 第二步：使用批量输出结果重构每个图
            concated_graphs = concat_graphs(args, outputs, batch)

            # 5. 计算损失
            batch_losses = compute_loss(args, outputs, batch, gnn, y_cf, concated_graphs)

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
            epoch_losses['kl'] += batch_losses['kl'].item()
            epoch_losses['pred'] += batch_losses['pred'].item()
            # epoch_losses['fid'] += batch_losses['fid'].item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'recon_x': f'{batch_losses["recon_x"].item():.4f}',
                'recon_adj': f'{batch_losses["recon_adj"].item():.4f}',
                'kl': f'{batch_losses["kl"].item():.4f}',
                'pred': f'{batch_losses["pred"].item():.4f}',
                # 'fid': f'{batch_losses["fid"]:.4f}',
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
        print(f"  KL Loss: {epoch_losses['kl']:.4f}")
        print(f"  Pred Loss: {epoch_losses['pred']:.4f}")
        # print(f"  FID Loss: {epoch_losses['fid']:.4f}")

        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{epochs}:\n")
            f.write(f"  Total Loss:     {epoch_losses['total']:.6f}\n")
            f.write(f"  Recon X Loss:   {epoch_losses['recon_x']:.6f}\n")
            f.write(f"  Recon Adj Loss: {epoch_losses['recon_adj']:.6f}\n")
            f.write(f"  KL Loss:        {epoch_losses['kl']:.6f}\n")
            f.write(f"  Pred Loss:      {epoch_losses['pred']:.6f}\n")
            # f.write(f"  FID Loss:       {epoch_losses['fid']:.6f}\n")

        # 保存最佳模型
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'param/myexplainer_{args.dataset}_best.pt')
            print(f"  *** Saved best model with loss {best_loss:.4f} ***")

            with open(log_file, 'a') as f:
                f.write(f"  >>> Best model saved! (loss: {best_loss:.6f})\n")

        with open(log_file, 'a') as f:
            f.write("\n")

        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'param/myexplainer_{args.dataset}_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total'],
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

            with open(log_file, 'a') as f:
                f.write(f"  Checkpoint saved: {checkpoint_path}\n\n")

            # evaluation_metrics = evaluate(args, model, gnn, eval_loader)
            #
            # print(
            #     "  Validity ↑: {:.4f} (successful: {}/total: {})".format(
            #         evaluation_metrics["validity"],
            #         int(evaluation_metrics["successful"]),
            #         int(evaluation_metrics["total"]),
            #     )
            # )
            # print(
            #     "  Proximity ↓: {:.4f}".format(
            #         evaluation_metrics["proximity"]
            #     )
            # )
            # print(
            #     "  Fidelity+ ↑: {:.4f}".format(
            #         evaluation_metrics["fidelity+"]
            #     )
            # )
            # print(
            #     "  Fidelity- ↓: {:.4f}".format(
            #         evaluation_metrics["fidelity-"]
            #     )
            # )
            # print(
            #     "  Fidelity_prob ↑: {:.4f}".format(
            #         evaluation_metrics["fidelity"]
            #     )
            # )
            # print(
            #     "  Sparsity ↑: {:.4f}".format(
            #         evaluation_metrics["sparsity"]
            #     )
            # )

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
    model.load_state_dict(torch.load(f'param/myexplainer_{args.dataset}_best.pt'))
    print("\nTraining completed! Loaded best model.")
    print(f"Best model from epoch {best_epoch} with loss {best_loss:.4f}")
    print(f"Training log saved to: {log_file}")
    if viz_enabled:
        print(f"Visualizations saved to: {viz_dir}")

    # 绘制损失曲线
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_range, losses['total'], label='Total Loss', linewidth=2)
    plt.plot(epochs_range, losses['recon_x'], label='Recon X Loss')
    plt.plot(epochs_range, losses['recon_adj'], label='Recon Adj Loss')
    plt.plot(epochs_range, losses['kl'], label='KL Loss')
    plt.plot(epochs_range, losses['pred'], label='Pred Loss')
    # plt.plot(epochs_range, losses['fid'], label='FID Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(log_dir, f"loss_curves_{timestamp}.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Loss curves saved to: {loss_plot_path}")

    return model, losses

def train_myexplainer_with_causality(args, model, gnn, train_loader, eval_loader, optimizer, epochs=200, log_dir='logs'):
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
        'kl': [],
        'pred': [],
        # 'ortho': [],
    }

    best_loss = float('inf')
    best_epoch = 0

    # 可视化配置：选择训练集中第一个batch的第一个图作为示例图，并在每个epoch结束时可视化其变换
    viz_enabled = False  # 可切换是否启用可视化
    viz_dir = os.path.join(log_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    device = args.device

    for epoch in range(epochs):
        model.train()
        gnn.eval()
        epoch_losses = {
            'total': 0.0,
            'recon_x': 0.0,
            'recon_adj': 0.0,
            'kl': 0.0,
            'pred': 0.0,
            # 'ortho': 0.0,
        }

        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')


        for batch_idx, batch in enumerate(progress_bar):
            # 1. 准备数据
            graphs_batch = batch['graphs'].to(args.device)
            subgraphs_batch = batch['subgraphs'].to(args.device)


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



            # 第一步：提取批量信息
            # 使用 to_dense_batch 将 [total_nodes, F] 转换为 [batch_size, max_num_nodes, F]
            all_graph_x, _ = to_dense_batch(graphs_batch.x, graphs_batch.batch, max_num_nodes=args.max_num_nodes)
            all_graph_x = all_graph_x.to(device)  # [batch_size, max_num_nodes, x_dim]

            # 将邻接矩阵转换为密集格式 [batch_size, max_num_nodes, max_num_nodes]
            all_graph_adj = to_dense_adj(graphs_batch.edge_index, graphs_batch.batch, max_num_nodes=args.max_num_nodes).to(device)

            # 提取子图数据
            all_subgraph_x, all_subgraph_adj, _ = core_data_from_batch(args, batch)

            if args.edge_attr_dim != 0:
                # 如果有边特征，使用 to_dense_adj 的 edge_attr 参数
                all_graph_edge_attr = to_dense_adj(
                    graphs_batch.edge_index,
                    graphs_batch.batch,
                    edge_attr=graphs_batch.edge_attr,
                    max_num_nodes=args.max_num_nodes
                ).to(device)

                outputs = model(
                    x=all_graph_x,
                    adj=all_graph_adj,
                    y_cf=y_cf,
                    edge_attr=all_graph_edge_attr,
                    # x_sub=all_subgraph_x,
                    # adj_sub=all_subgraph_adj
                )
            else:
                outputs = model(
                    x=all_graph_x,
                    adj=all_graph_adj,
                    y_cf=y_cf,
                    # x_sub=all_subgraph_x,
                    # adj_sub=all_subgraph_adj
                )


            args.tmp = False
            # 第二步：使用批量输出结果中的每一个重构图
            output_graphs_batch = process_outputs(args, outputs)
            if batch_idx == 0:
                for i in range(3,4):
                    graph = output_graphs_batch.get_example(i)
                    # print(graph)
                    # print(graph.x)
                    # print(graph.edge_index)
                    G = to_networkx(graph).to_undirected()
                    pos = nx.spring_layout(G)
                    plt.title(f"图{i}")
                    nx.draw(G, pos, with_labels=True, node_color='lightgreen',node_size=50, font_size=10, font_weight='bold')

                    plt.show()
                    args.tmp = True

            # 5. 计算损失
            batch_losses = compute_loss_causality(args, outputs, batch, gnn, y_cf, output_graphs_batch)

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
            epoch_losses['kl'] += batch_losses['kl'].item()
            epoch_losses['pred'] += batch_losses['pred'].item()
            # epoch_losses['ortho'] += batch_losses['ortho'].item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'recon_x': f'{batch_losses["recon_x"].item():.4f}',
                'recon_adj': f'{batch_losses["recon_adj"].item():.4f}',
                'kl': f'{batch_losses["kl"].item():.4f}',
                'pred': f'{batch_losses["pred"].item():.4f}',
                # 'ortho': f'{batch_losses["ortho"]:.4f}',
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
        print(f"  KL Loss: {epoch_losses['kl']:.4f}")
        print(f"  Pred Loss: {epoch_losses['pred']:.4f}")
        # print(f"  FID Loss: {epoch_losses['fid']:.4f}")
        # print(f"  Ortho Loss: {epoch_losses['ortho']:.4f}")

        # 写入日志
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{epochs}:\n")
            f.write(f"  Total Loss:     {epoch_losses['total']:.6f}\n")
            f.write(f"  Recon X Loss:   {epoch_losses['recon_x']:.6f}\n")
            f.write(f"  Recon Adj Loss: {epoch_losses['recon_adj']:.6f}\n")
            f.write(f"  KL Loss:        {epoch_losses['kl']:.6f}\n")
            f.write(f"  Pred Loss:      {epoch_losses['pred']:.6f}\n")
            # f.write(f"  FID Loss:       {epoch_losses['fid']:.6f}\n")
            # f.write(f"  Ortho Loss:     {epoch_losses['ortho']:.6f}\n")

        # 保存最佳模型
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'param/myexplainer_{args.dataset}_best.pt')
            print(f"  *** Saved best model with loss {best_loss:.4f} ***")

            with open(log_file, 'a') as f:
                f.write(f"  >>> Best model saved! (loss: {best_loss:.6f})\n")

        with open(log_file, 'a') as f:
            f.write("\n")

        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'param/myexplainer_{args.dataset}_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total'],
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")

            with open(log_file, 'a') as f:
                f.write(f"  Checkpoint saved: {checkpoint_path}\n\n")

            # evaluation_metrics = evaluate(args, model, gnn, eval_loader)
            #
            # print(
            #     "  Validity ↑: {:.4f} (successful: {}/total: {})".format(
            #         evaluation_metrics["validity"],
            #         int(evaluation_metrics["successful"]),
            #         int(evaluation_metrics["total"]),
            #     )
            # )
            # print(
            #     "  Proximity ↓: {:.4f}".format(
            #         evaluation_metrics["proximity"]
            #     )
            # )
            # print(
            #     "  Fidelity+ ↑: {:.4f}".format(
            #         evaluation_metrics["fidelity+"]
            #     )
            # )
            # print(
            #     "  Fidelity- ↓: {:.4f}".format(
            #         evaluation_metrics["fidelity-"]
            #     )
            # )
            # print(
            #     "  Fidelity_prob ↑: {:.4f}".format(
            #         evaluation_metrics["fidelity"]
            #     )
            # )
            # print(
            #     "  Sparsity ↑: {:.4f}".format(
            #         evaluation_metrics["sparsity"]
            #     )
            # )

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
    model.load_state_dict(torch.load(f'param/myexplainer_{args.dataset}_best.pt'))
    print("\nTraining completed! Loaded best model.")
    print(f"Best model from epoch {best_epoch} with loss {best_loss:.4f}")
    print(f"Training log saved to: {log_file}")
    if viz_enabled:
        print(f"Visualizations saved to: {viz_dir}")

    # 绘制损失曲线
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs_range, losses['total'], label='Total Loss', linewidth=2)
    plt.plot(epochs_range, losses['recon_x'], label='Recon X Loss')
    plt.plot(epochs_range, losses['recon_adj'], label='Recon Adj Loss')
    plt.plot(epochs_range, losses['kl'], label='KL Loss')
    plt.plot(epochs_range, losses['pred'], label='Pred Loss')
    # plt.plot(epochs_range, losses['fid'], label='FID Loss')
    # plt.plot(epochs_range, losses['ortho'], label='Ortho Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(log_dir, f"loss_curves_{timestamp}.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Loss curves saved to: {loss_plot_path}")

    return model, losses





def train_myexplainerV2(args, model, gnn, train_loader, eval_loader, optimizer, epochs=30):
    # 记录损失历史
    losses = {
        'total': [],
        'recon': [],
        'mask': [],
        'edit_inside': [],
        'edit_outside': [],
        'cf': [],
        'kl': [],
    }

    best_loss = float('inf')
    best_epoch = 0


    y_desired_cache = []
    gnn.eval()
    with torch.no_grad():
        for batch in train_loader:
            origraphs = batch['graphs'].to(args.device)
            ori_pred_logits, _ = gnn.get_pred(origraphs.x, origraphs.edge_index, origraphs.batch)
            ori_pred = ori_pred_logits.argmax(dim=1)
            y_desired = (1 - ori_pred).float().unsqueeze(1)
            y_desired_cache.append(y_desired.cpu())

    for epoch in range(epochs):
        model.train()
        gnn.eval()
        epoch_losses = {
            'total': 0.0,
            'recon': 0.0,
            'mask': 0.0,
            'edit_inside': 0.0,
            'edit_outside': 0.0,
            'cf': 0.0,
            'kl': 0.0,
        }

        num_batches = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')

        for batch_idx, batch in enumerate(progress_bar):

            origraphs = batch['graphs'].to(args.device)
            subgraphs = [g.to(args.device) for g in batch['subgraphs']]

            x = origraphs.x
            edge_index = origraphs.edge_index
            batch_vec = origraphs.batch

            # ✅ 使用预计算的y_desired，确保每个epoch一致
            y_desired = y_desired_cache[batch_idx].to(args.device)
            y_hat = (1 - y_desired).float()  # 原始预测 = 1 - 反事实标签


            # 4. 处理每个图：提取子图 -> 准备批量数据
            optimizer.zero_grad()

            # outputs = model(
            #     x=x,
            #     edge_index=edge_index,
            #     batch=batch_vec,
            #     y_desired=y_desired.view(-1, 1),
            #     edge_attr=getattr(origraphs, 'edge_attr', None)
            # )
            outputs = model(origraphs, subgraphs)
            loss_dict = model.compute_loss(args, origraphs, y_desired, outputs)

            if batch_idx in [0,1,2,3,4]:
                visualize_explainer_graph(origraphs, y_desired, outputs)



            # loss_dict = model.compute_loss(args, origraphs, subgraphs, gnn, y_desired, outputs)


            loss = loss_dict["total"]



            loss.backward()

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm().item()
            #         print(f"{name}: grad_norm = {grad_norm:.6f}")
            #
            # total_norm = 0.0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f"Total gradient norm: {total_norm:.4f}")

            # 7. 梯度裁剪（可选，防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 8. 优化器更新参数
            optimizer.step()

            # 9. 累积损失到epoch_losses
            total_loss = loss_dict["total"]
            for key in loss_dict:
                epoch_losses[key] += loss_dict[key].item()
            num_batches += 1

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                # 'recon': f'{loss_dict["recon"]:.4f}',
                # 'mask': f'{loss_dict["mask"]:.4f}',
                # 'edit_inside': f'{loss_dict["edit_inside"]:.4f}',
                # 'edit_outside': f'{loss_dict["edit_outside"]:.4f}',
                'cf': f'{loss_dict["cf"]:.4f}',
                # 'kl': f'{loss_dict["kl"]:.4f}',
            })

        # 计算epoch平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            losses[key].append(epoch_losses[key])

        # 可视化：每个epoch结束时，绘制示例图的原始和重构版本

        # 打印epoch总结
        print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
        for loss_name, loss_value in epoch_losses.items():
            # Skip the mask loss if it's commented out in the original
            if loss_name != 'mask':
                print(f"  {loss_name.replace('_', ' ').title()} Loss: {loss_value:.4f}")


        # 保存最佳模型
        if epoch_losses['total'] < best_loss:
            best_loss = epoch_losses['total']
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'param/myexplainer_{args.dataset}_best.pt')
            print(f"  *** Saved best model with loss {best_loss:.4f} ***")


        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'param/myexplainer_{args.dataset}_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_losses['total'],
            }, checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")


    # 加载最佳模型
    model.load_state_dict(torch.load(f'param/myexplainer_{args.dataset}_best.pt'))
    print("\nTraining completed! Loaded best model.")
    print(f"Best model from epoch {best_epoch} with loss {best_loss:.4f}")

    # 绘制损失曲线
    # epochs_range = range(1, epochs + 1)
    # plt.figure(figsize=(12, 8))
    # plt.plot(epochs_range, losses['total'], label='Total Loss', linewidth=2)
    # plt.plot(epochs_range, losses['recon'], label='Recon Loss')
    # plt.plot(epochs_range, losses['kl'], label='KL Loss')
    # plt.plot(epochs_range, losses['cf'], label='cf Loss')
    # plt.plot(epochs_range, losses['mask'], label='mask Loss')
    # plt.plot(epochs_range, losses['edit_inside'], label='edit_inside Loss')
    # plt.plot(epochs_range, losses['edit_outside'], label='edit_outside Loss')
    #
    #
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss Curves')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    # plt.close()

    epochs_range = range(1, epochs + 1)
    loss_types = ['total', 'recon', 'kl', 'cf', 'edit_inside', 'edit_outside']

    for loss_type in loss_types:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs_range, losses[loss_type], linewidth=2, color='blue')
        plt.xlabel('Epoch')
        plt.ylabel(f'{loss_type.capitalize()} Loss')
        plt.title(f'Training {loss_type.capitalize()} Loss Curve')
        plt.grid(True)
        plt.show()
        plt.close()
        formatted_string = ", ".join(f"{x:.2f}" for x in losses[loss_type])
        print(f"{loss_type}: [{formatted_string}]")

    return model, losses


def scale_losses_by_grad_norm(
    losses: Dict[str, torch.Tensor],
    model: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    根据每个损失项的梯度范数对损失进行比例缩放。

    参数:
        losses (dict[str, torch.Tensor]): 包含多个损失项的字典，
            形如 {"loss1": loss1, "loss2": loss2, ...}。
        model (torch.nn.Module): 用于计算梯度的模型。

    返回:
        dict[str, torch.Tensor]: 缩放后的损失项字典。
    """
    # 只取需要梯度的参数
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("model 没有任何 requires_grad=True 的参数，无法计算梯度范数。")

    grad_norms: Dict[str, torch.Tensor] = {}

    # 1. 计算每个 loss 相对于模型参数的梯度范数
    for name, loss in losses.items():
        if name == 'total':
            continue

        if not isinstance(loss, torch.Tensor):
            raise TypeError(f"losses['{name}'] 不是 torch.Tensor。")

        if loss.grad_fn is None:
            # 一般说明 loss 被 .item() 过，或者在 no_grad() 环境里
            raise RuntimeError(f"losses['{name}'] 无 grad_fn，无法求梯度。")

        # autograd.grad 不会把梯度写入 param.grad，副作用更小
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,   # 多个 loss 共用计算图时需要保留
            allow_unused=True
        )

        # 计算二范数
        total_sq = torch.zeros([], device=loss.device)
        for g in grads:
            if g is not None:
                total_sq = total_sq + g.pow(2).sum()

        grad_norm = torch.sqrt(total_sq + 1e-12)
        grad_norms[name] = grad_norm

    # 2. 根据梯度范数计算缩放系数
    # 这里采用“梯度大的 loss 权重更小”的方式来平衡各项：
    #   weight_i = mean_norm / grad_norm_i
    norms = torch.stack(list(grad_norms.values()))
    mean_norm = norms.mean()

    scaled_losses: Dict[str, torch.Tensor] = {}
    for name, loss in losses.items():
        gn = grad_norms[name]
        weight = mean_norm / (gn + 1e-12)   # 梯度大 → weight 小
        scaled_losses[name] = loss * weight

    return scaled_losses