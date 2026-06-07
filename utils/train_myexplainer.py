import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.vis_utils import visualize_explainer_graph

def train_myexplainerV2(args, model, gnn, train_loader, eval_loader, optimizer, scheduler, epochs=30):
    # 记录损失历史
    losses = {
        'total': [],
        'recon': [],
        # 'mask': [],
        # 'edit_inside': [],
        # 'edit_outside': [],
        'cf': [],
        'kl': [],
        'val_total': []
    }

    best_val_loss = float('inf')
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
            # 'mask': 0.0,
            # 'edit_inside': 0.0,
            # 'edit_outside': 0.0,
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
                'recon': f'{loss_dict["recon"]:.4f}',
                # 'mask': f'{loss_dict["mask"]:.4f}',
                # 'edit_inside': f'{loss_dict["edit_inside"]:.4f}',
                # 'edit_outside': f'{loss_dict["edit_outside"]:.4f}',
                'cf': f'{loss_dict["cf"]:.4f}',
                'kl': f'{loss_dict["kl"]:.4f}',
            })

        # 计算epoch平均损失
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            losses[key].append(epoch_losses[key])

        model.eval()  # 切换到评估模式
        val_loss_accum = 0.0
        val_batches = 0

        with torch.no_grad():  # 不计算梯度
            for batch in eval_loader:
                origraphs = batch['graphs'].to(args.device)
                subgraphs = [g.to(args.device) for g in batch['subgraphs']]

                # 验证集也需要计算目标标签 (实时计算，因为cache里只有train的)
                ori_pred_logits, _ = gnn.get_pred(origraphs.x, origraphs.edge_index, origraphs.batch)
                ori_pred = ori_pred_logits.argmax(dim=1)
                y_desired = (1 - ori_pred).float().unsqueeze(1)

                outputs = model(origraphs, subgraphs)
                loss_dict = model.compute_loss(args, origraphs, y_desired, outputs)

                val_loss_accum += loss_dict["total"].item()
                val_batches += 1

        # 计算验证集平均 Loss
        val_epoch_loss = val_loss_accum / val_batches if val_batches > 0 else 0.0
        losses['val_total'].append(val_epoch_loss)

        scheduler.step(val_epoch_loss)


        # 打印epoch总结
        print(f"\nEpoch {epoch + 1}/{epochs} Summary:")
        print(f"  Train Total Loss: {epoch_losses['total']:.4f}")
        print(f"  Val   Total Loss: {val_epoch_loss:.4f}")
        for loss_name, loss_value in epoch_losses.items():
            # Skip the mask loss if it's commented out in the original
            if loss_name != 'mask':
                print(f"  {loss_name.replace('_', ' ').title()} Loss: {loss_value:.4f}")


        # 保存最佳模型
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), f'param/myexplainer_{args.dataset}_best.pt')
            print(f"  *** Saved Best Model (Val Loss: {best_val_loss:.4f}) ***")


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
    print(f"Best model from epoch {best_epoch} with loss {best_val_loss:.4f}")

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
    loss_types = ['total', 'recon', 'kl', 'cf','val_total']

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