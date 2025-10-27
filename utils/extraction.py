import torch
import os
from tqdm import tqdm
from utils.dataset import get_datasets
from torch_geometric.loader import DataLoader


def extract_data(args, dataloader, model):
    all_pred_classes = []
    all_high_confidence_mask = []

    for i, data in enumerate(dataloader):
        data = data.to(args.device)
        with torch.no_grad():
            if args.task == 'gc':
                pred, _ = model.get_pred(data.x, data.edge_index, data.batch)
            else:
                # 处理其他任务类型
                pass

            # 获取每个样本的最大预测概率及其预测类别
            max_probs, predicted_classes = torch.max(pred, dim=1)

            # 记录当前批次的预测结果
            all_pred_classes.append(predicted_classes)

            # 生成高置信度样本的掩码（概率 > threshold）
            high_confidence_mask = max_probs > args.threshold
            all_high_confidence_mask.append(high_confidence_mask)

    # 合并所有批次的结果
    all_pred_classes = torch.cat(all_pred_classes)
    all_high_confidence_mask = torch.cat(all_high_confidence_mask)

    return all_pred_classes, all_high_confidence_mask




