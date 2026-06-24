"""Evaluation utilities for MyExplainer.

Currently this module provides a validity metric implementation which is
responsible for measuring the proportion of generated counterfactual graphs
that obtain the desired labels when evaluated by a pre-trained GNN model.
"""

import torch
from tqdm import tqdm

from utils.output_to_batch import output_to_batch
from eval.metrics import proximity as compute_proximity, fidelity as compute_fidelity_prob, sparsity as compute_sparsity
import torch.nn.functional as F

from utils.vis_utils import visualize_explainer_graph


def evaluate(config, model, gnn, data_loader):
    model.eval()
    gnn.eval()


    y_desired_all = []
    ori_prob_all = []
    with torch.no_grad():
        for batch in data_loader:
            origraphs = batch['graphs'].to(config.device)
            _, ori_pred_logits = gnn.get_pred(origraphs.x, origraphs.edge_index, origraphs.batch)
            ori_prob = F.softmax(ori_pred_logits, dim=1)
            ori_pred = ori_pred_logits.argmax(dim=1)
            y_desired = (1 - ori_pred).float().unsqueeze(1)
            y_desired_all.append(y_desired.cpu())
            ori_prob_all.append(ori_prob.cpu())
    device = config.device



    proximity = 0.0
    valid_cf = 0
    fidel_sum = 0.00
    sparsity_sum = 0.00


    total = data_loader.dataset.__len__()
    num_batches = 0  # 添加batch计数

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Evaluating:")):
            origraphs = batch['graphs'].to(config.device)
            subgraphs = batch['subgraphs']

            x = origraphs.x
            edge_index = origraphs.edge_index
            batch_vec = origraphs.batch

            # ✅ 使用预计算的y_desired，确保每个epoch一致
            y_desired = y_desired_all[batch_idx].to(config.device)
            y_hat = (1 - y_desired).float()  # 原始预测 = 1 - 反事实标签


            outputs = model(
                graphs=origraphs,
                subgraphs=subgraphs
            )

            if getattr(config, 'visualize', False):
                visualize_explainer_graph(origraphs, y_desired, outputs, dataset_name=config.dataset)

            cf_graphs = output_to_batch(origraphs, outputs)

            valid_cf += count_valid(y_desired, cf_graphs, gnn)
            proximity += compute_proximity(config, cf_graphs, origraphs)
            fidel_sum += compute_fidelity_prob(config, origraphs, cf_graphs, ori_prob_all[batch_idx], gnn)
            sparsity_sum += compute_sparsity(config, origraphs, cf_graphs)


            num_batches += 1

    validity = valid_cf / total if total > 0 else 0.0
    sparsity = sparsity_sum / total if total > 0 else 0.0
    avg_proximity = proximity / total if total > 0 else 0.0
    fidelity = fidel_sum / total if total > 0 else 0.0



    return {
        "validity": validity,
        "proximity": avg_proximity,  # 返回平均值
        "fidelity": fidelity,
        "sparsity": sparsity,
        "successful": valid_cf,
        "total": total,
    }


def count_valid(target_lables, cf_graphs, gnn):
    gnn.eval()

    pred_logits_cf = gnn.get_pred(cf_graphs.x, cf_graphs.edge_index, cf_graphs.batch)[0]
    pred_labels_cf = pred_logits_cf.argmax(dim=1).view(-1,1)

    flipped_lables = (pred_labels_cf == target_lables).sum().item()

    return flipped_lables

