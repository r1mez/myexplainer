"""Model output conversion: reconstructed tensors to PyG Batch."""

import torch
from torch_geometric.data import Data, Batch


def process_outputs(args, outputs):
    """
    将重构的节点特征 x_recon 和邻接矩阵 adj_recon 转换为 PyG 格式的 Batch。

    参数:
        args: 包含配置信息的参数对象
            - max_subgraph_nodes: 最大节点数
            - x_dim: 节点特征维度
            - device: 设备
        outputs: dict，模型输出，包含
            - 'x_recon': 重构的节点特征，shape [B, N*F] 或 [B, N, F]
            - 'adj_recon': 重构的邻接矩阵，shape [B, N*N] 或 [B, N, N]

    返回:
        Batch: PyG的Batch对象，包含batch_size个重构图
    """
    device = args.device
    max_num_nodes = args.max_subgraph_nodes
    x_dim = args.x_dim

    # 从outputs中提取重构结果
    x_recon = outputs['x_recon']  # [B, N*F] 或 [B, N, F]
    adj_recon = outputs['adj_recon']  # [B, N*N] 或 [B, N, N]

    # 1. 确保形状正确：将扁平化的张量reshape为 (batch_size, max_num_nodes, *)
    if x_recon.dim() == 2:
        # [B, N*F] -> [B, N, F]
        batch_size = x_recon.shape[0]
        x_recon = x_recon.view(batch_size, max_num_nodes, x_dim)
    else:
        # 已经是 [B, N, F]
        batch_size = x_recon.shape[0]

    if adj_recon.dim() == 2:
        # [B, N*N] -> [B, N, N]
        adj_recon = adj_recon.view(batch_size, max_num_nodes, max_num_nodes)
    # 否则已经是 [B, N, N]

    # 2. 为每个batch中的图创建PyG Data对象
    graphs = []
    for i in range(batch_size):
        x_i = x_recon[i]  # [N, F]
        adj_i = adj_recon[i]  # [N, N]

        # 3. 将邻接矩阵转换为edge_index
        # 设置阈值，将概率值转换为0/1（可根据需要调整阈值）
        threshold = 0.5
        mask = adj_i > threshold

        # 获取所有边的索引 [2, E]
        edge_index = torch.nonzero(mask, as_tuple=False).t().contiguous()

        # 4. 创建PyG Data对象
        graph = Data(
            x=x_i,  # [N, F]
            edge_index=edge_index,  # [2, E]
            num_nodes=max_num_nodes
        )
        graphs.append(graph)

    # 5. 将所有图合并为一个Batch
    batch = Batch.from_data_list(graphs).to(device)

    return batch
