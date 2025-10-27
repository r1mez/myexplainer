import time
import collections

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from datasets import Mutagenicity
from .graphedx import GRAPHEDX_xor_on_edge
from .conf import get_conf, get_gmn_conf
import os

from pathlib import Path
# 定义GraphData命名元组
GraphData = collections.namedtuple(
    "GraphData",
    ["from_idx", "to_idx", "node_features", "edge_features", "n_graphs", "graph_idx"],
)

# 设置默认配置和设备

conf = get_conf()

gmn_config = get_gmn_conf()

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

current_dir = Path(__file__).parent
model_path = os.path.join(current_dir, 'pretrained', 'GRAPHEDX.pt')

model = GRAPHEDX_xor_on_edge(conf, gmn_config).to(device)
loaded = torch.load(model_path, map_location=torch.device(device))
if isinstance(loaded, torch.nn.Module):
    model = loaded
else:
    model.load_state_dict(loaded['model_state_dict'])
model.eval()

def edge_list_to_adj_matrix(from_nodes, to_nodes, num_nodes=None, device='cpu'):
    """
    Convert edge list to a 3D adjacency matrix as a torch.Tensor with shape [1, num_nodes, num_nodes].

    Args:
        from_nodes (torch.Tensor): Source nodes of edges.
        to_nodes (torch.Tensor): Target nodes of edges.
        num_nodes (int, optional): Number of nodes in the graph. If None, inferred from max node index.
        device (str, optional): Device to place the tensor on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        torch.Tensor: Adjacency matrix of shape [1, num_nodes, num_nodes] as a torch.Tensor.
    """
    # Ensure inputs are torch tensors
    # from_nodes = torch.as_tensor(from_nodes, dtype=torch.long, device=device)
    # to_nodes = torch.as_tensor(to_nodes, dtype=torch.long, device=device)

    # Infer number of nodes if not provided
    if num_nodes is None:
        num_nodes = max(from_nodes.max(), to_nodes.max()) + 1

    # Initialize 2D adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes))

    # Set edges in adjacency matrix
    adj_matrix[from_nodes, to_nodes] = 1

    # Reshape to 3D tensor with shape [1, num_nodes, num_nodes]
    adj_matrix = adj_matrix.unsqueeze(0)

    return adj_matrix

def compute_ged(
        edge_index_A, edge_index_B, device=device
):
    a_from = edge_index_A[0]
    a_to = edge_index_A[1]
    b_from = edge_index_B[0]
    b_to = edge_index_B[1]
    query_adj = edge_list_to_adj_matrix(a_from, a_to, device=device)
    target_adj = edge_list_to_adj_matrix(b_from, b_to, device=device)

    # 数据预处理
    a_from = a_from.to(device)
    a_to = a_to.to(device)
    b_from = b_from.to(device)
    b_to = b_to.to(device)
    query_adj = query_adj.to(device)
    target_adj = target_adj.to(device)

    num_nodes_a = len(torch.cat([a_from, a_to]).unique())
    num_nodes_b = len(torch.cat([b_from, b_to]).unique())
    num_edges_a = len(a_from)
    num_edges_b = len(b_from)
    nodes_diff = abs(num_nodes_a - num_nodes_b)
    print(f"节点数差：{nodes_diff}")
    edges_diff = abs(num_edges_a - num_edges_b)
    print(f"边数差：{edges_diff}")
    if nodes_diff > 10 or edges_diff > 10:
        print("节点或边数差异过大，可能导致计算不准确。请检查输入数据。")
        return float('inf')

    from_idx = torch.cat((a_from, b_from), dim=-1)
    to_idx = torch.cat((a_to, b_to), dim=-1)
    node_features = torch.ones((num_nodes_a + num_nodes_b, 1), device=device)
    edge_features = torch.ones((num_edges_a + num_edges_b, 1), device=device)
    graph_idx = torch.tensor([0] * num_nodes_a + [1] * num_nodes_b, device=device)
    n_graphs = 2

    batch_data = GraphData(
        from_idx=from_idx,
        to_idx=to_idx,
        node_features=node_features,
        edge_features=edge_features,
        n_graphs=n_graphs,
        graph_idx=graph_idx
    )
    batch_data_sizes = torch.tensor([num_nodes_a, num_nodes_b], device=device)

    # 推理

    print("开始计算GED...")
    with torch.no_grad():
        ged = model(batch_data, batch_data_sizes, query_adj, target_adj)
    print(ged)
    print("计算GED完成。")

    return ged.item()


# 示例用法
if __name__ == "__main__":
    model_path = 'pretrained/GRAPHEDX.pt'
    dataset = Mutagenicity("../data/mutag", mode="training")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    # 初始化对称矩阵 (188个图 -> 188x188)
    ged_matrix = np.zeros((len(dataset), len(dataset)), dtype=float)

    # 计算结果存储示例（i<j对称填充）
    for i in range(200,len(dataset)):
        for j in range(i + 1, len(dataset)):
            try:
                # print(dataset[i]["edge_index"], dataset[j]["edge_index"])
                ged = compute_ged(dataset[i]["edge_index"], dataset[j]["edge_index"])  # 自定义GED计算函数
            except:
                ged = float('inf')
            ged_matrix[i, j] = ged
            ged_matrix[j, i] = ged  # 利用对称性
            print(f'图对[{i}][{j}]计算完成:{ged}')

    # 保存（高效二进制）
    np.save('mutag_ged_matrix.npy', ged_matrix)

    # 加载使用
    loaded_matrix = np.load('mutag_ged_matrix.npy')
    print(loaded_matrix[10, 20])  # 秒级访问任意图对距离
    # 图A的边数据
    # edge_index_A = torch.tensor([[ 0,  1,  0,  2,  0,  3,  1,  4,  1,  5,  2,  6,  3,  7,  4,  8,  5,  9,
    #       6, 10,  6, 11,  7, 12,  8, 13, 10, 14, 11, 15, 13, 16, 16, 17, 16, 18,
    #       7, 10,  9, 13, 14, 15,  3, 19,  4, 20,  5, 21,  8, 22,  9, 23, 12, 24,
    #      12, 25, 15, 26],
    #     [ 1,  0,  2,  0,  3,  0,  4,  1,  5,  1,  6,  2,  7,  3,  8,  4,  9,  5,
    #      10,  6, 11,  6, 12,  7, 13,  8, 14, 10, 15, 11, 16, 13, 17, 16, 18, 16,
    #      10,  7, 13,  9, 15, 14, 19,  3, 20,  4, 21,  5, 22,  8, 23,  9, 24, 12,
    #      25, 12, 26, 15]])
    # edge_index_B = torch.tensor([[   0, 1,  0,  2,  0,  3,  1,  4,  1,  5,  2,  6,  3,  7,  4,  8,  5,  9,
    #       6, 10,  6, 11,  7, 12,  8, 13, 10, 14, 11, 15, 13, 16, 16, 17, 16, 18,
    #       7, 10,  9, 13, 14, 15,  3, 19,  4, 20,  5, 21,  8, 22,  9, 23, 12, 24,
    #      12, 25, 15, 26],
    #                                 [ 1,  0,  2,  0,  3,  0,  4,  1,  5,  1,  6,  2,  7,  3,  8,  4,  9,  5,
    #      10,  6, 11,  6, 12,  7, 13,  8, 14, 10, 15, 11, 16, 13, 17, 16, 18, 16,
    #      10,  7, 13,  9, 15, 14, 19,  3, 20,  4, 21,  5, 22,  8, 23,  9, 24, 12,
    #      25, 12, 26, 15]])
    #
    # edge_index_A, edge_index_B = edge_index_B, edge_index_A
    #
    # ged_value = compute_ged(
    #     edge_index_A, edge_index_B
    # )
    # print(f"Predicted GED: {ged_value}")
