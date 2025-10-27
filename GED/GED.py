# import time
# import collections
#
# from graphedx import GRAPHEDX_xor_on_edge
# from ged_utils.utils import *
#
# from conf import get_conf, get_gmn_conf
#
# model_path = 'pretrained/GRAPHEDX_xor_on_edge_mutagenicity_1.pt'
#
# conf = get_conf()
# gmn_config = get_gmn_conf()
#
# # set_seed(conf.training.seed)
# start = time.time()
# model = GRAPHEDX_xor_on_edge(conf, gmn_config).to('cuda:0' if torch.cuda.is_available() else 'cpu')
# end = time.time()
# print("Model initialization time:", end - start)
# start = time.time()
# loaded = torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
# if isinstance(loaded, torch.nn.Module):
#     model = loaded
# else:
#     model.load_state_dict(loaded['model_state_dict'])
#
# model.eval()
#
# # 输入数据构造
# GraphData = collections.namedtuple(
#     "GraphData",
#     ["from_idx", "to_idx", "node_features", "edge_features", "n_graphs", "graph_idx"],
# )
# a_from = torch.tensor([ 0,  1,  0,  2,  0,  3,  1,  4,  1,  5,  2,  6,  3,  7,  4,  8,  4,  9,
#           5, 10,  6, 11,  8, 12, 11, 13, 12, 14,  6,  8,  7, 10, 13, 14,  3, 15,
#           4, 16,  5, 17,  7, 18,  9, 19, 10, 20, 11, 21, 12, 22, 13, 23, 14, 24], device = 'cuda:0')
# a_to = torch.tensor([ 1,  0,  2,  0,  3,  0,  4,  1,  5,  1,  6,  2,  7,  3,  8,  4,  9,  4,
#          10,  5, 11,  6, 12,  8, 13, 11, 14, 12,  8,  6, 10,  7, 14, 13, 15,  3,
#          16,  4, 17,  5, 18,  7, 19,  9, 20, 10, 21, 11, 22, 12, 23, 13, 24, 14], device = 'cuda:0')
# # b_from = a_from
# # b_to = a_to
# b_from = torch.tensor([ 0,  1,  0,  2,  0,  3,  0,  4,  1,  5,  1,  6,  3,  7,  3,  8,  4,  9,
#           4, 10,  5, 11,  5, 12,  9, 13,  1,  2,  5,  7,  6, 14,  6, 15,  6, 16,
#          11, 17, 11, 18, 11, 19, 12, 20, 13, 21, 13, 22, 13, 23], device = 'cuda:0')
# b_to = torch.tensor([ 1,  0,  2,  0,  3,  0,  4,  0,  5,  1,  6,  1,  7,  3,  8,  3,  9,  4,
#          10,  4, 11,  5, 12,  5, 13,  9,  2,  1,  7,  5, 14,  6, 15,  6, 16,  6,
#          17, 11, 18, 11, 19, 11, 20, 12, 21, 13, 22, 13, 23, 13], device = 'cuda:0')
# num_nodes_a = len(torch.cat([a_from, a_to]).unique())
# num_nodes_b = len(torch.cat([b_from, b_to]).unique())
# num_edges_a = len(a_from)
# num_edges_b = len(b_from)
# from_idx = torch.cat((a_from, b_from), -1)
# to_idx = torch.cat((a_to, b_to), -1)
# node_features = torch.ones((num_nodes_a + num_nodes_b, 1), device='cuda:0')
# edge_features = torch.ones((num_edges_a + num_edges_b, 1), device='cuda:0')
# graph_idx = torch.tensor([0 for i in range(num_nodes_a)] + [1 for i in range(num_nodes_b)], device='cuda:0')
# n_graphs = 2
# batch_data = GraphData(
#     from_idx=from_idx,
#     to_idx=to_idx,
#     node_features=node_features,
#     edge_features=edge_features,
#     n_graphs=n_graphs,
#     graph_idx=graph_idx
# )
# batch_data_sizes = torch.tensor([num_nodes_a, num_nodes_b], device='cuda:0')
# query_adj = torch.tensor([[[0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
#                           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]],device = 'cuda:0')
#
# target_adj = torch.tensor([[[0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
#                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]], device='cuda:0')
#
# # 推理
# r = num_edges_a + num_edges_b + num_nodes_a + num_nodes_b
# ged = model(batch_data, batch_data_sizes, query_adj, target_adj) / r
# end = time.time()
# print(f"Predicted GED: {ged.item()}")
# print(f"Prediction time taken: {end - start:.9f} seconds")

import time
import collections
import torch
from graphedx import GRAPHEDX_xor_on_edge
from conf import get_conf, get_gmn_conf

# 定义GraphData命名元组
GraphData = collections.namedtuple(
    "GraphData",
    ["from_idx", "to_idx", "node_features", "edge_features", "n_graphs", "graph_idx"],
)

# 设置默认配置和设备

conf = get_conf()

gmn_config = get_gmn_conf()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_path = 'pretrained/GRAPHEDX.pt'
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
    from_nodes = torch.as_tensor(from_nodes, dtype=torch.long, device=device)
    to_nodes = torch.as_tensor(to_nodes, dtype=torch.long, device=device)

    # Infer number of nodes if not provided
    if num_nodes is None:
        num_nodes = max(from_nodes.max(), to_nodes.max()) + 1

    # Initialize 2D adjacency matrix
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.long, device=device)

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
    start = time.time()

    ged = model(batch_data, batch_data_sizes, query_adj, target_adj)
    pred_time = time.time() - start

    return ged.item()


# 示例用法
if __name__ == "__main__":
    model_path = 'pretrained/GRAPHEDX.pt'

    # 图A的边数据
    edge_index_A = torch.tensor([[ 0,  1,  0,  2,  0,  3,  1,  4,  1,  5,  2,  6,  3,  7,  4,  8,  5,  9,
          6, 10,  6, 11,  7, 12,  8, 13, 10, 14, 11, 15, 13, 16, 16, 17, 16, 18,
          7, 10,  9, 13, 14, 15,  3, 19,  4, 20,  5, 21,  8, 22,  9, 23, 12, 24,
         12, 25, 15, 26],
        [ 1,  0,  2,  0,  3,  0,  4,  1,  5,  1,  6,  2,  7,  3,  8,  4,  9,  5,
         10,  6, 11,  6, 12,  7, 13,  8, 14, 10, 15, 11, 16, 13, 17, 16, 18, 16,
         10,  7, 13,  9, 15, 14, 19,  3, 20,  4, 21,  5, 22,  8, 23,  9, 24, 12,
         25, 12, 26, 15]])
    edge_index_B = torch.tensor([[   1,  0,  2,  0,  3,  1,  4,  1,  5,  2,  6,  3,  7,  4,  8,  5,  9,
          6, 10,  6, 11,  7, 12,  8, 13, 10, 14, 11, 15, 13, 16, 16, 17, 16, 18,
          7, 10,  9, 13, 14, 15,  3, 19,  4, 20,  5, 21,  8, 22,  9, 23, 12, 24,
         12, 25, 15, 26],
        [ 1,  0,  2,  0,  3,  0,  4,  1,  5,  1,  6,  2,  7,  3,  8,  4,  9,  5,
         10,  6, 11,  6, 12,  7, 13,  8, 14, 10, 15, 11, 16, 13, 17, 16, 18, 16,
         10,  7, 13,  9, 15, 14, 19,  3, 20,  4, 21,  5, 22,  8, 23,  9, 24, 12,
         25, 12, 26, 15]])

    edge_index_A, edge_index_B = edge_index_B, edge_index_A

    ged_value = compute_ged(
        edge_index_A, edge_index_B
    )
    print(f"Predicted GED: {ged_value}")
    # print(f"Model initialization time: {init_time:.9f} seconds")
    # print(f"Prediction time taken: {pred_time:.9f} seconds")